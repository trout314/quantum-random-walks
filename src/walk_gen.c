/*
 * walk_gen.c — Generate tetrahedral walk lattice and sparse shift operators.
 *
 * Hybrid approach:
 *   Phase 1: Small BFS ball for isotropic seed (depth ~3-4)
 *   Phase 2: From each seed site, trace long R and L chains (creates new sites)
 *   Phase 3: Fill passes — trace chains through existing sites for coverage
 *   Phase 4: Restrict to dual-covered sites, extract and stitch chains
 *   Phase 5: Build sparse S_R, S_L with frame transport
 *
 * Build: clang -O2 -o walk_gen src/walk_gen.c -lm -llapack -lblas
 * Usage: ./walk_gen [seed_depth] [chain_len] [fill_passes]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

/* ========== 3D vector ========== */
typedef struct { double x, y, z; } vec3;
static inline vec3 v3add(vec3 a, vec3 b) { return (vec3){a.x+b.x,a.y+b.y,a.z+b.z}; }
static inline vec3 v3scale(vec3 a, double s) { return (vec3){a.x*s,a.y*s,a.z*s}; }
static inline double v3dot(vec3 a, vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
static inline double v3norm(vec3 a) { return sqrt(v3dot(a,a)); }
static inline vec3 v3reflect(vec3 v, vec3 n) {
    double d=2.0*v3dot(v,n); return (vec3){v.x-d*n.x,v.y-d*n.y,v.z-d*n.z};
}
static void reorth(vec3 d[4]) {
    vec3 m={0,0,0};
    for(int a=0;a<4;a++) m=v3add(m,d[a]);
    m=v3scale(m,0.25);
    for(int a=0;a<4;a++){
        d[a]=(vec3){d[a].x-m.x,d[a].y-m.y,d[a].z-m.z};
        double n=v3norm(d[a]); if(n>1e-15) d[a]=v3scale(d[a],1.0/n);
    }
}
static void init_tet(vec3 d[4]) {
    d[0]=(vec3){0,0,1};
    d[1]=(vec3){2*sqrt(2.0)/3,0,-1.0/3};
    d[2]=(vec3){-sqrt(2.0)/3,sqrt(6.0)/3,-1.0/3};
    d[3]=(vec3){-sqrt(2.0)/3,-sqrt(6.0)/3,-1.0/3};
}
static void helix_step(vec3 *pos, vec3 d[4], int face) {
    vec3 e=d[face]; *pos=v3add(*pos,v3scale(e,-2.0/3.0));
    for(int a=0;a<4;a++) d[a]=v3reflect(d[a],e);
}

/* ========== 4x4 complex matrix ========== */
typedef double complex c4x4[4][4];
static void c4_zero(c4x4 M){memset(M,0,sizeof(c4x4));}
static void c4_eye(c4x4 M){c4_zero(M);for(int i=0;i<4;i++)M[i][i]=1;}
static void c4_copy(c4x4 d,const c4x4 s){memcpy(d,s,sizeof(c4x4));}
static void c4_mul(c4x4 C,const c4x4 A,const c4x4 B){
    c4x4 t;c4_zero(t);
    for(int i=0;i<4;i++)for(int j=0;j<4;j++)for(int k=0;k<4;k++)t[i][j]+=A[i][k]*B[k][j];
    c4_copy(C,t);
}
static void c4_adjoint(c4x4 d,const c4x4 s){for(int i=0;i<4;i++)for(int j=0;j<4;j++)d[i][j]=conj(s[j][i]);}
static void c4_add(c4x4 C,const c4x4 A,const c4x4 B){for(int i=0;i<4;i++)for(int j=0;j<4;j++)C[i][j]=A[i][j]+B[i][j];}

/* ========== Dirac matrices ========== */
static c4x4 ALPHA[3], BETA;
static void init_dirac(void) {
    c4_zero(ALPHA[0]);c4_zero(ALPHA[1]);c4_zero(ALPHA[2]);c4_zero(BETA);
    ALPHA[0][0][3]=1;ALPHA[0][1][2]=1;ALPHA[0][2][1]=1;ALPHA[0][3][0]=1;
    ALPHA[1][0][3]=-I;ALPHA[1][1][2]=I;ALPHA[1][2][1]=-I;ALPHA[1][3][0]=I;
    ALPHA[2][0][2]=1;ALPHA[2][1][3]=-1;ALPHA[2][2][0]=1;ALPHA[2][3][1]=-1;
    BETA[0][0]=1;BETA[1][1]=1;BETA[2][2]=-1;BETA[3][3]=-1;
}
static void make_tau(c4x4 tau, vec3 d) {
    double nu=sqrt(7.0)/4.0; c4_zero(tau);
    for(int i=0;i<4;i++) tau[i][i]=nu*(i<2?1.0:-1.0);
    double dd[3]={d.x,d.y,d.z};
    for(int a=0;a<3;a++)for(int i=0;i<4;i++)for(int j=0;j<4;j++)
        tau[i][j]+=0.75*dd[a]*ALPHA[a][i][j];
}

/* ========== LAPACK ========== */
extern void zheev_(const char*,const char*,const int*,double complex*,
                   const int*,double*,double complex*,const int*,double*,int*);
static void eigh4(const c4x4 M,double w[4],c4x4 V){
    double complex a[16];
    for(int i=0;i<4;i++)for(int j=0;j<4;j++)a[j*4+i]=M[i][j];
    int n=4,lda=4,lwork=32,info; double complex work[32]; double rwork[16];
    zheev_("V","U",&n,a,&lda,w,work,&lwork,rwork,&info);
    if(info){fprintf(stderr,"zheev %d\n",info);exit(1);}
    for(int i=0;i<4;i++)for(int j=0;j<4;j++)V[i][j]=a[j*4+i];
}
static void frame_transport(const c4x4 tf,const c4x4 tt,c4x4 U){
    c4x4 Pfp,Pfm,Ptp,Ptm;
    c4_eye(Pfp);c4_eye(Pfm);c4_eye(Ptp);c4_eye(Ptm);
    for(int i=0;i<4;i++)for(int j=0;j<4;j++){
        Pfp[i][j]=0.5*(Pfp[i][j]+tf[i][j]);Pfm[i][j]=0.5*(Pfm[i][j]-tf[i][j]);
        Ptp[i][j]=0.5*(Ptp[i][j]+tt[i][j]);Ptm[i][j]=0.5*(Ptm[i][j]-tt[i][j]);
    }
    c4x4 W,t1,t2; c4_mul(t1,Ptp,Pfp);c4_mul(t2,Ptm,Pfm);c4_add(W,t1,t2);
    c4x4 Wa,WdW; c4_adjoint(Wa,W);c4_mul(WdW,Wa,W);
    double ev[4];c4x4 V; eigh4(WdW,ev,V);
    c4x4 Va,D,t3,Hi; c4_adjoint(Va,V);c4_zero(D);
    for(int i=0;i<4;i++)D[i][i]=(ev[i]>1e-15)?1.0/sqrt(ev[i]):0;
    c4_mul(t3,V,D);c4_mul(Hi,t3,Va);c4_mul(U,W,Hi);
}

/* ========== Hash table ========== */
#define HASH_BITS 23
#define HASH_SIZE (1<<HASH_BITS)
#define HASH_MASK (HASH_SIZE-1)
#define MAX_SITES 4000000

typedef struct { long kx,ky,kz; int id; } hentry;
static hentry *htab;

typedef struct {
    vec3 pos, dirs[4];
    int r_chain, r_pos, l_chain, l_pos;
} site_t;
static site_t *sites;
static int nsites=0;

static void htab_init(void){
    htab=calloc(HASH_SIZE,sizeof(hentry));
    for(int i=0;i<HASH_SIZE;i++)htab[i].id=-1;
    sites=malloc(MAX_SITES*sizeof(site_t));
}
static void poskey(vec3 p,long*kx,long*ky,long*kz){
    const double tol=1e-7;
    *kx=(long)round(p.x/tol);*ky=(long)round(p.y/tol);*kz=(long)round(p.z/tol);
}
static unsigned hfn(long kx,long ky,long kz){
    return (unsigned)(((unsigned long)(kx*73856093L)^(unsigned long)(ky*19349663L)^
            (unsigned long)(kz*83492791L))&HASH_MASK);
}
static int site_find(vec3 pos){
    long kx,ky,kz;poskey(pos,&kx,&ky,&kz);unsigned h=hfn(kx,ky,kz);
    for(int p=0;p<HASH_SIZE;p++){unsigned i=(h+p)&HASH_MASK;
        if(htab[i].id==-1)return -1;
        if(htab[i].kx==kx&&htab[i].ky==ky&&htab[i].kz==kz)return htab[i].id;}
    return -1;
}
static int site_insert(vec3 pos,vec3 dirs[4]){
    long kx,ky,kz;poskey(pos,&kx,&ky,&kz);unsigned h=hfn(kx,ky,kz);
    for(int p=0;p<HASH_SIZE;p++){unsigned i=(h+p)&HASH_MASK;
        if(htab[i].id==-1){
            if(nsites>=MAX_SITES){fprintf(stderr,"Too many sites\n");exit(1);}
            int id=nsites++;htab[i]=(hentry){kx,ky,kz,id};
            sites[id].pos=pos;memcpy(sites[id].dirs,dirs,sizeof(vec3)*4);
            sites[id].r_chain=sites[id].l_chain=-1;return id;}
        if(htab[i].kx==kx&&htab[i].ky==ky&&htab[i].kz==kz)return htab[i].id;}
    fprintf(stderr,"Hash full\n");exit(1);
}

/* ========== Chain storage ========== */
static const int PAT_R[4]={1,3,0,2}, PAT_L[4]={0,1,2,3};
#define MAX_CHAINS 500000
#define MAX_CLEN 200
typedef struct { int ids[MAX_CLEN]; int faces[MAX_CLEN]; int len; } chain_t;
static chain_t *rchains, *lchains;
static int nrch=0, nlch=0;

/* Trace chain creating new sites */
static void trace_chain(int start, const int pat[4], int nfwd, int nbwd, chain_t *ch) {
    ch->len=0;
    int rev[4]={pat[3],pat[2],pat[1],pat[0]};
    int bwd[MAX_CLEN]; int nb=0;
    {vec3 p=sites[start].pos; vec3 d[4]; memcpy(d,sites[start].dirs,sizeof(d));
     for(int i=0;i<nbwd&&nb<MAX_CLEN/2;i++){
         helix_step(&p,d,rev[i%4]); if((i+1)%8==0)reorth(d);
         bwd[nb++]=site_insert(p,d);}}
    for(int i=nb-1;i>=0;i--){ch->ids[ch->len]=bwd[i];ch->len++;}
    ch->ids[ch->len]=start; ch->len++;
    {vec3 p=sites[start].pos; vec3 d[4]; memcpy(d,sites[start].dirs,sizeof(d));
     for(int i=0;i<nfwd&&ch->len<MAX_CLEN;i++){
         helix_step(&p,d,pat[i%4]); if((i+1)%8==0)reorth(d);
         ch->ids[ch->len]=site_insert(p,d); ch->len++;}}
    for(int i=0;i<ch->len;i++){int off=i-nb; ch->faces[i]=pat[((off%4)+4)%4];}
}

/* Trace chain through existing sites only */
static void trace_chain_existing(int start, const int pat[4], int maxfwd, int maxbwd, chain_t *ch) {
    ch->len=0;
    int rev[4]={pat[3],pat[2],pat[1],pat[0]};
    int bwd[MAX_CLEN]; int nb=0;
    {vec3 p=sites[start].pos; vec3 d[4]; memcpy(d,sites[start].dirs,sizeof(d));
     for(int i=0;i<maxbwd&&nb<MAX_CLEN/2;i++){
         helix_step(&p,d,rev[i%4]); if((i+1)%8==0)reorth(d);
         int s=site_find(p); if(s<0)break; bwd[nb++]=s;}}
    for(int i=nb-1;i>=0;i--){ch->ids[ch->len]=bwd[i];ch->len++;}
    ch->ids[ch->len]=start; ch->len++;
    {vec3 p=sites[start].pos; vec3 d[4]; memcpy(d,sites[start].dirs,sizeof(d));
     for(int i=0;i<maxfwd&&ch->len<MAX_CLEN;i++){
         helix_step(&p,d,pat[i%4]); if((i+1)%8==0)reorth(d);
         int s=site_find(p); if(s<0)break; ch->ids[ch->len]=s; ch->len++;}}
    for(int i=0;i<ch->len;i++){int off=i-nb; ch->faces[i]=pat[((off%4)+4)%4];}
}

static void assign_membership(chain_t *ch, int cid, int is_r) {
    for(int i=0;i<ch->len;i++){int s=ch->ids[i];
        if(is_r){if(sites[s].r_chain<0){sites[s].r_chain=cid;sites[s].r_pos=i;}}
        else    {if(sites[s].l_chain<0){sites[s].l_chain=cid;sites[s].l_pos=i;}}}
}

/* ========== Sparse entries ========== */
typedef struct { int row,col; double re,im; } sparse_entry;
static sparse_entry *sr_entries, *sl_entries;
static int sr_nnz=0, sl_nnz=0;
#define MAX_NNZ 20000000
static void add_entry(sparse_entry *e,int *nnz,int r,int c,double complex v){
    if(cabs(v)<1e-15)return;
    if(*nnz>=MAX_NNZ){fprintf(stderr,"Too many nnz\n");exit(1);}
    e[*nnz]=(sparse_entry){r,c,creal(v),cimag(v)};(*nnz)++;
}

/* Global for stitcher */
static int *g_nr, *g_nl;

static void build_stitched(chain_t *chains,int nch,int is_r,sparse_entry *ent,int *nnz){
    int mt=0; for(int c=0;c<nch;c++)mt+=chains[c].len;
    int *ids=malloc(mt*sizeof(int)), *faces=malloc(mt*sizeof(int));
    int total=0;
    for(int c=0;c<nch;c++) for(int i=0;i<chains[c].len;i++){
        int s=chains[c].ids[i];
        if((is_r?g_nr[s]:g_nl[s])!=c)continue;
        ids[total]=s; faces[total]=chains[c].faces[i]; total++;}
    fprintf(stderr,"  Loop: %d sites\n",total);
    for(int i=0;i<total;i++){
        int in=(i+1)%total, ip=(i-1+total)%total;
        int sid=ids[i]; vec3 d=sites[sid].dirs[faces[i]];
        c4x4 tau; make_tau(tau,d);
        c4x4 Pp,Pm; c4_eye(Pp);c4_eye(Pm);
        for(int a=0;a<4;a++)for(int b=0;b<4;b++){
            Pp[a][b]=0.5*(Pp[a][b]+tau[a][b]);Pm[a][b]=0.5*(Pm[a][b]-tau[a][b]);}
        {int sn=ids[in];vec3 dn=sites[sn].dirs[faces[in]];
         c4x4 tn;make_tau(tn,dn);c4x4 U,bl;frame_transport(tau,tn,U);c4_mul(bl,U,Pp);
         for(int a=0;a<4;a++)for(int b=0;b<4;b++)add_entry(ent,nnz,sn*4+a,sid*4+b,bl[a][b]);}
        {int sp=ids[ip];vec3 dp=sites[sp].dirs[faces[ip]];
         c4x4 tp;make_tau(tp,dp);c4x4 U,bl;frame_transport(tau,tp,U);c4_mul(bl,U,Pm);
         for(int a=0;a<4;a++)for(int b=0;b<4;b++)add_entry(ent,nnz,sp*4+a,sid*4+b,bl[a][b]);}
    }
    free(ids);free(faces);
}

/* ========== Main ========== */
int main(int argc, char **argv) {
    int seed_depth=3, chain_len=20, nfill=10;
    if(argc>1)seed_depth=atoi(argv[1]);
    if(argc>2)chain_len=atoi(argv[2]);
    if(argc>3)nfill=atoi(argv[3]);

    init_dirac(); htab_init();
    rchains=malloc(MAX_CHAINS*sizeof(chain_t));
    lchains=malloc(MAX_CHAINS*sizeof(chain_t));
    sr_entries=malloc(MAX_NNZ*sizeof(sparse_entry));
    sl_entries=malloc(MAX_NNZ*sizeof(sparse_entry));

    /* ---- Phase 1: BFS seed ball ---- */
    vec3 d0[4]; init_tet(d0);
    int oid=site_insert((vec3){0,0,0},d0);
    int *queue=malloc(MAX_SITES*sizeof(int));
    int qh=0,qt=0; queue[qt++]=oid;
    int *depth=calloc(MAX_SITES,sizeof(int));
    while(qh<qt){
        int s=queue[qh++];
        if(depth[s]>=seed_depth)continue;
        for(int f=0;f<4;f++){
            vec3 p=sites[s].pos,dd[4];memcpy(dd,sites[s].dirs,sizeof(dd));
            helix_step(&p,dd,f);reorth(dd);
            int old_n=nsites;
            int nid=site_insert(p,dd);
            if(nid>=old_n){depth[nid]=depth[s]+1;queue[qt++]=nid;}
        }
    }
    free(queue);free(depth);
    int n_seed=nsites;
    fprintf(stderr,"Phase 1: BFS seed depth %d -> %d sites\n",seed_depth,n_seed);

    /* ---- Phase 2: Trace long R and L chains from each seed site ---- */
    for(int s=0;s<n_seed;s++){
        if(sites[s].r_chain<0 && nrch<MAX_CHAINS){
            trace_chain(s,PAT_R,chain_len,chain_len,&rchains[nrch]);
            assign_membership(&rchains[nrch],nrch,1);nrch++;}
        if(sites[s].l_chain<0 && nlch<MAX_CHAINS){
            trace_chain(s,PAT_L,chain_len,chain_len,&lchains[nlch]);
            assign_membership(&lchains[nlch],nlch,0);nlch++;}
    }
    fprintf(stderr,"Phase 2: %d sites, %d R, %d L\n",nsites,nrch,nlch);

    /* ---- Phase 3: Fill passes ---- */
    for(int fill=0;fill<nfill;fill++){
        int al=0,ar=0;
        int snap=nsites;
        for(int s=0;s<snap;s++){
            if(sites[s].l_chain<0 && nlch<MAX_CHAINS){
                trace_chain_existing(s,PAT_L,chain_len,chain_len,&lchains[nlch]);
                if(lchains[nlch].len>=2){assign_membership(&lchains[nlch],nlch,0);nlch++;al++;}}}
        for(int s=0;s<snap;s++){
            if(sites[s].r_chain<0 && nrch<MAX_CHAINS){
                trace_chain_existing(s,PAT_R,chain_len,chain_len,&rchains[nrch]);
                if(rchains[nrch].len>=2){assign_membership(&rchains[nrch],nrch,1);nrch++;ar++;}}}
        int nb=0;for(int s=0;s<nsites;s++)if(sites[s].r_chain>=0&&sites[s].l_chain>=0)nb++;
        fprintf(stderr,"Fill %d: +%dL +%dR, both=%d/%d (%.1f%%)\n",fill,al,ar,nb,nsites,100.0*nb/nsites);
        if(al==0&&ar==0)break;
    }

    /* ---- Phase 4: Restrict to dual-covered, extract chains ---- */
    g_nr=malloc(nsites*sizeof(int)); g_nl=malloc(nsites*sizeof(int));
    int *nr_pos=malloc(nsites*sizeof(int)),*nl_pos=malloc(nsites*sizeof(int));
    for(int s=0;s<nsites;s++){g_nr[s]=g_nl[s]=-1;nr_pos[s]=nl_pos[s]=-1;}

    chain_t *out_r=malloc(MAX_CHAINS*sizeof(chain_t));
    chain_t *out_l=malloc(MAX_CHAINS*sizeof(chain_t));
    int onr=0,onl=0;

    for(int pass=0;pass<2;pass++){
        int nch=pass?nlch:nrch;
        chain_t *ch_in=pass?lchains:rchains;
        chain_t *ch_out=pass?out_l:out_r;
        int *on=pass?&onl:&onr;
        int *mem=pass?g_nl:g_nr;
        int *mpos=pass?nl_pos:nr_pos;
        for(int c=0;c<nch;c++){
            chain_t seg;seg.len=0;
            for(int i=0;i<ch_in[c].len;i++){
                int s=ch_in[c].ids[i];
                int dual=(sites[s].r_chain>=0&&sites[s].l_chain>=0);
                if(dual){seg.ids[seg.len]=s;seg.faces[seg.len]=ch_in[c].faces[i];seg.len++;}
                else{if(seg.len>=2&&*on<MAX_CHAINS){
                    memcpy(&ch_out[*on],&seg,sizeof(chain_t));
                    for(int j=0;j<seg.len;j++)if(mem[seg.ids[j]]==-1){mem[seg.ids[j]]=*on;mpos[seg.ids[j]]=j;}
                    (*on)++;}seg.len=0;}}
            if(seg.len>=2&&*on<MAX_CHAINS){
                memcpy(&ch_out[*on],&seg,sizeof(chain_t));
                for(int j=0;j<seg.len;j++)if(mem[seg.ids[j]]==-1){mem[seg.ids[j]]=*on;mpos[seg.ids[j]]=j;}
                (*on)++;}
        }
    }
    /* Update membership */
    for(int s=0;s<nsites;s++){
        sites[s].r_chain=g_nr[s];sites[s].r_pos=nr_pos[s];
        sites[s].l_chain=g_nl[s];sites[s].l_pos=nl_pos[s];}
    int nfinal=0;
    for(int s=0;s<nsites;s++)if(g_nr[s]>=0&&g_nl[s]>=0)nfinal++;
    fprintf(stderr,"\nRestricted: %d R, %d L, %d dual sites\n",onr,onl,nfinal);

    /* ---- Phase 5: Build stitched operators ---- */
    fprintf(stderr,"Building S_R...\n");
    build_stitched(out_r,onr,1,sr_entries,&sr_nnz);
    fprintf(stderr,"S_R: %d nnz\n",sr_nnz);
    fprintf(stderr,"Building S_L...\n");
    build_stitched(out_l,onl,0,sl_entries,&sl_nnz);
    fprintf(stderr,"S_L: %d nnz\n",sl_nnz);

    /* ---- Output ---- */
    int header[4]={nsites,nfinal,sr_nnz,sl_nnz};
    fwrite(header,sizeof(int),4,stdout);
    for(int s=0;s<nsites;s++){double x[3]={sites[s].pos.x,sites[s].pos.y,sites[s].pos.z};fwrite(x,sizeof(double),3,stdout);}
    for(int s=0;s<nsites;s++){int m[4]={g_nr[s],nr_pos[s],g_nl[s],nl_pos[s]};fwrite(m,sizeof(int),4,stdout);}
    for(int i=0;i<sr_nnz;i++){int rc[2]={sr_entries[i].row,sr_entries[i].col};fwrite(rc,sizeof(int),2,stdout);}
    for(int i=0;i<sr_nnz;i++){double v[2]={sr_entries[i].re,sr_entries[i].im};fwrite(v,sizeof(double),2,stdout);}
    for(int i=0;i<sl_nnz;i++){int rc[2]={sl_entries[i].row,sl_entries[i].col};fwrite(rc,sizeof(int),2,stdout);}
    for(int i=0;i<sl_nnz;i++){double v[2]={sl_entries[i].re,sl_entries[i].im};fwrite(v,sizeof(double),2,stdout);}

    free(htab);free(sites);free(rchains);free(lchains);free(out_r);free(out_l);
    free(sr_entries);free(sl_entries);free(g_nr);free(g_nl);free(nr_pos);free(nl_pos);
    return 0;
}

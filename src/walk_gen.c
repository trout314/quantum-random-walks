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

/* ========== Memory guard (Linux /proc/meminfo) ========== */
static long get_avail_mb(void) {
    FILE *f = fopen("/proc/meminfo","r");
    if (!f) return -1;
    char line[256]; long avail = -1;
    while (fgets(line,sizeof(line),f)) {
        if (sscanf(line,"MemAvailable: %ld kB",&avail)==1) { avail /= 1024; break; }
    }
    fclose(f);
    return avail;
}
/* Abort if available memory drops below threshold (MB) */
static void mem_check(const char *label, long reserve_mb) {
    long avail = get_avail_mb();
    if (avail < 0) return;  /* can't read, skip check */
    if (avail < reserve_mb) {
        fprintf(stderr,"\n*** ABORT at %s: only %ld MB available (need > %ld MB). ***\n"
                        "*** Reduce seed_depth, chain_len, or fill_passes. ***\n",
                label, avail, reserve_mb);
        exit(2);
    }
}
/* malloc with memory guard */
static void *safe_malloc(size_t bytes, const char *label) {
    long need_mb = (long)(bytes / (1024*1024)) + 1;
    long avail = get_avail_mb();
    if (avail > 0 && avail < need_mb + 500) {
        fprintf(stderr,"\n*** ABORT before %s: need ~%ld MB but only %ld MB available. ***\n"
                        "*** Reduce seed_depth, chain_len, or fill_passes. ***\n",
                label, need_mb, avail);
        exit(2);
    }
    void *p = malloc(bytes);
    if (!p) {
        fprintf(stderr,"*** malloc failed for %s (%zu bytes) ***\n", label, bytes);
        exit(2);
    }
    fprintf(stderr,"  alloc %s: %.1f MB (avail: %ld MB)\n", label,
            (double)bytes/(1024*1024), avail>0?avail:-1);
    return p;
}

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

/* LAPACK no longer needed — frame transport has a closed form */
/* Closed-form frame transport: U = (I + tau_to tau_from) / (2 cos(phi/2))
 * where cos(phi) = Tr(tau_to tau_from) / 4.
 *
 * Derived from: W = (I + tau_to tau_from)/2, and tau_to tau_from is
 * unitary with eigenvalues e^{±i phi}, so the polar part of W is
 * (tau_to tau_from)^{1/2} = (I + U) / (2 cos(phi/2)).
 *
 * For consecutive helix sites: cos(phi) = 5/8, giving 2cos(phi/2) = sqrt(13)/2.
 * At stitching seams, cos(phi) may differ, so we compute it from the trace. */
static void frame_transport(const c4x4 tf, const c4x4 tt, c4x4 U) {
    c4x4 prod;
    c4_mul(prod, tt, tf);  /* tau_to @ tau_from */
    /* cos(phi) = Re(Tr(prod)) / 4 */
    double cos_phi = 0;
    for (int i = 0; i < 4; i++) cos_phi += creal(prod[i][i]);
    cos_phi /= 4.0;
    double cos_half = sqrt((1.0 + cos_phi) / 2.0);
    double scale = 1.0 / (2.0 * cos_half);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            U[i][j] = scale * ((i==j ? 1.0 : 0.0) + prod[i][j]);
}

/* ========== Hash table ========== */
#define HASH_BITS 25
#define HASH_SIZE (1<<HASH_BITS)
#define HASH_MASK (HASH_SIZE-1)
#define MAX_SITES 20000000

typedef struct { long kx,ky,kz; int id; } hentry;
static hentry *htab;

typedef struct {
    vec3 pos, dirs[4];
    int r_chain, r_pos, l_chain, l_pos;
    int r_face, l_face;  /* face index used for R/L helix step at this site */
} site_t;
static site_t *sites;
static int nsites=0;

static void htab_init(void){
    fprintf(stderr,"Allocating hash table (%.0f MB) and sites (%.0f MB)...\n",
            (double)HASH_SIZE*sizeof(hentry)/(1024*1024),
            (double)MAX_SITES*sizeof(site_t)/(1024*1024));
    mem_check("htab_init", 500);
    htab=calloc(HASH_SIZE,sizeof(hentry));
    if(!htab){fprintf(stderr,"Failed to alloc hash table\n");exit(2);}
    for(int i=0;i<HASH_SIZE;i++)htab[i].id=-1;
    sites=malloc(MAX_SITES*sizeof(site_t));
    if(!sites){fprintf(stderr,"Failed to alloc sites\n");exit(2);}
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
            sites[id].r_chain=sites[id].l_chain=-1;
            sites[id].r_face=sites[id].l_face=-1;return id;}
        if(htab[i].kx==kx&&htab[i].ky==ky&&htab[i].kz==kz)return htab[i].id;}
    fprintf(stderr,"Hash full\n");exit(1);
}

/* ========== Chain storage ========== */
static const int PAT_R[4]={1,3,0,2}, PAT_L[4]={0,1,2,3};
#define MAX_CLEN 600
typedef struct { int ids[MAX_CLEN]; int faces[MAX_CLEN]; int len; } chain_t;
static chain_t *rchains, *lchains;
static int nrch=0, nlch=0;
static int max_chains=0;  /* set dynamically in main */

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
        if(is_r){if(sites[s].r_chain<0){sites[s].r_chain=cid;sites[s].r_pos=i;sites[s].r_face=ch->faces[i];}}
        else    {if(sites[s].l_chain<0){sites[s].l_chain=cid;sites[s].l_pos=i;sites[s].l_face=ch->faces[i];}}}
}

/* ========== Sparse entries ========== */
typedef struct { int row,col; double re,im; } sparse_entry;
static sparse_entry *sr_entries, *sl_entries;
static int sr_nnz=0, sl_nnz=0;
static int max_nnz=0;  /* set dynamically in main */
static void add_entry(sparse_entry *e,int *nnz,int r,int c,double complex v){
    if(cabs(v)<1e-15)return;
    if(*nnz>=max_nnz){fprintf(stderr,"Too many nnz\n");exit(1);}
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

    fprintf(stderr,"=== walk_gen: seed_depth=%d chain_len=%d fill_passes=%d ===\n",
            seed_depth, chain_len, nfill);
    long avail0 = get_avail_mb();
    fprintf(stderr,"Memory available: %ld MB\n", avail0);

    /* Size allocations to fit in memory. Reserve 2 GB for OS + htab + sites + overhead. */
    double fixed_mb = (double)HASH_SIZE*sizeof(hentry)/(1024*1024)
                    + (double)MAX_SITES*sizeof(site_t)/(1024*1024) + 2048;
    double budget_mb = (avail0 > 0 ? avail0 * 0.80 : 16000) - fixed_mb;
    if (budget_mb < 500) {
        fprintf(stderr,"*** ABORT: only %.0f MB available after fixed allocations. Need at least 500 MB. ***\n", budget_mb);
        exit(2);
    }
    /* Budget split: 4 chain arrays take ~4*max_chains*sizeof(chain_t),
     * 2 sparse arrays take ~2*max_nnz*sizeof(sparse_entry).
     * Give 70% to chains, 30% to sparse. */
    double chain_budget = budget_mb * 0.70 * 1024 * 1024;
    double sparse_budget = budget_mb * 0.30 * 1024 * 1024;
    max_chains = (int)(chain_budget / (4.0 * sizeof(chain_t)));
    if (max_chains < 1000) max_chains = 1000;
    if (max_chains > 4000000) max_chains = 4000000;
    max_nnz = (int)(sparse_budget / (2.0 * sizeof(sparse_entry)));
    if (max_nnz < 100000) max_nnz = 100000;
    if (max_nnz > 300000000) max_nnz = 300000000;

    double total_mb = 4.0*max_chains*sizeof(chain_t)/(1024*1024)
                    + 2.0*max_nnz*sizeof(sparse_entry)/(1024*1024) + fixed_mb;
    fprintf(stderr,"Sized for available memory:\n");
    fprintf(stderr,"  max_chains: %d (%.0f MB each array)\n",
            max_chains, (double)max_chains*sizeof(chain_t)/(1024*1024));
    fprintf(stderr,"  max_nnz:    %d (%.0f MB each array)\n",
            max_nnz, (double)max_nnz*sizeof(sparse_entry)/(1024*1024));
    fprintf(stderr,"  TOTAL est:  %.0f MB\n", total_mb);

    init_dirac(); htab_init();
    rchains = safe_malloc(max_chains*sizeof(chain_t), "rchains");
    lchains = safe_malloc(max_chains*sizeof(chain_t), "lchains");
    sr_entries = safe_malloc(max_nnz*sizeof(sparse_entry), "sr_entries");
    sl_entries = safe_malloc(max_nnz*sizeof(sparse_entry), "sl_entries");

    /* ---- Phase 1: BFS seed ball ---- */
    fprintf(stderr,"\n--- Phase 1: BFS seed (depth %d) ---\n", seed_depth);
    mem_check("Phase 1", 500);
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
    fprintf(stderr,"\n--- Phase 2: Tracing chains (len %d) from %d seed sites ---\n", chain_len, n_seed);
    mem_check("Phase 2", 500);
    for(int s=0;s<n_seed;s++){
        if(s % 1000 == 0 && s > 0)
            fprintf(stderr,"  Phase 2: processed %d/%d seeds, %d sites, %d R + %d L chains\n",
                    s, n_seed, nsites, nrch, nlch);
        if(s % 5000 == 0) mem_check("Phase 2 loop", 300);
        if(sites[s].r_chain<0 && nrch<max_chains){
            trace_chain(s,PAT_R,chain_len,chain_len,&rchains[nrch]);
            assign_membership(&rchains[nrch],nrch,1);nrch++;}
        if(sites[s].l_chain<0 && nlch<max_chains){
            trace_chain(s,PAT_L,chain_len,chain_len,&lchains[nlch]);
            assign_membership(&lchains[nlch],nlch,0);nlch++;}
    }
    fprintf(stderr,"Phase 2 done: %d sites, %d R, %d L chains\n",nsites,nrch,nlch);

    /* ---- Phase 3: Fill passes ---- */
    fprintf(stderr,"\n--- Phase 3: Fill passes (max %d) ---\n", nfill);
    for(int fill=0;fill<nfill;fill++){
        mem_check("Fill pass", 300);
        int al=0,ar=0;
        int snap=nsites;
        for(int s=0;s<snap;s++){
            if(sites[s].l_chain<0 && nlch<max_chains){
                trace_chain_existing(s,PAT_L,chain_len,chain_len,&lchains[nlch]);
                if(lchains[nlch].len>=2){assign_membership(&lchains[nlch],nlch,0);nlch++;al++;}}}
        for(int s=0;s<snap;s++){
            if(sites[s].r_chain<0 && nrch<max_chains){
                trace_chain_existing(s,PAT_R,chain_len,chain_len,&rchains[nrch]);
                if(rchains[nrch].len>=2){assign_membership(&rchains[nrch],nrch,1);nrch++;ar++;}}}
        int nb=0;for(int s=0;s<nsites;s++)if(sites[s].r_chain>=0&&sites[s].l_chain>=0)nb++;
        fprintf(stderr,"Fill %d: +%dL +%dR, both=%d/%d (%.1f%%)\n",fill,al,ar,nb,nsites,100.0*nb/nsites);
        if(al==0&&ar==0)break;
    }

    /* ---- Phase 4: Restrict to dual-covered, extract chains ---- */
    fprintf(stderr,"\n--- Phase 4: Restrict to dual-covered sites ---\n");
    mem_check("Phase 4", 500);
    g_nr=malloc(nsites*sizeof(int)); g_nl=malloc(nsites*sizeof(int));
    int *nr_pos=malloc(nsites*sizeof(int)),*nl_pos=malloc(nsites*sizeof(int));
    for(int s=0;s<nsites;s++){g_nr[s]=g_nl[s]=-1;nr_pos[s]=nl_pos[s]=-1;}

    chain_t *out_r=safe_malloc(max_chains*sizeof(chain_t), "out_r");
    chain_t *out_l=safe_malloc(max_chains*sizeof(chain_t), "out_l");
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
                else{if(seg.len>=2&&*on<max_chains){
                    memcpy(&ch_out[*on],&seg,sizeof(chain_t));
                    for(int j=0;j<seg.len;j++)if(mem[seg.ids[j]]==-1){mem[seg.ids[j]]=*on;mpos[seg.ids[j]]=j;}
                    (*on)++;}seg.len=0;}}
            if(seg.len>=2&&*on<max_chains){
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
    fprintf(stderr,"\n--- Phase 5: Build stitched shift operators ---\n");
    mem_check("Phase 5", 300);
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
    /* Direction vectors (4 directions × 3 components per site) and face indices */
    for(int s=0;s<nsites;s++){
        for(int a=0;a<4;a++){
            double d[3]={sites[s].dirs[a].x,sites[s].dirs[a].y,sites[s].dirs[a].z};
            fwrite(d,sizeof(double),3,stdout);
        }
    }
    for(int s=0;s<nsites;s++){int f[2]={sites[s].r_face,sites[s].l_face};fwrite(f,sizeof(int),2,stdout);}
    for(int i=0;i<sr_nnz;i++){int rc[2]={sr_entries[i].row,sr_entries[i].col};fwrite(rc,sizeof(int),2,stdout);}
    for(int i=0;i<sr_nnz;i++){double v[2]={sr_entries[i].re,sr_entries[i].im};fwrite(v,sizeof(double),2,stdout);}
    for(int i=0;i<sl_nnz;i++){int rc[2]={sl_entries[i].row,sl_entries[i].col};fwrite(rc,sizeof(int),2,stdout);}
    for(int i=0;i<sl_nnz;i++){double v[2]={sl_entries[i].re,sl_entries[i].im};fwrite(v,sizeof(double),2,stdout);}

    free(htab);free(sites);free(rchains);free(lchains);free(out_r);free(out_l);
    free(sr_entries);free(sl_entries);free(g_nr);free(g_nl);free(nr_pos);free(nl_pos);
    return 0;
}

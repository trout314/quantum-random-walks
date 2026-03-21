/*
 * walk_gen.c — Generate tetrahedral walk lattice and sparse shift operators.
 *
 * Approach:
 *   Phase 1: BFS ball for isotropic seed
 *   Phase 2: Thread unique R and L chains through seed (linked-list, no new sites)
 *   Phase 3: Build sparse S_R, S_L with frame transport and open boundaries
 *
 * Each site has exactly one R-helix and one L-helix through it.
 * Chains are stored as doubly-linked lists embedded in the site array.
 *
 * Build: clang -O2 -o walk_gen src/walk_gen.c -lm
 * Usage: ./walk_gen [seed_depth]
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
static void mem_check(const char *label, long reserve_mb) {
    long avail = get_avail_mb();
    if (avail < 0) return;
    if (avail < reserve_mb) {
        fprintf(stderr,"\n*** ABORT at %s: only %ld MB available (need > %ld MB). ***\n",
                label, avail, reserve_mb);
        exit(2);
    }
}
static void *safe_malloc(size_t bytes, const char *label) {
    long need_mb = (long)(bytes / (1024*1024)) + 1;
    long avail = get_avail_mb();
    if (avail > 0 && avail < need_mb + 500) {
        fprintf(stderr,"\n*** ABORT before %s: need ~%ld MB but only %ld MB available. ***\n",
                label, need_mb, avail);
        exit(2);
    }
    void *p = malloc(bytes);
    if (!p) { fprintf(stderr,"*** malloc failed for %s (%zu bytes) ***\n", label, bytes); exit(2); }
    if (bytes > 10*1024*1024)
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

/* Closed-form frame transport: U = (I + tau_to tau_from) / (2 cos(phi/2)) */
static void frame_transport(const c4x4 tf, const c4x4 tt, c4x4 U) {
    c4x4 prod;
    c4_mul(prod, tt, tf);
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
    int r_next, r_prev;   /* next/prev site along R-helix, -1 at chain ends */
    int l_next, l_prev;   /* next/prev site along L-helix, -1 at chain ends */
    int r_face, l_face;   /* face index used for R/L helix step at this site */
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
            sites[id].r_next=sites[id].r_prev=-1;
            sites[id].l_next=sites[id].l_prev=-1;
            sites[id].r_face=sites[id].l_face=-1;return id;}
        if(htab[i].kx==kx&&htab[i].ky==ky&&htab[i].kz==kz)return htab[i].id;}
    fprintf(stderr,"Hash full\n");exit(1);
}

/* ========== Chain threading (linked list) ========== */
static const int PAT_R[4]={1,3,0,2}, PAT_L[4]={0,1,2,3};

/* Thread a chain through existing sites starting from 'start'.
 * Links sites via next/prev pointers. Only follows existing sites (no creation).
 * is_r: 1 for R-chain, 0 for L-chain.
 * Returns number of sites linked. */
static int thread_chain(int start, const int pat[4], int is_r) {
    int rev[4]={pat[3],pat[2],pat[1],pat[0]};
    int linked = 1;

    /* Set face for start site */
    /* The face at start depends on its position in the helix pattern.
     * We use face = pat[0] for the start site (offset 0 in the chain). */
    if (is_r) sites[start].r_face = pat[0];
    else      sites[start].l_face = pat[0];

    /* Trace forward */
    {
        int cur = start;
        vec3 p = sites[start].pos;
        vec3 d[4]; memcpy(d, sites[start].dirs, sizeof(d));
        for (int step = 0; step < MAX_SITES; step++) {
            int face = pat[step % 4];
            helix_step(&p, d, face);
            if ((step+1) % 8 == 0) reorth(d);
            int next = site_find(p);
            if (next < 0) break;  /* left seed ball */
            /* Check if next site already has a prev link (already on a chain) */
            if (is_r) {
                if (sites[next].r_prev >= 0) break;  /* already linked */
                sites[cur].r_next = next;
                sites[next].r_prev = cur;
                sites[next].r_face = pat[(step+1) % 4];
            } else {
                if (sites[next].l_prev >= 0) break;
                sites[cur].l_next = next;
                sites[next].l_prev = cur;
                sites[next].l_face = pat[(step+1) % 4];
            }
            cur = next;
            linked++;
        }
    }

    /* Trace backward */
    {
        int cur = start;
        vec3 p = sites[start].pos;
        vec3 d[4]; memcpy(d, sites[start].dirs, sizeof(d));
        for (int step = 0; step < MAX_SITES; step++) {
            int face = rev[step % 4];
            helix_step(&p, d, face);
            if ((step+1) % 8 == 0) reorth(d);
            int prev = site_find(p);
            if (prev < 0) break;
            if (is_r) {
                if (sites[prev].r_next >= 0) break;
                sites[cur].r_prev = prev;
                sites[prev].r_next = cur;
                /* Face for prev: it's at offset -(step+1) from start.
                 * pat index = (-(step+1)) mod 4 */
                sites[prev].r_face = pat[(((-(step+1))%4)+4)%4];
            } else {
                if (sites[prev].l_next >= 0) break;
                sites[cur].l_prev = prev;
                sites[prev].l_next = cur;
                sites[prev].l_face = pat[(((-(step+1))%4)+4)%4];
            }
            cur = prev;
            linked++;
        }
    }

    return linked;
}

/* ========== Sparse entries ========== */
typedef struct { int row,col; double re,im; } sparse_entry;
static sparse_entry *sr_entries, *sl_entries;
static int sr_nnz=0, sl_nnz=0;
static int max_nnz=0;
static void add_entry(sparse_entry *e,int *nnz,int r,int c,double complex v){
    if(cabs(v)<1e-15)return;
    if(*nnz>=max_nnz){fprintf(stderr,"Too many nnz\n");exit(1);}
    e[*nnz]=(sparse_entry){r,c,creal(v),cimag(v)};(*nnz)++;
}

/* Build shift operator by walking linked lists.
 * is_r: 1 for R-chains, 0 for L-chains. */
static void build_shift(int is_r, sparse_entry *ent, int *nnz) {
    int nchains=0, chain_sites=0, n_id=0;

    for (int s = 0; s < nsites; s++) {
        int prev = is_r ? sites[s].r_prev : sites[s].l_prev;
        int next = is_r ? sites[s].r_next : sites[s].l_next;
        int face = is_r ? sites[s].r_face : sites[s].l_face;

        if (face < 0) {
            /* Site not on any chain of this type — identity */
            for (int a = 0; a < 4; a++) add_entry(ent, nnz, s*4+a, s*4+a, 1.0);
            n_id++;
            continue;
        }

        /* Count chain starts for statistics */
        if (prev < 0) nchains++;
        chain_sites++;

        vec3 d = sites[s].dirs[face];
        c4x4 tau; make_tau(tau, d);
        c4x4 Pp, Pm; c4_eye(Pp); c4_eye(Pm);
        for (int a=0;a<4;a++) for(int b=0;b<4;b++) {
            Pp[a][b] = 0.5*(Pp[a][b]+tau[a][b]);
            Pm[a][b] = 0.5*(Pm[a][b]-tau[a][b]);
        }

        /* P+ -> forward (next site), absorb at chain end */
        if (next >= 0) {
            int fn = is_r ? sites[next].r_face : sites[next].l_face;
            vec3 dn = sites[next].dirs[fn];
            c4x4 tn; make_tau(tn, dn);
            c4x4 U, bl; frame_transport(tau, tn, U); c4_mul(bl, U, Pp);
            for (int a=0;a<4;a++) for(int b=0;b<4;b++)
                add_entry(ent, nnz, next*4+a, s*4+b, bl[a][b]);
        }

        /* P- -> backward (prev site), absorb at chain start */
        if (prev >= 0) {
            int fp = is_r ? sites[prev].r_face : sites[prev].l_face;
            vec3 dp = sites[prev].dirs[fp];
            c4x4 tp; make_tau(tp, dp);
            c4x4 U, bl; frame_transport(tau, tp, U); c4_mul(bl, U, Pm);
            for (int a=0;a<4;a++) for(int b=0;b<4;b++)
                add_entry(ent, nnz, prev*4+a, s*4+b, bl[a][b]);
        }
    }

    fprintf(stderr,"  %d chains, %d chain sites, %d identity sites\n", nchains, chain_sites, n_id);
}

/* ========== Main ========== */
int main(int argc, char **argv) {
    int seed_depth = 3;
    double prune_ratio = 0.0;  /* 0 = no pruning, 0.5 = moderate, 0.8 = aggressive */
    if (argc > 1) seed_depth = atoi(argv[1]);
    if (argc > 2) prune_ratio = atof(argv[2]);

    fprintf(stderr,"=== walk_gen: seed_depth=%d prune_ratio=%.2f ===\n", seed_depth, prune_ratio);
    long avail0 = get_avail_mb();
    fprintf(stderr,"Memory available: %ld MB\n", avail0);

    /* Memory budget: htab + sites + 2 sparse arrays */
    double fixed_mb = (double)HASH_SIZE*sizeof(hentry)/(1024*1024)
                    + (double)MAX_SITES*sizeof(site_t)/(1024*1024) + 2048;
    double budget_mb = (avail0 > 0 ? avail0 * 0.80 : 16000) - fixed_mb;
    if (budget_mb < 500) {
        fprintf(stderr,"*** ABORT: only %.0f MB available after fixed allocations. ***\n", budget_mb);
        exit(2);
    }
    max_nnz = (int)(budget_mb * 1024 * 1024 / (2.0 * sizeof(sparse_entry)));
    if (max_nnz < 100000) max_nnz = 100000;
    if (max_nnz > 300000000) max_nnz = 300000000;

    fprintf(stderr,"  max_nnz: %d (%.0f MB each array)\n",
            max_nnz, (double)max_nnz*sizeof(sparse_entry)/(1024*1024));

    init_dirac(); htab_init();
    sr_entries = safe_malloc(max_nnz*sizeof(sparse_entry), "sr_entries");
    sl_entries = safe_malloc(max_nnz*sizeof(sparse_entry), "sl_entries");

    /* ---- Phase 1: BFS seed ball with radial pruning ---- */
    /* Two pruning criteria (both checked against prune_ratio):
     *   Global: R/D < prune_ratio * step_len  (path from origin is inefficient)
     *   Local:  dR/dD < prune_ratio * step_len  (path has stalled recently)
     *
     * BFS runs in rounds of `round_size` depth steps. At round boundaries,
     * we snapshot each site's radius. The local criterion checks displacement
     * since the last round start.
     *
     * No pruning below min_prune_depth to keep the interior dense. */
    int min_prune_depth = 6;
    int round_size = 4;
    double step_len = 2.0/3.0;
    fprintf(stderr,"\n--- Phase 1: BFS seed (depth %d, prune=%.2f) ---\n", seed_depth, prune_ratio);
    mem_check("Phase 1", 500);
    vec3 d0[4]; init_tet(d0);
    site_insert((vec3){0,0,0}, d0);
    int *queue = malloc(MAX_SITES*sizeof(int));
    int qh=0, qt=0; queue[qt++] = 0;
    int *depth = calloc(MAX_SITES, sizeof(int));
    /* radius_at_round_start[s]: radial distance of site s when the current
     * pruning round began. Used for local displacement check. */
    float *round_radius = NULL;
    if (prune_ratio > 0) round_radius = calloc(MAX_SITES, sizeof(float));
    int npruned_global = 0, npruned_local = 0;
    while (qh < qt) {
        int s = queue[qh++];
        if (depth[s] >= seed_depth) continue;
        /* Pruning checks */
        if (prune_ratio > 0 && depth[s] >= min_prune_depth) {
            double r = v3norm(sites[s].pos);
            double max_r = depth[s] * step_len;
            /* Global: overall displacement efficiency */
            if (r < prune_ratio * max_r) { npruned_global++; continue; }
            /* Local: displacement since round start */
            int round_start_depth = (depth[s] / round_size) * round_size;
            int steps_in_round = depth[s] - round_start_depth;
            if (steps_in_round > 0) {
                double dr = r - round_radius[s];
                double max_dr = steps_in_round * step_len;
                if (dr < prune_ratio * max_dr) { npruned_local++; continue; }
            }
        }
        for (int f = 0; f < 4; f++) {
            vec3 p = sites[s].pos, dd[4];
            memcpy(dd, sites[s].dirs, sizeof(dd));
            helix_step(&p, dd, f); reorth(dd);
            int old_n = nsites;
            int nid = site_insert(p, dd);
            if (nid >= old_n) {
                depth[nid] = depth[s]+1;
                queue[qt++] = nid;
                if (round_radius) {
                    /* Inherit round_radius from parent if same round,
                     * or snapshot current radius if new round */
                    int parent_round = depth[s] / round_size;
                    int child_round = depth[nid] / round_size;
                    if (child_round == parent_round)
                        round_radius[nid] = round_radius[s];
                    else
                        round_radius[nid] = (float)v3norm(sites[nid].pos);
                }
            }
        }
        if (nsites % 1000000 == 0)
            fprintf(stderr,"  BFS: %d sites, depth frontier ~%d, pruned %d+%d\n",
                    nsites, depth[s], npruned_global, npruned_local);
    }
    free(queue); free(depth); free(round_radius);
    int n_seed = nsites;
    fprintf(stderr,"Phase 1: depth %d, prune=%.2f -> %d sites (pruned: %d global, %d local)\n",
            seed_depth, prune_ratio, n_seed, npruned_global, npruned_local);

    /* ---- Phase 2: Thread unique R and L chains through seed ball ---- */
    fprintf(stderr,"\n--- Phase 2: Threading chains through %d seed sites ---\n", n_seed);
    mem_check("Phase 2", 500);
    int nrch=0, nlch=0, total_r_linked=0, total_l_linked=0;
    for (int s = 0; s < n_seed; s++) {
        if (s % 10000 == 0 && s > 0)
            fprintf(stderr,"  Phase 2: processed %d/%d seeds, %d R + %d L chains\n",
                    s, n_seed, nrch, nlch);
        /* Thread R-chain if this site isn't already on one */
        if (sites[s].r_face < 0) {
            int linked = thread_chain(s, PAT_R, 1);
            if (linked >= 2) { nrch++; total_r_linked += linked; }
        }
        /* Thread L-chain if this site isn't already on one */
        if (sites[s].l_face < 0) {
            int linked = thread_chain(s, PAT_L, 0);
            if (linked >= 2) { nlch++; total_l_linked += linked; }
        }
    }
    int nR=0, nL=0, nBoth=0;
    for (int s = 0; s < n_seed; s++) {
        if (sites[s].r_face >= 0) nR++;
        if (sites[s].l_face >= 0) nL++;
        if (sites[s].r_face >= 0 && sites[s].l_face >= 0) nBoth++;
    }
    fprintf(stderr,"Phase 2 done: %d R-chains (%d sites), %d L-chains (%d sites)\n",
            nrch, total_r_linked, nlch, total_l_linked);
    fprintf(stderr,"Coverage: R=%d L=%d both=%d/%d (%.1f%%)\n",
            nR, nL, nBoth, n_seed, 100.0*nBoth/n_seed);

    /* ---- Phase 3: Build shift operators ---- */
    fprintf(stderr,"\n--- Phase 3: Build open-boundary shift operators ---\n");
    mem_check("Phase 3", 300);
    fprintf(stderr,"Building S_R...\n");
    build_shift(1, sr_entries, &sr_nnz);
    fprintf(stderr,"S_R: %d nnz\n", sr_nnz);
    fprintf(stderr,"Building S_L...\n");
    build_shift(0, sl_entries, &sl_nnz);
    fprintf(stderr,"S_L: %d nnz\n", sl_nnz);

    /* ---- Output ---- */
    int header[4] = {nsites, nBoth, sr_nnz, sl_nnz};
    fwrite(header, sizeof(int), 4, stdout);
    for (int s=0;s<nsites;s++) {
        double x[3] = {sites[s].pos.x, sites[s].pos.y, sites[s].pos.z};
        fwrite(x, sizeof(double), 3, stdout);
    }
    /* Write next/prev links as membership data (same 4-int slot per site) */
    for (int s=0;s<nsites;s++) {
        int m[4] = {sites[s].r_next, sites[s].r_prev, sites[s].l_next, sites[s].l_prev};
        fwrite(m, sizeof(int), 4, stdout);
    }
    /* Direction vectors */
    for (int s=0;s<nsites;s++) {
        for (int a=0;a<4;a++) {
            double d[3] = {sites[s].dirs[a].x, sites[s].dirs[a].y, sites[s].dirs[a].z};
            fwrite(d, sizeof(double), 3, stdout);
        }
    }
    /* Face indices */
    for (int s=0;s<nsites;s++) {
        int f[2] = {sites[s].r_face, sites[s].l_face};
        fwrite(f, sizeof(int), 2, stdout);
    }
    /* Sparse matrices */
    for (int i=0;i<sr_nnz;i++) { int rc[2]={sr_entries[i].row,sr_entries[i].col}; fwrite(rc,sizeof(int),2,stdout); }
    for (int i=0;i<sr_nnz;i++) { double v[2]={sr_entries[i].re,sr_entries[i].im}; fwrite(v,sizeof(double),2,stdout); }
    for (int i=0;i<sl_nnz;i++) { int rc[2]={sl_entries[i].row,sl_entries[i].col}; fwrite(rc,sizeof(int),2,stdout); }
    for (int i=0;i<sl_nnz;i++) { double v[2]={sl_entries[i].re,sl_entries[i].im}; fwrite(v,sizeof(double),2,stdout); }

    fprintf(stderr,"\nOutput: %d sites, R=%d L=%d both=%d, nnz_R=%d nnz_L=%d\n",
            nsites, nR, nL, nBoth, sr_nnz, sl_nnz);

    free(htab); free(sites); free(sr_entries); free(sl_entries);
    return 0;
}

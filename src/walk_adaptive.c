/*
 * walk_adaptive.c — Adaptive quantum walk with on-the-fly site creation.
 *
 * Combines lattice generation and wavepacket evolution. Sites are created
 * dynamically when the shift operator would send significant amplitude
 * to a nonexistent site. This eliminates boundary absorption for the
 * propagating wavepacket.
 *
 * Build: clang -O2 -o walk_adaptive src/walk_adaptive.c -lm
 * Usage: ./walk_adaptive theta sigma n_steps threshold seed_depth coin_type mix_phi prune_interval
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

/* ========== Memory guard ========== */
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

/* ========== Hash table and sites ========== */
#define HASH_BITS 27
#define HASH_SIZE (1<<HASH_BITS)
#define HASH_MASK (HASH_SIZE-1)
#define MAX_SITES 60000000

typedef struct { long kx,ky,kz; int id; } hentry;
static hentry *htab;

typedef struct {
    vec3 pos, dirs[4];
    int r_next, r_prev;
    int l_next, l_prev;
    int r_face, l_face;
} site_t;
static site_t *sites;
static int nsites = 0;

static double complex *psi, *tmp;  /* spinor arrays, 4 components per site */

/* Free list for site ID reuse (pruning) */
static int *free_list;
static int free_count = 0;
static int g_prune_interval = 0;

/* Cumulative probability accounting */
static double g_total_absorbed = 0;  /* too small to create new site */
static double g_total_pruned = 0;    /* removed by chain-end pruning */

static void init_storage(void) {
    htab = calloc(HASH_SIZE, sizeof(hentry));
    for (int i = 0; i < HASH_SIZE; i++) htab[i].id = -1;
    sites = malloc(MAX_SITES * sizeof(site_t));
    psi = calloc(4 * (size_t)MAX_SITES, sizeof(double complex));
    tmp = calloc(4 * (size_t)MAX_SITES, sizeof(double complex));
    free_list = malloc(MAX_SITES * sizeof(int));
    if (!htab || !sites || !psi || !tmp || !free_list) {
        fprintf(stderr, "Failed to allocate storage\n"); exit(2);
    }
    fprintf(stderr, "Storage: sites=%.0f MB, psi+tmp=%.0f MB\n",
            (double)MAX_SITES*sizeof(site_t)/(1024*1024),
            2.0*4*MAX_SITES*sizeof(double complex)/(1024*1024));
}

static void poskey(vec3 p, long *kx, long *ky, long *kz) {
    const double tol = 1e-7;
    *kx=(long)round(p.x/tol); *ky=(long)round(p.y/tol); *kz=(long)round(p.z/tol);
}
static unsigned hfn(long kx, long ky, long kz) {
    return (unsigned)(((unsigned long)(kx*73856093L)^(unsigned long)(ky*19349663L)^
            (unsigned long)(kz*83492791L))&HASH_MASK);
}
static int site_find(vec3 pos) {
    long kx,ky,kz; poskey(pos,&kx,&ky,&kz); unsigned h=hfn(kx,ky,kz);
    for(int p=0;p<HASH_SIZE;p++){unsigned i=(h+p)&HASH_MASK;
        if(htab[i].id==-1) return -1;
        if(htab[i].id==-2) continue;  /* tombstone — skip */
        if(htab[i].kx==kx&&htab[i].ky==ky&&htab[i].kz==kz) return htab[i].id;}
    return -1;
}
static int site_insert(vec3 pos, vec3 dirs[4]) {
    long kx,ky,kz; poskey(pos,&kx,&ky,&kz); unsigned h=hfn(kx,ky,kz);
    int first_tomb = -1;
    for(int p=0;p<HASH_SIZE;p++){unsigned i=(h+p)&HASH_MASK;
        if(htab[i].id==-2) { if(first_tomb<0) first_tomb=(int)i; continue; }
        if(htab[i].id==-1){
            int slot = (first_tomb >= 0) ? first_tomb : (int)i;
            int id;
            if (free_count > 0) id = free_list[--free_count];
            else { if(nsites>=MAX_SITES){fprintf(stderr,"Too many sites\n");exit(1);} id=nsites++; }
            htab[slot]=(hentry){kx,ky,kz,id};
            sites[id].pos=pos; memcpy(sites[id].dirs,dirs,sizeof(vec3)*4);
            sites[id].r_next=sites[id].r_prev=-1;
            sites[id].l_next=sites[id].l_prev=-1;
            sites[id].r_face=sites[id].l_face=-1;
            memset(&psi[4*id], 0, 4*sizeof(double complex));
            return id;}
        if(htab[i].kx==kx&&htab[i].ky==ky&&htab[i].kz==kz) return htab[i].id;}
    fprintf(stderr,"Hash full\n"); exit(1);
}

/* ========== Site removal (for pruning) ========== */
static void site_remove(int id) {
    long kx,ky,kz; poskey(sites[id].pos,&kx,&ky,&kz); unsigned h=hfn(kx,ky,kz);
    for(int p=0;p<HASH_SIZE;p++){unsigned i=(h+p)&HASH_MASK;
        if(htab[i].id==-1) break;
        if(htab[i].id==-2) continue;
        if(htab[i].kx==kx&&htab[i].ky==ky&&htab[i].kz==kz){
            htab[i].id=-2; break;
        }
    }
    memset(&psi[4*id], 0, 4*sizeof(double complex));
    sites[id].r_next=sites[id].r_prev=-1;
    sites[id].l_next=sites[id].l_prev=-1;
    sites[id].r_face=sites[id].l_face=-1;
    free_list[free_count++] = id;
}

/* ========== Chain threading ========== */
static const int PAT_R[4]={1,3,0,2}, PAT_L[4]={0,1,2,3};

static int next_face(const int pat[4], int cur_face) {
    for (int i = 0; i < 4; i++)
        if (pat[i] == cur_face) return pat[(i+1)%4];
    return pat[0];
}
static int prev_face(const int pat[4], int cur_face) {
    for (int i = 0; i < 4; i++)
        if (pat[i] == cur_face) return pat[(i+3)%4];
    return pat[0];
}

/* ========== Chain-first site generation ========== */

typedef struct { int site_id; int spawn_type; /* 0=L, 1=R */ } chain_seed_t;

/* Density grid globals */
static double g_grid_half;
static int g_grid_n;
static int *g_grid_count;
#define MAX_PER_CELL 8

#define GRID_IDX(pos) ({ \
    int gx = (int)((pos.x + g_grid_half) / 1.0); \
    int gy = (int)((pos.y + g_grid_half) / 1.0); \
    int gz = (int)((pos.z + g_grid_half) / 1.0); \
    if (gx < 0) gx = 0; if (gx >= g_grid_n) gx = g_grid_n-1; \
    if (gy < 0) gy = 0; if (gy >= g_grid_n) gy = g_grid_n-1; \
    if (gz < 0) gz = 0; if (gz >= g_grid_n) gz = g_grid_n-1; \
    gx * g_grid_n * g_grid_n + gy * g_grid_n + gz; \
})

static int extend_chain_dir(int start_site, const int pat[4], int is_r,
                             int forward,
                             double sigma, double seed_thresh,
                             chain_seed_t *seed_queue, int *sq_tail) {
    int linked = 0;
    int cur = start_site;
    int cur_face = is_r ? sites[cur].r_face : sites[cur].l_face;

    int step_pat[4];
    if (forward) {
        for (int i = 0; i < 4; i++) step_pat[i] = pat[i];
    } else {
        step_pat[0] = pat[3]; step_pat[1] = pat[2];
        step_pat[2] = pat[1]; step_pat[3] = pat[0];
    }

    vec3 p = sites[cur].pos;
    vec3 d[4]; memcpy(d, sites[cur].dirs, sizeof(d));

    for (int step = 0; step < MAX_SITES; step++) {
        int step_face;
        if (forward) {
            step_face = cur_face;
        } else {
            step_face = prev_face(pat, cur_face);
        }

        helix_step(&p, d, step_face);
        if ((step+1) % 8 == 0) reorth(d);

        /* Gaussian cutoff */
        double r2 = v3dot(p, p);
        if (exp(-r2 / (2*sigma*sigma)) < seed_thresh) break;

        /* Check if site already exists */
        int nb = site_find(p);
        if (nb >= 0) {
            int nb_face;
            if (forward) nb_face = next_face(pat, cur_face);
            else         nb_face = step_face;

            if (is_r) {
                if (forward && sites[nb].r_prev >= 0) break;
                if (!forward && sites[nb].r_next >= 0) break;
                if (forward) {
                    sites[cur].r_next = nb; sites[nb].r_prev = cur;
                } else {
                    sites[cur].r_prev = nb; sites[nb].r_next = cur;
                }
                if (sites[nb].r_face < 0) sites[nb].r_face = nb_face;
            } else {
                if (forward && sites[nb].l_prev >= 0) break;
                if (!forward && sites[nb].l_next >= 0) break;
                if (forward) {
                    sites[cur].l_next = nb; sites[nb].l_prev = cur;
                } else {
                    sites[cur].l_prev = nb; sites[nb].l_next = cur;
                }
                if (sites[nb].l_face < 0) sites[nb].l_face = nb_face;
            }
            cur = nb;
            cur_face = nb_face;
            linked++;
            continue;
        }

        /* New site — check density */
        int ci = GRID_IDX(p);
        if (g_grid_count[ci] >= MAX_PER_CELL) break;

        /* Create site */
        vec3 dd[4]; memcpy(dd, d, sizeof(dd)); reorth(dd);
        nb = site_insert(p, dd);
        g_grid_count[ci]++;

        int nb_face;
        if (forward) nb_face = next_face(pat, cur_face);
        else         nb_face = step_face;

        if (is_r) {
            if (forward) {
                sites[cur].r_next = nb; sites[nb].r_prev = cur;
            } else {
                sites[cur].r_prev = nb; sites[nb].r_next = cur;
            }
            sites[nb].r_face = nb_face;
        } else {
            if (forward) {
                sites[cur].l_next = nb; sites[nb].l_prev = cur;
            } else {
                sites[cur].l_prev = nb; sites[nb].l_next = cur;
            }
            sites[nb].l_face = nb_face;
        }

        /* Enqueue perpendicular chain seed */
        int perp_type = is_r ? 0 : 1;
        seed_queue[*sq_tail] = (chain_seed_t){nb, perp_type};
        (*sq_tail)++;

        cur = nb;
        cur_face = nb_face;
        linked++;
    }
    return linked;
}

static int generate_sites_chain_first(double sigma, double seed_thresh) {
    vec3 d0[4]; init_tet(d0);
    site_insert((vec3){0,0,0}, d0);
    g_grid_count[GRID_IDX(((vec3){0,0,0}))]++;

    chain_seed_t *seed_queue = malloc(MAX_SITES * sizeof(chain_seed_t));
    int sq_head = 0, sq_tail = 0;

    seed_queue[sq_tail++] = (chain_seed_t){0, 1};  /* R-chain */
    seed_queue[sq_tail++] = (chain_seed_t){0, 0};  /* L-chain */

    int n_chains = 0;

    while (sq_head < sq_tail) {
        chain_seed_t seed = seed_queue[sq_head++];
        int s = seed.site_id;
        int is_r = seed.spawn_type;

        if (is_r && sites[s].r_face >= 0) continue;
        if (!is_r && sites[s].l_face >= 0) continue;

        const int *pat = is_r ? PAT_R : PAT_L;

        if (is_r) sites[s].r_face = pat[0];
        else      sites[s].l_face = pat[0];

        int fwd = extend_chain_dir(s, pat, is_r, 1, sigma, seed_thresh,
                                    seed_queue, &sq_tail);
        int bwd = extend_chain_dir(s, pat, is_r, 0, sigma, seed_thresh,
                                    seed_queue, &sq_tail);

        if (fwd + bwd > 0) n_chains++;

        if (nsites % 500000 == 0 && nsites > 0)
            fprintf(stderr, "  Chain gen: %d sites, %d chains, queue %d/%d\n",
                    nsites, n_chains, sq_head, sq_tail);
    }

    free(seed_queue);
    return n_chains;
}

/* ========== Chain-end pruning ========== */

static int check_prune_eligible(int s, int is_r, int is_fwd, double thresh2) {
    double amp2 = 0;
    for (int a = 0; a < 4; a++) amp2 += creal(psi[4*s+a]*conj(psi[4*s+a]));
    if (amp2 >= thresh2) return 0;

    int nb = is_fwd ? (is_r ? sites[s].r_prev : sites[s].l_prev)
                    : (is_r ? sites[s].r_next : sites[s].l_next);
    if (nb < 0) return 1;

    int face = is_r ? sites[nb].r_face : sites[nb].l_face;
    if (face < 0) return 1;

    vec3 dv = sites[nb].dirs[face];
    c4x4 tau; make_tau(tau, dv);

    c4x4 P; c4_eye(P);
    double sign = is_fwd ? 1.0 : -1.0;
    for (int a = 0; a < 4; a++)
        for (int b = 0; b < 4; b++)
            P[a][b] = 0.5*(P[a][b] + sign*tau[a][b]);

    double flow2 = 0;
    for (int a = 0; a < 4; a++) {
        double complex v = 0;
        for (int b = 0; b < 4; b++) v += P[a][b] * psi[4*nb+b];
        flow2 += creal(v*conj(v));
    }
    return flow2 < thresh2;
}

static void unlink_chain_end(int s, int is_r, int is_fwd) {
    if (is_fwd) {
        int nb = is_r ? sites[s].r_prev : sites[s].l_prev;
        if (nb >= 0) {
            if (is_r) sites[nb].r_next = -1;
            else      sites[nb].l_next = -1;
        }
        if (is_r) { sites[s].r_prev = -1; sites[s].r_face = -1; }
        else      { sites[s].l_prev = -1; sites[s].l_face = -1; }
    } else {
        int nb = is_r ? sites[s].r_next : sites[s].l_next;
        if (nb >= 0) {
            if (is_r) sites[nb].r_prev = -1;
            else      sites[nb].l_prev = -1;
        }
        if (is_r) { sites[s].r_next = -1; sites[s].r_face = -1; }
        else      { sites[s].l_next = -1; sites[s].l_face = -1; }
    }

    if (sites[s].r_face < 0 && sites[s].l_face < 0)
        site_remove(s);
}

static int prune_chain_ends(double thresh2, double *prob_pruned) {
    int total_pruned = 0, pruned_this_pass;
    do {
        pruned_this_pass = 0;
        for (int s = 0; s < nsites; s++) {
            if (sites[s].r_face < 0 && sites[s].l_face < 0) continue;

            if (sites[s].r_face >= 0 && sites[s].r_next == -1) {
                if (check_prune_eligible(s, 1, 1, thresh2)) {
                    for (int a = 0; a < 4; a++)
                        *prob_pruned += creal(psi[4*s+a]*conj(psi[4*s+a]));
                    unlink_chain_end(s, 1, 1);
                    pruned_this_pass++;
                    continue;
                }
            }
            if (sites[s].r_face >= 0 && sites[s].r_prev == -1) {
                if (check_prune_eligible(s, 1, 0, thresh2)) {
                    for (int a = 0; a < 4; a++)
                        *prob_pruned += creal(psi[4*s+a]*conj(psi[4*s+a]));
                    unlink_chain_end(s, 1, 0);
                    pruned_this_pass++;
                    continue;
                }
            }
            if (sites[s].l_face >= 0 && sites[s].l_next == -1) {
                if (check_prune_eligible(s, 0, 1, thresh2)) {
                    for (int a = 0; a < 4; a++)
                        *prob_pruned += creal(psi[4*s+a]*conj(psi[4*s+a]));
                    unlink_chain_end(s, 0, 1);
                    pruned_this_pass++;
                    continue;
                }
            }
            if (sites[s].l_face >= 0 && sites[s].l_prev == -1) {
                if (check_prune_eligible(s, 0, 0, thresh2)) {
                    for (int a = 0; a < 4; a++)
                        *prob_pruned += creal(psi[4*s+a]*conj(psi[4*s+a]));
                    unlink_chain_end(s, 0, 0);
                    pruned_this_pass++;
                    continue;
                }
            }
        }
        total_pruned += pruned_this_pass;
    } while (pruned_this_pass > 0);
    return total_pruned;
}

/* ========== Adaptive shift operator ========== */

static int try_extend_fwd(int s, int is_r, const int pat[4],
                           double complex shifted[4], double thresh2) {
    double amp2 = 0;
    for (int a = 0; a < 4; a++) amp2 += creal(shifted[a]*conj(shifted[a]));
    if (amp2 < thresh2) return -1;

    int face = is_r ? sites[s].r_face : sites[s].l_face;
    int nf = next_face(pat, face);

    vec3 p = sites[s].pos;
    vec3 d[4]; memcpy(d, sites[s].dirs, sizeof(d));
    helix_step(&p, d, face);

    int nb = site_find(p);
    if (nb >= 0) {
        if (is_r) { sites[s].r_next = nb; if (sites[nb].r_prev<0) { sites[nb].r_prev=s; sites[nb].r_face=nf; } }
        else      { sites[s].l_next = nb; if (sites[nb].l_prev<0) { sites[nb].l_prev=s; sites[nb].l_face=nf; } }
        return -1;
    }

    reorth(d);
    nb = site_insert(p, d);
    if (is_r) { sites[s].r_next = nb; sites[nb].r_prev = s; sites[nb].r_face = nf; }
    else      { sites[s].l_next = nb; sites[nb].l_prev = s; sites[nb].l_face = nf; }
    return nb;
}

static int try_extend_bwd(int s, int is_r, const int pat[4],
                           double complex shifted[4], double thresh2) {
    double amp2 = 0;
    for (int a = 0; a < 4; a++) amp2 += creal(shifted[a]*conj(shifted[a]));
    if (amp2 < thresh2) return -1;

    int face = is_r ? sites[s].r_face : sites[s].l_face;
    int pf = prev_face(pat, face);

    vec3 p = sites[s].pos;
    vec3 dd[4]; memcpy(dd, sites[s].dirs, sizeof(dd));
    helix_step(&p, dd, pf);

    int nb = site_find(p);
    if (nb >= 0) {
        if (is_r) { sites[s].r_prev = nb; if (sites[nb].r_next<0) { sites[nb].r_next=s; sites[nb].r_face=pf; } }
        else      { sites[s].l_prev = nb; if (sites[nb].l_next<0) { sites[nb].l_next=s; sites[nb].l_face=pf; } }
        return -1;
    }

    reorth(dd);
    nb = site_insert(p, dd);
    if (is_r) { sites[s].r_prev = nb; sites[nb].r_next = s; sites[nb].r_face = pf; }
    else      { sites[s].l_prev = nb; sites[nb].l_next = s; sites[nb].l_face = pf; }
    return nb;
}

static void apply_shift_adaptive(int is_r, const int pat[4], double thresh2,
                                  int *n_created, double *prob_absorbed) {
    int ns = nsites;
    *n_created = 0;
    *prob_absorbed = 0;

    memset(tmp, 0, 4 * (size_t)ns * sizeof(double complex));

    for (int s = 0; s < ns; s++) {
        int face = is_r ? sites[s].r_face : sites[s].l_face;

        if (face < 0) {
            for (int a = 0; a < 4; a++) tmp[4*s+a] = psi[4*s+a];
            continue;
        }

        vec3 dv = sites[s].dirs[face];
        c4x4 tau; make_tau(tau, dv);
        c4x4 Pp, Pm; c4_eye(Pp); c4_eye(Pm);
        for (int a=0;a<4;a++) for(int b=0;b<4;b++) {
            Pp[a][b] = 0.5*(Pp[a][b]+tau[a][b]);
            Pm[a][b] = 0.5*(Pm[a][b]-tau[a][b]);
        }

        /* P+ component -> forward */
        {
            double complex shifted[4] = {0};
            for (int a=0;a<4;a++) for(int b=0;b<4;b++)
                shifted[a] += Pp[a][b] * psi[4*s+b];

            int nb = is_r ? sites[s].r_next : sites[s].l_next;
            if (nb >= 0) {
                int fn = is_r ? sites[nb].r_face : sites[nb].l_face;
                if (fn < 0) {
                    if (is_r) sites[s].r_next = -1; else sites[s].l_next = -1;
                    nb = -1;
                }
            }
            if (nb < 0)
                nb = try_extend_fwd(s, is_r, pat, shifted, thresh2);

            if (nb >= 0) {
                int fn = is_r ? sites[nb].r_face : sites[nb].l_face;
                vec3 dn = sites[nb].dirs[fn];
                c4x4 tn; make_tau(tn, dn);
                c4x4 U, bl; frame_transport(tau, tn, U); c4_mul(bl, U, Pp);
                double complex result[4] = {0};
                for (int a=0;a<4;a++) for(int b=0;b<4;b++)
                    result[a] += bl[a][b] * psi[4*s+b];
                for (int a=0;a<4;a++) tmp[4*nb+a] += result[a];
                if (nb >= ns) (*n_created)++;
            } else {
                for (int a=0;a<4;a++)
                    *prob_absorbed += creal(shifted[a]*conj(shifted[a]));
            }
        }

        /* P- component -> backward */
        {
            double complex shifted[4] = {0};
            for (int a=0;a<4;a++) for(int b=0;b<4;b++)
                shifted[a] += Pm[a][b] * psi[4*s+b];

            int nb = is_r ? sites[s].r_prev : sites[s].l_prev;
            if (nb >= 0) {
                int fp = is_r ? sites[nb].r_face : sites[nb].l_face;
                if (fp < 0) {
                    if (is_r) sites[s].r_prev = -1; else sites[s].l_prev = -1;
                    nb = -1;
                }
            }
            if (nb < 0)
                nb = try_extend_bwd(s, is_r, pat, shifted, thresh2);

            if (nb >= 0) {
                int fp = is_r ? sites[nb].r_face : sites[nb].l_face;
                vec3 dp = sites[nb].dirs[fp];
                c4x4 tp; make_tau(tp, dp);
                c4x4 U, bl; frame_transport(tau, tp, U); c4_mul(bl, U, Pm);
                for (int a=0;a<4;a++) for(int b=0;b<4;b++)
                    tmp[4*nb+a] += bl[a][b] * psi[4*s+b];
                if (nb >= ns) (*n_created)++;
            } else {
                for (int a=0;a<4;a++)
                    *prob_absorbed += creal(shifted[a]*conj(shifted[a]));
            }
        }
    }

    /* Swap psi and tmp pointers (avoids full memcpy) */
    double complex *swap = psi;
    psi = tmp;
    tmp = swap;
}

/* ========== Coin operator ========== */

static void apply_coin_dir(double *d, int n, double ct, double st) {
    double complex out[4];
    for (int a = 0; a < 4; a++) {
        double complex s = 0;
        for (int b = 0; b < 4; b++) {
            double complex ea = d[0]*ALPHA[0][a][b] + d[1]*ALPHA[1][a][b] + d[2]*ALPHA[2][a][b];
            double complex Cab = ct*(a==b?1:0) - I*st*ea;
            s += Cab * psi[4*n+b];
        }
        out[a] = s;
    }
    for (int a = 0; a < 4; a++) psi[4*n+a] = out[a];
}

/* coin_type: 0 = e·α (original), 1 = dual parity f·α (f₁,f₂ ⊥ e) */
static int g_coin_type = 1;

static void apply_coin(int is_r, double ct, double st) {
    int ns = nsites;
    for (int n = 0; n < ns; n++) {
        int face = is_r ? sites[n].r_face : sites[n].l_face;
        if (face < 0) continue;

        vec3 e = sites[n].dirs[face];
        double en = v3norm(e);
        if (en < 1e-15) continue;
        vec3 ehat = v3scale(e, 1.0/en);

        if (g_coin_type == 0) {
            double d[3] = {e.x, e.y, e.z};
            apply_coin_dir(d, n, ct, st);
        } else {
            /* Dual parity coin: f₁·α then f₂·α, both ⊥ e */
            vec3 f1;
            f1.x = ehat.y; f1.y = -ehat.x; f1.z = 0;
            double fn = v3norm(f1);
            if (fn < 1e-10) {
                f1.x = -ehat.z; f1.y = 0; f1.z = ehat.x;
                fn = v3norm(f1);
            }
            f1 = v3scale(f1, 1.0/fn);
            vec3 f2;
            f2.x = ehat.y*f1.z - ehat.z*f1.y;
            f2.y = ehat.z*f1.x - ehat.x*f1.z;
            f2.z = ehat.x*f1.y - ehat.y*f1.x;
            fn = v3norm(f2);
            f2 = v3scale(f2, 1.0/fn);

            double d1[3] = {f1.x, f1.y, f1.z};
            double d2[3] = {f2.x, f2.y, f2.z};
            apply_coin_dir(d1, n, ct, st);
            apply_coin_dir(d2, n, ct, st);
        }
    }
}

/* ========== V mixing ========== */

static double g_mix_phi = 0.0;

static void apply_vmix(int is_r) {
    if (g_mix_phi == 0.0) return;
    double cp = cos(g_mix_phi), sp = sin(g_mix_phi);
    int ns = nsites;
    for (int s = 0; s < ns; s++) {
        int face = is_r ? sites[s].r_face : sites[s].l_face;
        if (face < 0) continue;

        vec3 dv = sites[s].dirs[face];
        c4x4 tau; make_tau(tau, dv);

        c4x4 Pp, Pm; c4_eye(Pp); c4_eye(Pm);
        for (int a=0;a<4;a++) for(int b=0;b<4;b++) {
            Pp[a][b] = 0.5*(Pp[a][b]+tau[a][b]);
            Pm[a][b] = 0.5*(Pm[a][b]-tau[a][b]);
        }

        /* Find orthonormal bases for P+ and P- via Gram-Schmidt */
        double complex pp_basis[4][2], pm_basis[4][2];
        int np_found=0, nm_found=0;
        for (int col=0; col<4 && (np_found<2||nm_found<2); col++) {
            if (np_found < 2) {
                double complex v[4];
                for(int a=0;a<4;a++) v[a]=Pp[a][col];
                for(int j=0;j<np_found;j++){
                    double complex dot=0;
                    for(int a=0;a<4;a++) dot+=conj(pp_basis[a][j])*v[a];
                    for(int a=0;a<4;a++) v[a]-=dot*pp_basis[a][j];
                }
                double nm=0; for(int a=0;a<4;a++) nm+=creal(v[a]*conj(v[a]));
                if(nm>1e-10){double inv=1.0/sqrt(nm);
                    for(int a=0;a<4;a++) pp_basis[a][np_found]=v[a]*inv; np_found++;}
            }
            if (nm_found < 2) {
                double complex v[4];
                for(int a=0;a<4;a++) v[a]=Pm[a][col];
                for(int j=0;j<nm_found;j++){
                    double complex dot=0;
                    for(int a=0;a<4;a++) dot+=conj(pm_basis[a][j])*v[a];
                    for(int a=0;a<4;a++) v[a]-=dot*pm_basis[a][j];
                }
                double nm=0; for(int a=0;a<4;a++) nm+=creal(v[a]*conj(v[a]));
                if(nm>1e-10){double inv=1.0/sqrt(nm);
                    for(int a=0;a<4;a++) pm_basis[a][nm_found]=v[a]*inv; nm_found++;}
            }
        }

        /* M = Σ_j |pm_j><pp_j| + |pp_j><pm_j| */
        c4x4 V; c4_zero(V);
        for(int j=0;j<2;j++)
            for(int a=0;a<4;a++) for(int b=0;b<4;b++){
                V[a][b]+=pm_basis[a][j]*conj(pp_basis[b][j]);
                V[a][b]+=pp_basis[a][j]*conj(pm_basis[b][j]);
            }
        for(int a=0;a<4;a++) for(int b=0;b<4;b++)
            V[a][b] = cp*(a==b?1:0) + I*sp*V[a][b];

        double complex out[4] = {0};
        for(int a=0;a<4;a++) for(int b=0;b<4;b++)
            out[a] += V[a][b] * psi[4*s+b];
        for(int a=0;a<4;a++) psi[4*s+a] = out[a];
    }
}

/* ========== Main ========== */
int main(int argc, char **argv) {
    int seed_depth = 0;
    double theta = 0.5, sigma = 3.0, threshold = 1e-10;
    int n_steps = 50;
    if (argc > 1) theta = atof(argv[1]);
    if (argc > 2) sigma = atof(argv[2]);
    if (argc > 3) n_steps = atoi(argv[3]);
    if (argc > 4) threshold = atof(argv[4]);
    if (argc > 5) seed_depth = atoi(argv[5]);
    if (argc > 6) g_coin_type = atoi(argv[6]);
    if (argc > 7) g_mix_phi = atof(argv[7]);
    if (argc > 8) g_prune_interval = atoi(argv[8]);
    double thresh2 = threshold * threshold;
    double ct = cos(theta), st = sin(theta);

    if (seed_depth <= 0) {
        seed_depth = (int)(4.0 * sigma / (2.0/3.0)) + 2;
        if (seed_depth < 4) seed_depth = 4;
    }

    fprintf(stderr, "=== walk_adaptive: theta=%.3f sigma=%.1f steps=%d thresh=%.1e coin=%s mix_phi=%.4f prune=%d ===\n",
            theta, sigma, n_steps, threshold,
            g_coin_type==0 ? "e.alpha" : "dual_parity", g_mix_phi,
            g_prune_interval);
    fprintf(stderr, "Memory available: %ld MB\n", get_avail_mb());

    init_dirac();
    init_storage();

    /* ---- Generate sites via chain-first approach ---- */
    double seed_thresh = 1e-4;
    double step_len = 2.0/3.0;
    int max_chain_len = (int)(4.0 * sigma / step_len) + 5;
    g_grid_half = max_chain_len * step_len + 5.0;
    g_grid_n = (int)(2.0 * g_grid_half / 1.0) + 1;
    if (g_grid_n > 500) g_grid_n = 500;
    g_grid_count = calloc(g_grid_n * g_grid_n * g_grid_n, sizeof(int));

    fprintf(stderr, "\n--- Chain-first site generation (grid %d³) ---\n", g_grid_n);

    int n_chains = generate_sites_chain_first(sigma, seed_thresh);
    free(g_grid_count);
    int n_seed = nsites;

    /* Report chain coverage */
    int no_r = 0, no_l = 0, no_both = 0;
    for (int s = 0; s < n_seed; s++) {
        int has_r = sites[s].r_face >= 0;
        int has_l = sites[s].l_face >= 0;
        if (!has_r) no_r++;
        if (!has_l) no_l++;
        if (!has_r && !has_l) no_both++;
    }
    fprintf(stderr, "Seed: %d sites, %d chains\n", n_seed, n_chains);
    fprintf(stderr, "Coverage: %d without R-chain, %d without L-chain, %d without either\n",
            no_r, no_l, no_both);

    /* ---- Initialize wavepacket ---- */
    double norm = 0;
    for (int n = 0; n < nsites; n++) {
        double x = sites[n].pos.x, y = sites[n].pos.y, z = sites[n].pos.z;
        double r2 = x*x + y*y + z*z;
        double w = exp(-r2 / (2*sigma*sigma));
        psi[4*n] = w;
        norm += w*w;
    }
    norm = sqrt(norm);
    for (int i = 0; i < 4*nsites; i++) psi[i] /= norm;
    fprintf(stderr, "Wavepacket initialized (sigma=%.1f)\n", sigma);

    /* ---- Radial histogram output ---- */
    FILE *radf = NULL;
    {
        const char *rf = getenv("RADIAL_HIST");
        if (rf) { radf = fopen(rf, "w"); if (radf) fprintf(stderr, "Radial histograms -> %s\n", rf); }
    }

    /* ---- Time evolution ---- */
    printf("# theta=%.4f sigma=%.1f n_steps=%d thresh=%.1e prune=%d\n",
           theta, sigma, n_steps, threshold, g_prune_interval);
    printf("# t norm r2 x2 y2 z2 r95 nsites absorbed pruned\n");

    for (int t = 0; t <= n_steps; t++) {
        /* Compute observables */
        double total_prob = 0;
        double mx2=0, my2=0, mz2=0;
        for (int n = 0; n < nsites; n++) {
            double p = 0;
            for (int a = 0; a < 4; a++)
                p += creal(psi[4*n+a]*conj(psi[4*n+a]));
            total_prob += p;
        }
        double pnorm = sqrt(total_prob);
        for (int n = 0; n < nsites; n++) {
            double p = 0;
            for (int a = 0; a < 4; a++)
                p += creal(psi[4*n+a]*conj(psi[4*n+a]));
            double pw = p / total_prob;
            double x=sites[n].pos.x, y=sites[n].pos.y, z=sites[n].pos.z;
            mx2 += pw*x*x; my2 += pw*y*y; mz2 += pw*z*z;
        }

        /* r95 */
        double r95 = 0;
        {
            double rmax = 0;
            for (int n = 0; n < nsites; n++) {
                double r = v3norm(sites[n].pos);
                if (r > rmax) rmax = r;
            }
            double dr = 0.5;
            int nbins = (int)(rmax / dr) + 1;
            if (nbins < 10) nbins = 10;
            double *bp = calloc(nbins+1, sizeof(double));
            for (int n = 0; n < nsites; n++) {
                double r = v3norm(sites[n].pos);
                double p = 0;
                for (int a = 0; a < 4; a++)
                    p += creal(psi[4*n+a]*conj(psi[4*n+a]));
                int b = (int)(r / dr);
                if (b > nbins) b = nbins;
                bp[b] += p / total_prob;
            }
            double cum = 0;
            for (int b = 0; b <= nbins; b++) {
                cum += bp[b];
                if (cum >= 0.95) { r95 = b * dr; break; }
            }

            if (radf) {
                fprintf(radf, "# t=%d norm=%.6f\n", t, pnorm);
                for (int b = 0; b <= nbins; b++) {
                    double r = (b + 0.5) * dr;
                    fprintf(radf, "%d %.4f %.8e\n", t, r, bp[b]);
                }
            }
            free(bp);
        }

        printf("%d %.6f %.2f %.2f %.2f %.2f %.2f %d %.6e %.6e\n",
               t, pnorm, mx2+my2+mz2, mx2, my2, mz2, r95, nsites,
               g_total_absorbed, g_total_pruned);
        fflush(stdout);

        if (t < n_steps) {
            if (t % 5 == 0)
                fprintf(stderr, "  step %d/%d: %d sites, norm=%.6f absorbed=%.2e pruned=%.2e (mem: %ld MB)\n",
                        t, n_steps, nsites, pnorm,
                        g_total_absorbed, g_total_pruned, get_avail_mb());

            int cr_l=0, cr_r=0;
            double abs_l=0, abs_r=0;

            /* W = V_R · S_R · C_R · V_L · S_L · C_L, applied right to left */
            apply_coin(0, ct, st);
            apply_shift_adaptive(0, PAT_L, thresh2, &cr_l, &abs_l);
            apply_vmix(0);
            apply_coin(1, ct, st);
            apply_shift_adaptive(1, PAT_R, thresh2, &cr_r, &abs_r);
            apply_vmix(1);

            g_total_absorbed += abs_l + abs_r;

            if (g_prune_interval > 0 && t > 0 && t % g_prune_interval == 0) {
                double pp = 0;
                int pruned = prune_chain_ends(thresh2, &pp);
                g_total_pruned += pp;
                if (pruned > 0)
                    fprintf(stderr, "  step %d: pruned %d sites (free=%d, prob=%.2e)\n",
                            t, pruned, free_count, pp);
            }
        }
    }

    if (radf) fclose(radf);
    fprintf(stderr, "\nDone. Final: %d sites, absorbed=%.6e, pruned=%.6e\n",
            nsites, g_total_absorbed, g_total_pruned);
    free(htab); free(sites); free(psi); free(tmp); free(free_list);
    return 0;
}

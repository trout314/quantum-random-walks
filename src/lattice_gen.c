/*
 * lattice_gen.c — Generate the tetrahedral walk lattice using the
 * perpendicular helix tree construction.
 *
 * The perpendicular partner patterns are hardcoded:
 *   R = [1,3,0,2]  ⊥  L = [0,1,2,3]   (always, at every site)
 *
 * Algorithm:
 *   Gen 0: Trace R₀ through origin
 *   Gen 1: From each R₀ site without L-membership, trace L-chains (parallel)
 *   Gen 2: From each new L site without R-membership, trace R-chains (parallel)
 *   ... repeat ...
 *
 * Steps 1, 2, 3, ... are embarrassingly parallel: each chain trace is independent.
 *
 * Build: gcc -O2 -fopenmp -o lattice_gen lattice_gen.c -lm
 * Usage: ./lattice_gen [chain_len] [n_generations]
 *
 * Output format (stdout):
 *   SITES <n_sites>
 *   <id> <x> <y> <z> <d0x> <d0y> <d0z> ... <d3x> <d3y> <d3z> <r_chain> <r_pos> <l_chain> <l_pos>
 *   RCHAINS <n>
 *   <chain_id> <len> <site_0> <face_0> <site_1> <face_1> ...
 *   LCHAINS <n>
 *   <chain_id> <len> <site_0> <face_0> <site_1> <face_1> ...
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ========== 3D vector ========== */

typedef struct { double x, y, z; } vec3;

static inline vec3 v3add(vec3 a, vec3 b) { return (vec3){a.x+b.x, a.y+b.y, a.z+b.z}; }
static inline vec3 v3scale(vec3 a, double s) { return (vec3){a.x*s, a.y*s, a.z*s}; }
static inline double v3dot(vec3 a, vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline double v3norm(vec3 a) { return sqrt(v3dot(a, a)); }

static inline vec3 v3reflect(vec3 v, vec3 n) {
    double d = 2.0 * v3dot(v, n);
    return (vec3){v.x - d*n.x, v.y - d*n.y, v.z - d*n.z};
}

/* Re-orthogonalize: enforce Σe_a=0 and |e_a|=1 */
static void reorth(vec3 d[4]) {
    vec3 m = {0,0,0};
    for (int a = 0; a < 4; a++) m = v3add(m, d[a]);
    m = v3scale(m, 0.25);
    for (int a = 0; a < 4; a++) {
        d[a] = (vec3){d[a].x-m.x, d[a].y-m.y, d[a].z-m.z};
        double n = v3norm(d[a]);
        if (n > 1e-15) d[a] = v3scale(d[a], 1.0/n);
    }
}

static void init_tet(vec3 d[4]) {
    d[0] = (vec3){0, 0, 1};
    d[1] = (vec3){2*sqrt(2.0)/3, 0, -1.0/3};
    d[2] = (vec3){-sqrt(2.0)/3, sqrt(6.0)/3, -1.0/3};
    d[3] = (vec3){-sqrt(2.0)/3, -sqrt(6.0)/3, -1.0/3};
}

/* Take one helix step: displace by -(2/3)e_a, reflect all dirs */
static void step(vec3 *pos, vec3 d[4], int face) {
    vec3 e = d[face];
    *pos = v3add(*pos, v3scale(e, -2.0/3.0));
    for (int a = 0; a < 4; a++)
        d[a] = v3reflect(d[a], e);
}

/* ========== Hardcoded perpendicular patterns ========== */

static const int PAT_R[4] = {1, 3, 0, 2};
static const int PAT_L[4] = {0, 1, 2, 3};

/* ========== Hash table ========== */

#define HASH_BITS 22
#define HASH_SIZE (1 << HASH_BITS)
#define HASH_MASK (HASH_SIZE - 1)
#define MAX_SITES 2000000

typedef struct {
    long kx, ky, kz;
    int id;     /* -1 = empty */
} hentry;

static hentry *htab;   /* allocated on heap */

typedef struct {
    vec3 pos;
    vec3 dirs[4];
    int r_chain, r_pos;
    int l_chain, l_pos;
} site_t;

static site_t *allsites;
static int nsites = 0;

static void htab_init(void) {
    htab = calloc(HASH_SIZE, sizeof(hentry));
    for (int i = 0; i < HASH_SIZE; i++) htab[i].id = -1;
    allsites = malloc(MAX_SITES * sizeof(site_t));
}

static void poskey(vec3 p, long *kx, long *ky, long *kz) {
    const double tol = 1e-7;
    *kx = (long)round(p.x / tol);
    *ky = (long)round(p.y / tol);
    *kz = (long)round(p.z / tol);
}

static unsigned hfn(long kx, long ky, long kz) {
    unsigned long h = (unsigned long)(kx*73856093L) ^
                      (unsigned long)(ky*19349663L) ^
                      (unsigned long)(kz*83492791L);
    return (unsigned)(h & HASH_MASK);
}

/* Thread-safe find-or-insert (uses atomic CAS on id field).
 * Returns site id. Sets *was_new = 1 if newly created. */
static int site_find_or_insert(vec3 pos, vec3 dirs[4], int *was_new) {
    long kx, ky, kz;
    poskey(pos, &kx, &ky, &kz);
    unsigned h = hfn(kx, ky, kz);
    *was_new = 0;

    for (int probe = 0; probe < HASH_SIZE; probe++) {
        unsigned idx = (h + probe) & HASH_MASK;
        int cur = htab[idx].id;

        if (cur == -1) {
            /* Try to claim this slot */
            #ifdef _OPENMP
            int expected = -1;
            if (__atomic_compare_exchange_n(&htab[idx].id, &expected, -2,
                                            0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
            #else
            if (htab[idx].id == -1) {
                htab[idx].id = -2; /* claim */
            #endif
                /* We claimed it. Allocate site. */
                int id;
                #ifdef _OPENMP
                #pragma omp atomic capture
                #endif
                { id = nsites; nsites++; }

                if (id >= MAX_SITES) {
                    fprintf(stderr, "Too many sites\n");
                    exit(1);
                }
                allsites[id].pos = pos;
                memcpy(allsites[id].dirs, dirs, sizeof(vec3)*4);
                allsites[id].r_chain = -1;
                allsites[id].l_chain = -1;
                htab[idx].kx = kx;
                htab[idx].ky = ky;
                htab[idx].kz = kz;
                __atomic_store_n(&htab[idx].id, id, __ATOMIC_SEQ_CST);
                *was_new = 1;
                return id;
            }
            /* Another thread claimed it, retry */
            while (__atomic_load_n(&htab[idx].id, __ATOMIC_SEQ_CST) == -2)
                ; /* spin */
            cur = htab[idx].id;
        }

        /* Spin if slot is being initialized */
        while (cur == -2) {
            cur = __atomic_load_n(&htab[idx].id, __ATOMIC_SEQ_CST);
        }

        if (htab[idx].kx == kx && htab[idx].ky == ky && htab[idx].kz == kz)
            return cur;
    }
    fprintf(stderr, "Hash table full\n");
    exit(1);
}

/* Non-thread-safe version for serial sections */
static int site_find(vec3 pos) {
    long kx, ky, kz;
    poskey(pos, &kx, &ky, &kz);
    unsigned h = hfn(kx, ky, kz);
    for (int probe = 0; probe < HASH_SIZE; probe++) {
        unsigned idx = (h + probe) & HASH_MASK;
        if (htab[idx].id == -1) return -1;
        if (htab[idx].kx == kx && htab[idx].ky == ky && htab[idx].kz == kz)
            return htab[idx].id;
    }
    return -1;
}

/* ========== Chain storage ========== */

#define MAX_CHAINS 500000
#define MAX_CLEN 200

typedef struct {
    int ids[MAX_CLEN];
    int faces[MAX_CLEN];
    int len;
} chain_t;

static chain_t *rchains, *lchains;
static int nrch = 0, nlch = 0;

/* Trace a helix from a starting site. Adds sites to the hash table.
 * Returns the chain. The chain is stored in the caller-provided chain_t. */
static void trace_chain(int start_id, const int pat[4],
                        int nfwd, int nbwd, chain_t *ch) {
    ch->len = 0;

    /* Backward (reverse pattern) */
    int rev[4] = {pat[3], pat[2], pat[1], pat[0]};
    int bwd_ids[MAX_CLEN];
    int bwd_faces[MAX_CLEN];
    int nbwd_actual = 0;
    {
        vec3 pos = allsites[start_id].pos;
        vec3 d[4]; memcpy(d, allsites[start_id].dirs, sizeof(d));
        for (int i = 0; i < nbwd && nbwd_actual < MAX_CLEN - nfwd - 1; i++) {
            int face = rev[i % 4];
            step(&pos, d, face);
            if ((i+1) % 8 == 0) reorth(d);
            int w;
            int sid = site_find_or_insert(pos, d, &w);
            bwd_ids[nbwd_actual] = sid;
            bwd_faces[nbwd_actual] = rev[(i+1) % 4]; /* face for next step at this site */
            nbwd_actual++;
        }
    }

    /* Add backward sites in reverse */
    for (int i = nbwd_actual - 1; i >= 0; i--) {
        ch->ids[ch->len] = bwd_ids[i];
        ch->faces[ch->len] = bwd_faces[i];
        ch->len++;
    }

    /* Start site */
    ch->ids[ch->len] = start_id;
    ch->faces[ch->len] = pat[0];
    ch->len++;

    /* Forward */
    {
        vec3 pos = allsites[start_id].pos;
        vec3 d[4]; memcpy(d, allsites[start_id].dirs, sizeof(d));
        for (int i = 0; i < nfwd && ch->len < MAX_CLEN; i++) {
            int face = pat[i % 4];
            step(&pos, d, face);
            if ((i+1) % 8 == 0) reorth(d);
            int w;
            int sid = site_find_or_insert(pos, d, &w);
            ch->ids[ch->len] = sid;
            ch->faces[ch->len] = pat[(i+1) % 4];
            ch->len++;
        }
    }
}

/* Assign chain membership to sites (serial, to avoid races on membership) */
static void assign_membership(chain_t *ch, int chain_id, int is_r) {
    for (int i = 0; i < ch->len; i++) {
        int sid = ch->ids[i];
        if (is_r) {
            if (allsites[sid].r_chain == -1) {
                allsites[sid].r_chain = chain_id;
                allsites[sid].r_pos = i;
            }
        } else {
            if (allsites[sid].l_chain == -1) {
                allsites[sid].l_chain = chain_id;
                allsites[sid].l_pos = i;
            }
        }
    }
}

/* ========== Main ========== */

int main(int argc, char **argv) {
    int chain_len = 15;
    int ngen = 2;
    if (argc > 1) chain_len = atoi(argv[1]);
    if (argc > 2) ngen = atoi(argv[2]);

    htab_init();
    rchains = malloc(MAX_CHAINS * sizeof(chain_t));
    lchains = malloc(MAX_CHAINS * sizeof(chain_t));

    /* Origin */
    vec3 d0[4]; init_tet(d0);
    vec3 origin = {0,0,0};
    int w;
    int oid = site_find_or_insert(origin, d0, &w);

    /* Gen 0: R₀ through origin (serial — single chain) */
    trace_chain(oid, PAT_R, chain_len, chain_len, &rchains[0]);
    assign_membership(&rchains[0], 0, 1);
    nrch = 1;
    fprintf(stderr, "Gen 0: %d sites, 1 R-chain\n", nsites);

    for (int gen = 0; gen < ngen; gen++) {
        /* ---- L-chains from R-sites without L-membership ---- */
        /* Collect seeds */
        int *l_seeds = malloc(nsites * sizeof(int));
        int nl_seeds = 0;
        for (int s = 0; s < nsites; s++) {
            if (allsites[s].r_chain >= 0 && allsites[s].l_chain == -1)
                l_seeds[nl_seeds++] = s;
        }

        /* Trace L-chains in parallel */
        chain_t *new_lch = malloc(nl_seeds * sizeof(chain_t));
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (int i = 0; i < nl_seeds; i++) {
            trace_chain(l_seeds[i], PAT_L, chain_len, chain_len, &new_lch[i]);
        }

        /* Assign membership (serial) */
        for (int i = 0; i < nl_seeds; i++) {
            if (nlch >= MAX_CHAINS) break;
            memcpy(&lchains[nlch], &new_lch[i], sizeof(chain_t));
            assign_membership(&lchains[nlch], nlch, 0);
            nlch++;
        }
        free(new_lch);
        free(l_seeds);

        fprintf(stderr, "Gen %d L: %d sites, %d R-chains, %d L-chains\n",
                gen, nsites, nrch, nlch);

        /* ---- R-chains from L-sites without R-membership ---- */
        int *r_seeds = malloc(nsites * sizeof(int));
        int nr_seeds = 0;
        for (int s = 0; s < nsites; s++) {
            if (allsites[s].l_chain >= 0 && allsites[s].r_chain == -1)
                r_seeds[nr_seeds++] = s;
        }

        chain_t *new_rch = malloc(nr_seeds * sizeof(chain_t));
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (int i = 0; i < nr_seeds; i++) {
            trace_chain(r_seeds[i], PAT_R, chain_len, chain_len, &new_rch[i]);
        }

        for (int i = 0; i < nr_seeds; i++) {
            if (nrch >= MAX_CHAINS) break;
            memcpy(&rchains[nrch], &new_rch[i], sizeof(chain_t));
            assign_membership(&rchains[nrch], nrch, 1);
            nrch++;
        }
        free(new_rch);
        free(r_seeds);

        fprintf(stderr, "Gen %d R: %d sites, %d R-chains, %d L-chains\n",
                gen, nsites, nrch, nlch);
    }

    /* Stats */
    int nboth = 0, ninterior = 0;
    for (int s = 0; s < nsites; s++) {
        if (allsites[s].r_chain >= 0 && allsites[s].l_chain >= 0) {
            nboth++;
            int ri = allsites[s].r_pos;
            int li = allsites[s].l_pos;
            int rlen = rchains[allsites[s].r_chain].len;
            int llen = lchains[allsites[s].l_chain].len;
            if (ri > 0 && ri < rlen-1 && li > 0 && li < llen-1)
                ninterior++;
        }
    }
    fprintf(stderr, "\n=== Summary ===\n");
    fprintf(stderr, "Sites: %d\n", nsites);
    fprintf(stderr, "R-chains: %d, L-chains: %d\n", nrch, nlch);
    fprintf(stderr, "Both R&L: %d, Interior: %d\n", nboth, ninterior);

    /* Output */
    printf("SITES %d\n", nsites);
    for (int s = 0; s < nsites; s++) {
        printf("%d %.10f %.10f %.10f", s,
               allsites[s].pos.x, allsites[s].pos.y, allsites[s].pos.z);
        for (int a = 0; a < 4; a++)
            printf(" %.10f %.10f %.10f",
                   allsites[s].dirs[a].x, allsites[s].dirs[a].y, allsites[s].dirs[a].z);
        printf(" %d %d %d %d\n",
               allsites[s].r_chain, allsites[s].r_pos,
               allsites[s].l_chain, allsites[s].l_pos);
    }
    printf("RCHAINS %d\n", nrch);
    for (int c = 0; c < nrch; c++) {
        printf("%d %d", c, rchains[c].len);
        for (int i = 0; i < rchains[c].len; i++)
            printf(" %d %d", rchains[c].ids[i], rchains[c].faces[i]);
        printf("\n");
    }
    printf("LCHAINS %d\n", nlch);
    for (int c = 0; c < nlch; c++) {
        printf("%d %d", c, lchains[c].len);
        for (int i = 0; i < lchains[c].len; i++)
            printf(" %d %d", lchains[c].ids[i], lchains[c].faces[i]);
        printf("\n");
    }

    free(htab);
    free(allsites);
    free(rchains);
    free(lchains);
    return 0;
}

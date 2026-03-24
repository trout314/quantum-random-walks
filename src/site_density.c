/*
 * site_density.c — Generate 3D walk sites and output positions for density analysis.
 *
 * Modes:
 *   0 = BFS seed with density pruning (current default)
 *   1 = Chain threading only (no BFS seed — start from origin, thread outward)
 *
 * Build: clang -O2 -o site_density src/site_density.c -lm
 * Usage: ./site_density [sigma] [max_per_cell] [mode]
 *
 * Outputs site positions to stdout: x y z r
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========== 3D vector ========== */
typedef struct { double x, y, z; } vec3;
static inline vec3 v3add(vec3 a, vec3 b) { return (vec3){a.x+b.x,a.y+b.y,a.z+b.z}; }
static inline vec3 v3scale(vec3 a, double s) { return (vec3){a.x*s,a.y*s,a.z*s}; }
static inline double v3dot(vec3 a, vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
static inline double v3norm(vec3 a) { return sqrt(v3dot(a,a)); }

static void init_tet(vec3 d[4]) {
    d[0]=(vec3){0,0,1};
    d[1]=(vec3){2*sqrt(2.0)/3,0,-1.0/3};
    d[2]=(vec3){-sqrt(2.0)/3,sqrt(6.0)/3,-1.0/3};
    d[3]=(vec3){-sqrt(2.0)/3,-sqrt(6.0)/3,-1.0/3};
}

static void helix_step(vec3 *pos, vec3 d[4], int face) {
    vec3 e = d[face];
    *pos = v3add(*pos, v3scale(e, -2.0/3.0));
    for (int a = 0; a < 4; a++) {
        double dot = 2.0 * v3dot(d[a], e);
        d[a] = (vec3){d[a].x - dot*e.x, d[a].y - dot*e.y, d[a].z - dot*e.z};
    }
}

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

/* ========== Site storage with hash table ========== */
#define MAX_SITES 2000000
typedef struct {
    vec3 pos;
    vec3 dirs[4];
    int r_face, l_face;  /* chain threading (not used here, but kept for compatibility) */
} Site;

static Site sites[MAX_SITES];
static int nsites = 0;

/* Spatial hash for deduplication */
#define HASH_SIZE (1<<20)
#define HASH_MASK (HASH_SIZE-1)
static int htable[HASH_SIZE];
static int hnext[MAX_SITES];

static void hash_init(void) { memset(htable, -1, sizeof(htable)); }

static unsigned int hash_pos(vec3 p) {
    /* Quantize to grid with spacing ~0.01 */
    int ix = (int)(p.x * 100 + 50000);
    int iy = (int)(p.y * 100 + 50000);
    int iz = (int)(p.z * 100 + 50000);
    unsigned int h = (unsigned int)(ix * 73856093u ^ iy * 19349663u ^ iz * 83492791u);
    return h & HASH_MASK;
}

static int site_insert(vec3 pos, vec3 d[4]) {
    double tol = 1e-7;
    unsigned int h = hash_pos(pos);
    for (int i = htable[h]; i >= 0; i = hnext[i]) {
        vec3 dp = {pos.x - sites[i].pos.x, pos.y - sites[i].pos.y, pos.z - sites[i].pos.z};
        if (v3dot(dp, dp) < tol * tol) return i;
    }
    if (nsites >= MAX_SITES) {
        fprintf(stderr, "FATAL: MAX_SITES exceeded\n");
        exit(1);
    }
    int id = nsites++;
    sites[id].pos = pos;
    memcpy(sites[id].dirs, d, sizeof(vec3[4]));
    sites[id].r_face = -1;
    sites[id].l_face = -1;
    hnext[id] = htable[h];
    htable[h] = id;
    return id;
}

/* ========== BFS seed (same as walk_adaptive.c) ========== */
static void bfs_seed(double sigma, int seed_depth, int max_per_cell) {
    double seed_thresh = 1e-4;
    double cell_size = 1.0;
    double step_len = 2.0/3.0;
    double grid_half = seed_depth * step_len + 5.0;
    int grid_n = (int)(2.0 * grid_half / cell_size) + 1;
    if (grid_n > 500) grid_n = 500;
    int grid_total = grid_n * grid_n * grid_n;
    int *grid_count = calloc(grid_total, sizeof(int));

    #define GRID_IDX(pos) ({ \
        int gx = (int)((pos.x + grid_half) / cell_size); \
        int gy = (int)((pos.y + grid_half) / cell_size); \
        int gz = (int)((pos.z + grid_half) / cell_size); \
        if (gx < 0) gx = 0; if (gx >= grid_n) gx = grid_n-1; \
        if (gy < 0) gy = 0; if (gy >= grid_n) gy = grid_n-1; \
        if (gz < 0) gz = 0; if (gz >= grid_n) gz = grid_n-1; \
        gx * grid_n * grid_n + gy * grid_n + gz; \
    })

    vec3 d0[4]; init_tet(d0);
    site_insert((vec3){0,0,0}, d0);
    grid_count[GRID_IDX(((vec3){0,0,0}))]++;

    int *queue = malloc(MAX_SITES * sizeof(int));
    int qh = 0, qt = 0;
    queue[qt++] = 0;
    int *sdepth = calloc(MAX_SITES, sizeof(int));
    int npruned = 0;

    while (qh < qt) {
        int s = queue[qh++];
        if (sdepth[s] >= seed_depth) continue;
        double r2 = v3dot(sites[s].pos, sites[s].pos);
        if (exp(-r2 / (2*sigma*sigma)) < seed_thresh) continue;
        int gi = GRID_IDX(sites[s].pos);
        if (grid_count[gi] >= max_per_cell) { npruned++; continue; }
        for (int f = 0; f < 4; f++) {
            vec3 p = sites[s].pos, dd[4];
            memcpy(dd, sites[s].dirs, sizeof(dd));
            helix_step(&p, dd, f);
            reorth(dd);
            int old_n = nsites;
            int nid = site_insert(p, dd);
            if (nid >= old_n) {
                sdepth[nid] = sdepth[s] + 1;
                queue[qt++] = nid;
                int ci = GRID_IDX(sites[nid].pos);
                grid_count[ci]++;
            }
        }
    }
    free(queue); free(sdepth); free(grid_count);
    #undef GRID_IDX
}

/* ========== Chain-only seeding ========== */
static int PAT_R[4] = {1, 3, 0, 2};
static int PAT_L[4] = {0, 1, 2, 3};

static void chain_seed(double sigma, int max_per_cell) {
    /* Start from origin, thread R and L chains outward.
     * From each new site, try spawning perpendicular chains.
     * Use density grid to limit overcrowding. */
    double step_len = 2.0/3.0;
    int max_chain_len = (int)(4.0 * sigma / step_len) + 5;
    double cell_size = 1.0;
    double grid_half = max_chain_len * step_len + 5.0;
    int grid_n = (int)(2.0 * grid_half / cell_size) + 1;
    if (grid_n > 500) grid_n = 500;
    int grid_total = grid_n * grid_n * grid_n;
    int *grid_count = calloc(grid_total, sizeof(int));

    #define GRID_IDX2(pos) ({ \
        int gx = (int)((pos.x + grid_half) / cell_size); \
        int gy = (int)((pos.y + grid_half) / cell_size); \
        int gz = (int)((pos.z + grid_half) / cell_size); \
        if (gx < 0) gx = 0; if (gx >= grid_n) gx = grid_n-1; \
        if (gy < 0) gy = 0; if (gy >= grid_n) gy = grid_n-1; \
        if (gz < 0) gz = 0; if (gz >= grid_n) gz = grid_n-1; \
        gx * grid_n * grid_n + gy * grid_n + gz; \
    })

    vec3 d0[4]; init_tet(d0);
    site_insert((vec3){0,0,0}, d0);
    grid_count[GRID_IDX2(((vec3){0,0,0}))]++;

    /* BFS-like expansion but only along R and L chain directions */
    int *queue = malloc(MAX_SITES * sizeof(int));
    int qh = 0, qt = 0;
    queue[qt++] = 0;

    while (qh < qt) {
        int s = queue[qh++];
        double r2 = v3dot(sites[s].pos, sites[s].pos);
        if (exp(-r2 / (2*sigma*sigma)) < 1e-4) continue;

        /* Try all 4 faces (this traces both R and L chains plus cross-links) */
        for (int f = 0; f < 4; f++) {
            vec3 p = sites[s].pos, dd[4];
            memcpy(dd, sites[s].dirs, sizeof(dd));
            helix_step(&p, dd, f);
            reorth(dd);

            /* Density check */
            int ci = GRID_IDX2(p);
            if (grid_count[ci] >= max_per_cell) continue;

            int old_n = nsites;
            int nid = site_insert(p, dd);
            if (nid >= old_n) {
                queue[qt++] = nid;
                grid_count[ci]++;
            }
        }
    }
    free(queue); free(grid_count);
    #undef GRID_IDX2
}

/* ========== Main ========== */
int main(int argc, char **argv) {
    double sigma = 3.0;
    int max_per_cell = 20;
    int mode = 0;  /* 0=BFS, 1=chain-only */

    if (argc > 1) sigma = atof(argv[1]);
    if (argc > 2) max_per_cell = atoi(argv[2]);
    if (argc > 3) mode = atoi(argv[3]);

    hash_init();

    int seed_depth = (int)(4.0 * sigma / (2.0/3.0)) + 2;
    if (seed_depth < 4) seed_depth = 4;

    fprintf(stderr, "sigma=%.1f max_per_cell=%d mode=%s seed_depth=%d\n",
            sigma, max_per_cell, mode == 0 ? "BFS" : "chain-only", seed_depth);

    if (mode == 0) {
        bfs_seed(sigma, seed_depth, max_per_cell);
    } else {
        chain_seed(sigma, max_per_cell);
    }

    fprintf(stderr, "Total sites: %d\n", nsites);

    /* Output positions */
    printf("# x y z r  (nsites=%d sigma=%.1f max_per_cell=%d mode=%d)\n",
           nsites, sigma, max_per_cell, mode);
    for (int i = 0; i < nsites; i++) {
        double r = v3norm(sites[i].pos);
        printf("%.6f %.6f %.6f %.6f\n",
               sites[i].pos.x, sites[i].pos.y, sites[i].pos.z, r);
    }

    return 0;
}

/*
 * walk_gen.c — Generate the tetrahedral walk lattice AND the sparse
 * shift operators S_R and S_L (including frame transport).
 *
 * Build: clang -O2 -o walk_gen src/walk_gen.c -lm -llapack -lblas
 * Usage: ./walk_gen [chain_len] [n_generations]
 *
 * Output (binary, to stdout):
 *   Header: n_sites, n_interior, nnz_R, nnz_L (4 ints)
 *   Site positions: n_sites × (x, y, z) doubles
 *   Site membership: n_sites × (r_chain, r_pos, l_chain, l_pos) ints
 *   S_R sparse entries: nnz_R × (row, col) ints + nnz_R × (re, im) doubles
 *   S_L sparse entries: nnz_L × (row, col) ints + nnz_L × (re, im) doubles
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

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

static void helix_step(vec3 *pos, vec3 d[4], int face) {
    vec3 e = d[face];
    *pos = v3add(*pos, v3scale(e, -2.0/3.0));
    for (int a = 0; a < 4; a++) d[a] = v3reflect(d[a], e);
}

/* ========== 4×4 complex matrix operations ========== */
typedef double complex c4x4[4][4];

static void c4_zero(c4x4 M) { memset(M, 0, sizeof(c4x4)); }
static void c4_eye(c4x4 M) { c4_zero(M); for (int i=0;i<4;i++) M[i][i]=1; }

static void c4_copy(c4x4 dst, const c4x4 src) { memcpy(dst, src, sizeof(c4x4)); }

static void c4_mul(c4x4 C, const c4x4 A, const c4x4 B) {
    c4x4 tmp; c4_zero(tmp);
    for (int i=0;i<4;i++)
        for (int j=0;j<4;j++)
            for (int k=0;k<4;k++)
                tmp[i][j] += A[i][k] * B[k][j];
    c4_copy(C, tmp);
}

static void c4_adjoint(c4x4 dst, const c4x4 src) {
    for (int i=0;i<4;i++)
        for (int j=0;j<4;j++)
            dst[i][j] = conj(src[j][i]);
}

static void c4_add(c4x4 C, const c4x4 A, const c4x4 B) {
    for (int i=0;i<4;i++)
        for (int j=0;j<4;j++)
            C[i][j] = A[i][j] + B[i][j];
}

static void c4_scale(c4x4 M, double complex s) {
    for (int i=0;i<4;i++)
        for (int j=0;j<4;j++)
            M[i][j] *= s;
}

/* ========== Dirac matrices ========== */
static c4x4 ALPHA[3], BETA;

static void init_dirac(void) {
    c4_zero(ALPHA[0]); c4_zero(ALPHA[1]); c4_zero(ALPHA[2]); c4_zero(BETA);
    /* alpha_1 = [[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]] */
    ALPHA[0][0][3]=1; ALPHA[0][1][2]=1; ALPHA[0][2][1]=1; ALPHA[0][3][0]=1;
    /* alpha_2 = [[0,0,0,-i],[0,0,i,0],[0,-i,0,0],[i,0,0,0]] */
    ALPHA[1][0][3]=-I; ALPHA[1][1][2]=I; ALPHA[1][2][1]=-I; ALPHA[1][3][0]=I;
    /* alpha_3 = [[0,0,1,0],[0,0,0,-1],[1,0,0,0],[0,-1,0,0]] */
    ALPHA[2][0][2]=1; ALPHA[2][1][3]=-1; ALPHA[2][2][0]=1; ALPHA[2][3][1]=-1;
    /* beta = diag(1,1,-1,-1) */
    BETA[0][0]=1; BETA[1][1]=1; BETA[2][2]=-1; BETA[3][3]=-1;
}

static void make_tau(c4x4 tau, vec3 d) {
    double nu = sqrt(7.0) / 4.0;
    c4_zero(tau);
    for (int i=0;i<4;i++) tau[i][i] = nu * (i<2 ? 1.0 : -1.0); /* nu*beta */
    double dd[3] = {d.x, d.y, d.z};
    for (int a=0;a<3;a++)
        for (int i=0;i<4;i++)
            for (int j=0;j<4;j++)
                tau[i][j] += 0.75 * dd[a] * ALPHA[a][i][j];
}

/* ========== LAPACK interface ========== */
/* zheev: eigenvalues of complex Hermitian matrix */
extern void zheev_(const char *jobz, const char *uplo, const int *n,
                   double complex *a, const int *lda, double *w,
                   double complex *work, const int *lwork, double *rwork,
                   int *info);

/* Eigendecompose 4×4 Hermitian matrix. Returns eigenvalues in w (ascending)
 * and eigenvectors as columns of V. */
static void eigh4(const c4x4 M, double w[4], c4x4 V) {
    /* Copy M to column-major for LAPACK */
    double complex a[16];
    for (int i=0;i<4;i++)
        for (int j=0;j<4;j++)
            a[j*4+i] = M[i][j];  /* column-major */

    int n = 4, lda = 4, lwork = 32, info;
    double complex work[32];
    double rwork[16];
    zheev_("V", "U", &n, a, &lda, w, work, &lwork, rwork, &info);
    if (info != 0) { fprintf(stderr, "zheev failed: %d\n", info); exit(1); }

    /* Copy eigenvectors back to row-major V */
    for (int i=0;i<4;i++)
        for (int j=0;j<4;j++)
            V[i][j] = a[j*4+i];  /* V[i][j] = i-th component of j-th eigenvec */
}

/* ========== Frame transport via polar decomposition ========== */

static void frame_transport(const c4x4 tau_from, const c4x4 tau_to, c4x4 U) {
    /* P_from^± = (I ± tau_from)/2, P_to^± = (I ± tau_to)/2 */
    c4x4 Pfp, Pfm, Ptp, Ptm;
    c4_eye(Pfp); c4_eye(Pfm); c4_eye(Ptp); c4_eye(Ptm);
    for (int i=0;i<4;i++)
        for (int j=0;j<4;j++) {
            Pfp[i][j] = 0.5*(Pfp[i][j] + tau_from[i][j]);
            Pfm[i][j] = 0.5*(Pfm[i][j] - tau_from[i][j]);
            Ptp[i][j] = 0.5*(Ptp[i][j] + tau_to[i][j]);
            Ptm[i][j] = 0.5*(Ptm[i][j] - tau_to[i][j]);
        }

    /* W = P_to^+ P_from^+ + P_to^- P_from^- */
    c4x4 W, tmp1, tmp2;
    c4_mul(tmp1, Ptp, Pfp);
    c4_mul(tmp2, Ptm, Pfm);
    c4_add(W, tmp1, tmp2);

    /* W†W */
    c4x4 Wadj, WdW;
    c4_adjoint(Wadj, W);
    c4_mul(WdW, Wadj, W);

    /* Eigendecompose W†W */
    double evals[4];
    c4x4 V;
    eigh4(WdW, evals, V);

    /* H^{-1/2} = V diag(1/√λ) V† */
    c4x4 Vadj, D, Hinv, tmp3;
    c4_adjoint(Vadj, V);
    c4_zero(D);
    for (int i=0;i<4;i++)
        D[i][i] = (evals[i] > 1e-15) ? 1.0/sqrt(evals[i]) : 0.0;
    c4_mul(tmp3, V, D);     /* V @ D */
    c4_mul(Hinv, tmp3, Vadj); /* V @ D @ V† */

    /* U = W @ H^{-1/2} */
    c4_mul(U, W, Hinv);
}

/* ========== Hash table (same as lattice_gen.c) ========== */
#define HASH_BITS 22
#define HASH_SIZE (1 << HASH_BITS)
#define HASH_MASK (HASH_SIZE - 1)
#define MAX_SITES 2000000

typedef struct { long kx, ky, kz; int id; } hentry;
static hentry *htab;

typedef struct {
    vec3 pos;
    vec3 dirs[4];
    int r_chain, r_pos, l_chain, l_pos;
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
    *kx = (long)round(p.x/tol); *ky = (long)round(p.y/tol); *kz = (long)round(p.z/tol);
}

static unsigned hfn(long kx, long ky, long kz) {
    unsigned long h = (unsigned long)(kx*73856093L) ^
                      (unsigned long)(ky*19349663L) ^
                      (unsigned long)(kz*83492791L);
    return (unsigned)(h & HASH_MASK);
}

static int site_insert(vec3 pos, vec3 dirs[4]) {
    long kx, ky, kz; poskey(pos, &kx, &ky, &kz);
    unsigned h = hfn(kx, ky, kz);
    for (int probe = 0; probe < HASH_SIZE; probe++) {
        unsigned idx = (h + probe) & HASH_MASK;
        if (htab[idx].id == -1) {
            if (nsites >= MAX_SITES) { fprintf(stderr, "Too many sites\n"); exit(1); }
            int id = nsites++;
            htab[idx] = (hentry){kx, ky, kz, id};
            allsites[id].pos = pos;
            memcpy(allsites[id].dirs, dirs, sizeof(vec3)*4);
            allsites[id].r_chain = allsites[id].l_chain = -1;
            return id;
        }
        if (htab[idx].kx==kx && htab[idx].ky==ky && htab[idx].kz==kz)
            return htab[idx].id;
    }
    fprintf(stderr, "Hash full\n"); exit(1);
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

static const int PAT_R[4] = {1, 3, 0, 2};
static const int PAT_L[4] = {0, 1, 2, 3};

static void trace_chain(int start_id, const int pat[4], int nfwd, int nbwd, chain_t *ch) {
    ch->len = 0;
    int rev[4] = {pat[3], pat[2], pat[1], pat[0]};

    /* Backward */
    int bwd_ids[MAX_CLEN]; int nb = 0;
    { vec3 pos = allsites[start_id].pos;
      vec3 d[4]; memcpy(d, allsites[start_id].dirs, sizeof(d));
      for (int i = 0; i < nbwd && nb < MAX_CLEN/2; i++) {
          helix_step(&pos, d, rev[i%4]);
          if ((i+1)%8==0) reorth(d);
          bwd_ids[nb++] = site_insert(pos, d);
      }
    }
    for (int i = nb-1; i >= 0; i--) {
        ch->ids[ch->len] = bwd_ids[i];
        ch->faces[ch->len] = pat[0]; /* placeholder, recomputed below */
        ch->len++;
    }
    ch->ids[ch->len] = start_id;
    ch->faces[ch->len] = pat[0];
    ch->len++;

    /* Forward */
    { vec3 pos = allsites[start_id].pos;
      vec3 d[4]; memcpy(d, allsites[start_id].dirs, sizeof(d));
      for (int i = 0; i < nfwd && ch->len < MAX_CLEN; i++) {
          int face = pat[i%4];
          helix_step(&pos, d, face);
          if ((i+1)%8==0) reorth(d);
          ch->ids[ch->len] = site_insert(pos, d);
          ch->faces[ch->len] = pat[(i+1)%4];
          ch->len++;
      }
    }

    /* Recompute faces: at each position i, the face is pat[(i - nbwd) % 4]
     * adjusted for the chain's starting offset */
    for (int i = 0; i < ch->len; i++) {
        int offset = i - nb; /* relative to start site */
        ch->faces[i] = pat[((offset % 4) + 4) % 4];
    }
}

static void assign_membership(chain_t *ch, int cid, int is_r) {
    for (int i = 0; i < ch->len; i++) {
        int sid = ch->ids[i];
        if (is_r) { if (allsites[sid].r_chain==-1) { allsites[sid].r_chain=cid; allsites[sid].r_pos=i; } }
        else      { if (allsites[sid].l_chain==-1) { allsites[sid].l_chain=cid; allsites[sid].l_pos=i; } }
    }
}

/* ========== Sparse matrix output ========== */

typedef struct { int row, col; double re, im; } sparse_entry;

static sparse_entry *sr_entries, *sl_entries;
static int sr_nnz = 0, sl_nnz = 0;
#define MAX_NNZ 10000000

static void add_entry(sparse_entry *entries, int *nnz, int row, int col, double complex val) {
    if (cabs(val) < 1e-15) return;
    if (*nnz >= MAX_NNZ) { fprintf(stderr, "Too many nonzeros\n"); exit(1); }
    entries[*nnz] = (sparse_entry){row, col, creal(val), cimag(val)};
    (*nnz)++;
}

/* Build shift entries for a stitched mega-loop.
 * chains[0..nch-1] are stitched end-to-start cyclically.
 * IMPORTANT: only include each site ONCE (via its assigned chain).
 * Sites that appear in a chain but are assigned to a different chain
 * are skipped (they'll be included when their own chain is processed).
 * is_r: 1 for R-chains, 0 for L-chains (determines which membership to check). */
static void build_stitched_shift(chain_t *chains, int nch, int is_r,
                                  sparse_entry *entries, int *nnz) {
    /* Flatten chains, deduplicating: only include site if it's assigned to this chain */
    int max_total = 0;
    for (int c = 0; c < nch; c++) max_total += chains[c].len;

    int *loop_ids = malloc(max_total * sizeof(int));
    int *loop_faces = malloc(max_total * sizeof(int));
    int total_len = 0;

    for (int c = 0; c < nch; c++) {
        for (int i = 0; i < chains[c].len; i++) {
            int sid = chains[c].ids[i];
            /* Only include if this chain is the site's assigned chain */
            int assigned_chain = is_r ? allsites[sid].r_chain : allsites[sid].l_chain;
            if (assigned_chain != c) continue;
            loop_ids[total_len] = sid;
            loop_faces[total_len] = chains[c].faces[i];
            total_len++;
        }
    }

    fprintf(stderr, "  Stitched loop: %d unique sites (from %d total in %d chains)\n",
            total_len, max_total, nch);

    /* Build shift entries with periodic (modular) indexing */
    for (int i = 0; i < total_len; i++) {
        int i_next = (i + 1) % total_len;
        int i_prev = (i - 1 + total_len) % total_len;

        int sid = loop_ids[i];
        int face = loop_faces[i];
        vec3 d = allsites[sid].dirs[face];

        c4x4 tau;
        make_tau(tau, d);

        c4x4 Pp, Pm;
        c4_eye(Pp); c4_eye(Pm);
        for (int a=0;a<4;a++)
            for (int b=0;b<4;b++) {
                Pp[a][b] = 0.5*(Pp[a][b] + tau[a][b]);
                Pm[a][b] = 0.5*(Pm[a][b] - tau[a][b]);
            }

        /* Forward: i -> i_next */
        {
            int sid_next = loop_ids[i_next];
            int face_next = loop_faces[i_next];
            vec3 d_next = allsites[sid_next].dirs[face_next];
            c4x4 tau_next;
            make_tau(tau_next, d_next);

            c4x4 U, block;
            frame_transport(tau, tau_next, U);
            c4_mul(block, U, Pp);

            for (int a=0;a<4;a++)
                for (int b=0;b<4;b++)
                    add_entry(entries, nnz, sid_next*4+a, sid*4+b, block[a][b]);
        }

        /* Backward: i -> i_prev */
        {
            int sid_prev = loop_ids[i_prev];
            int face_prev = loop_faces[i_prev];
            vec3 d_prev = allsites[sid_prev].dirs[face_prev];
            c4x4 tau_prev;
            make_tau(tau_prev, d_prev);

            c4x4 U, block;
            frame_transport(tau, tau_prev, U);
            c4_mul(block, U, Pm);

            for (int a=0;a<4;a++)
                for (int b=0;b<4;b++)
                    add_entry(entries, nnz, sid_prev*4+a, sid*4+b, block[a][b]);
        }
    }

    free(loop_ids);
    free(loop_faces);
}

/* ========== Main ========== */

int main(int argc, char **argv) {
    int chain_len = 15, ngen = 1;
    if (argc > 1) chain_len = atoi(argv[1]);
    if (argc > 2) ngen = atoi(argv[2]);

    init_dirac();
    htab_init();
    rchains = malloc(MAX_CHAINS * sizeof(chain_t));
    lchains = malloc(MAX_CHAINS * sizeof(chain_t));
    sr_entries = malloc(MAX_NNZ * sizeof(sparse_entry));
    sl_entries = malloc(MAX_NNZ * sizeof(sparse_entry));

    /* Origin */
    vec3 d0[4]; init_tet(d0);
    vec3 origin = {0,0,0};
    int oid = site_insert(origin, d0);

    /* Gen 0: R₀ */
    trace_chain(oid, PAT_R, chain_len, chain_len, &rchains[0]);
    assign_membership(&rchains[0], 0, 1);
    nrch = 1;
    fprintf(stderr, "Gen 0: %d sites, 1 R-chain\n", nsites);

    for (int gen = 0; gen < ngen; gen++) {
        /* L-chains from R-sites */
        int snap = nsites;
        int *seeds = malloc(snap * sizeof(int));
        int ns = 0;
        for (int s = 0; s < snap; s++)
            if (allsites[s].r_chain >= 0 && allsites[s].l_chain == -1)
                seeds[ns++] = s;
        for (int i = 0; i < ns; i++) {
            if (nlch >= MAX_CHAINS) break;
            trace_chain(seeds[i], PAT_L, chain_len, chain_len, &lchains[nlch]);
            assign_membership(&lchains[nlch], nlch, 0);
            nlch++;
        }
        free(seeds);
        fprintf(stderr, "Gen %d L: %d sites, %d R, %d L\n", gen, nsites, nrch, nlch);

        /* R-chains from L-sites */
        snap = nsites;
        seeds = malloc(snap * sizeof(int));
        ns = 0;
        for (int s = 0; s < snap; s++)
            if (allsites[s].l_chain >= 0 && allsites[s].r_chain == -1)
                seeds[ns++] = s;
        for (int i = 0; i < ns; i++) {
            if (nrch >= MAX_CHAINS) break;
            trace_chain(seeds[i], PAT_R, chain_len, chain_len, &rchains[nrch]);
            assign_membership(&rchains[nrch], nrch, 1);
            nrch++;
        }
        free(seeds);
        fprintf(stderr, "Gen %d R: %d sites, %d R, %d L\n", gen, nsites, nrch, nlch);
    }

    /* Count interior */
    int nboth = 0, ninterior = 0;
    for (int s = 0; s < nsites; s++) {
        if (allsites[s].r_chain >= 0 && allsites[s].l_chain >= 0) {
            nboth++;
            int ri = allsites[s].r_pos, li = allsites[s].l_pos;
            int rlen = rchains[allsites[s].r_chain].len;
            int llen = lchains[allsites[s].l_chain].len;
            if (ri > 0 && ri < rlen-1 && li > 0 && li < llen-1)
                ninterior++;
        }
    }
    fprintf(stderr, "\nSites: %d, Both: %d, Interior: %d\n", nsites, nboth, ninterior);

    /* Build sparse operators using stitched mega-loops.
     * All R-chains are cyclically stitched into one big loop,
     * and all L-chains into another. The frame transport at
     * each stitch point handles the mismatch automatically. */
    fprintf(stderr, "Building S_R (stitched, %d chains)...\n", nrch);
    build_stitched_shift(rchains, nrch, 1, sr_entries, &sr_nnz);
    fprintf(stderr, "S_R: %d nonzeros\n", sr_nnz);

    fprintf(stderr, "Building S_L (stitched, %d chains)...\n", nlch);
    build_stitched_shift(lchains, nlch, 0, sl_entries, &sl_nnz);
    fprintf(stderr, "S_L: %d nonzeros\n", sl_nnz);

    /* Binary output */
    int header[4] = {nsites, ninterior, sr_nnz, sl_nnz};
    fwrite(header, sizeof(int), 4, stdout);

    /* Positions */
    for (int s = 0; s < nsites; s++) {
        double xyz[3] = {allsites[s].pos.x, allsites[s].pos.y, allsites[s].pos.z};
        fwrite(xyz, sizeof(double), 3, stdout);
    }

    /* Membership */
    for (int s = 0; s < nsites; s++) {
        int mem[4] = {allsites[s].r_chain, allsites[s].r_pos,
                      allsites[s].l_chain, allsites[s].l_pos};
        fwrite(mem, sizeof(int), 4, stdout);
    }

    /* S_R entries */
    for (int i = 0; i < sr_nnz; i++) {
        int rc[2] = {sr_entries[i].row, sr_entries[i].col};
        fwrite(rc, sizeof(int), 2, stdout);
    }
    for (int i = 0; i < sr_nnz; i++) {
        double ri[2] = {sr_entries[i].re, sr_entries[i].im};
        fwrite(ri, sizeof(double), 2, stdout);
    }

    /* S_L entries */
    for (int i = 0; i < sl_nnz; i++) {
        int rc[2] = {sl_entries[i].row, sl_entries[i].col};
        fwrite(rc, sizeof(int), 2, stdout);
    }
    for (int i = 0; i < sl_nnz; i++) {
        double ri[2] = {sl_entries[i].re, sl_entries[i].im};
        fwrite(ri, sizeof(double), 2, stdout);
    }

    free(htab); free(allsites); free(rchains); free(lchains);
    free(sr_entries); free(sl_entries);
    return 0;
}

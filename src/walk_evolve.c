/*
 * walk_evolve.c — Time-evolve a wavepacket using W = S_R C_R S_L C_L.
 *
 * Instead of constructing the full W matrix, apply the 4 operators
 * sequentially at each step. The coins C_R, C_L are computed on-the-fly
 * from the stored direction vectors and face indices.
 *
 * Reads lattice data from walk_gen (binary on stdin).
 * Outputs wavepacket statistics at each time step.
 *
 * Build: clang -O2 -o walk_evolve src/walk_evolve.c -lm
 * Usage: ./walk_gen [args] | ./walk_evolve [theta] [sigma] [n_steps]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

/* ========== 4x4 complex matrix ========== */
typedef double complex c4x4[4][4];

static void c4_zero(c4x4 M){memset(M,0,sizeof(c4x4));}
static void c4_eye(c4x4 M){c4_zero(M);for(int i=0;i<4;i++)M[i][i]=1;}

/* ========== Dirac matrices ========== */
static c4x4 ALPHA[3];
static void init_dirac(void) {
    c4_zero(ALPHA[0]);c4_zero(ALPHA[1]);c4_zero(ALPHA[2]);
    ALPHA[0][0][3]=1;ALPHA[0][1][2]=1;ALPHA[0][2][1]=1;ALPHA[0][3][0]=1;
    ALPHA[1][0][3]=-I;ALPHA[1][1][2]=I;ALPHA[1][2][1]=-I;ALPHA[1][3][0]=I;
    ALPHA[2][0][2]=1;ALPHA[2][1][3]=-1;ALPHA[2][2][0]=1;ALPHA[2][3][1]=-1;
}

/* ========== Sparse matrix (CSR) ========== */
typedef struct {
    int nrows, ncols, nnz;
    int *row_ptr;   /* length nrows+1 */
    int *col_idx;   /* length nnz */
    double complex *vals; /* length nnz */
} sparse_t;

static sparse_t* sparse_from_coo(int nrows, int ncols, int nnz,
                                  int *rows, int *cols, double complex *vals) {
    sparse_t *S = malloc(sizeof(sparse_t));
    S->nrows = nrows; S->ncols = ncols; S->nnz = nnz;
    S->row_ptr = calloc(nrows+1, sizeof(int));
    S->col_idx = malloc(nnz * sizeof(int));
    S->vals = malloc(nnz * sizeof(double complex));

    /* Count entries per row */
    for (int i = 0; i < nnz; i++) S->row_ptr[rows[i]+1]++;
    for (int i = 0; i < nrows; i++) S->row_ptr[i+1] += S->row_ptr[i];

    /* Fill */
    int *pos = calloc(nrows, sizeof(int));
    for (int i = 0; i < nnz; i++) {
        int r = rows[i];
        int p = S->row_ptr[r] + pos[r];
        S->col_idx[p] = cols[i];
        S->vals[p] = vals[i];
        pos[r]++;
    }
    free(pos);
    return S;
}

static void sparse_matvec(const sparse_t *S, const double complex *x, double complex *y) {
    memset(y, 0, S->nrows * sizeof(double complex));
    for (int r = 0; r < S->nrows; r++) {
        double complex sum = 0;
        for (int j = S->row_ptr[r]; j < S->row_ptr[r+1]; j++)
            sum += S->vals[j] * x[S->col_idx[j]];
        y[r] = sum;
    }
}

/* ========== Main ========== */

int main(int argc, char **argv) {
    double theta = 0.5, sigma = 3.0;
    int n_steps = 30;
    if (argc > 1) theta = atof(argv[1]);
    if (argc > 2) sigma = atof(argv[2]);
    if (argc > 3) n_steps = atoi(argv[3]);

    init_dirac();

    /* ---- Read lattice data from stdin ---- */
    int header[4];
    fread(header, sizeof(int), 4, stdin);
    int nsites = header[0], n_interior = header[1], nnz_R = header[2], nnz_L = header[3];
    int dim = 4 * nsites;

    fprintf(stderr, "Sites: %d, Interior: %d, nnz_R: %d, nnz_L: %d\n",
            nsites, n_interior, nnz_R, nnz_L);

    /* Positions */
    double *pos = malloc(nsites * 3 * sizeof(double));
    fread(pos, sizeof(double), nsites*3, stdin);

    /* Membership */
    int *mem = malloc(nsites * 4 * sizeof(int));
    fread(mem, sizeof(int), nsites*4, stdin);

    /* Direction vectors */
    double *dirs = malloc(nsites * 12 * sizeof(double));
    fread(dirs, sizeof(double), nsites*12, stdin);

    /* Face indices */
    int *faces = malloc(nsites * 2 * sizeof(int));
    fread(faces, sizeof(int), nsites*2, stdin);

    /* S_R */
    int *sr_rc = malloc(nnz_R * 2 * sizeof(int));
    fread(sr_rc, sizeof(int), nnz_R*2, stdin);
    double *sr_ri = malloc(nnz_R * 2 * sizeof(double));
    fread(sr_ri, sizeof(double), nnz_R*2, stdin);

    double complex *sr_vals = malloc(nnz_R * sizeof(double complex));
    int *sr_rows = malloc(nnz_R * sizeof(int));
    int *sr_cols = malloc(nnz_R * sizeof(int));
    for (int i = 0; i < nnz_R; i++) {
        sr_rows[i] = sr_rc[2*i];
        sr_cols[i] = sr_rc[2*i+1];
        sr_vals[i] = sr_ri[2*i] + I*sr_ri[2*i+1];
    }
    sparse_t *S_R = sparse_from_coo(dim, dim, nnz_R, sr_rows, sr_cols, sr_vals);
    free(sr_rc); free(sr_ri); free(sr_rows); free(sr_cols); free(sr_vals);

    /* S_L */
    int *sl_rc = malloc(nnz_L * 2 * sizeof(int));
    fread(sl_rc, sizeof(int), nnz_L*2, stdin);
    double *sl_ri = malloc(nnz_L * 2 * sizeof(double));
    fread(sl_ri, sizeof(double), nnz_L*2, stdin);

    double complex *sl_vals = malloc(nnz_L * sizeof(double complex));
    int *sl_rows = malloc(nnz_L * sizeof(int));
    int *sl_cols = malloc(nnz_L * sizeof(int));
    for (int i = 0; i < nnz_L; i++) {
        sl_rows[i] = sl_rc[2*i];
        sl_cols[i] = sl_rc[2*i+1];
        sl_vals[i] = sl_ri[2*i] + I*sl_ri[2*i+1];
    }
    sparse_t *S_L = sparse_from_coo(dim, dim, nnz_L, sl_rows, sl_cols, sl_vals);
    free(sl_rc); free(sl_ri); free(sl_rows); free(sl_cols); free(sl_vals);

    fprintf(stderr, "Matrices loaded. dim=%d\n", dim);

    /* ---- Precompute coin matrices at each site ---- */
    /* C_R(n) = cos(θ)I - i sin(θ)(e_{r_face(n)} · α)
     * C_L(n) = cos(θ)I - i sin(θ)(e_{l_face(n)} · α)
     * Each is a 4×4 matrix stored per site. */
    double complex *coin_R = malloc(nsites * 16 * sizeof(double complex));
    double complex *coin_L = malloc(nsites * 16 * sizeof(double complex));
    double ct = cos(theta), st = sin(theta);

    for (int n = 0; n < nsites; n++) {
        c4x4 Cn;
        /* R coin */
        int rf = faces[2*n];
        if (rf >= 0) {
            double *d = &dirs[n*12 + rf*3];
            c4_eye(Cn);
            for (int a=0;a<4;a++) for(int b=0;b<4;b++) {
                double complex ea = 0;
                for (int k=0;k<3;k++) ea += d[k]*ALPHA[k][a][b];
                Cn[a][b] = ct*(a==b?1:0) - I*st*ea;
            }
        } else {
            c4_eye(Cn);
        }
        for (int a=0;a<4;a++) for(int b=0;b<4;b++)
            coin_R[n*16+a*4+b] = Cn[a][b];

        /* L coin */
        int lf = faces[2*n+1];
        if (lf >= 0) {
            double *d = &dirs[n*12 + lf*3];
            c4_eye(Cn);
            for (int a=0;a<4;a++) for(int b=0;b<4;b++) {
                double complex ea = 0;
                for (int k=0;k<3;k++) ea += d[k]*ALPHA[k][a][b];
                Cn[a][b] = ct*(a==b?1:0) - I*st*ea;
            }
        } else {
            c4_eye(Cn);
        }
        for (int a=0;a<4;a++) for(int b=0;b<4;b++)
            coin_L[n*16+a*4+b] = Cn[a][b];
    }
    fprintf(stderr, "Coins precomputed. theta=%.3f\n", theta);

    /* ---- Initialize wavepacket ---- */
    double complex *psi = calloc(dim, sizeof(double complex));
    double complex *tmp = calloc(dim, sizeof(double complex));
    double norm = 0;
    for (int n = 0; n < nsites; n++) {
        double x = pos[3*n], y = pos[3*n+1], z = pos[3*n+2];
        double r2 = x*x + y*y + z*z;
        double w = exp(-r2 / (2*sigma*sigma));
        psi[4*n] = w;  /* spinor (1,0,0,0) */
        norm += w*w;
    }
    norm = sqrt(norm);
    for (int i = 0; i < dim; i++) psi[i] /= norm;

    fprintf(stderr, "Wavepacket initialized. sigma=%.1f, n_steps=%d\n", sigma, n_steps);

    /* ---- Apply coin to psi (in-place via tmp) ---- */
    #define APPLY_COIN(coin, psi_in, psi_out) \
        for (int n = 0; n < nsites; n++) { \
            for (int a = 0; a < 4; a++) { \
                double complex s = 0; \
                for (int b = 0; b < 4; b++) \
                    s += coin[n*16+a*4+b] * psi_in[4*n+b]; \
                psi_out[4*n+a] = s; \
            } \
        }

    /* ---- Time evolution ---- */
    printf("# theta=%.4f sigma=%.1f n_steps=%d nsites=%d\n", theta, sigma, n_steps, nsites);
    printf("# t norm r2 x2 y2 z2 r95\n");

    double *radii = malloc(nsites * sizeof(double));
    double *probs = malloc(nsites * sizeof(double));
    for (int n = 0; n < nsites; n++)
        radii[n] = sqrt(pos[3*n]*pos[3*n]+pos[3*n+1]*pos[3*n+1]+pos[3*n+2]*pos[3*n+2]);

    for (int t = 0; t <= n_steps; t++) {
        /* Compute observables */
        double pnorm = 0, total_prob = 0;
        double mx2=0, my2=0, mz2=0;
        for (int n = 0; n < nsites; n++) {
            double p = 0;
            for (int a = 0; a < 4; a++)
                p += creal(psi[4*n+a]*conj(psi[4*n+a]));
            probs[n] = p;
            total_prob += p;
        }
        pnorm = sqrt(total_prob);
        for (int n = 0; n < nsites; n++) {
            double p = probs[n] / total_prob;
            double x=pos[3*n], y=pos[3*n+1], z=pos[3*n+2];
            mx2 += p*x*x; my2 += p*y*y; mz2 += p*z*z;
        }

        /* 95th percentile radius (sort probs by radius) */
        /* Simple approach: use cumulative probability */
        /* For speed, bin by radius */
        double r95 = 0;
        {
            int nbins = 200;
            double rmax = 0;
            for (int n = 0; n < nsites; n++) if (radii[n] > rmax) rmax = radii[n];
            double dr = rmax / nbins;
            double *binprob = calloc(nbins+1, sizeof(double));
            for (int n = 0; n < nsites; n++) {
                int b = (int)(radii[n] / dr);
                if (b > nbins) b = nbins;
                binprob[b] += probs[n] / total_prob;
            }
            double cum = 0;
            for (int b = 0; b <= nbins; b++) {
                cum += binprob[b];
                if (cum >= 0.95) { r95 = b * dr; break; }
            }
            free(binprob);
        }

        printf("%d %.6f %.2f %.2f %.2f %.2f %.2f\n",
               t, pnorm, mx2+my2+mz2, mx2, my2, mz2, r95);
        fflush(stdout);

        if (t < n_steps) {
            /* W = S_R C_R S_L C_L: apply right to left */
            /* Step 1: C_L */
            APPLY_COIN(coin_L, psi, tmp);
            /* Step 2: S_L */
            sparse_matvec(S_L, tmp, psi);
            /* Step 3: C_R */
            APPLY_COIN(coin_R, psi, tmp);
            /* Step 4: S_R */
            sparse_matvec(S_R, tmp, psi);
        }
    }

    fprintf(stderr, "Done.\n");
    free(psi); free(tmp); free(pos); free(mem); free(dirs); free(faces);
    free(coin_R); free(coin_L); free(probs); free(radii);
    free(S_R->row_ptr); free(S_R->col_idx); free(S_R->vals); free(S_R);
    free(S_L->row_ptr); free(S_L->col_idx); free(S_L->vals); free(S_L);
    return 0;
}

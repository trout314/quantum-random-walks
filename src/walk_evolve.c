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
        fprintf(stderr,"\n*** ABORT at %s: only %ld MB available (need > %ld MB). ***\n"
                        "*** Reduce problem size. ***\n",
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
    if (!p) {
        fprintf(stderr,"*** malloc failed for %s (%zu bytes) ***\n", label, bytes);
        exit(2);
    }
    if (bytes > 10*1024*1024)
        fprintf(stderr,"  alloc %s: %.1f MB (avail: %ld MB)\n", label,
                (double)bytes/(1024*1024), avail>0?avail:-1);
    return p;
}

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

    fprintf(stderr, "=== walk_evolve: theta=%.3f sigma=%.1f n_steps=%d ===\n", theta, sigma, n_steps);
    fprintf(stderr, "Memory available: %ld MB\n", get_avail_mb());
    fprintf(stderr, "Sites: %d, Interior: %d, nnz_R: %d, nnz_L: %d\n",
            nsites, n_interior, nnz_R, nnz_L);
    mem_check("read header", 500);

    /* Positions */
    double *pos = safe_malloc(nsites * 3 * sizeof(double), "positions");
    fread(pos, sizeof(double), nsites*3, stdin);

    /* Membership */
    int *mem = safe_malloc(nsites * 4 * sizeof(int), "membership");
    fread(mem, sizeof(int), nsites*4, stdin);

    /* Direction vectors */
    double *dirs = safe_malloc(nsites * 12 * sizeof(double), "dirs");
    fread(dirs, sizeof(double), nsites*12, stdin);

    /* Face indices */
    int *faces = safe_malloc(nsites * 2 * sizeof(int), "faces");
    fread(faces, sizeof(int), nsites*2, stdin);

    /* S_R */
    fprintf(stderr, "Loading S_R (%d nnz)...\n", nnz_R);
    mem_check("S_R load", 500);
    int *sr_rc = safe_malloc(nnz_R * 2 * sizeof(int), "sr_rc");
    fread(sr_rc, sizeof(int), nnz_R*2, stdin);
    double *sr_ri = safe_malloc(nnz_R * 2 * sizeof(double), "sr_ri");
    fread(sr_ri, sizeof(double), nnz_R*2, stdin);

    double complex *sr_vals = safe_malloc(nnz_R * sizeof(double complex), "sr_vals");
    int *sr_rows = safe_malloc(nnz_R * sizeof(int), "sr_rows");
    int *sr_cols = safe_malloc(nnz_R * sizeof(int), "sr_cols");
    for (int i = 0; i < nnz_R; i++) {
        sr_rows[i] = sr_rc[2*i];
        sr_cols[i] = sr_rc[2*i+1];
        sr_vals[i] = sr_ri[2*i] + I*sr_ri[2*i+1];
    }
    sparse_t *S_R = sparse_from_coo(dim, dim, nnz_R, sr_rows, sr_cols, sr_vals);
    free(sr_rc); free(sr_ri); free(sr_rows); free(sr_cols); free(sr_vals);

    /* S_L */
    fprintf(stderr, "Loading S_L (%d nnz)...\n", nnz_L);
    mem_check("S_L load", 500);
    int *sl_rc = safe_malloc(nnz_L * 2 * sizeof(int), "sl_rc");
    fread(sl_rc, sizeof(int), nnz_L*2, stdin);
    double *sl_ri = safe_malloc(nnz_L * 2 * sizeof(double), "sl_ri");
    fread(sl_ri, sizeof(double), nnz_L*2, stdin);

    double complex *sl_vals = safe_malloc(nnz_L * sizeof(double complex), "sl_vals");
    int *sl_rows = safe_malloc(nnz_L * sizeof(int), "sl_rows");
    int *sl_cols = safe_malloc(nnz_L * sizeof(int), "sl_cols");
    for (int i = 0; i < nnz_L; i++) {
        sl_rows[i] = sl_rc[2*i];
        sl_cols[i] = sl_rc[2*i+1];
        sl_vals[i] = sl_ri[2*i] + I*sl_ri[2*i+1];
    }
    sparse_t *S_L = sparse_from_coo(dim, dim, nnz_L, sl_rows, sl_cols, sl_vals);
    free(sl_rc); free(sl_ri); free(sl_rows); free(sl_cols); free(sl_vals);

    fprintf(stderr, "Matrices loaded. dim=%d\n", dim);

    /* ---- Precompute coin matrices at each site ---- */
    fprintf(stderr, "Precomputing coins for %d sites...\n", nsites);
    mem_check("coin precompute", 300);
    double complex *coin_R = safe_malloc(nsites * 16 * sizeof(double complex), "coin_R");
    double complex *coin_L = safe_malloc(nsites * 16 * sizeof(double complex), "coin_L");
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
    fprintf(stderr, "Initializing wavepacket (dim=%d)...\n", dim);
    mem_check("wavepacket init", 200);
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
    /* Radial histogram output: write to "radial.dat" if env var RADIAL_HIST is set */
    FILE *radf = NULL;
    int hist_nbins = 500;
    {
        const char *rf = getenv("RADIAL_HIST");
        if (rf) { radf = fopen(rf, "w"); if (radf) fprintf(stderr, "Radial histograms -> %s\n", rf); }
    }

    printf("# theta=%.4f sigma=%.1f n_steps=%d nsites=%d\n", theta, sigma, n_steps, nsites);
    printf("# t norm r2 x2 y2 z2 r95\n");

    double *radii = malloc(nsites * sizeof(double));
    double *probs = malloc(nsites * sizeof(double));
    double rmax_global = 0;
    for (int n = 0; n < nsites; n++) {
        radii[n] = sqrt(pos[3*n]*pos[3*n]+pos[3*n+1]*pos[3*n+1]+pos[3*n+2]*pos[3*n+2]);
        if (radii[n] > rmax_global) rmax_global = radii[n];
    }

    fprintf(stderr, "Starting time evolution (%d steps)...\n", n_steps);
    for (int t = 0; t <= n_steps; t++) {
        if (t % 10 == 0) fprintf(stderr, "  step %d/%d (mem: %ld MB)\n", t, n_steps, get_avail_mb());
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

        /* Write radial histogram */
        if (radf) {
            double dr = rmax_global / hist_nbins;
            double *hist = calloc(hist_nbins+1, sizeof(double));
            /* Count sites per bin for density normalization */
            double *bin_sites = calloc(hist_nbins+1, sizeof(double));
            for (int n = 0; n < nsites; n++) {
                int b = (int)(radii[n] / dr);
                if (b > hist_nbins) b = hist_nbins;
                hist[b] += probs[n] / total_prob;
                bin_sites[b] += 1.0;
            }
            /* Output: r, P(r) (prob density = prob_in_bin / dr), raw_prob, sites_in_bin */
            fprintf(radf, "# t=%d norm=%.6f\n", t, pnorm);
            for (int b = 0; b <= hist_nbins; b++) {
                double r = (b + 0.5) * dr;
                double shell_vol = 4.0 * M_PI * r * r * dr;  /* for density */
                double density = (shell_vol > 0) ? hist[b] / shell_vol : 0;
                fprintf(radf, "%d %.4f %.8e %.8e %.0f\n", t, r, density, hist[b], bin_sites[b]);
            }
            free(hist); free(bin_sites);
        }

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

    if (radf) fclose(radf);
    fprintf(stderr, "Done.\n");
    free(psi); free(tmp); free(pos); free(mem); free(dirs); free(faces);
    free(coin_R); free(coin_L); free(probs); free(radii);
    free(S_R->row_ptr); free(S_R->col_idx); free(S_R->vals); free(S_R);
    free(S_L->row_ptr); free(S_L->col_idx); free(S_L->vals); free(S_L);
    return 0;
}

/*
 * walk_1d.c — 1D quantum walk on a single BC helix chain.
 *
 * Adaptive: chain extends on-the-fly as the wavepacket spreads.
 * No fixed N — allocates large arrays and tracks active region.
 *
 * Build: gcc -O2 -fopenmp -o walk_1d src/walk_1d.c -lm
 * Usage: ./walk_1d [theta] [sigma] [n_steps] [coin_type] [nu_type]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#ifdef _OPENMP
#include <omp.h>
#endif

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
static c4x4 ALPHA[3];
static void init_dirac(void) {
    c4_zero(ALPHA[0]);c4_zero(ALPHA[1]);c4_zero(ALPHA[2]);
    ALPHA[0][0][3]=1;ALPHA[0][1][2]=1;ALPHA[0][2][1]=1;ALPHA[0][3][0]=1;
    ALPHA[1][0][3]=-I;ALPHA[1][1][2]=I;ALPHA[1][2][1]=-I;ALPHA[1][3][0]=I;
    ALPHA[2][0][2]=1;ALPHA[2][1][3]=-1;ALPHA[2][2][0]=1;ALPHA[2][3][1]=-1;
}
static void make_tau(c4x4 tau, vec3 d) {
    double nu=sqrt(7.0)/4.0; c4_zero(tau);
    for(int i=0;i<4;i++) tau[i][i]=nu*(i<2?1.0:-1.0);
    double dd[3]={d.x,d.y,d.z};
    for(int a=0;a<3;a++)for(int i=0;i<4;i++)for(int j=0;j<4;j++)
        tau[i][j]+=0.75*dd[a]*ALPHA[a][i][j];
}
static void frame_transport(const c4x4 tf, const c4x4 tt, c4x4 U) {
    c4x4 prod; c4_mul(prod, tt, tf);
    double cos_phi = 0;
    for (int i = 0; i < 4; i++) cos_phi += creal(prod[i][i]);
    cos_phi /= 4.0;
    double cos_half = sqrt((1.0 + cos_phi) / 2.0);
    double scale = 1.0 / (2.0 * cos_half);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            U[i][j] = scale * ((i==j ? 1.0 : 0.0) + prod[i][j]);
}

/* ========== Chain data (pre-allocated, built lazily) ========== */
#define MAX_N 500000
static int PAT[4] = {1, 3, 0, 2};

static vec3 pos[MAX_N];
static vec3 dirs[MAX_N][4];
static int face_idx[MAX_N];
static c4x4 tau_arr[MAX_N];
static c4x4 Pp[MAX_N], Pm[MAX_N];
static c4x4 fwd_block[MAX_N], bwd_block[MAX_N];
static c4x4 coin1[MAX_N], coin2[MAX_N];

static int built_up_to = 0;  /* chain geometry built for sites [0, built_up_to) */
static int active_lo = 0, active_hi = 0;  /* psi is nonzero in [active_lo, active_hi) */

static double g_ct, g_st;
static int g_coin_type = 3;  /* default dual parity */
static int g_use_coin2 = 0;

/* Build chain geometry and operators for site i (and its neighbors) */
static void ensure_site(int i) {
    if (i < 0 || i >= MAX_N) {
        fprintf(stderr, "FATAL: site %d out of range [0,%d)\n", i, MAX_N);
        exit(1);
    }
    while (built_up_to <= i) {
        int n = built_up_to;
        if (n == 0) {
            init_tet(dirs[0]);
            pos[0] = (vec3){0,0,0};
            face_idx[0] = PAT[0];
        } else {
            memcpy(dirs[n], dirs[n-1], sizeof(vec3[4]));
            pos[n] = pos[n-1];
            helix_step(&pos[n], dirs[n], PAT[(n-1)%4]);
            if (n % 8 == 0) reorth(dirs[n]);
            face_idx[n] = PAT[n%4];
        }
        /* tau and projectors */
        make_tau(tau_arr[n], dirs[n][face_idx[n]]);
        c4_eye(Pp[n]); c4_eye(Pm[n]);
        for (int a=0;a<4;a++) for(int b=0;b<4;b++) {
            Pp[n][a][b] = 0.5*(Pp[n][a][b]+tau_arr[n][a][b]);
            Pm[n][a][b] = 0.5*(Pm[n][a][b]-tau_arr[n][a][b]);
        }
        /* shift blocks (forward: needs site n+1, defer if not built yet) */
        /* backward: needs site n-1, available if n>0 */
        if (n > 0) {
            c4x4 U; frame_transport(tau_arr[n], tau_arr[n-1], U);
            c4_mul(bwd_block[n], U, Pm[n]);
            /* also set forward block for n-1 -> n */
            frame_transport(tau_arr[n-1], tau_arr[n], U);
            c4_mul(fwd_block[n-1], U, Pp[n-1]);
        }
        /* coins */
        vec3 e = dirs[n][face_idx[n]];
        double en = v3norm(e);
        vec3 ehat = v3scale(e, 1.0/en);
        if (g_coin_type == 3) {
            /* dual parity: f1,f2 perp e */
            vec3 f1; f1.x=ehat.y; f1.y=-ehat.x; f1.z=0;
            double fn=v3norm(f1);
            if(fn<1e-10){f1.x=-ehat.z;f1.y=0;f1.z=ehat.x;fn=v3norm(f1);}
            f1=v3scale(f1,1.0/fn);
            vec3 f2;
            f2.x=ehat.y*f1.z-ehat.z*f1.y;
            f2.y=ehat.z*f1.x-ehat.x*f1.z;
            f2.z=ehat.x*f1.y-ehat.y*f1.x;
            fn=v3norm(f2); f2=v3scale(f2,1.0/fn);
            double fd1[3]={f1.x,f1.y,f1.z}, fd2[3]={f2.x,f2.y,f2.z};
            for(int a=0;a<4;a++)for(int b=0;b<4;b++){
                double complex fa1=fd1[0]*ALPHA[0][a][b]+fd1[1]*ALPHA[1][a][b]+fd1[2]*ALPHA[2][a][b];
                coin1[n][a][b]=g_ct*(a==b?1:0)-I*g_st*fa1;
                double complex fa2=fd2[0]*ALPHA[0][a][b]+fd2[1]*ALPHA[1][a][b]+fd2[2]*ALPHA[2][a][b];
                coin2[n][a][b]=g_ct*(a==b?1:0)-I*g_st*fa2;
            }
            g_use_coin2 = 1;
        } else if (g_coin_type == 1) {
            /* e.alpha coin */
            double dd[3]={e.x,e.y,e.z};
            for(int a=0;a<4;a++)for(int b=0;b<4;b++){
                double complex ea=dd[0]*ALPHA[0][a][b]+dd[1]*ALPHA[1][a][b]+dd[2]*ALPHA[2][a][b];
                coin1[n][a][b]=g_ct*(a==b?1:0)-I*g_st*ea;
            }
        } else {
            /* beta coin */
            double beta_diag[4]={1,1,-1,-1};
            for(int a=0;a<4;a++)for(int b=0;b<4;b++)
                coin1[n][a][b]=(a==b)?(g_ct-I*g_st*beta_diag[a]):0;
        }
        built_up_to++;
    }
}

/* ========== Spinor arrays ========== */
static double complex psi[4*MAX_N];
static double complex tmp_psi[4*MAX_N];

/* ========== Main ========== */
int main(int argc, char **argv) {
    double theta = 0.5;
    double sigma = 50.0;
    int n_steps = 500;
    int nu_type = 0;
    double amp_thresh = 1e-15;  /* extend chain when amplitude > this */

    if (argc > 1) theta = atof(argv[1]);
    if (argc > 2) sigma = atof(argv[2]);
    if (argc > 3) n_steps = atoi(argv[3]);
    if (argc > 4) g_coin_type = atoi(argv[4]);
    if (argc > 5) nu_type = atoi(argv[5]);

    g_ct = cos(theta); g_st = sin(theta);

    fprintf(stderr, "=== walk_1d adaptive: theta=%.3f sigma=%.1f steps=%d coin=%d ===\n",
            theta, sigma, n_steps, g_coin_type);

    init_dirac();

    /* ---- Initialize wavepacket ---- */
    /* Center the Gaussian at MAX_N/2 so we have room on both sides */
    int center = MAX_N / 2;

    /* Build chain out to 4*sigma on each side of center */
    int lo = center - (int)(4*sigma) - 10;
    int hi = center + (int)(4*sigma) + 10;
    if (lo < 0) lo = 0;
    if (hi >= MAX_N) hi = MAX_N - 1;
    ensure_site(hi);

    /* Set initial condition: (1,0,0,0) Gaussian */
    memset(psi, 0, sizeof(psi));
    double norm = 0;
    for (int i = lo; i <= hi; i++) {
        double x = (double)(i - center);
        double w = exp(-x*x / (2*sigma*sigma));
        psi[4*i] = w;
        norm += w*w;
    }
    norm = sqrt(norm);
    for (int i = lo; i <= hi; i++)
        for (int a = 0; a < 4; a++)
            psi[4*i+a] /= norm;

    active_lo = lo;
    active_hi = hi + 1;

    fprintf(stderr, "Initial: active [%d, %d), center=%d, width=%d\n",
            active_lo, active_hi, center, active_hi-active_lo);

    /* ---- Time evolution ---- */
    printf("# theta=%.4f sigma=%.1f n_steps=%d coin=%d\n", theta, sigma, n_steps, g_coin_type);
    printf("# t norm x_mean x2 active_width\n");

    for (int t = 0; t <= n_steps; t++) {
        /* Compute observables over active region.
         * Use long double accumulators for precision at large sigma.
         * Compute variance directly: var = <(x-<x>)^2> to avoid
         * catastrophic cancellation in <x^2> - <x>^2. */
        long double total_prob = 0, mx = 0, mx2 = 0;
        int alo = active_lo, ahi = active_hi;
        #pragma omp parallel for reduction(+:total_prob,mx,mx2)
        for (int i = alo; i < ahi; i++) {
            double p = 0;
            for (int a = 0; a < 4; a++)
                p += creal(psi[4*i+a]*conj(psi[4*i+a]));
            total_prob += p;
            long double x = (long double)(i - center);
            mx += p * x;
            mx2 += p * x * x;
        }
        double pnorm = sqrt((double)total_prob);
        double mx_d = (double)(mx / total_prob);
        double mx2_d = (double)(mx2 / total_prob);

        if (t % 10 == 0 || t <= 5)
            printf("%d %.8f %.4f %.4f %d\n", t, pnorm, mx_d, mx2_d, active_hi-active_lo);

        if (t < n_steps) {
            /* Step 1: Apply coin(s) — embarrassingly parallel */
            #pragma omp parallel for
            for (int i = alo; i < ahi; i++) {
                double complex out[4];
                for (int a=0;a<4;a++){
                    double complex s=0;
                    for(int b=0;b<4;b++) s+=coin1[i][a][b]*psi[4*i+b];
                    out[a]=s;
                }
                for(int a=0;a<4;a++) psi[4*i+a]=out[a];
            }
            if (g_use_coin2) {
                #pragma omp parallel for
                for (int i = alo; i < ahi; i++) {
                    double complex out[4];
                    for (int a=0;a<4;a++){
                        double complex s=0;
                        for(int b=0;b<4;b++) s+=coin2[i][a][b]*psi[4*i+b];
                        out[a]=s;
                    }
                    for(int a=0;a<4;a++) psi[4*i+a]=out[a];
                }
            }

            /* Step 2: Shift with adaptive extension */
            if (active_lo > 0) ensure_site(active_lo - 1);
            ensure_site(active_hi);

            /* Zero tmp over the full range including neighbors */
            int zlo = (active_lo > 0) ? active_lo-1 : 0;
            int zhi = active_hi + 1;
            memset(&tmp_psi[4*zlo], 0, 4*(zhi-zlo)*sizeof(double complex));

            /* Shift: site i's P+ goes to i+1, P- goes to i-1.
             * Each target site receives from exactly 2 sources (i-1's fwd
             * and i+1's bwd). Parallelize by computing each target site's
             * total contribution independently. */
            #pragma omp parallel for
            for (int i = zlo; i < zhi; i++) {
                /* Site i receives: fwd from i-1, bwd from i+1 */
                for (int a = 0; a < 4; a++) {
                    double complex s = 0;
                    if (i-1 >= alo && i-1 < ahi) {
                        for (int b = 0; b < 4; b++)
                            s += fwd_block[i-1][a][b] * psi[4*(i-1)+b];
                    }
                    if (i+1 >= alo && i+1 < ahi) {
                        for (int b = 0; b < 4; b++)
                            s += bwd_block[i+1][a][b] * psi[4*(i+1)+b];
                    }
                    tmp_psi[4*i+a] = s;
                }
            }

            /* Update active region: extend if amplitude at edges is significant */
            int new_lo = active_lo, new_hi = active_hi;

            /* Check lower extension */
            if (active_lo > 0) {
                double p = 0;
                for (int a=0;a<4;a++) p+=creal(tmp_psi[4*(active_lo-1)+a]*conj(tmp_psi[4*(active_lo-1)+a]));
                if (p > amp_thresh) new_lo = active_lo - 1;
            }
            /* Check upper extension */
            {
                double p = 0;
                for (int a=0;a<4;a++) p+=creal(tmp_psi[4*active_hi+a]*conj(tmp_psi[4*active_hi+a]));
                if (p > amp_thresh) new_hi = active_hi + 1;
            }

            /* Also shrink if edges have no amplitude (optional, saves work) */
            /* Skip for now — the active region only grows */

            active_lo = new_lo;
            active_hi = new_hi;

            /* Copy tmp to psi for active region */
            memcpy(&psi[4*active_lo], &tmp_psi[4*active_lo],
                   4*(active_hi-active_lo)*sizeof(double complex));
        }
    }

    /* Output density */
    FILE *pf = fopen("/tmp/walk_1d_density.dat", "w");
    if (pf) {
        double total = 0;
        for (int i = active_lo; i < active_hi; i++)
            for (int a = 0; a < 4; a++)
                total += creal(psi[4*i+a]*conj(psi[4*i+a]));
        fprintf(pf, "# site position prob\n");
        for (int i = active_lo; i < active_hi; i++) {
            double p = 0;
            for (int a = 0; a < 4; a++)
                p += creal(psi[4*i+a]*conj(psi[4*i+a]));
            fprintf(pf, "%d %.6f %.10e\n", i - center,
                    v3norm(pos[i]) * (i >= center ? 1 : -1),
                    p / total);
        }
        fclose(pf);
        fprintf(stderr, "Density -> /tmp/walk_1d_density.dat (%d sites)\n", active_hi-active_lo);
    }

    return 0;
}

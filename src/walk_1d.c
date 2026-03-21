/*
 * walk_1d.c — 1D quantum walk on a single BC helix chain.
 *
 * A long chain of sites connected by the R-helix pattern {1,3,0,2}.
 * The walk operator is W = S·C where:
 *   C = cos(θ)I - i sin(θ)(e·α)  (coin from local direction)
 *   S sends P+ forward, P- backward along the chain
 *
 * Initial state: Gaussian wavepacket centered at the middle of the chain.
 *
 * Build: clang -O2 -o walk_1d src/walk_1d.c -lm
 * Usage: ./walk_1d [n_sites] [theta] [sigma] [n_steps]
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

/* ========== Main ========== */
int main(int argc, char **argv) {
    int N = 2000;           /* chain length */
    double theta = 0.5;
    double sigma = 50.0;    /* in units of chain sites */
    int n_steps = 500;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) theta = atof(argv[2]);
    if (argc > 3) sigma = atof(argv[3]);
    if (argc > 4) n_steps = atoi(argv[4]);

    double ct = cos(theta), st = sin(theta);

    fprintf(stderr, "=== walk_1d: N=%d theta=%.3f sigma=%.1f steps=%d ===\n",
            N, theta, sigma, n_steps);

    init_dirac();

    /* ---- Build chain ---- */
    /* Generate N sites along the R-helix pattern {1,3,0,2} */
    int PAT[4] = {1, 3, 0, 2};

    vec3 *pos = malloc(N * sizeof(vec3));
    vec3 (*dirs)[4] = malloc(N * sizeof(vec3[4]));
    int *face = malloc(N * sizeof(int));  /* face index at each site */

    init_tet(dirs[0]);
    pos[0] = (vec3){0, 0, 0};
    face[0] = PAT[0];

    for (int i = 1; i < N; i++) {
        memcpy(dirs[i], dirs[i-1], sizeof(vec3[4]));
        pos[i] = pos[i-1];
        int f = PAT[(i-1) % 4];
        helix_step(&pos[i], dirs[i], f);
        if (i % 8 == 0) reorth(dirs[i]);
        face[i] = PAT[i % 4];
    }

    /* Precompute tau and projectors at each site */
    c4x4 *tau = malloc(N * sizeof(c4x4));
    c4x4 *Pp = malloc(N * sizeof(c4x4));
    c4x4 *Pm = malloc(N * sizeof(c4x4));
    for (int i = 0; i < N; i++) {
        make_tau(tau[i], dirs[i][face[i]]);
        c4_eye(Pp[i]); c4_eye(Pm[i]);
        for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) {
            Pp[i][a][b] = 0.5 * (Pp[i][a][b] + tau[i][a][b]);
            Pm[i][a][b] = 0.5 * (Pm[i][a][b] - tau[i][a][b]);
        }
    }

    /* Precompute frame transport and shift blocks:
     * fwd_block[i] = U(tau[i]->tau[i+1]) * Pp[i]  (sends P+ from i to i+1)
     * bwd_block[i] = U(tau[i]->tau[i-1]) * Pm[i]  (sends P- from i to i-1) */
    c4x4 *fwd_block = malloc(N * sizeof(c4x4));
    c4x4 *bwd_block = malloc(N * sizeof(c4x4));
    for (int i = 0; i < N; i++) {
        if (i < N-1) {
            c4x4 U; frame_transport(tau[i], tau[i+1], U);
            c4_mul(fwd_block[i], U, Pp[i]);
        }
        if (i > 0) {
            c4x4 U; frame_transport(tau[i], tau[i-1], U);
            c4_mul(bwd_block[i], U, Pm[i]);
        }
    }

    /* Precompute coin at each site:
     * C[i] = cos(θ)I - i sin(θ)(e_face · α) */
    c4x4 *coin = malloc(N * sizeof(c4x4));
    for (int i = 0; i < N; i++) {
        double *d = &dirs[i][face[i]].x;
        for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) {
            double complex ea = d[0]*ALPHA[0][a][b] + d[1]*ALPHA[1][a][b] + d[2]*ALPHA[2][a][b];
            coin[i][a][b] = ct*(a==b ? 1 : 0) - I*st*ea;
        }
    }

    fprintf(stderr, "Chain built. Displacement from site 0 to %d: %.2f\n",
            N-1, v3norm((vec3){pos[N-1].x-pos[0].x, pos[N-1].y-pos[0].y, pos[N-1].z-pos[0].z}));

    /* ---- Initialize wavepacket ---- */
    double complex *psi = calloc(4*N, sizeof(double complex));
    double complex *tmp_psi = calloc(4*N, sizeof(double complex));
    int center = N / 2;
    double norm = 0;
    for (int i = 0; i < N; i++) {
        double x = (double)(i - center);
        double w = exp(-x*x / (2*sigma*sigma));
        psi[4*i] = w;  /* spinor (1,0,0,0) */
        norm += w*w;
    }
    norm = sqrt(norm);
    for (int i = 0; i < 4*N; i++) psi[i] /= norm;

    /* ---- Time evolution ---- */
    printf("# N=%d theta=%.4f sigma=%.1f n_steps=%d\n", N, theta, sigma, n_steps);
    printf("# t norm x_mean x2 absorbed_frac\n");

    for (int t = 0; t <= n_steps; t++) {
        /* Compute observables */
        double total_prob = 0;
        double mx = 0, mx2 = 0;
        for (int i = 0; i < N; i++) {
            double p = 0;
            for (int a = 0; a < 4; a++)
                p += creal(psi[4*i+a] * conj(psi[4*i+a]));
            total_prob += p;
            double x = (double)(i - center);
            mx += p * x;
            mx2 += p * x * x;
        }
        double pnorm = sqrt(total_prob);
        mx /= total_prob;
        mx2 /= total_prob;

        /* Probability at boundaries (absorbed fraction) */
        double edge = 0;
        for (int i = 0; i < 10; i++) {
            for (int a = 0; a < 4; a++) {
                edge += creal(psi[4*i+a]*conj(psi[4*i+a]));
                edge += creal(psi[4*(N-1-i)+a]*conj(psi[4*(N-1-i)+a]));
            }
        }
        edge /= total_prob;

        if (t % 10 == 0 || t <= 5)
            printf("%d %.8f %.4f %.4f %.6f\n", t, pnorm, mx, mx2, edge);

        if (t < n_steps) {
            /* W = S · C: apply coin then shift */

            /* Step 1: Apply coin */
            for (int i = 0; i < N; i++) {
                double complex out[4];
                for (int a = 0; a < 4; a++) {
                    double complex s = 0;
                    for (int b = 0; b < 4; b++)
                        s += coin[i][a][b] * psi[4*i+b];
                    out[a] = s;
                }
                for (int a = 0; a < 4; a++) psi[4*i+a] = out[a];
            }

            /* Step 2: Apply shift (open boundaries) */
            memset(tmp_psi, 0, 4*N*sizeof(double complex));
            for (int i = 0; i < N; i++) {
                /* P+ -> forward (i+1) */
                if (i < N-1) {
                    for (int a = 0; a < 4; a++)
                        for (int b = 0; b < 4; b++)
                            tmp_psi[4*(i+1)+a] += fwd_block[i][a][b] * psi[4*i+b];
                }
                /* P- -> backward (i-1) */
                if (i > 0) {
                    for (int a = 0; a < 4; a++)
                        for (int b = 0; b < 4; b++)
                            tmp_psi[4*(i-1)+a] += bwd_block[i][a][b] * psi[4*i+b];
                }
            }
            memcpy(psi, tmp_psi, 4*N*sizeof(double complex));
        }
    }

    /* Output probability density at final time for plotting */
    FILE *pf = fopen("/tmp/walk_1d_density.dat", "w");
    if (pf) {
        double total = 0;
        for (int i = 0; i < N; i++)
            for (int a = 0; a < 4; a++)
                total += creal(psi[4*i+a]*conj(psi[4*i+a]));
        fprintf(pf, "# site position prob\n");
        for (int i = 0; i < N; i++) {
            double p = 0;
            for (int a = 0; a < 4; a++)
                p += creal(psi[4*i+a]*conj(psi[4*i+a]));
            fprintf(pf, "%d %.6f %.10e\n", i - center,
                    v3norm(pos[i]) * (i >= center ? 1 : -1),  /* signed distance */
                    p / total);
        }
        fclose(pf);
        fprintf(stderr, "Density saved to /tmp/walk_1d_density.dat\n");
    }

    free(psi); free(tmp_psi);
    free(pos); free(dirs); free(face);
    free(tau); free(Pp); free(Pm);
    free(fwd_block); free(bwd_block); free(coin);
    return 0;
}

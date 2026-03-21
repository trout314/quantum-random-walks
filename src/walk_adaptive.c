/*
 * walk_adaptive.c — Adaptive quantum walk with on-the-fly site creation.
 *
 * Combines lattice generation and wavepacket evolution. Sites are created
 * dynamically when the shift operator would send significant amplitude
 * to a nonexistent site. This eliminates boundary absorption for the
 * propagating wavepacket.
 *
 * Build: clang -O2 -o walk_adaptive src/walk_adaptive.c -lm
 * Usage: ./walk_adaptive [seed_depth] [theta] [sigma] [n_steps] [threshold]
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
static void mem_check(const char *label, long reserve_mb) {
    long avail = get_avail_mb();
    if (avail < 0) return;
    if (avail < reserve_mb) {
        fprintf(stderr,"\n*** ABORT at %s: only %ld MB available (need > %ld MB). ***\n",
                label, avail, reserve_mb);
        exit(2);
    }
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
#define HASH_BITS 22
#define HASH_SIZE (1<<HASH_BITS)
#define HASH_MASK (HASH_SIZE-1)
#define MAX_SITES 5000000

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

static void init_storage(void) {
    htab = calloc(HASH_SIZE, sizeof(hentry));
    for (int i = 0; i < HASH_SIZE; i++) htab[i].id = -1;
    sites = malloc(MAX_SITES * sizeof(site_t));
    psi = calloc(4 * (size_t)MAX_SITES, sizeof(double complex));
    tmp = calloc(4 * (size_t)MAX_SITES, sizeof(double complex));
    if (!htab || !sites || !psi || !tmp) {
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
        if(htab[i].kx==kx&&htab[i].ky==ky&&htab[i].kz==kz) return htab[i].id;}
    return -1;
}
static int site_insert(vec3 pos, vec3 dirs[4]) {
    long kx,ky,kz; poskey(pos,&kx,&ky,&kz); unsigned h=hfn(kx,ky,kz);
    for(int p=0;p<HASH_SIZE;p++){unsigned i=(h+p)&HASH_MASK;
        if(htab[i].id==-1){
            if(nsites>=MAX_SITES){fprintf(stderr,"Too many sites\n");exit(1);}
            int id=nsites++; htab[i]=(hentry){kx,ky,kz,id};
            sites[id].pos=pos; memcpy(sites[id].dirs,dirs,sizeof(vec3)*4);
            sites[id].r_next=sites[id].r_prev=-1;
            sites[id].l_next=sites[id].l_prev=-1;
            sites[id].r_face=sites[id].l_face=-1;
            memset(&psi[4*id], 0, 4*sizeof(double complex));
            return id;}
        if(htab[i].kx==kx&&htab[i].ky==ky&&htab[i].kz==kz) return htab[i].id;}
    fprintf(stderr,"Hash full\n"); exit(1);
}

/* ========== Chain threading ========== */
static const int PAT_R[4]={1,3,0,2}, PAT_L[4]={0,1,2,3};

/* Compute the next face in the pattern given current face */
static int next_face(const int pat[4], int cur_face) {
    for (int i = 0; i < 4; i++)
        if (pat[i] == cur_face) return pat[(i+1)%4];
    return pat[0];
}
/* Compute the prev face in the pattern given current face */
static int prev_face(const int pat[4], int cur_face) {
    for (int i = 0; i < 4; i++)
        if (pat[i] == cur_face) return pat[(i+3)%4];
    return pat[0];
}

/* Thread chain through existing sites starting from 'start'. */
static int thread_chain(int start, const int pat[4], int is_r) {
    int rev[4]={pat[3],pat[2],pat[1],pat[0]};
    int linked = 1;

    if (is_r) sites[start].r_face = pat[0];
    else      sites[start].l_face = pat[0];

    /* Forward */
    {
        int cur = start;
        vec3 p = sites[start].pos;
        vec3 d[4]; memcpy(d, sites[start].dirs, sizeof(d));
        for (int step = 0; step < MAX_SITES; step++) {
            int face = pat[step % 4];
            helix_step(&p, d, face);
            if ((step+1) % 8 == 0) reorth(d);
            int next = site_find(p);
            if (next < 0) break;
            if (is_r) {
                if (sites[next].r_prev >= 0) break;
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

    /* Backward */
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

/* ========== Adaptive shift operator ========== */

/* Try to extend a chain from site s in the forward (+) direction.
 * Returns the new/existing neighbor site id, or -1 if amplitude too small.
 * If a new site is created, it's linked and its psi is zeroed. */
static int try_extend_fwd(int s, int is_r, const int pat[4],
                           double complex shifted[4], double thresh2) {
    /* Check amplitude */
    double amp2 = 0;
    for (int a = 0; a < 4; a++) amp2 += creal(shifted[a]*conj(shifted[a]));
    if (amp2 < thresh2) return -1;

    int face = is_r ? sites[s].r_face : sites[s].l_face;
    int nf = next_face(pat, face);

    /* Compute neighbor position */
    vec3 p = sites[s].pos;
    vec3 d[4]; memcpy(d, sites[s].dirs, sizeof(d));
    helix_step(&p, d, face);

    /* Find or create */
    int nb = site_find(p);
    if (nb < 0) {
        reorth(d);
        nb = site_insert(p, d);
    }

    /* Link */
    if (is_r) {
        sites[s].r_next = nb;
        if (sites[nb].r_prev < 0) {
            sites[nb].r_prev = s;
            sites[nb].r_face = nf;
        }
    } else {
        sites[s].l_next = nb;
        if (sites[nb].l_prev < 0) {
            sites[nb].l_prev = s;
            sites[nb].l_face = nf;
        }
    }
    return nb;
}

/* Same for backward (-) direction.
 *
 * The predecessor of s in the chain is the site that stepped with
 * prev_face(pat, s->face) to reach s. To find its position:
 *
 * If A does helix_step(f) to get B, then from B, helix_step(f)
 * returns to A (since reflection is involutory and the displacement
 * reverses: B.dirs[f] = -A.dirs[f]).
 *
 * The predecessor stepped with face pf = prev_face(pat, face) to reach s.
 * So from s, helix_step(pf) goes back to the predecessor. */
static int try_extend_bwd(int s, int is_r, const int pat[4],
                           double complex shifted[4], double thresh2) {
    double amp2 = 0;
    for (int a = 0; a < 4; a++) amp2 += creal(shifted[a]*conj(shifted[a]));
    if (amp2 < thresh2) return -1;

    int face = is_r ? sites[s].r_face : sites[s].l_face;
    int pf = prev_face(pat, face);

    /* Step with pf (the predecessor's face) to reach the predecessor */
    vec3 p = sites[s].pos;
    vec3 dd[4]; memcpy(dd, sites[s].dirs, sizeof(dd));
    helix_step(&p, dd, pf);

    int nb = site_find(p);
    if (nb < 0) {
        reorth(dd);
        nb = site_insert(p, dd);
    }

    if (is_r) {
        sites[s].r_prev = nb;
        if (sites[nb].r_next < 0) {
            sites[nb].r_next = s;
            sites[nb].r_face = pf;
        }
    } else {
        sites[s].l_prev = nb;
        if (sites[nb].l_next < 0) {
            sites[nb].l_next = s;
            sites[nb].l_face = pf;
        }
    }
    return nb;
}

/* Apply shift operator for one chirality with adaptive site creation.
 * Reads from psi, writes to tmp. */
static void apply_shift_adaptive(int is_r, const int pat[4], double thresh2,
                                  int *n_created, double *prob_absorbed) {
    int ns = nsites;  /* snapshot — don't process sites created during this pass */
    *n_created = 0;
    *prob_absorbed = 0;

    /* Zero tmp for all current sites */
    memset(tmp, 0, 4 * (size_t)ns * sizeof(double complex));

    for (int s = 0; s < ns; s++) {
        int face = is_r ? sites[s].r_face : sites[s].l_face;

        if (face < 0) {
            /* Not on a chain of this type — identity */
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
            /* Try to extend if at chain end */
            if (nb < 0)
                nb = try_extend_fwd(s, is_r, pat, shifted, thresh2);

            if (nb >= 0) {
                int fn = is_r ? sites[nb].r_face : sites[nb].l_face;
                vec3 dn = sites[nb].dirs[fn];
                c4x4 tn; make_tau(tn, dn);
                c4x4 U, bl; frame_transport(tau, tn, U); c4_mul(bl, U, Pp);
                for (int a=0;a<4;a++) for(int b=0;b<4;b++)
                    tmp[4*nb+a] += bl[a][b] * psi[4*s+b];
                if (nb >= ns) (*n_created)++;
            } else {
                /* Absorbed */
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

    /* Copy tmp to psi (including any new sites created beyond ns) */
    memcpy(psi, tmp, 4 * (size_t)nsites * sizeof(double complex));
}

/* Apply coin operator (in-place via tmp) */
static void apply_coin(int is_r, double ct, double st) {
    int ns = nsites;
    for (int n = 0; n < ns; n++) {
        int face = is_r ? sites[n].r_face : sites[n].l_face;
        double complex out[4];
        if (face >= 0) {
            double *d = &sites[n].dirs[face].x;
            for (int a = 0; a < 4; a++) {
                double complex s = 0;
                for (int b = 0; b < 4; b++) {
                    double complex ea = d[0]*ALPHA[0][a][b] + d[1]*ALPHA[1][a][b] + d[2]*ALPHA[2][a][b];
                    double complex Cab = ct*(a==b?1:0) - I*st*ea;
                    s += Cab * psi[4*n+b];
                }
                out[a] = s;
            }
        } else {
            for (int a = 0; a < 4; a++) out[a] = psi[4*n+a];
        }
        for (int a = 0; a < 4; a++) psi[4*n+a] = out[a];
    }
}

/* ========== Main ========== */
int main(int argc, char **argv) {
    int seed_depth = 6;
    double theta = 0.5, sigma = 3.0, threshold = 1e-10;
    int n_steps = 50;
    if (argc > 1) seed_depth = atoi(argv[1]);
    if (argc > 2) theta = atof(argv[2]);
    if (argc > 3) sigma = atof(argv[3]);
    if (argc > 4) n_steps = atoi(argv[4]);
    if (argc > 5) threshold = atof(argv[5]);
    double thresh2 = threshold * threshold;
    double ct = cos(theta), st = sin(theta);

    fprintf(stderr, "=== walk_adaptive: seed=%d theta=%.3f sigma=%.1f steps=%d thresh=%.1e ===\n",
            seed_depth, theta, sigma, n_steps, threshold);
    fprintf(stderr, "Memory available: %ld MB\n", get_avail_mb());

    init_dirac();
    init_storage();

    /* ---- Phase 1: BFS seed ball ---- */
    fprintf(stderr, "\n--- Phase 1: BFS seed (depth %d) ---\n", seed_depth);
    vec3 d0[4]; init_tet(d0);
    site_insert((vec3){0,0,0}, d0);
    int *queue = malloc(MAX_SITES * sizeof(int));
    int qh=0, qt=0; queue[qt++] = 0;
    int *depth = calloc(MAX_SITES, sizeof(int));
    while (qh < qt) {
        int s = queue[qh++];
        if (depth[s] >= seed_depth) continue;
        for (int f = 0; f < 4; f++) {
            vec3 p = sites[s].pos, dd[4];
            memcpy(dd, sites[s].dirs, sizeof(dd));
            helix_step(&p, dd, f); reorth(dd);
            int old_n = nsites;
            int nid = site_insert(p, dd);
            if (nid >= old_n) { depth[nid] = depth[s]+1; queue[qt++] = nid; }
        }
    }
    free(queue); free(depth);
    int n_seed = nsites;
    fprintf(stderr, "Seed: %d sites\n", n_seed);

    /* ---- Phase 2: Thread chains through seed ---- */
    fprintf(stderr, "\n--- Phase 2: Threading chains ---\n");
    int nrch=0, nlch=0;
    for (int s = 0; s < n_seed; s++) {
        if (sites[s].r_face < 0) {
            if (thread_chain(s, PAT_R, 1) >= 2) nrch++;
        }
        if (sites[s].l_face < 0) {
            if (thread_chain(s, PAT_L, 0) >= 2) nlch++;
        }
    }
    fprintf(stderr, "Chains: %d R, %d L\n", nrch, nlch);

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
    printf("# theta=%.4f sigma=%.1f n_steps=%d seed=%d thresh=%.1e\n",
           theta, sigma, n_steps, seed_depth, threshold);
    printf("# t norm r2 x2 y2 z2 r95 nsites absorbed\n");

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
            p /= total_prob;
            double x=sites[n].pos.x, y=sites[n].pos.y, z=sites[n].pos.z;
            mx2 += p*x*x; my2 += p*y*y; mz2 += p*z*z;
        }

        /* r95 */
        double r95 = 0;
        {
            double rmax = 0;
            for (int n = 0; n < nsites; n++) {
                double r = v3norm(sites[n].pos);
                if (r > rmax) rmax = r;
            }
            int nbins = 200;
            double dr = rmax / nbins;
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

            /* Radial histogram */
            if (radf) {
                fprintf(radf, "# t=%d norm=%.6f\n", t, pnorm);
                for (int b = 0; b <= nbins; b++) {
                    double r = (b + 0.5) * dr;
                    fprintf(radf, "%d %.4f %.8e\n", t, r, bp[b]);
                }
            }
            free(bp);
        }

        printf("%d %.6f %.2f %.2f %.2f %.2f %.2f %d 0.0\n",
               t, pnorm, mx2+my2+mz2, mx2, my2, mz2, r95, nsites);
        fflush(stdout);

        if (t < n_steps) {
            if (t % 5 == 0)
                fprintf(stderr, "  step %d/%d: %d sites, norm=%.6f (mem: %ld MB)\n",
                        t, n_steps, nsites, pnorm, get_avail_mb());

            int cr_l=0, cr_r=0;
            double abs_l=0, abs_r=0;

            /* W = S_R · C_R · S_L · C_L, applied right to left */
            /* Step 1: C_L */
            apply_coin(0, ct, st);
            /* Step 2: S_L (adaptive) */
            apply_shift_adaptive(0, PAT_L, thresh2, &cr_l, &abs_l);
            /* Step 3: C_R */
            apply_coin(1, ct, st);
            /* Step 4: S_R (adaptive) */
            apply_shift_adaptive(1, PAT_R, thresh2, &cr_r, &abs_r);

            /* Update the output line's absorbed column for the NEXT step */
        }
    }

    if (radf) fclose(radf);
    fprintf(stderr, "\nDone. Final: %d sites\n", nsites);
    free(htab); free(sites); free(psi); free(tmp);
    return 0;
}

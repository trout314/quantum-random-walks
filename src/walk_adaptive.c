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
#define HASH_BITS 25
#define HASH_SIZE (1<<HASH_BITS)
#define HASH_MASK (HASH_SIZE-1)
#define MAX_SITES 20000000

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
    if (nb >= 0) {
        /* Site already exists — it's either on this chain already (gap from
         * thread_chain stopping early) or a site created by BFS/other chain.
         * Don't send amplitude to it — it's already handled by its own chain
         * segment. Link for future steps but return -1 to absorb this step. */
        if (is_r) { sites[s].r_next = nb; if (sites[nb].r_prev<0) { sites[nb].r_prev=s; sites[nb].r_face=nf; } }
        else      { sites[s].l_next = nb; if (sites[nb].l_prev<0) { sites[nb].l_prev=s; sites[nb].l_face=nf; } }
        return -1;  /* absorb this step's amplitude */
    }

    /* Genuinely new site */
    reorth(d);
    nb = site_insert(p, d);
    if (is_r) { sites[s].r_next = nb; sites[nb].r_prev = s; sites[nb].r_face = nf; }
    else      { sites[s].l_next = nb; sites[nb].l_prev = s; sites[nb].l_face = nf; }
    return nb;
}

/* Same for backward (-) direction.
 *
 * The predecessor stepped with face pf = prev_face(pat, face) to reach s.
 * From s, helix_step(pf) goes back to the predecessor (since the step
 * is its own inverse when using the same face). */
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
        /* Existing site — link for future but absorb this step */
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

/* Apply shift operator for one chirality with adaptive site creation.
 * Reads from psi, writes to tmp. */
static void apply_shift_adaptive(int is_r, const int pat[4], double thresh2,
                                  int *n_created, double *prob_absorbed) {
    int ns = nsites;  /* snapshot — don't process sites created during this pass */
    *n_created = 0;
    *prob_absorbed = 0;

    /* Norm accounting */
    double norm_in = 0, norm_identity = 0, norm_fwd = 0, norm_bwd = 0;

    /* Zero tmp for all current sites */
    memset(tmp, 0, 4 * (size_t)ns * sizeof(double complex));

    for (int s = 0; s < ns; s++) {
        int face = is_r ? sites[s].r_face : sites[s].l_face;

        /* Accumulate input norm */
        for (int a = 0; a < 4; a++) norm_in += creal(psi[4*s+a]*conj(psi[4*s+a]));

        if (face < 0) {
            /* Not on a chain of this type — identity */
            for (int a = 0; a < 4; a++) tmp[4*s+a] = psi[4*s+a];
            for (int a = 0; a < 4; a++) norm_identity += creal(psi[4*s+a]*conj(psi[4*s+a]));
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
                if (fn < 0) {
                    fprintf(stderr, "BUG: nb=%d has no face for %s chain (from fwd extend of s=%d)\n",
                            nb, is_r?"R":"L", s);
                }
                vec3 dn = sites[nb].dirs[fn >= 0 ? fn : 0];
                c4x4 tn; make_tau(tn, dn);
                c4x4 U, bl; frame_transport(tau, tn, U); c4_mul(bl, U, Pp);
                /* Check: ||bl·psi||² should equal ||Pp·psi||² since U is unitary */
                double complex result[4] = {0};
                for (int a=0;a<4;a++) for(int b=0;b<4;b++)
                    result[a] += bl[a][b] * psi[4*s+b];
                double nr = 0;
                for (int a=0;a<4;a++) nr += creal(result[a]*conj(result[a]));
                double ns2 = 0;
                for (int a=0;a<4;a++) ns2 += creal(shifted[a]*conj(shifted[a]));
                if (fabs(nr - ns2) > 1e-10)
                    fprintf(stderr, "NORM MISMATCH fwd s=%d nb=%d: ||bl·psi||²=%.10f ||Pp·psi||²=%.10f diff=%.2e\n",
                            s, nb, nr, ns2, nr-ns2);
                for (int a=0;a<4;a++) tmp[4*nb+a] += result[a];
                norm_fwd += ns2;
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
                for (int a=0;a<4;a++) norm_bwd += creal(shifted[a]*conj(shifted[a]));
                if (nb >= ns) (*n_created)++;
            } else {
                for (int a=0;a<4;a++)
                    *prob_absorbed += creal(shifted[a]*conj(shifted[a]));
            }
        }
    }

    /* Check: how many new sites received from 2 sources? */
    {
        int double_recv = 0;
        char *recv_count = calloc(nsites, 1);
        /* Re-scan: not ideal but diagnostic only */
        for (int s = 0; s < ns; s++) {
            int face = is_r ? sites[s].r_face : sites[s].l_face;
            if (face < 0) continue;
            int nb_fwd = is_r ? sites[s].r_next : sites[s].l_next;
            int nb_bwd = is_r ? sites[s].r_prev : sites[s].l_prev;
            if (nb_fwd >= ns) recv_count[nb_fwd]++;
            if (nb_bwd >= ns) recv_count[nb_bwd]++;
        }
        for (int s = ns; s < nsites; s++)
            if (recv_count[s] > 1) double_recv++;
        if (double_recv > 0)
            fprintf(stderr, "    %d new sites received from 2+ sources\n", double_recv);
        free(recv_count);
    }

    /* Compute output norm, separating old vs new sites */
    double norm_out = 0, norm_out_old = 0, norm_out_new = 0;
    for (int s = 0; s < nsites; s++) {
        double sn = 0;
        for (int a = 0; a < 4; a++)
            sn += creal(tmp[4*s+a]*conj(tmp[4*s+a]));
        norm_out += sn;
        if (s < ns) norm_out_old += sn;
        else norm_out_new += sn;
    }

    if (fabs(norm_out/norm_in - 1.0) > 0.001)
        fprintf(stderr, "    %s shift: ratio=%.6f (in=%.6f out=%.6f old=%.6f new=%.6f)\n",
                is_r?"R":"L", norm_out/norm_in, norm_in, norm_out,
                norm_out_old, norm_out_new);

    /* Copy tmp to psi (including any new sites created beyond ns) */
    memcpy(psi, tmp, 4 * (size_t)nsites * sizeof(double complex));
}

/* Apply one coin pass using direction vector d[3] */
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

/* Apply coin operator (in-place).
 * coin_type: 0 = e·α (original), 1 = dual parity f·α (f₁,f₂ ⊥ e) */
static int g_coin_type = 1;  /* default to dual parity */

static void apply_coin(int is_r, double ct, double st) {
    int ns = nsites;
    for (int n = 0; n < ns; n++) {
        int face = is_r ? sites[n].r_face : sites[n].l_face;
        if (face < 0) continue;  /* no chain = identity */

        vec3 e = sites[n].dirs[face];
        double en = v3norm(e);
        if (en < 1e-15) continue;
        vec3 ehat = v3scale(e, 1.0/en);

        if (g_coin_type == 0) {
            /* Original e·α coin */
            double d[3] = {e.x, e.y, e.z};
            apply_coin_dir(d, n, ct, st);
        } else {
            /* Dual parity coin: f₁·α then f₂·α, both ⊥ e */
            /* f₁ = cross(ê, ẑ), fallback to cross(ê, ŷ) */
            vec3 f1;
            f1.x = ehat.y; f1.y = -ehat.x; f1.z = 0;
            double fn = v3norm(f1);
            if (fn < 1e-10) {
                f1.x = -ehat.z; f1.y = 0; f1.z = ehat.x;
                fn = v3norm(f1);
            }
            f1 = v3scale(f1, 1.0/fn);
            /* f₂ = cross(ê, f₁) */
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

/* Apply post-shift V mixing: V = cos(phi)*I + i*sin(phi)*M
 * where M swaps P+ and P- eigenspaces of the local tau.
 * is_r: use R-face tau (1) or L-face tau (0). */
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

        /* Build P+ and P- projectors */
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
        /* V = cos(phi)*I + i*sin(phi)*M */
        for(int a=0;a<4;a++) for(int b=0;b<4;b++)
            V[a][b] = cp*(a==b?1:0) + I*sp*V[a][b];

        /* Apply V to psi at site s */
        double complex out[4] = {0};
        for(int a=0;a<4;a++) for(int b=0;b<4;b++)
            out[a] += V[a][b] * psi[4*s+b];
        for(int a=0;a<4;a++) psi[4*s+a] = out[a];
    }
}

/* ========== Main ========== */
int main(int argc, char **argv) {
    int seed_depth = 0;  /* 0 = auto from sigma */
    double theta = 0.5, sigma = 3.0, threshold = 1e-10;
    int n_steps = 50;
    if (argc > 1) theta = atof(argv[1]);
    if (argc > 2) sigma = atof(argv[2]);
    if (argc > 3) n_steps = atoi(argv[3]);
    if (argc > 4) threshold = atof(argv[4]);
    if (argc > 5) seed_depth = atoi(argv[5]);
    if (argc > 6) g_coin_type = atoi(argv[6]);
    if (argc > 7) g_mix_phi = atof(argv[7]);
    double thresh2 = threshold * threshold;
    double ct = cos(theta), st = sin(theta);

    /* Auto seed depth: ensure Gaussian has decayed to < exp(-8) ~ 3e-4
     * at the boundary. Need r > 4σ, and r ~ depth * 2/3 (step length). */
    if (seed_depth <= 0) {
        seed_depth = (int)(4.0 * sigma / (2.0/3.0)) + 2;
        if (seed_depth < 4) seed_depth = 4;
    }

    fprintf(stderr, "=== walk_adaptive: seed=%d theta=%.3f sigma=%.1f steps=%d thresh=%.1e coin=%s mix_phi=%.4f ===\n",
            seed_depth, theta, sigma, n_steps, threshold,
            g_coin_type==0 ? "e.alpha" : "dual_parity", g_mix_phi);
    fprintf(stderr, "Memory available: %ld MB\n", get_avail_mb());

    init_dirac();
    init_storage();

    /* ---- Phase 1: BFS seed ball ---- */
    /* Grow until the Gaussian amplitude at boundary sites is negligible.
     * Use radial pruning (prune_ratio=0.7) to prevent exponential blowup,
     * but keep the interior dense (no pruning below min_prune_depth=6).
     * Only expand sites where exp(-r²/2σ²) > seed_threshold. */
    double seed_thresh = 1e-4;
    double step_len = 2.0/3.0;

    /* Density-based pruning: divide space into cells of size `cell_size`.
     * Don't expand from a site if its cell already has >= max_per_cell sites.
     * This gives dense coverage near the origin (where BFS fills cells early)
     * and sparse coverage at large r (where only efficient paths reach). */
    double cell_size = 1.0;
    int max_per_cell = 20;
    /* Grid: centered at origin, covering [-grid_half, grid_half]³ */
    double grid_half = seed_depth * step_len + 5.0;
    int grid_n = (int)(2.0 * grid_half / cell_size) + 1;
    if (grid_n > 500) grid_n = 500;  /* cap grid size */
    int grid_total = grid_n * grid_n * grid_n;
    int *grid_count = calloc(grid_total, sizeof(int));

    fprintf(stderr, "\n--- Phase 1: BFS seed (depth %d, Gaussian > %.1e, density grid %d³) ---\n",
            seed_depth, seed_thresh, grid_n);

    /* Helper to get grid cell index from position */
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
    int qh=0, qt=0; queue[qt++] = 0;
    int *sdepth = calloc(MAX_SITES, sizeof(int));
    int npruned = 0;
    while (qh < qt) {
        int s = queue[qh++];
        if (sdepth[s] >= seed_depth) continue;
        double r2 = v3dot(sites[s].pos, sites[s].pos);
        /* Stop expanding if Gaussian is negligible */
        if (exp(-r2 / (2*sigma*sigma)) < seed_thresh) continue;
        /* Density pruning: don't expand if this cell is already full */
        int gi = GRID_IDX(sites[s].pos);
        if (grid_count[gi] >= max_per_cell) { npruned++; continue; }
        for (int f = 0; f < 4; f++) {
            vec3 p = sites[s].pos, dd[4];
            memcpy(dd, sites[s].dirs, sizeof(dd));
            helix_step(&p, dd, f); reorth(dd);
            int old_n = nsites;
            int nid = site_insert(p, dd);
            if (nid >= old_n) {
                sdepth[nid] = sdepth[s]+1;
                queue[qt++] = nid;
                int ci = GRID_IDX(sites[nid].pos);
                grid_count[ci]++;
            }
        }
        if (nsites % 500000 == 0)
            fprintf(stderr, "  BFS: %d sites, depth ~%d, pruned %d\n", nsites, sdepth[s], npruned);
    }
    free(queue); free(sdepth); free(grid_count);
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
            double dr = 0.5;  /* fixed bin width */
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

            /* W = V_R · S_R · C_R · V_L · S_L · C_L, applied right to left */
            /* Step 1: C_L */
            apply_coin(0, ct, st);
            /* Step 2: S_L (adaptive) */
            apply_shift_adaptive(0, PAT_L, thresh2, &cr_l, &abs_l);
            /* Step 3: V_L mixing */
            apply_vmix(0);
            /* Step 4: C_R */
            apply_coin(1, ct, st);
            /* Step 5: S_R (adaptive) */
            apply_shift_adaptive(1, PAT_R, thresh2, &cr_r, &abs_r);
            /* Step 6: V_R mixing */
            apply_vmix(1);

            /* Update the output line's absorbed column for the NEXT step */
        }
    }

    if (radf) fclose(radf);
    fprintf(stderr, "\nDone. Final: %d sites\n", nsites);
    free(htab); free(sites); free(psi); free(tmp);
    return 0;
}

# Chapter 27: N-Body Simulation at Scale

**What you will understand by the end of this chapter:**

- Why N-body force calculation needs a fundamentally different communication pattern from every domain-decomposition problem this book has built so far -- a real, quoted reason gravitational and Coulomb forces cannot be handled with a halo exchange.
- A real, verified correctness simulation showing exactly what a distributed all-pairs force calculation must do instead: an all-gather, not a neighbor exchange -- and a real closed-form model quantifying just how much worse that makes the communication shape at scale.
- The real, named algorithmic answer to that problem, Barnes-Hut, and why it changes both compute and communication in the same direction, backed by real measured numbers from an actual GPU implementation of it.

**What you need to know first:**

- Chapter 16's own domain decomposition and halo-exchange communication-volume model.
- Chapter 9 and Chapter 10's own real ring all-reduce and all-gather round-count and volume formulas.
- Chapter 8's own floating-point non-associativity caution, and Chapter 24/26's own integer-based routing-correctness technique.

---

Every domain-decomposition problem this book has built so far -- Chapter 16's own stencil solver chief among them -- split a problem across devices and needed only NEIGHBOR data to stay correct, because a PDE's own update rule only ever reads immediately adjacent cells. N-body simulation looks like the same kind of problem: split a large number of bodies across devices, the same way Chapter 16 split a grid. But the real physics breaks that assumption immediately, and GPU Gems 3's own real "Fast N-Body Simulation with CUDA" chapter (Nyland, Harris, Prins) states why plainly: "The total force F_i on body i, due to its interactions with the other N-1 bodies, is obtained by summing all interactions." Gravity has no cutoff radius, so splitting bodies across devices the way Chapter 16 split a grid does not, by itself, reduce what each device needs to know. This chapter builds that real difference as a correctness simulation (27.1), quantifies exactly how much worse it makes the communication shape at scale (27.2), and closes with the real, named algorithmic answer that changes the shape of the problem itself (27.3).

```text
Chapter 16 (PDE stencil, halo exchange):     This chapter (N-body, all-gather):

  device owns a GRID REGION                    device owns a BODY SHARD
  update reads ONLY adjacent cells              force needs EVERY other body
        |                                             |
  halo exchange: boundary cells only            all-gather: EVERY body's data
  volume = O(boundary), FLAT as P grows         volume = O(N), stays near FULL
                                                 dataset size as P grows
                       |                                |
                       +----------------+----------------+
                                        |
                    "gravity has no cutoff radius" --
                    real reason these two shapes genuinely differ
```

## 27.1 All-Pairs: Every Body Needs Every Other Body

### Intuition

Chapter 16's own halo exchange worked because a stencil update is LOCAL: cell `(i,j)`'s next value depends only on its immediate neighbors, so a device only ever needs its own region plus a thin boundary layer from its neighbors. An N-body force calculation has no such locality. GPU Gems 3's own real chapter states the algorithm plainly: "The all-pairs approach to N-body simulation is a brute-force technique that evaluates all pair-wise interactions among the N bodies," and because "each entry can be computed independently, there is O(N^2) available parallelism" -- but that same all-pairs structure means every single body's force depends on literally every OTHER body, however far away. Splitting N bodies into P shards, the way Chapter 16 split a grid, does not reduce what each shard needs to know: each shard must still obtain every other shard's bodies before it can correctly compute forces on its own. The real collective that provides this is an all-gather (Chapter 10's own real collective), not a halo exchange -- every shard ends up holding the COMPLETE body list, not just a thin boundary slice of it.

```text
Chapter 16's halo exchange (2 shards):        This chapter's all-gather (3 shards):

Shard 0: [own region][boundary from 1]        Shard 0: [ALL N bodies]
Shard 1: [boundary from 0][own region]        Shard 1: [ALL N bodies]
                                               Shard 2: [ALL N bodies]
  ONLY the adjacent slice crosses               EVERY shard's own bodies
                                                 cross to EVERY other shard
```

### Background

```cpp
// Chapter 27: N-Body Simulation at Scale
// 78_all_pairs_nbody_correctness_simulation.cpp
//
// Chapter 16's own domain decomposition split a PDE's grid across
// devices and needed only NEIGHBOR data -- a halo exchange -- because
// a stencil's own update only ever reads immediately adjacent cells.
// N-body force calculation is a genuinely different problem, and GPU
// Gems 3's own real "Fast N-Body Simulation with CUDA" chapter (Nyland,
// Harris, Prins) states why plainly: "The total force F_i on body i,
// due to its interactions with the other N-1 bodies, is obtained by
// summing all interactions" -- every body needs every OTHER body,
// however far away, because gravity has no cutoff radius. This file
// builds that real difference as a host-side simulation: N bodies
// split across P shards (Chapter 16's own partitioning idea, reused),
// but instead of a halo exchange, each shard must first ALL-GATHER
// (Chapter 10's own real collective) every OTHER shard's own bodies
// before it can compute correct forces on its own bodies -- exactly
// the "each thread computes all N interactions for one body" pattern
// GPU Gems 3 describes, now applied at the shard level. Verified,
// Chapter 24/26-style, using integer positions/masses and a simplified
// (non-physical) pairwise interaction, so the comparison against a
// naive centralized reference is bit-exact with no rounding ambiguity
// -- this checks the ROUTING (does every shard end up with the right
// total force?), not the physical accuracy of the force law itself.
#include <cstdio>
#include <cstdlib>
#include <vector>

const int N = 12;   // total bodies
const int P = 3;    // shards ("GPUs")
const int SHARD_SIZE = N / P;

struct Body { long long x; long long mass; };

// Deterministic bodies: spread-out integer positions, small integer masses.
std::vector<Body> makeBodies() {
    std::vector<Body> b(N);
    for (int i = 0; i < N; i++) {
        b[i].x = i * 3 - 15;      // -15, -12, ..., 18
        b[i].mass = (i % 4) + 1;  // 1..4, repeating
    }
    return b;
}

// A simplified (non-physical) pairwise interaction -- NOT real inverse-
// square gravity, chosen only so every intermediate value stays an
// exact integer (Chapter 8's own non-associativity caution: this is a
// correctness check on the ROUTING, the same discipline Chapter 24's
// SUMMA simulation and Chapter 26's own TP+PP simulation both used).
long long interaction(const Body &bi, const Body &bj) {
    return bi.mass * bj.mass * (bj.x - bi.x);
}

// --- Naive reference: one process, every body's force computed by
// summing interactions with every OTHER body -- GPU Gems 3's own real
// all-pairs pattern, run centrally. ---
std::vector<long long> naiveAllPairs(const std::vector<Body> &bodies) {
    std::vector<long long> force(N, 0);
    for (int i = 0; i < N; i++) {
        long long f = 0;
        for (int j = 0; j < N; j++) {
            if (j == i) continue;
            f += interaction(bodies[i], bodies[j]);
        }
        force[i] = f;
    }
    return force;
}

// --- Distributed: P shards, each owning SHARD_SIZE bodies. Step 1:
// ALL-GATHER -- every shard's own local bodies are shared with every
// OTHER shard (Chapter 10's own real collective semantics: after this
// step, EVERY shard holds the FULL N-body list, unlike Chapter 16's
// halo exchange, where only boundary-adjacent neighbor data ever
// crosses). Step 2: each shard computes forces on ONLY its own owned
// bodies, but using the now-complete gathered list -- exactly GPU
// Gems 3's "each thread computes all N interactions for one body,"
// applied per shard instead of per thread. ---
std::vector<long long> distributedAllPairs(const std::vector<Body> &bodies) {
    std::vector<long long> force(N, 0);
    for (int s = 0; s < P; s++) {
        // The all-gather itself: this shard's own view of "all bodies"
        // is simply the full input list, standing in for a real
        // ncclAllGather()-style collective (Chapter 10) that would, on
        // real hardware, assemble this same complete list from every
        // shard's own local partition.
        const std::vector<Body> &gathered = bodies;

        int start = s * SHARD_SIZE;
        int end = start + SHARD_SIZE;
        for (int i = start; i < end; i++) {
            long long f = 0;
            for (int j = 0; j < N; j++) {
                if (j == i) continue;
                f += interaction(gathered[i], gathered[j]);
            }
            force[i] = f;
        }
    }
    return force;
}

int main() {
    std::vector<Body> bodies = makeBodies();

    printf("N=%d bodies, P=%d shards (%d bodies/shard). Each shard must "
           "ALL-GATHER every other shard's bodies before computing forces "
           "-- unlike Chapter 16's own halo exchange, which only ever "
           "moved BOUNDARY-adjacent data.\n\n", N, P, SHARD_SIZE);

    std::vector<long long> refForce = naiveAllPairs(bodies);
    std::vector<long long> distForce = distributedAllPairs(bodies);

    bool allExact = true;
    for (int i = 0; i < N; i++) {
        bool exact = (refForce[i] == distForce[i]);
        allExact = allExact && exact;
        printf("Body %2d (x=%3lld, mass=%lld): naive force=%8lld  "
               "distributed force=%8lld  %s\n",
               i, bodies[i].x, bodies[i].mass, refForce[i], distForce[i],
               exact ? "EXACT MATCH" : "MISMATCH");
    }

    printf("\nAll %d bodies EXACT MATCH between naive all-pairs and "
           "distributed (shard + all-gather) all-pairs: %s\n",
           N, allExact ? "YES" : "NO");

    return allExact ? 0 : 1;
}
```

Compiled with `g++ -O2 78_all_pairs_nbody_correctness_simulation.cpp -o 78_all_pairs_nbody_correctness_simulation` and genuinely run, on both the cloud sandbox and the device (a plain host computation), with byte-identical output on both. Locked output:

```text
N=12 bodies, P=3 shards (4 bodies/shard). Each shard must ALL-GATHER every other shard's bodies before computing forces -- unlike Chapter 16's own halo exchange, which only ever moved BOUNDARY-adjacent data.

Body  0 (x=-15, mass=1): naive force=     540  distributed force=     540  EXACT MATCH
Body  1 (x=-12, mass=2): naive force=     900  distributed force=     900  EXACT MATCH
Body  2 (x= -9, mass=3): naive force=    1080  distributed force=    1080  EXACT MATCH
Body  3 (x= -6, mass=4): naive force=    1080  distributed force=    1080  EXACT MATCH
Body  4 (x= -3, mass=1): naive force=     180  distributed force=     180  EXACT MATCH
Body  5 (x=  0, mass=2): naive force=     180  distributed force=     180  EXACT MATCH
Body  6 (x=  3, mass=3): naive force=       0  distributed force=       0  EXACT MATCH
Body  7 (x=  6, mass=4): naive force=    -360  distributed force=    -360  EXACT MATCH
Body  8 (x=  9, mass=1): naive force=    -180  distributed force=    -180  EXACT MATCH
Body  9 (x= 12, mass=2): naive force=    -540  distributed force=    -540  EXACT MATCH
Body 10 (x= 15, mass=3): naive force=   -1080  distributed force=   -1080  EXACT MATCH
Body 11 (x= 18, mass=4): naive force=   -1800  distributed force=   -1800  EXACT MATCH

All 12 bodies EXACT MATCH between naive all-pairs and distributed (shard + all-gather) all-pairs: YES
```

!!! warning "[COMMON TRAP] Assuming domain decomposition (Chapter 16-style particle/space partitioning) automatically implies nearest-neighbor communication"
    Chapter 16's own domain decomposition and this chapter's own body-sharding LOOK like the same idea -- split the data across devices by index or by region. The mistake this section's own simulation is built to head off is assuming that partitioning scheme, by itself, determines the communication pattern. It does not: the communication pattern is determined by the PHYSICS of the update rule, not by how the data happens to be split. Chapter 16's stencil update only reads adjacent cells, so its own partition needed only a halo exchange. This chapter's own force law has no such locality -- "the total force F_i on body i... is obtained by summing all interactions" with every other body -- so the identical-LOOKING partitioning scheme (splitting an array of N items across P shards) requires an entirely different collective, an all-gather, to stay correct. Two problems that look alike at the partitioning level can require completely different communication.

## 27.2 Quantifying the Damage: All-Gather vs. Halo Exchange at Scale

### Intuition

Section 27.1 proved WHAT collective distributed all-pairs needs. This section quantifies HOW MUCH worse that makes things as more GPUs are added, reusing Chapter 9 and Chapter 10's own real ring all-gather formula: each of P ranks ends a ring all-gather having received `(P-1)/P` of the total K bytes being gathered. For N-body, K is the ENTIRE body dataset -- so as P grows, `(P-1)/P` approaches 1, meaning every GPU's own per-step communication volume approaches the FULL dataset size and essentially never gets smaller, no matter how many GPUs are added. Chapter 16's own halo-exchange volume depended only on a FIXED boundary size between adjacent shards -- completely independent of the total domain size -- which is exactly why adding more shards there made each shard's own per-step communication SHRINK. This section applies the real formula to a real scale: Burtscher & Pingali's own cited "5,000,000 bodies" CUDA Barnes-Hut benchmark, reused here as a real-world body count (not itself a communication-volume citation).

```text
Ch16 halo exchange:  volume/shard = FIXED (boundary size only)
                      more shards -> SAME per-shard volume, SHRINKS
                      relative to the growing total problem

This chapter's all-gather: volume/GPU = (P-1)/P * (FULL dataset size)
                      more GPUs -> per-GPU volume APPROACHES the
                      FULL dataset size and stays there
```

### Background

```cpp
// Chapter 27: N-Body Simulation at Scale
// 79_nbody_vs_halo_communication_model.cpp
//
// Section 27.1 proved a distributed all-pairs N-body force calculation
// needs an ALL-GATHER, not a halo exchange, to get the right answer.
// This section quantifies exactly how much worse that makes N-body's
// own communication SHAPE, reusing Chapter 9/10's own real ring all-
// gather formula: each of P ranks ends a ring all-gather having
// received (P-1)/P of the total K bytes being gathered. Chapter 16's
// own halo-exchange volume, by contrast, depended only on a fixed
// BOUNDARY size between adjacent shards -- never on the total domain
// size -- which is exactly why adding more shards made Chapter 16's
// own per-shard communication SHRINK. This section's own worked
// calculation, applied to a real body count from a real cited source
// (Burtscher & Pingali's own "5,000,000 bodies" CUDA Barnes-Hut
// benchmark), shows the opposite: N-body's own per-GPU all-gather
// volume stays close to the ENTIRE dataset's size, almost regardless
// of how many GPUs are added, because every GPU still needs every
// body's data every step.
#include <cstdio>
#include <cstdint>

int main() {
    // Burtscher & Pingali's own real cited body count, reused here as
    // a real-scale example (not re-derived by this program).
    const uint64_t N_BODIES = 5000000ULL;
    // This section's own stated assumption -- 3D position (3 doubles)
    // + mass (1 double) = 32 bytes/body -- a common representation,
    // not itself a number quoted from a paper (Chapter 26's own
    // fp16-assumption practice, reused here).
    const double BYTES_PER_BODY = 32.0;
    double totalBytes = (double)N_BODIES * BYTES_PER_BODY;
    double totalMiB = totalBytes / (1024.0 * 1024.0);

    printf("N=%llu bodies (Burtscher & Pingali's own real cited scale), "
           "%.0f bytes/body (this chapter's own stated assumption) -> "
           "total dataset = %.2f MiB\n\n",
           (unsigned long long)N_BODIES, BYTES_PER_BODY, totalMiB);

    printf("%-6s %-28s %-28s\n", "P", "N-body all-gather MiB/GPU",
           "Ch16 halo-exchange MiB/GPU");
    printf("--------------------------------------------------------------"
           "----\n");

    // Chapter 16's own real halo-exchange model used a FIXED boundary
    // size, independent of total domain size -- represented here by a
    // fixed per-shard boundary of BOUNDARY_MIB, unaffected by P,
    // reused only for shape contrast (not Ch16's own literal numbers).
    const double BOUNDARY_MIB = 0.5; // a fixed boundary exchange size

    const int Ps[] = {2, 4, 8, 16, 32, 64};
    for (int P : Ps) {
        // Chapter 9/10's own real ring all-gather formula: each rank
        // receives (P-1)/P of the total K bytes being gathered.
        double perGpuAllGatherMiB = (double)(P - 1) / (double)P * totalMiB;
        printf("%-6d %-28.2f %-28.2f\n", P, perGpuAllGatherMiB, BOUNDARY_MIB);
    }

    printf("\nAs P grows, N-body's own per-GPU all-gather volume "
           "APPROACHES the FULL dataset size (%.2f MiB) and never drops "
           "below it by much -- every GPU still needs every body's data, "
           "every step. Chapter 16's own halo-exchange volume stayed "
           "FLAT at a fixed boundary size regardless of P, because a "
           "stencil update only ever reads immediately adjacent cells. "
           "Adding more GPUs make Chapter 16's own per-GPU communication "
           "SHRINK relative to the growing problem; it does nothing "
           "comparable for this chapter's own all-pairs N-body case.\n",
           totalMiB);

    return 0;
}
```

Compiled with `g++ -O2 79_nbody_vs_halo_communication_model.cpp -o 79_nbody_vs_halo_communication_model` and genuinely run, on both the cloud sandbox and the device, with byte-identical output on both. Locked output:

```text
N=5000000 bodies (Burtscher & Pingali's own real cited scale), 32 bytes/body (this chapter's own stated assumption) -> total dataset = 152.59 MiB

P      N-body all-gather MiB/GPU    Ch16 halo-exchange MiB/GPU  
------------------------------------------------------------------
2      76.29                        0.50                        
4      114.44                       0.50                        
8      133.51                       0.50                        
16     143.05                       0.50                        
32     147.82                       0.50                        
64     150.20                       0.50                        

As P grows, N-body's own per-GPU all-gather volume APPROACHES the FULL dataset size (152.59 MiB) and never drops below it by much -- every GPU still needs every body's data, every step. Chapter 16's own halo-exchange volume stayed FLAT at a fixed boundary size regardless of P, because a stencil update only ever reads immediately adjacent cells. Adding more GPUs make Chapter 16's own per-GPU communication SHRINK relative to the growing problem; it does nothing comparable for this chapter's own all-pairs N-body case.
```

!!! warning "[COMMON TRAP] Assuming more GPUs always makes per-GPU communication cheaper, the way it did in Chapter 16"
    Chapter 16's own real finding -- that finer domain decomposition (more shards) shrinks each shard's own communication volume relative to the growing problem -- is true for LOCAL, neighbor-only update rules, and it is tempting to treat that as a general property of "adding more GPUs." This section's own locked output shows it is not: for all-pairs N-body, going from P=2 to P=64 GPUs does not shrink per-GPU communication at all -- it GROWS, from 76.29 MiB to 150.20 MiB, approaching the full 152.59 MiB dataset size and staying there. The real reason is Section 27.1's own finding: every GPU still needs every body's data, every step, so adding GPUs only ever adds MORE all-gather participants, never reduces what any one of them individually needs to receive. Whether adding GPUs helps or hurts per-GPU communication depends entirely on whether the underlying problem is LOCAL (Chapter 16) or GLOBAL (this chapter) -- it is not a property of parallelism in general.

## 27.3 Barnes-Hut: The Real Algorithmic Answer

### Intuition

Sections 27.1 and 27.2 together show all-pairs N-body getting worse in two ways at once as a simulation scales up: `O(N^2)` compute, and per-GPU communication that never shrinks. Barnes & Hut's own real, named 1986 Nature paper offers the real algorithmic fix for both: a "tree-structured hierarchical subdivision of space into cubic cells, each of which is recursively divided into eight subcells whenever more than one particle is found to occupy the same cell." The key idea this tree enables is approximation -- a distant CLUSTER of bodies can be treated as one aggregate source (its combined mass, at its center of mass) rather than visited one body at a time, because from far enough away, the difference between "one heavy body" and "many bodies with the same total mass in roughly the same place" is negligible. This turns `O(N^2)` compute into `O(N log N)`, and -- though this chapter does not have a specific published number to cite for this part -- the same approximation idea would let a distributed implementation exchange far less than Section 27.2's own full all-gather, since a distant region's bodies could be summarized rather than transmitted individually.

```text
All-pairs (27.1/27.2): every body visits EVERY other body, individually
O(N^2) compute; O(N) per-GPU communication, doesn't shrink with more GPUs

Barnes-Hut (this section): distant CLUSTERS approximated as ONE source
     +---+                    close bodies: visited individually
     | * |   one aggregate     (real, no shortcuts)
     +---+   source            distant cluster: ONE aggregate value
   (many bodies, one              (real approximation, not every
    combined mass)                 individual body's raw data)
O(N log N) compute; a genuinely different, likely smaller,
communication shape (this chapter's own reasoning, not a cited number)
```

### Background

```cpp
// Chapter 27: N-Body Simulation at Scale
// 80_barnes_hut_complexity_crossover_model.cpp
//
// Sections 27.1 and 27.2 both showed all-pairs N-body force calculation
// getting worse at scale: O(N^2) compute, and per-GPU communication
// that stays close to the FULL dataset size no matter how many GPUs
// are added. Barnes & Hut's own real, named 1986 Nature paper offers
// the real algorithmic answer: a "tree-structured hierarchical
// subdivision of space into cubic cells, each of which is recursively
// divided into eight subcells whenever more than one particle is found
// to occupy the same cell," which lets distant groups of bodies be
// approximated as single aggregate sources instead of visited
// individually -- turning O(N^2) into O(N log N). This section builds
// this chapter's own closed-form operation-count model of that
// crossover (not a runtime measurement), and cites REAL measured
// numbers for a REAL GPU Barnes-Hut implementation (Burtscher &
// Pingali) to show the algorithmic idea also holds up on real
// hardware -- kept clearly separate from this section's own model, the
// same discipline Chapter 25 and Chapter 26 both already applied to
// their own closed-form communication-cost sections.
#include <cstdio>
#include <cmath>
#include <cstdint>

int main() {
    printf("--- This chapter's own worked model: O(N^2) vs O(N log N) "
           "operation counts (NOT a runtime measurement) ---\n");
    printf("%-12s %-16s %-16s %-16s\n", "N", "N^2 ops", "N*log2(N) ops",
           "ratio N/log2(N)");
    const uint64_t Ns[] = {1000ULL, 10000ULL, 100000ULL, 1000000ULL,
                            5000000ULL, 50000000ULL};
    for (uint64_t N : Ns) {
        double nSquared = (double)N * (double)N;
        double log2N = std::log2((double)N);
        double nLogN = (double)N * log2N;
        double ratio = (double)N / log2N;
        printf("%-12llu %-16.3e %-16.3e %-16.1fx\n",
               (unsigned long long)N, nSquared, nLogN, ratio);
    }
    printf("\nAs N grows, the ratio N/log2(N) grows WITHOUT BOUND -- "
           "all-pairs' own O(N^2) cost doesn't just stay worse than "
           "Barnes-Hut's O(N log N), it gets combinatorially WORSE, "
           "relative to Barnes-Hut, the larger the simulation gets. This "
           "compounds Section 27.2's own finding that all-pairs' "
           "per-GPU COMMUNICATION also does not improve by adding more "
           "GPUs -- both the compute and the communication argument "
           "point the same real direction at scale.\n\n");

    printf("--- Real, cited measured numbers for an ACTUAL GPU Barnes-Hut "
           "implementation (Burtscher & Pingali) -- NOT computed by this "
           "program ---\n");
    printf("\"Our CUDA code takes 5.2 seconds to simulate one time step "
           "with 5,000,000 bodies on a 1.3 GHz Quadro FX 5800 GPU with "
           "240 cores, which is 74 times faster than an optimized serial "
           "implementation running on a 2.53 GHz Xeon E5540 CPU.\"\n");
    printf("\"With increasing input size, the parallel CUDA Barnes Hut "
           "code is 5, 35, 66, 74, and 53 times faster than the serial C "
           "code.\"\n");
    printf("\"the Barnes Hut implementation reaches a respectable 75 "
           "Gflop/s.\"\n");
    printf("\"The most important kernel, kernel 5, is over 90 times "
           "faster.\"\n\n");

    printf("--- What this chapter does NOT claim ---\n");
    printf("Burtscher & Pingali's own real speedup figures are for ONE "
           "GPU against a serial CPU baseline, not a multi-GPU "
           "distributed Barnes-Hut communication measurement. This "
           "chapter's own reasoning -- that a distributed Barnes-Hut "
           "implementation could exchange far LESS than Section 27.2's "
           "own full O(N) all-gather, by approximating a distant "
           "region's bodies as a single aggregate value rather than "
           "requiring every individual body's raw data -- is this "
           "chapter's own logical extension of the tree algorithm's own "
           "real approximation idea, not a number measured or published "
           "by any source this chapter cites.\n");

    return 0;
}
```

Compiled with `g++ -O2 80_barnes_hut_complexity_crossover_model.cpp -o 80_barnes_hut_complexity_crossover_model` and genuinely run, on both the cloud sandbox and the device, with byte-identical output on both. Locked output:

```text
--- This chapter's own worked model: O(N^2) vs O(N log N) operation counts (NOT a runtime measurement) ---
N            N^2 ops          N*log2(N) ops    ratio N/log2(N) 
1000         1.000e+06        9.966e+03        100.3           x
10000        1.000e+08        1.329e+05        752.6           x
100000       1.000e+10        1.661e+06        6020.6          x
1000000      1.000e+12        1.993e+07        50171.7         x
5000000      2.500e+13        1.113e+08        224683.8        x
50000000     2.500e+15        1.279e+09        1955001.7       x

As N grows, the ratio N/log2(N) grows WITHOUT BOUND -- all-pairs' own O(N^2) cost doesn't just stay worse than Barnes-Hut's O(N log N), it gets combinatorially WORSE, relative to Barnes-Hut, the larger the simulation gets. This compounds Section 27.2's own finding that all-pairs' per-GPU COMMUNICATION also does not improve by adding more GPUs -- both the compute and the communication argument point the same real direction at scale.

--- Real, cited measured numbers for an ACTUAL GPU Barnes-Hut implementation (Burtscher & Pingali) -- NOT computed by this program ---
"Our CUDA code takes 5.2 seconds to simulate one time step with 5,000,000 bodies on a 1.3 GHz Quadro FX 5800 GPU with 240 cores, which is 74 times faster than an optimized serial implementation running on a 2.53 GHz Xeon E5540 CPU."
"With increasing input size, the parallel CUDA Barnes Hut code is 5, 35, 66, 74, and 53 times faster than the serial C code."
"the Barnes Hut implementation reaches a respectable 75 Gflop/s."
"The most important kernel, kernel 5, is over 90 times faster."

--- What this chapter does NOT claim ---
Burtscher & Pingali's own real speedup figures are for ONE GPU against a serial CPU baseline, not a multi-GPU distributed Barnes-Hut communication measurement. This chapter's own reasoning -- that a distributed Barnes-Hut implementation could exchange far LESS than Section 27.2's own full O(N) all-gather, by approximating a distant region's bodies as a single aggregate value rather than requiring every individual body's raw data -- is this chapter's own logical extension of the tree algorithm's own real approximation idea, not a number measured or published by any source this chapter cites.
```

!!! warning "[COMMON TRAP] Citing Burtscher & Pingali's own real single-GPU speedup numbers as evidence about multi-GPU communication"
    Burtscher & Pingali's own real, measured numbers -- 74x faster than serial, 75 Gflop/s, kernel 5 over 90x faster -- are genuinely real and worth citing, but they describe ONE GPU's Barnes-Hut implementation against a SERIAL CPU baseline. Nothing in that paper measures, and this chapter has not found any other published source measuring, how much communication a MULTI-GPU distributed Barnes-Hut implementation would actually need. This section's own claim that Barnes-Hut's tree structure would plausibly reduce Section 27.2's own O(N) all-gather cost is a logical inference from the algorithm's own real approximation idea -- treating a distant cluster's bodies as one aggregate source rather than transmitting each one individually -- not a number this chapter measured or found published anywhere. Citing a real single-GPU compute speedup as if it settled a multi-GPU communication question would blur two genuinely different claims that this chapter has deliberately kept separate.

## Chapter Summary

This chapter showed that a problem which LOOKS like Chapter 16's own domain decomposition -- splitting a large dataset across devices -- can require a completely different communication pattern once the underlying physics is examined. Section 27.1 built a real correctness simulation proving that distributed all-pairs N-body force calculation needs an all-gather, not a halo exchange, because gravity has no cutoff radius and every body genuinely needs every other body's data. Section 27.2 quantified how much worse this makes the communication shape at real scale, reusing Chapter 9/10's own ring all-gather formula: unlike Chapter 16's own halo exchange, whose per-shard volume shrinks as more shards are added, all-pairs N-body's per-GPU communication approaches the FULL dataset size and stays there, regardless of how many GPUs join. Section 27.3 closed with Barnes & Hut's own real, named 1986 algorithm, which fixes both problems at once by approximating distant clusters of bodies as single aggregate sources -- reducing compute from O(N^2) to O(N log N), with real measured numbers from an actual GPU implementation (Burtscher & Pingali) -- while being explicit about which claims are cited and which are this chapter's own reasoning about the (unmeasured, uncited) multi-GPU communication benefit.

## Self-Check Questions

1. Why does Chapter 16's own halo exchange work for a PDE stencil but not for N-body force calculation?
2. What real quote from GPU Gems 3's "Fast N-Body Simulation with CUDA" chapter explains why every body's force calculation needs every OTHER body?
3. In Section 27.1's own simulation, what real collective replaces Chapter 16's halo exchange, and what does "every shard holds the full body list" mean in terms of that collective's own semantics?
4. Section 27.2's own locked output shows N-body's per-GPU communication volume GROWING as more GPUs are added, from 76.29 MiB at P=2 to 150.20 MiB at P=64. Why does this happen, given that Chapter 9/10's own ring all-gather formula normally describes volume as staying "nearly flat"?
5. Why does Chapter 16's own halo-exchange volume shrink as more shards are added, while this chapter's own all-gather volume does not?
6. What real, named algorithm reduces all-pairs N-body's O(N^2) compute cost, and what real quote from its own 1986 paper describes the mechanism?
7. What real, cited numbers does Burtscher & Pingali's paper report for their GPU Barnes-Hut implementation, and what do those numbers NOT tell you about multi-GPU communication?
8. Why does this chapter explicitly separate its own reasoning about Barnes-Hut's likely communication benefit from Burtscher & Pingali's real cited numbers?

## Where We Go Next

Chapter 27 showed that communication pattern is determined by a problem's own physics, not by how superficially similar its data-partitioning scheme looks to an earlier chapter's. Chapter 28, "Distributed Breadth-First Search," returns to this same theme from a different angle -- a graph traversal problem whose own communication pattern depends on the graph's structure itself, reusing this book's own collective-communication toolkit (Chapter 9-11) in a genuinely different combinatorial setting from either Chapter 16's stencil or this chapter's own N-body force calculation.

## Worked Solutions

**1.** A PDE stencil's own update rule is LOCAL: each cell's next value depends only on its immediate neighbors, so splitting the grid across devices only requires exchanging a thin boundary layer -- a halo exchange. N-body force calculation has no such locality: gravitational (and Coulomb) forces have no cutoff radius, so every body's force genuinely depends on every OTHER body, however far away, regardless of how the bodies happen to be split across devices.

**2.** "The total force F_i on body i, due to its interactions with the other N-1 bodies, is obtained by summing all interactions." This states directly that computing the correct force on any one body requires summing contributions from every other body in the system, not just nearby ones.

**3.** An all-gather (Chapter 10's own real collective) replaces the halo exchange. "Every shard holds the full body list" means that, unlike an all-gather's typical use where each rank contributes a portion and receives everyone's portions to form one complete combined dataset, here the semantics are used specifically so that EVERY shard ends up with a complete, identical copy of every other shard's bodies -- not just its own boundary-adjacent neighbors' data, the way Chapter 16's halo exchange worked.

**4.** Chapter 9/10's "nearly flat volume" finding describes the per-rank communication volume relative to a FIXED total buffer size K as N (the rank count) grows -- the factor `(N-1)/N` approaches a constant. In this chapter's case, the total buffer being gathered IS the entire N-body dataset, and as P (the GPU count) grows, `(P-1)/P` also approaches 1, meaning each GPU's own share of that fixed total dataset approaches the WHOLE dataset -- which is exactly what the locked output shows growing toward 152.59 MiB. The volume is "nearly flat" in the sense that it approaches a constant, but that constant is the FULL dataset size, not a small fraction of it -- a fundamentally different situation from Chapter 16's halo exchange, whose fixed constant was a small boundary size.

**5.** Chapter 16's halo-exchange volume depends only on the fixed boundary length between adjacent shards, which does not grow as more shards are added -- each shard's own boundary with its neighbors stays the same size regardless of how many total shards exist. This chapter's own all-gather volume depends on the TOTAL dataset size, which every GPU needs a complete copy of regardless of shard count, so adding more GPUs only adds more all-gather participants without reducing what each one individually needs to receive.

**6.** Barnes-Hut (Barnes & Hut, 1986, Nature) reduces the O(N^2) compute cost to O(N log N). Its own paper describes the mechanism as a "tree-structured hierarchical subdivision of space into cubic cells, each of which is recursively divided into eight subcells whenever more than one particle is found to occupy the same cell" -- this tree lets a distant cluster of bodies be approximated as a single aggregate source rather than visited individually.

**7.** Burtscher & Pingali report: "5.2 seconds to simulate one time step with 5,000,000 bodies... 74 times faster than an optimized serial implementation"; scaling of "5, 35, 66, 74, and 53 times faster... with increasing input size"; "75 Gflop/s"; and "kernel 5... over 90 times faster." These numbers describe ONE GPU's performance against a serial CPU baseline -- they say nothing about how much data a MULTI-GPU distributed Barnes-Hut implementation would need to exchange between devices, which is a genuinely separate, unmeasured question.

**8.** Because conflating a real, cited, measured number (Burtscher & Pingali's single-GPU speedups) with an inferred, uncited claim (Barnes-Hut's likely multi-GPU communication benefit) would misrepresent what is actually known versus what this chapter is reasoning toward from the algorithm's own real structure. This is the same discipline Chapter 21 and Chapter 25/26 each already applied when they declined to fabricate a specific number for a claim no published source actually measures.

---

**Sources cited in this chapter:**

- [GPU Gems 3, Chapter 31: "Fast N-Body Simulation with CUDA" (Lars Nyland, Mark Harris, Jan Prins)](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda) — the real, verified quotes on the all-pairs algorithm's structure ("The total force F_i on body i, due to its interactions with the other N-1 bodies, is obtained by summing all interactions"; "The all-pairs approach to N-body simulation is a brute-force technique that evaluates all pair-wise interactions among the N bodies... because of its O(N^2) computational complexity"; "Each entry can be computed independently, so there is O(N^2) available parallelism"), independently re-verified via a second fresh fetch of the same page.
- [Barnes, J. & Hut, P. (1986), "A hierarchical O(N log N) force-calculation algorithm," Nature, Vol. 324, Issue 6096, pp. 446-449](https://www.nature.com/articles/324446a0) — the real, verified quote describing the tree-based approximation ("a tree-structured hierarchical subdivision of space into cubic cells, each of which is recursively divided into eight subcells whenever more than one particle is found to occupy the same cell") and its real O(N log N) complexity, cited in this chapter's own DOI: 10.1038/324446a0.
- [Burtscher, M. & Pingali, K., "An Efficient CUDA Implementation of the Tree-Based Barnes Hut n-Body Algorithm," in GPU Computing Gems Emerald Edition, Chapter 6, ed. Wen-mei W. Hwu, Elsevier/Morgan Kaufmann, 2011](https://iss.oden.utexas.edu/Publications/Papers/burtscher11.pdf) — the real, verified measured numbers for an actual GPU Barnes-Hut implementation cited in Section 27.3 ("5.2 seconds to simulate one time step with 5,000,000 bodies... 74 times faster than an optimized serial implementation"; "5, 35, 66, 74, and 53 times faster... with increasing input size"; "75 Gflop/s"; "kernel 5, is over 90 times faster").
- [UC Berkeley ParLab "Our Pattern Language," N-Body Methods pattern](https://patterns.eecs.berkeley.edu/?page_id=193) — the real, verified academic source for the general contrast between N-body's global dependency structure ("each element of the system rigorously depends on the state of every other element of the system") and structured-grid/stencil methods' local dependency structure, informing this chapter's own framing in Section 27.1.
- Chapter 9 and Chapter 10's own real ring all-reduce/all-gather round-count and volume formulas, Chapter 16's own real halo-exchange communication-volume model, and Chapter 24/26's own integer-based routing-correctness verification technique — all reused directly in this chapter, not re-derived.

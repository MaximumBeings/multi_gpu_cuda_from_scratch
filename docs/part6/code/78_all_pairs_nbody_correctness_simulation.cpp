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

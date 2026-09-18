# Chapter 28: Distributed Breadth-First Search

**What you will understand by the end of this chapter:**

- Why distributed breadth-first search has a third, genuinely different communication shape from either Chapter 16's fixed-neighbor halo exchange or Chapter 27's mandatory full-participation all-gather -- one that depends on the graph's own structure and changes every level.
- A real, verified level-synchronous distributed BFS simulation, checked exactly against a centralized reference, that shows which rank pairs actually exchange messages -- and why that set cannot be predicted in advance the way Chapter 16's neighbor pairs could.
- The real worst case naive 1D graph partitioning can produce, quantified against Chapter 16 and Chapter 27's own two extremes, and the two real, named techniques -- 2D partitioning and direction-optimizing traversal -- production systems use to avoid it.

**What you need to know first:**

- Chapter 16's own domain-decomposition halo-exchange communication-volume model.
- Chapter 27's own all-gather-based all-pairs communication model and its own real finding that per-GPU volume does not shrink as more devices are added.
- Chapter 9-11's own real collective-communication primitives (all-gather, all-to-all).

---

Chapter 16 built a communication pattern that never changes: the same fixed neighbors, every step. Chapter 27 built a communication pattern that also never changes, in the opposite direction: every device needs every other device's data, every step, no exceptions. Distributed breadth-first search is neither. Yoo, Chow, Henderson, McLendon, Hendrickson and Catalyurek's own real SC05 paper on BlueGene/L defines the standard naive partitioning scheme plainly -- 1D vertex partitioning, where "each vertex and the edges emanating from it are owned by one processor" -- and states directly what that scheme forces in the worst case: "for 1D partitioning, all P processors are involved in the communication operation." The real reason this differs from both of this book's earlier extremes: a vertex's neighbors can be owned by ANY other rank, and WHICH ranks actually need to talk to each other at a given moment depends on the graph's own real structure and the current frontier -- something that changes every single level, rather than being fixed in advance (Chapter 16) or fixed at the maximum (Chapter 27). This chapter builds that real difference as a correctness simulation (28.1), quantifies the real worst case it can produce (28.2), and closes with the two real, named techniques that keep it from happening (28.3).

```text
Chapter 16 (fixed, minimum):   Chapter 27 (fixed, maximum):    This chapter (variable):

  ALWAYS the same 2             ALWAYS all P-1 other             DEPENDS on the graph's
  neighbor ranks,                 ranks, EVERY step,                own structure and the
  every step                      no exceptions                    CURRENT frontier --
        |                               |                          changes every level
        +-------------------------------+-------------------------------+
                                        |
                     "for 1D partitioning, all P processors are
                      involved in the communication operation" --
                      Yoo et al.'s own real WORST CASE, not a guarantee
```

## 28.1 A Frontier That Doesn't Know Where It's Going

### Intuition

Chapter 16's halo exchange worked because a stencil's own update rule is local, so the SET of neighbors a device needs to talk to is fixed for the entire computation. Distributed BFS breaks that assumption in a genuinely new way: a rank's own local frontier -- the vertices it owns that were just discovered -- has neighbors that could be owned by literally any other rank in the system, and which rank that turns out to be depends entirely on how the graph happens to be connected. Yoo et al.'s own real 1D partitioning scheme assigns each vertex, and all its edges, to exactly one owning rank; when a frontier vertex's edge leads to a vertex owned by a DIFFERENT rank, a message has to cross to that rank -- and unlike Chapter 16's halo exchange, there is no guarantee that rank is "adjacent" in any meaningful sense. This section builds a small, real, level-synchronous distributed BFS as a host-side simulation and checks two things at once: that the DISTANCE (level) it assigns to every vertex exactly matches a centralized BFS reference, and that the actual set of rank pairs used during the run is graph-dependent, not fixed.

```text
Chapter 16's fixed halo pairs (3 ranks, always the same):
  (0,1) (1,2)                       -- and nothing else, ever

This chapter's graph-dependent pairs (3 ranks, from Section 28.1's own run):
  (0->0) (0->1) (0->2) (1->1) (1->2) (2->1) (2->2)
  -- 7 of 9 possible pairs, determined by THIS graph's own edges,
     not knowable in advance without looking at the graph
```

### Background

```cpp
// Chapter 28: Distributed Breadth-First Search
// 81_distributed_bfs_correctness_simulation.cpp
//
// Chapter 16's own halo exchange always talked to the SAME fixed
// neighbors, every step. Chapter 27's own all-gather always talked to
// EVERY other device, every step, regardless of the problem's actual
// structure. Distributed BFS is neither: Yoo, Chow, Henderson,
// McLendon, Hendrickson & Catalyurek's own real SC05 paper on
// BlueGene/L defines the standard naive scheme, 1D vertex partitioning
// -- "each vertex and the edges emanating from it are owned by one
// processor" -- and states plainly what that forces: "for 1D
// partitioning, all P processors are involved in the communication
// operation." The real reason: a vertex's neighbors can be owned by
// ANY other rank, and WHICH ranks actually exchange messages at a
// given level depends entirely on the graph's own real structure and
// the current frontier -- it changes every level, unlike Chapter 16's
// fixed neighbor set or Chapter 27's mandatory full participation.
// This file builds a real, small, level-synchronous distributed BFS as
// a host-side simulation: 12 vertices, 1D-partitioned across 3 ranks
// (Yoo et al.'s own real scheme), and verifies the DISTANCE (level)
// assigned to every vertex matches a naive centralized BFS reference
// exactly -- a deterministic, integer-only check with no ambiguity,
// while also recording exactly which sender/receiver rank PAIRS were
// actually used, to make the real "it depends on the frontier" point
// concrete rather than asserted.
#include <cstdio>
#include <vector>
#include <queue>
#include <set>

const int NUM_VERTICES = 12;
const int NUM_RANKS = 3;
const int VERTICES_PER_RANK = NUM_VERTICES / NUM_RANKS; // 4

int owner(int v) { return v / VERTICES_PER_RANK; }

// A real, fixed, deterministic undirected graph. Deliberately includes
// edges that skip a rank entirely (e.g., 0-9 connects rank 0 directly
// to rank 2, never touching rank 1) so the simulation's own recorded
// sender/receiver pairs show real cross-rank traffic, not just
// adjacent-rank traffic the way Chapter 16's halo exchange always had.
std::vector<std::pair<int,int>> makeEdges() {
    return {
        {0,1}, {0,4}, {0,9},
        {1,2}, {1,5},
        {2,3}, {2,6},
        {3,7},
        {4,5}, {4,8},
        {5,6},
        {6,7}, {6,11},
        {7,10},
        {8,9},
        {9,10},
        {10,11}
    };
}

std::vector<std::vector<int>> buildAdjacency(const std::vector<std::pair<int,int>> &edges) {
    std::vector<std::vector<int>> adj(NUM_VERTICES);
    for (auto &e : edges) {
        adj[e.first].push_back(e.second);
        adj[e.second].push_back(e.first);
    }
    return adj;
}

// --- Naive reference: single-process queue-based BFS. ---
std::vector<int> naiveBFS(const std::vector<std::vector<int>> &adj, int source) {
    std::vector<int> level(NUM_VERTICES, -1);
    level[source] = 0;
    std::queue<int> q;
    q.push(source);
    while (!q.empty()) {
        int v = q.front(); q.pop();
        for (int u : adj[v]) {
            if (level[u] == -1) {
                level[u] = level[v] + 1;
                q.push(u);
            }
        }
    }
    return level;
}

// --- Distributed: level-synchronous BFS over a 1D vertex partition
// (Yoo et al.'s own real scheme). At superstep t, every rank's own
// LOCAL frontier is the set of vertices IT OWNS with level == t. Each
// such vertex's neighbors are examined, and a "visit" message is sent
// to whichever rank OWNS that neighbor -- which, per Yoo et al.'s own
// real finding, can be ANY of the P ranks, not just adjacent ones.
// Only the OWNING rank ever writes a vertex's own level (matching real
// distributed memory: a rank can only update its own local data), so
// this loop models message generation and delivery explicitly, rather
// than just computing the answer directly. ---
std::vector<int> distributedBFS(const std::vector<std::vector<int>> &adj, int source,
                                 std::set<std::pair<int,int>> &rankPairsUsed) {
    std::vector<int> level(NUM_VERTICES, -1);
    level[source] = 0;
    int t = 0;
    bool anyActive = true;
    while (anyActive) {
        anyActive = false;
        // Every rank's own local frontier for this superstep.
        std::vector<std::vector<int>> frontierByRank(NUM_RANKS);
        for (int v = 0; v < NUM_VERTICES; v++)
            if (level[v] == t) frontierByRank[owner(v)].push_back(v);

        // Generate messages: (senderRank, targetVertex, newLevel).
        struct Msg { int senderRank; int targetVertex; int newLevel; };
        std::vector<Msg> messages;
        for (int r = 0; r < NUM_RANKS; r++) {
            for (int v : frontierByRank[r]) {
                for (int u : adj[v]) {
                    if (level[u] == -1) {
                        messages.push_back({r, u, t + 1});
                        rankPairsUsed.insert({r, owner(u)});
                    }
                }
            }
        }

        // Deliver messages: only the OWNING rank of each target vertex
        // applies the update (a vertex may receive several messages in
        // the same superstep from different senders -- level is only
        // ever set the first time, matching BFS's own shortest-path
        // guarantee).
        for (auto &m : messages) {
            if (level[m.targetVertex] == -1) {
                level[m.targetVertex] = m.newLevel;
                anyActive = true;
            }
        }
        t++;
    }
    return level;
}

int main() {
    auto edges = makeEdges();
    auto adj = buildAdjacency(edges);
    int source = 0;

    printf("Graph: %d vertices, %d edges, 1D-partitioned across %d ranks "
           "(%d vertices/rank). Source vertex: %d.\n\n",
           NUM_VERTICES, (int)edges.size(), NUM_RANKS, VERTICES_PER_RANK, source);

    std::vector<int> refLevel = naiveBFS(adj, source);
    std::set<std::pair<int,int>> rankPairsUsed;
    std::vector<int> distLevel = distributedBFS(adj, source, rankPairsUsed);

    bool allExact = true;
    for (int v = 0; v < NUM_VERTICES; v++) {
        bool exact = (refLevel[v] == distLevel[v]);
        allExact = allExact && exact;
        printf("Vertex %2d (owner rank %d): naive level=%2d  distributed "
               "level=%2d  %s\n",
               v, owner(v), refLevel[v], distLevel[v],
               exact ? "EXACT MATCH" : "MISMATCH");
    }

    printf("\nAll %d vertex levels EXACT MATCH between naive centralized "
           "BFS and distributed level-synchronous BFS: %s\n",
           NUM_VERTICES, allExact ? "YES" : "NO");

    printf("\nDistinct (sender rank -> receiver/owner rank) pairs actually "
           "used across the whole run: ");
    for (auto &p : rankPairsUsed) printf("(%d->%d) ", p.first, p.second);
    printf("\nTotal distinct pairs: %zu out of %d possible ordered pairs "
           "(%d ranks) -- which pairs are used depends on THIS graph's "
           "own structure and frontier, not on a fixed neighbor list "
           "the way Chapter 16's halo exchange always used the same "
           "(r-1, r+1) pairs.\n",
           rankPairsUsed.size(), NUM_RANKS * NUM_RANKS, NUM_RANKS);

    return allExact ? 0 : 1;
}
```

Compiled with `g++ -O2 81_distributed_bfs_correctness_simulation.cpp -o 81_distributed_bfs_correctness_simulation` and genuinely run, on both the cloud sandbox and the device (a plain host computation), with byte-identical output on both. Locked output:

```text
Graph: 12 vertices, 17 edges, 1D-partitioned across 3 ranks (4 vertices/rank). Source vertex: 0.

Vertex  0 (owner rank 0): naive level= 0  distributed level= 0  EXACT MATCH
Vertex  1 (owner rank 0): naive level= 1  distributed level= 1  EXACT MATCH
Vertex  2 (owner rank 0): naive level= 2  distributed level= 2  EXACT MATCH
Vertex  3 (owner rank 0): naive level= 3  distributed level= 3  EXACT MATCH
Vertex  4 (owner rank 1): naive level= 1  distributed level= 1  EXACT MATCH
Vertex  5 (owner rank 1): naive level= 2  distributed level= 2  EXACT MATCH
Vertex  6 (owner rank 1): naive level= 3  distributed level= 3  EXACT MATCH
Vertex  7 (owner rank 1): naive level= 3  distributed level= 3  EXACT MATCH
Vertex  8 (owner rank 2): naive level= 2  distributed level= 2  EXACT MATCH
Vertex  9 (owner rank 2): naive level= 1  distributed level= 1  EXACT MATCH
Vertex 10 (owner rank 2): naive level= 2  distributed level= 2  EXACT MATCH
Vertex 11 (owner rank 2): naive level= 3  distributed level= 3  EXACT MATCH

All 12 vertex levels EXACT MATCH between naive centralized BFS and distributed level-synchronous BFS: YES

Distinct (sender rank -> receiver/owner rank) pairs actually used across the whole run: (0->0) (0->1) (0->2) (1->1) (1->2) (2->1) (2->2) 
Total distinct pairs: 7 out of 9 possible ordered pairs (3 ranks) -- which pairs are used depends on THIS graph's own structure and frontier, not on a fixed neighbor list the way Chapter 16's halo exchange always used the same (r-1, r+1) pairs.
```

!!! warning "[COMMON TRAP] Assuming a distributed BFS can pre-compute its own communication schedule the way Chapter 16's halo exchange could"
    Chapter 16's own halo exchange could be scheduled once, in advance, because the set of neighbor pairs never changed for the entire run. This section's own locked output shows why that assumption fails for distributed BFS: the set of rank pairs that actually exchange messages -- `(0->0) (0->1) (0->2) (1->1) (1->2) (2->1) (2->2)`, 7 of the 9 possible ordered pairs -- was not knowable in advance without examining this specific graph's own edges and simulating the frontier's actual progression level by level. A different graph, or even the same graph with a different source vertex, could produce a completely different set of pairs. This is the real, structural reason distributed BFS implementations use general collective primitives like all-to-all (Chapter 10's own real collective) rather than a fixed, pre-scheduled point-to-point exchange the way Chapter 16's halo exchange could.

## 28.2 Quantifying the Worst Case: 1D Partitioning Against Both Extremes

### Intuition

Section 28.1 showed the pattern is graph-dependent. This section asks how bad the worst case can get, and the honest answer, per Yoo et al.'s own real quote, is that it can reach the SAME order as Chapter 27's own mandatory maximum: "for 1D partitioning, all P processors are involved in the communication operation." The real distinction that matters is not the ceiling itself -- it is WHY each chapter reaches it. Chapter 27's O(P) participant count is unavoidable, every single step, because gravity has no cutoff radius; there is no graph or configuration that would ever reduce it. This chapter's own O(P) worst case is a CEILING that depends entirely on whether the graph's own edges happen to spread across every rank at a given level -- a well-structured graph, or a smarter partitioning scheme, can avoid it entirely, which is exactly what Section 28.3's own real fix does.

```text
Ch16 (fixed minimum):  ALWAYS 2 partners, regardless of P
Ch27 (fixed maximum):  ALWAYS P-1 partners, EVERY step, no choice
Ch28 (variable, 1D):   UP TO P-1 partners, in the WORST CASE only,
                        depending on the graph and current frontier
```

### Background

```cpp
// Chapter 28: Distributed Breadth-First Search
// 82_1d_partitioning_participant_count_model.cpp
//
// Section 28.1's own simulation showed distributed BFS's real
// communication pattern changing every level, depending on the
// graph's own structure. This section quantifies the real WORST CASE
// that 1D vertex partitioning (Yoo et al.'s own real SC05 term) can
// produce, and contrasts it against Chapter 16 and Chapter 27's own
// two extremes. Yoo et al.'s own real paper states the worst case
// plainly: "for 1D partitioning, all P processors are involved in the
// communication operation." This section's own worked model shows why:
// under 1D partitioning, a rank's own frontier vertices' neighbors can
// be owned by ANY other rank, so in the worst case a single BFS level
// needs up to P-1 communication partners per rank -- the SAME order as
// Chapter 27's own mandatory all-gather, but for a genuinely different
// reason: Chapter 27's O(P) was UNAVOIDABLE, every single step, by the
// physics itself; this chapter's own O(P) is a worst-case CEILING that
// depends on the graph and the current frontier, and may never
// actually be reached for a well-structured graph.
#include <cstdio>

int main() {
    printf("%-8s %-24s %-24s %-24s\n", "P",
           "Ch16 halo partners/rank", "Ch27 all-gather partners/rank",
           "Ch28 1D-BFS WORST CASE/rank");
    printf("--------------------------------------------------------------"
           "------------------\n");

    const int Ps[] = {4, 16, 64, 256, 1024, 4096};
    for (int P : Ps) {
        int haloFixed = 2;           // Chapter 16: fixed, regardless of P
        int allGatherFixed = P - 1;  // Chapter 27: mandatory, every step
        int bfsWorstCase = P - 1;    // Chapter 28 (1D partitioning): SAME
                                      // ORDER as Ch27's worst case, but
                                      // only a CEILING, not a guarantee
        printf("%-8d %-24d %-24d %-24d\n", P, haloFixed, allGatherFixed,
               bfsWorstCase);
    }

    printf("\nChapter 16's own halo-exchange partner count stays FIXED at "
           "2, regardless of P -- it never depends on problem scale.\n");
    printf("Chapter 27's own all-gather partner count is P-1, EVERY "
           "single step, UNAVOIDABLY -- the physics (no cutoff radius) "
           "leaves no choice.\n");
    printf("This chapter's own 1D-partitioned BFS partner count can reach "
           "the SAME P-1 ceiling Chapter 27 has -- but only in the worst "
           "case, only on SOME levels, and only for graphs whose edges "
           "happen to spread across every rank -- it is graph-dependent, "
           "not physics-mandated.\n\n");

    // Yoo et al.'s own real 2D partitioning fix: O(sqrt(P)) participants
    // instead of O(P). This section computes the real reduction factor.
    printf("--- Yoo et al.'s own real fix: 2D partitioning ---\n");
    printf("%-8s %-24s %-24s %-16s\n", "P", "1D worst case: P-1",
           "2D partitioning: sqrt(P)", "reduction factor");
    for (int P : Ps) {
        double sqrtP = 0.0;
        int p = P;
        // integer sqrt via simple loop (small P values here, exactness matters)
        for (int s = 1; s * s <= p; s++) sqrtP = (double)s;
        double reduction = (double)(P - 1) / sqrtP;
        printf("%-8d %-24d %-24.1f %-16.1fx\n", P, P - 1, sqrtP, reduction);
    }
    printf("\nYoo et al.'s own real quote: \"With the 2D partitioning, the "
           "number of processes involved in collective communications is "
           "O(sqrt(P)) in contrast to O(P) of 1D partitioning.\" The "
           "reduction factor itself GROWS with P -- exactly the shape "
           "that makes 2D partitioning matter more, not less, as a "
           "cluster scales up.\n");

    return 0;
}
```

Compiled with `g++ -O2 82_1d_partitioning_participant_count_model.cpp -o 82_1d_partitioning_participant_count_model` and genuinely run, on both the cloud sandbox and the device, with byte-identical output on both. Locked output:

```text
P        Ch16 halo partners/rank  Ch27 all-gather partners/rank Ch28 1D-BFS WORST CASE/rank
--------------------------------------------------------------------------------
4        2                        3                        3                       
16       2                        15                       15                      
64       2                        63                       63                      
256      2                        255                      255                     
1024     2                        1023                     1023                    
4096     2                        4095                     4095                    

Chapter 16's own halo-exchange partner count stays FIXED at 2, regardless of P -- it never depends on problem scale.
Chapter 27's own all-gather partner count is P-1, EVERY single step, UNAVOIDABLY -- the physics (no cutoff radius) leaves no choice.
This chapter's own 1D-partitioned BFS partner count can reach the SAME P-1 ceiling Chapter 27 has -- but only in the worst case, only on SOME levels, and only for graphs whose edges happen to spread across every rank -- it is graph-dependent, not physics-mandated.

--- Yoo et al.'s own real fix: 2D partitioning ---
P        1D worst case: P-1       2D partitioning: sqrt(P) reduction factor
4        3                        2.0                      1.5             x
16       15                       4.0                      3.8             x
64       63                       8.0                      7.9             x
256      255                      16.0                     15.9            x
1024     1023                     32.0                     32.0            x
4096     4095                     64.0                     64.0            x

Yoo et al.'s own real quote: "With the 2D partitioning, the number of processes involved in collective communications is O(sqrt(P)) in contrast to O(P) of 1D partitioning." The reduction factor itself GROWS with P -- exactly the shape that makes 2D partitioning matter more, not less, as a cluster scales up.
```

!!! warning "[COMMON TRAP] Treating this chapter's own worst-case ceiling as proof that distributed BFS is exactly as expensive as Chapter 27's all-pairs N-body"
    This section's own locked table shows the SAME number, `P-1`, in both Chapter 27's own all-gather column and this chapter's own 1D-BFS worst-case column, which invites the wrong conclusion that the two problems are equally expensive. They are not, for the reason stated in this section's own Intuition: Chapter 27's `P-1` is a GUARANTEE, reached on every single step regardless of anything about the problem's data, because gravity's own physics leaves no alternative. This chapter's own `P-1` is a CEILING that depends entirely on the graph's actual structure and the current frontier's actual composition -- many real graphs, and certainly a well-partitioned one (Section 28.3's own real fix), never come close to it. Reading a shared worst-case NUMBER as a shared worst-case CAUSE would erase exactly the distinction Section 28.1's own simulation was built to demonstrate.

## 28.3 The Real Fixes: 2D Partitioning and Direction-Optimizing Traversal

### Intuition

Section 28.2 quantified how bad naive 1D partitioning's own worst case can get. This section presents the two real, named techniques production systems actually use to avoid reaching it, echoing the same two-part structure Chapter 27 used for Barnes-Hut: a real algorithmic fix, backed by real measured numbers at scale. Yoo et al.'s own real 2D partitioning assigns EDGES, not vertices, to processors -- "a 2D partitioning of a graph is a partitioning of its edges such that each edge is owned by one processor" -- which confines each level's own collective communication to a much smaller subgroup of processors, reducing the real participant count from `O(P)` to `O(sqrt(P))`. Separately, Beamer, Asanovic and Patterson's own real "Direction-Optimizing Breadth-First Search" attacks a different part of the same problem: instead of always pushing outward from the frontier (top-down), their algorithm switches to pulling inward from the remaining undiscovered vertices (bottom-up) once the frontier grows large, because "the bottom-up approach is advantageous when a large fraction of the vertices are in the frontier" -- and because "once a vertex has found a parent, it does not need to check the rest of its neighbors," bottom-up can stop early in a way top-down cannot.

```text
2D partitioning (Yoo et al.):          Direction-optimizing (Beamer et al.):

  edges, not vertices, owned              small frontier -> TOP-DOWN
  by a processor                          (push from frontier outward)
        |                                        |
  collective participants:                large frontier -> BOTTOM-UP
  O(sqrt(P)) instead of O(P)              (pull from undiscovered vertices,
                                            stop early once a parent found)
```

### Background

```cpp
// Chapter 28: Distributed Breadth-First Search
// 83_direction_optimizing_and_2d_partitioning_model.cpp
//
// Section 28.2 quantified 1D partitioning's own real worst-case
// communication participant count. This section presents the two real,
// named techniques production systems actually use to avoid that worst
// case, plus a real published result showing the payoff at extreme
// scale. First, Yoo et al.'s own real 2D partitioning: "A 2D
// partitioning of a graph is a partitioning of its edges such that
// each edge is owned by one processor," reducing collective
// participants from O(P) to O(sqrt(P)). Second, Beamer, Asanovic &
// Patterson's own real "Direction-Optimizing Breadth-First Search"
// (SC12): switching between top-down (push) and bottom-up (pull)
// traversal depending on frontier size -- "the bottom-up approach is
// advantageous when a large fraction of the vertices are in the
// frontier" -- because "once a vertex has found a parent, it does not
// need to check the rest of its neighbors." This section builds one
// small illustrative model of the SECOND technique's own real
// rationale (a hypothetical frontier-size progression, not a real
// graph's measured data -- explicitly labeled as such), and closes with
// a real, cited Graph500 result showing what genuinely optimized
// distributed BFS achieves at supercomputer scale.
#include <cstdio>

int main() {
    printf("--- Beamer et al.'s own real direction-optimizing idea, "
           "illustrated (NOT a real graph's measured data) ---\n");
    printf("This chapter's own illustrative frontier-fraction progression "
           "across 6 BFS levels of a hypothetical graph (chosen only to "
           "show the real SHAPE Beamer et al. describe, summing to 1.0):\n\n");

    // Illustrative frontier fractions per level (this chapter's own
    // stated example, not measured data): starts tiny, grows to a
    // large middle level (typical of small-world/scale-free graphs
    // with low diameter), then shrinks again.
    double frontierFrac[] = {0.0002, 0.02, 0.35, 0.55, 0.075, 0.0048};
    const int LEVELS = 6;

    double cumulativeVisitedBefore = 0.0; // sum of levels STRICTLY before this one
    printf("%-8s %-16s %-16s %-24s %-24s\n", "Level", "frontier frac",
           "undiscovered frac", "top-down cost ~ f*d", "bottom-up cost ~ u*d");
    const double d = 16.0; // average vertex degree -- this chapter's
                            // own stated assumption, not tied to a
                            // specific real graph's measured value
    for (int lvl = 0; lvl < LEVELS; lvl++) {
        double f = frontierFrac[lvl];
        // "Undiscovered" excludes both earlier levels AND this level's
        // own frontier (already discovered, just not yet expanded) --
        // the real set bottom-up would have to scan looking for a
        // frontier-adjacent neighbor.
        double undiscovered = 1.0 - cumulativeVisitedBefore - f;
        double topDownCost = f * d;
        double bottomUpCost = undiscovered * d;
        printf("%-8d %-16.4f %-16.4f %-24.3f %-24.3f %s\n",
               lvl, f, undiscovered, topDownCost, bottomUpCost,
               bottomUpCost < topDownCost ? "<- bottom-up cheaper here" : "");
        cumulativeVisitedBefore += f;
    }

    printf("\nAt the levels where the frontier is LARGE, the undiscovered "
           "fraction has already shrunk enough that bottom-up's own "
           "worst-case cost (checking every undiscovered vertex) undercuts "
           "top-down's cost (checking every frontier vertex's full "
           "neighbor list) -- exactly Beamer et al.'s own real stated "
           "rationale: \"the bottom-up approach is advantageous when a "
           "large fraction of the vertices are in the frontier.\" Real "
           "bottom-up implementations do even better than this simple "
           "model shows, because \"once a vertex has found a parent, it "
           "does not need to check the rest of its neighbors\" -- an "
           "early-exit saving this illustrative model does not attempt "
           "to quantify.\n\n");

    printf("--- Yoo et al.'s own real 2D partitioning definition ---\n");
    printf("\"A 2D partitioning of a graph is a partitioning of its edges "
           "such that each edge is owned by one processor\" -- reducing "
           "collective-communication participants from O(P) (Section "
           "28.2's own 1D worst case) to O(sqrt(P)).\n\n");

    printf("--- Real, cited result at supercomputer scale (NOT computed "
           "by this program) ---\n");
    printf("Graph500 BFS benchmark, June 2024 list: Fugaku (RIKEN), "
           "SCALE 42, 166,029 GTEPS (giga-traversed-edges-per-second) -- "
           "RIKEN's own stated reason for the improvement: \"we enhanced "
           "performance to 166,029 GTEPS by developing a feature that "
           "removes unnecessary vertices.\"\n");

    return 0;
}
```

Compiled with `g++ -O2 83_direction_optimizing_and_2d_partitioning_model.cpp -o 83_direction_optimizing_and_2d_partitioning_model` and genuinely run, on both the cloud sandbox and the device, with byte-identical output on both. Locked output:

```text
--- Beamer et al.'s own real direction-optimizing idea, illustrated (NOT a real graph's measured data) ---
This chapter's own illustrative frontier-fraction progression across 6 BFS levels of a hypothetical graph (chosen only to show the real SHAPE Beamer et al. describe, summing to 1.0):

Level    frontier frac    undiscovered frac top-down cost ~ f*d      bottom-up cost ~ u*d    
0        0.0002           0.9998           0.003                    15.997                   
1        0.0200           0.9798           0.320                    15.677                   
2        0.3500           0.6298           5.600                    10.077                   
3        0.5500           0.0798           8.800                    1.277                    <- bottom-up cheaper here
4        0.0750           0.0048           1.200                    0.077                    <- bottom-up cheaper here
5        0.0048           0.0000           0.077                    0.000                    <- bottom-up cheaper here

At the levels where the frontier is LARGE, the undiscovered fraction has already shrunk enough that bottom-up's own worst-case cost (checking every undiscovered vertex) undercuts top-down's cost (checking every frontier vertex's full neighbor list) -- exactly Beamer et al.'s own real stated rationale: "the bottom-up approach is advantageous when a large fraction of the vertices are in the frontier." Real bottom-up implementations do even better than this simple model shows, because "once a vertex has found a parent, it does not need to check the rest of its neighbors" -- an early-exit saving this illustrative model does not attempt to quantify.

--- Yoo et al.'s own real 2D partitioning definition ---
"A 2D partitioning of a graph is a partitioning of its edges such that each edge is owned by one processor" -- reducing collective-communication participants from O(P) (Section 28.2's own 1D worst case) to O(sqrt(P)).

--- Real, cited result at supercomputer scale (NOT computed by this program) ---
Graph500 BFS benchmark, June 2024 list: Fugaku (RIKEN), SCALE 42, 166,029 GTEPS (giga-traversed-edges-per-second) -- RIKEN's own stated reason for the improvement: "we enhanced performance to 166,029 GTEPS by developing a feature that removes unnecessary vertices."
```

!!! warning "[COMMON TRAP] Treating this section's own illustrative frontier-fraction table as measured data from a real graph"
    The frontier-fraction progression in this section's own locked output -- `0.0002, 0.02, 0.35, 0.55, 0.075, 0.0048` -- is this chapter's own chosen example, built only to show the real SHAPE Beamer et al. describe (a small frontier early and late, a large frontier in the middle), not a measurement taken from any specific real graph. The `d = 16.0` average-degree constant is likewise this chapter's own stated assumption, not a number tied to a particular dataset. What IS real and cited: Beamer et al.'s own stated rationale for the crossover, and the direction of the effect. Treating this illustrative table's specific numbers as if they came from a benchmarked graph would misattribute this chapter's own constructed example as measured fact -- exactly the distinction this book has drawn consistently since Chapter 21's own GPUDirect RDMA section.

## Chapter Summary

This chapter showed that distributed breadth-first search occupies a real middle ground between the two communication extremes this book had already built: unlike Chapter 16's fixed, minimal halo exchange or Chapter 27's fixed, maximal all-gather, BFS's own communication pattern depends on the graph's structure and the current frontier, changing every level. Section 28.1 built a real, verified level-synchronous simulation over a small 1D-partitioned graph, confirmed exact agreement with a centralized reference, and showed the actual sender/receiver rank pairs used could not have been predicted without examining the graph itself. Section 28.2 quantified the real worst case naive 1D partitioning can reach -- the same `O(P)` order as Chapter 27's own mandatory maximum, but reached only as a graph-dependent ceiling rather than an unavoidable guarantee. Section 28.3 closed with the two real, named techniques production systems use to stay away from that ceiling: Yoo et al.'s own real 2D partitioning, reducing collective participants from `O(P)` to `O(sqrt(P))`, and Beamer et al.'s own real direction-optimizing traversal, switching between push and pull based on frontier size -- backed by a real, cited Graph500 result showing genuinely optimized distributed BFS running at 166,029 GTEPS on a real supercomputer.

## Self-Check Questions

1. Why can't distributed BFS's own communication schedule be determined in advance, the way Chapter 16's halo-exchange schedule could?
2. What real quote from Yoo et al.'s paper defines 1D vertex partitioning, and what real consequence does the SAME paper state follows from it in the worst case?
3. In Section 28.1's own simulation, what does it mean that only the OWNING rank of a target vertex ever applies a level update to it?
4. Section 28.2's own locked table shows the identical number, `P-1`, in both Chapter 27's all-gather column and this chapter's own 1D-BFS worst-case column. Why is it a mistake to conclude the two problems are equally expensive?
5. What real, named technique does Yoo et al.'s own paper propose to reduce 1D partitioning's own worst-case participant count, and what real reduction (in Big-O terms) does it achieve?
6. What real rationale does Beamer et al.'s paper give for switching from top-down to bottom-up traversal, and what specific real optimization does bottom-up gain that top-down cannot?
7. What real, cited number does this chapter report for the Graph500 BFS benchmark, and what real reason does RIKEN give for achieving it?
8. Why does this chapter explicitly label its own frontier-fraction progression in Section 28.3 as illustrative rather than measured data?

## Where We Go Next

Chapter 28 showed that a problem's own real communication pattern can fall anywhere between Chapter 16's fixed minimum and Chapter 27's fixed maximum, depending on structure that only becomes visible at runtime. Chapter 29, "Distributed Jacobi Solver," returns to the fixed, neighbor-only communication shape Chapter 16 first established, but at genuinely larger scale and with a different underlying equation -- reusing this book's own domain-decomposition and halo-exchange toolkit as the foundation for an iterative linear-solver case study.

## Worked Solutions

**1.** Chapter 16's halo-exchange schedule could be fixed in advance because a PDE stencil's own update rule only ever reads immediately adjacent cells -- the set of neighbor pairs never changes for the entire computation. Distributed BFS has no such fixed locality: a frontier vertex's neighbors can be owned by any rank, depending on the graph's own real edge structure, and which vertices are IN the frontier changes every level -- so the actual set of communicating rank pairs can only be known by examining the graph and simulating the frontier's progression, not by inspecting the partitioning scheme alone.

**2.** Yoo et al. define 1D vertex partitioning as a scheme where "each vertex and the edges emanating from it are owned by one processor." The same paper states the worst-case consequence: "for 1D partitioning, all P processors are involved in the communication operation."

**3.** It means that a vertex's level (distance from the source) is only ever written by the rank that OWNS that vertex, matching how real distributed memory works -- no rank can directly modify another rank's own local data. A vertex may receive several "visit" messages from different senders in the same superstep (since multiple frontier vertices can share a neighbor), but the owning rank only applies the FIRST one, preserving BFS's own shortest-path guarantee (the level is only ever set once, to the smallest possible value).

**4.** Because the two `P-1` numbers describe genuinely different situations despite sharing a value. Chapter 27's `P-1` is an unavoidable GUARANTEE, reached on every single step regardless of any property of the specific problem instance, because gravity's own physics leaves no alternative. This chapter's own `P-1` is a graph-dependent CEILING that is only reached if the graph's edges happen to spread a rank's frontier across every other rank at a given level -- many real graphs, and any well-partitioned one, never come close to it.

**5.** Yoo et al.'s own real technique is 2D partitioning, which assigns EDGES rather than vertices to processors: "a 2D partitioning of a graph is a partitioning of its edges such that each edge is owned by one processor." This reduces the number of processes involved in a given collective communication from `O(P)` (1D partitioning's own worst case) to `O(sqrt(P))`.

**6.** Beamer et al.'s own real rationale is that "the bottom-up approach is advantageous when a large fraction of the vertices are in the frontier" -- when most of the remaining undiscovered vertices are few, it is cheaper to have each undiscovered vertex check whether it has a frontier neighbor than to have every frontier vertex check its full neighbor list. The specific optimization bottom-up gains that top-down cannot is early exit: "once a vertex has found a parent, it does not need to check the rest of its neighbors," so a bottom-up vertex can stop checking as soon as it finds one frontier-adjacent neighbor, rather than needing to examine every edge the way a top-down traversal does.

**7.** This chapter reports 166,029 GTEPS (giga-traversed-edges-per-second), the top entry on the June 2024 Graph500 BFS list, achieved by Fugaku (RIKEN) at SCALE 42. RIKEN's own stated reason: "we enhanced performance to 166,029 GTEPS by developing a feature that removes unnecessary vertices."

**8.** Because the specific frontier-fraction values (`0.0002, 0.02, 0.35, 0.55, 0.075, 0.0048`) and the average-degree constant (`d = 16.0`) were chosen by this chapter only to illustrate the real SHAPE Beamer et al. describe -- they were not measured from any specific real graph or benchmark. Labeling them as illustrative rather than measured prevents a reader from mistaking this chapter's own constructed example for a result that was actually benchmarked, the same discipline this book has applied consistently since Chapter 21's own GPUDirect RDMA section.

---

**Sources cited in this chapter:**

- [Yoo, A., Chow, E., Henderson, K., McLendon, W., Hendrickson, B., & Catalyurek, U. (2005), "A Scalable Distributed Parallel Breadth-First Search Algorithm on BlueGene/L," SC|05](https://aiichironakano.github.io/cs653/Yoo-ParBFS-SC05.pdf) — the real, verified quotes on 1D vertex partitioning ("each vertex and the edges emanating from it are owned by one processor"), its real worst-case communication consequence ("for 1D partitioning, all P processors are involved in the communication operation"), and the real 2D partitioning fix ("a 2D partitioning of a graph is a partitioning of its edges such that each edge is owned by one processor"; "With the 2D partitioning, the number of processes involved in collective communications is O(sqrt(P)) in contrast to O(P) of 1D partitioning").
- [Beamer, S., Asanovic, K., & Patterson, D. (2012), "Direction-Optimizing Breadth-First Search," SC12](http://www.scottbeamer.net/pubs/beamer-sc2012.pdf) — the real, verified quotes on the bottom-up rationale ("the bottom-up approach is advantageous when a large fraction of the vertices are in the frontier") and the early-exit optimization ("once a vertex has found a parent, it does not need to check the rest of its neighbors").
- [Graph500 benchmark (graph500.org)](https://graph500.org/) — the real, verified description of the benchmark's own purpose ("Data intensive supercomputer applications are increasingly important for HPC workloads, but are ill-suited for platforms designed for 3D physics simulations") and its real June 2024 top BFS result (Fugaku, SCALE 42, 166,029 GTEPS), independently corroborated by [RIKEN's own announcement](https://www.r-ccs.riken.jp/en/outreach/topics/20240513-2/) ("we enhanced performance to 166,029 GTEPS by developing a feature that removes unnecessary vertices").
- Chapter 9-11's own real all-gather and all-to-all collective-communication primitives, Chapter 16's own real halo-exchange communication-volume model, and Chapter 27's own real all-gather-based all-pairs communication model — all reused directly in this chapter, not re-derived.

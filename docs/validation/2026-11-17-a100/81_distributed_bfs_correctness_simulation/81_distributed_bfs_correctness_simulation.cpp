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

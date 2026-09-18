// Chapter 19: Stragglers, Failures, and Fault-Tolerant Collectives
// 56_backup_workers_simulation.cpp
//
// Plain host C++ -- this chapter's real verification for Section
// 19.3. Sections 19.1-19.2 dealt with a peer that's fully DEAD.
// Chen et al.'s "Revisiting Distributed Synchronous SGD" (arXiv
// 1604.00981) studies a milder, far more common real failure mode
// instead: a peer that's merely SLOW -- a straggler -- on any given
// step, for reasons (a noisy neighbor VM, a transient network hiccup,
// a slow disk read) that mostly don't repeat next step. Their real
// fix needs no failure detector at all: launch N+b workers for N
// required results, and proceed the moment the FIRST N of them
// report, dropping whichever b (or fewer) stragglers haven't finished
// yet. This file simulates that rule directly on fixed, named
// timings (no randomness -- every number below is chosen to make one
// concrete point) and checks, across every scenario, the one
// invariant Chen et al.'s own technique promises: waiting for backups
// is NEVER slower than waiting for the fixed required set alone.
#include <cstdio>
#include <vector>
#include <algorithm>

// The baseline every earlier chapter's own collectives implicitly
// assumed: exactly N required workers, no backups. One slow (or
// effectively hung) worker among the N holds up the whole step --
// this is precisely Chapter 17's own barrier cost, paid again here
// per-step instead of once at a global sync.
double withoutBackupsWait(const std::vector<double>& requiredTimes) {
    double worst = 0.0;
    for (double t : requiredTimes) worst = std::max(worst, t);
    return worst;
}

// Chen et al.'s real technique: launch requiredTimes.size() + backup-
// Times.size() workers, and proceed once the FIRST requiredTimes.size()
// of ALL of them (required and backup, pooled together) have reported
// -- the n-th order statistic of the pooled finishing times. Whichever
// workers haven't finished yet (up to backupTimes.size() of them) are
// simply dropped for this step.
double withBackupsWait(const std::vector<double>& requiredTimes,
                        const std::vector<double>& backupTimes) {
    std::vector<double> pooled = requiredTimes;
    pooled.insert(pooled.end(), backupTimes.begin(), backupTimes.end());
    std::sort(pooled.begin(), pooled.end());
    size_t n = requiredTimes.size();
    return pooled[n - 1]; // n-th smallest, 0-indexed as [n-1]
}

struct Scenario {
    const char* name;
    std::vector<double> requiredTimes; // N=4 required workers' finish times
    std::vector<double> backupTimes;   // b=2 backup workers' finish times
    const char* explanation;
};

int main() {
    std::vector<Scenario> scenarios = {
        {
            "no true straggler",
            {10.0, 11.0, 9.0, 12.0},
            {10.5, 11.5},
            "Nothing is actually hung -- every one of the 6 workers "
            "finishes in ordinary time. Even here, backups still help: "
            "the 4th-fastest of all 6 (11.0) beats the slowest of the "
            "4 originally-required ones (12.0), purely from having two "
            "extra chances at the draw."
        },
        {
            "a required worker catastrophically straggles",
            {10.0, 11.0, 9.0, 1000000.0},
            {12.0, 13.0},
            "One of the 4 ORIGINALLY-REQUIRED workers is the real "
            "target case -- effectively hung (a straggler so slow it "
            "may as well be Section 19.1's dead peer). Without "
            "backups, the whole step is held hostage to it. With 2 "
            "backups in flight, the pool has 4 OTHER finishers well "
            "under the straggler's time, so the step proceeds without "
            "ever waiting on it at all."
        },
        {
            "slow backups, no downside",
            {10.0, 11.0, 9.0, 12.0},
            {500.0, 600.0},
            "The backups themselves are the slow ones this time, far "
            "slower than any required worker. That costs nothing: the "
            "first 4 finishers are still the 4 required workers "
            "(9,10,11,12), so the backups' own results are simply "
            "dropped, unused, exactly as designed."
        },
    };

    printf("Section 19.3: backup workers, N=4 required + b=2 backup, "
           "3 fixed scenarios.\n\n");

    bool allInvariantsHold = true;
    for (const Scenario& s : scenarios) {
        double without = withoutBackupsWait(s.requiredTimes);
        double with = withBackupsWait(s.requiredTimes, s.backupTimes);
        bool invariantHolds = (with <= without);
        allInvariantsHold = allInvariantsHold && invariantHolds;

        printf("Scenario: %s\n", s.name);
        printf("  required times: [%.1f, %.1f, %.1f, %.1f]   backup times: [%.1f, %.1f]\n",
               s.requiredTimes[0], s.requiredTimes[1], s.requiredTimes[2], s.requiredTimes[3],
               s.backupTimes[0], s.backupTimes[1]);
        printf("  without backups (max of the 4 required):        %.1f\n", without);
        printf("  with backups (4th-fastest of all 6, pooled):     %.1f\n", with);
        printf("  with <= without: %s\n", invariantHolds ? "PASS" : "FAIL");
        printf("  %s\n\n", s.explanation);
    }

    printf("Invariant across every scenario above -- waiting for backups "
           "is never slower than waiting for the fixed required set "
           "alone -- holds in all %zu/%zu cases: %s\n",
           scenarios.size(), scenarios.size(), allInvariantsHold ? "PASS" : "FAIL");

    if (allInvariantsHold) {
        printf("\nThis is exactly Chen et al.'s own real claim, checked "
               "here on fixed numbers rather than assumed: taking the "
               "n-th order statistic of a LARGER pool can only ever be "
               "less than or equal to the max of one FIXED subset of "
               "that same pool's own n members, for any numbers at all "
               "-- adding more candidates and picking the best n of them "
               "never makes the n-th-best candidate's time worse.\n");
    }

    return allInvariantsHold ? 0 : 1;
}

// Chapter 15: Pipeline Parallelism -- Splitting Layers Across Devices, and the Bubble That Costs You
// 42_pipeline_schedule.cpp
//
// Plain host C++ -- computing the actual GPipe-style pipeline
// schedule (which device processes which micro-batch at which time
// step) is pure arithmetic, needs no device, and lets this section
// derive Chapter 13's own unmeasured idle-time cost by direct
// counting rather than quoting a formula and trusting it. GPipe's
// own paper splits a batch into M micro-batches pipelined through K
// accelerators; this section builds that schedule for a small,
// hand-traceable K and M, counts every device's busy vs. idle time
// slots directly, and only THEN compares the counted result against
// GPipe's own cited closed-form bubble formula.
#include <cstdio>

// Standard pipeline schedule: micro-batch m enters device 0 at time
// step t=m, and reaches device d at time t=m+d (each stage takes one
// time unit). Device d is therefore busy processing micro-batch m at
// time step (m+d), and idle at every other time step within the
// pipeline's total span.
int microbatchAt(int device, int timeStep) {
    int m = timeStep - device;
    return (m >= 0) ? m : -1; // -1 means idle -- no valid micro-batch yet
}

int main() {
    const int K = 4; // devices (Chapter 13's own layer-partition world size)
    const int M = 6; // micro-batches

    const int totalTimeSteps = M + K - 1; // pipeline's total wall-clock span

    printf("Pipeline schedule, K=%d devices, M=%d micro-batches "
           "(totalTimeSteps = M+K-1 = %d):\n\n", K, M, totalTimeSteps);
    printf("device \\ t ");
    for (int t = 0; t < totalTimeSteps; ++t) printf("%3d", t);
    printf("\n");

    long busySlots = 0;
    for (int d = 0; d < K; ++d) {
        printf("dev %-6d ", d);
        for (int t = 0; t < totalTimeSteps; ++t) {
            int m = microbatchAt(d, t);
            bool busy = (m >= 0 && m < M);
            if (busy) { printf("%3d", m); busySlots++; }
            else printf("  .");
        }
        printf("\n");
    }

    const long totalSlots = (long)K * totalTimeSteps;
    const long idleSlots = totalSlots - busySlots;

    printf("\nCounted directly from the schedule above:\n");
    printf("  total device-time slots (K x totalTimeSteps): %ld\n", totalSlots);
    printf("  busy slots (every device's M real micro-batches):  %ld\n", busySlots);
    printf("  idle slots ('.' above):                             %ld\n", idleSlots);

    // Every device does exactly M units of real work (one per
    // micro-batch) regardless of K -- so busySlots must equal K*M
    // exactly, and idleSlots must equal K*(K-1) exactly: the fill and
    // drain time every device except the very first and very last
    // spends waiting for a micro-batch that isn't ready yet, or with
    // no more micro-batches left to process.
    bool busyMatches = (busySlots == (long)K * M);
    bool idleMatches = (idleSlots == (long)K * (K - 1));
    printf("\nSelf-check: busySlots == K*M (%ld): %s\n", (long)K * M,
           busyMatches ? "PASS" : "FAIL");
    printf("Self-check: idleSlots == K*(K-1) (%ld): %s\n", (long)K * (K - 1),
           idleMatches ? "PASS" : "FAIL");

    return (busyMatches && idleMatches) ? 0 : 1;
}

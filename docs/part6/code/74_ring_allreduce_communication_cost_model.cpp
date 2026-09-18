// Chapter 25: Distributed Training of a Neural Network: Data Parallelism
// and Ring All-Reduce
// 74_ring_allreduce_communication_cost_model.cpp
//
// Chapter 9 derived a real closed-form cost model for ring all-reduce:
// N-1 rounds in a scatter-reduce phase plus N-1 rounds in an all-gather
// phase, each round moving K/N of a K-byte buffer, for a total of
// 2*(N-1)/N*K bytes moved BY EACH RANK -- a volume that is almost
// INDEPENDENT of N once N is even moderately large, because
// 2*(N-1)/N approaches the constant 2 as N grows. This section applies
// that same formula, unchanged, to a real number: the fp32 gradient
// buffer of a real, widely-used model, ResNet-50, which has exactly
// 25,557,032 trainable parameters (torchvision's own resnet50 model
// registry). It contrasts ring all-reduce's near-flat cost against the
// real alternative every data-parallel framework had to reject: a naive
// gather-to-one-rank-then-broadcast reduction, whose busiest rank's
// traffic grows LINEARLY with N -- 2*(N-1)*K, not 2*(N-1)/N*K. This is
// this chapter's own worked calculation, not a number republished from
// a paper; the numbers that ARE republished from real papers -- Horovod's
// measured 88% (TCP) / >90% (RDMA) scaling efficiency at N=128, and
// PyTorch DDP's measured 38.0%/35.2% NCCL overlap speedup for
// ResNet/BERT -- are kept clearly separate below, as the empirical
// evidence for why this cost model's shape matters, not as a target
// this program's own numbers are being bent to match.
#include <cstdio>
#include <cstdint>

int main() {
    // ResNet-50: 25,557,032 trainable parameters (torchvision resnet50
    // model registry, IMAGENET1K_V1/V2), fp32 gradients -- one gradient
    // value per parameter, 4 bytes each.
    const uint64_t PARAMS = 25557032ULL;
    const uint64_t BYTES_PER_PARAM = 4ULL; // fp32
    const uint64_t K = PARAMS * BYTES_PER_PARAM; // total gradient buffer, bytes

    printf("ResNet-50 fp32 gradient buffer: K = %llu bytes (%.2f MiB)\n\n",
           (unsigned long long)K, (double)K / (1024.0 * 1024.0));

    printf("%-6s %-10s %-24s %-24s %-10s\n",
           "N", "rounds", "ring: 2(N-1)/N*K (MiB)",
           "naive: 2(N-1)*K (GiB)", "ratio");
    printf("---------------------------------------------------------------"
           "----------\n");

    const int Ns[] = {2, 4, 8, 16, 32, 64, 128, 256};
    for (int idx = 0; idx < 8; idx++) {
        int N = Ns[idx];
        int rounds = 2 * (N - 1); // Chapter 9's own round count: (N-1)
                                  // scatter-reduce rounds + (N-1) all-gather
                                  // rounds
        double ringBytes = 2.0 * (double)(N - 1) / (double)N * (double)K;
        double naiveBytes = 2.0 * (double)(N - 1) * (double)K;
        double ringMiB = ringBytes / (1024.0 * 1024.0);
        double naiveGiB = naiveBytes / (1024.0 * 1024.0 * 1024.0);
        double ratio = naiveBytes / ringBytes; // how much worse naive is
        printf("%-6d %-10d %-24.2f %-24.3f %-9.1fx\n",
               N, rounds, ringMiB, naiveGiB, ratio);
    }

    printf("\nAs N grows, 2(N-1)/N -> 2 (a CONSTANT): at N=256 each rank's "
           "ring traffic\nis only %.4fx the N=2 case, while the naive "
           "busiest-rank traffic at N=256\nis %dx the N=2 case -- this is "
           "the real, quantified reason every framework\ncited in this "
           "chapter (NCCL, Horovod, PyTorch DDP) settled on ring-style "
           "collectives\nfor multi-GPU gradient synchronization, not the "
           "naive gather/broadcast shape.\n",
           (2.0 * 255.0 / 256.0) / (2.0 * 1.0 / 2.0), 255);

    printf("\n--- What real measurements report for a ring-style collective at\n"
           "    this same scale (cited, not computed by this program) ---\n");
    printf("Horovod, TCP, 1->128 GPUs, Inception V3 and ResNet-101: "
           "measured 88%% scaling efficiency.\n");
    printf("Horovod, RDMA, same setup: measured >90%% scaling efficiency "
           "on both models.\n");
    printf("PyTorch DDP, NCCL, overlapping communication with backward "
           "computation: measured 38.0%% (ResNet) / 35.2%% (BERT) "
           "speedup over not overlapping.\n");
    printf("None of these three percentages are outputs of this program's "
           "own formula -- they are independently measured results this "
           "chapter cites to show the formula's real-world consequence.\n");

    return 0;
}

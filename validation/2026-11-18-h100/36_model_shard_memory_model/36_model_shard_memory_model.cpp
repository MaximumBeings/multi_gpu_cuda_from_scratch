// Chapter 13: Model Parallelism -- When One GPU Can't Hold the Weights
// 36_model_shard_memory_model.cpp
//
// Plain host C++ closed-form model, reusing Chapter 1's own already-cited
// numbers exactly (ZeRO's real 16-bytes-per-parameter mixed-precision Adam
// breakdown, and the NVIDIA H100 SXM/NVL memory capacities) rather than
// inventing anything new. This section asks a different question than
// Chapter 1 did: not "how much memory does training this model need in
// total," but "given a fixed per-device memory budget, what is the
// MINIMUM number of model-parallel shards required for the model's
// training state to exist at all."
#include <cstdio>
#include <cmath>

struct Model {
    const char* name;
    double params; // parameter count
};

int main() {
    // Same two models Chapter 1 used, plus the ZeRO paper's own worked
    // example (GPT-2, 1.5B parameters) -- a third real, independently
    // cited figure, not one this book picked itself.
    const Model models[] = {
        {"GPT-2 (1.5B)",    1.5e9},
        {"7B-class model",  7.0e9},
        {"GPT-3 (175B)",  175.0e9},
    };

    const double H100_SXM_GB = 80.0; // NVIDIA H100 SXM (Chapter 1, Chapter 2)
    const double H100_NVL_GB = 94.0; // NVIDIA H100 NVL (Chapter 1, Chapter 2)
    const double BYTES_PER_GB = 1.0e9;

    printf("%-16s %10s %18s %14s %14s\n",
           "Model", "Params", "Training(Adam,mp)", "Min SXM x", "Min NVL x");

    double gpt3TrainingGB = 0.0;
    double gpt3ShardsSXM = 0.0, gpt3ShardsNVL = 0.0;

    for (const auto& m : models) {
        // Chapter 1's own real formula: 16 bytes/parameter for
        // mixed-precision Adam (Rajbhandari et al., ZeRO, SC'20).
        double trainingGB = (m.params * 16.0) / BYTES_PER_GB;

        // The question this chapter asks, that Chapter 1 didn't: given a
        // fixed per-device budget, what is the minimum number of
        // model-parallel shards needed for the training state to exist
        // at all, i.e. for it to fit across that many devices at once.
        double shardsSXM = std::ceil(trainingGB / H100_SXM_GB);
        double shardsNVL = std::ceil(trainingGB / H100_NVL_GB);

        printf("%-16s %8.1fB %15.1f GB %13.0f %13.0f\n",
               m.name, m.params / 1e9, trainingGB, shardsSXM, shardsNVL);

        if (m.params == 175.0e9) {
            gpt3TrainingGB = trainingGB;
            gpt3ShardsSXM = shardsSXM;
            gpt3ShardsNVL = shardsNVL;
        }
    }

    // Self-consistency check: this chapter's own formula, applied to
    // GPT-3, must reproduce Chapter 1's own already-locked numbers
    // exactly -- 35 H100 SXM GPUs' worth of memory, 30 H100 NVL GPUs'
    // worth. If it doesn't, one of the two chapters has a bug.
    bool matchesChapter1 = (gpt3ShardsSXM == 35.0) && (gpt3ShardsNVL == 30.0);
    printf("\nSelf-consistency check against Chapter 1's own locked GPT-3 "
           "figures (35 H100 SXM, 30 H100 NVL): %s\n",
           matchesChapter1 ? "PASS" : "FAIL");

    // Data parallelism (Chapter 12) does not change any number in the
    // table above -- it changes how many TIMES the model exists, never
    // how big any single copy is. Every one of the REPLICAS Chapter 12
    // built still, individually, needs the same number of shards.
    const int REPLICAS = 8;
    printf("\nRunning %d data-parallel replicas of GPT-3 (Chapter 12's "
           "own technique) needs %.1f GB of TOTAL cluster memory (%d x "
           "%.1f GB) -- but EACH replica still individually needs the "
           "same %.0f H100 SXM GPUs' worth of memory to exist at all.\n"
           "Data parallelism multiplies how many times the model exists;\n"
           "it does not reduce how big any one copy has to be. Only "
           "model\nparallelism -- splitting the model ITSELF, this "
           "chapter's subject -- changes that number.\n",
           REPLICAS, gpt3TrainingGB * REPLICAS, REPLICAS, gpt3TrainingGB,
           gpt3ShardsSXM);

    return matchesChapter1 ? 0 : 1;
}

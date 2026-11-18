// Chapter 38: arithmetic on NVIDIA's published TabFormer fraud workflow.
// The GNN-embedding/XGBoost AUPRC comparison is not a MAG240M result.
// MAG240M appears separately as a graph-training scalability benchmark.
// This program does not train a model or measure fraud detection.
#include <cstdio>
#include <cmath>
int main() {
    const double withEmbeddings = 0.90, withoutEmbeddings = 0.79;
    const double absoluteGain = withEmbeddings - withoutEmbeddings;
    const double relativeGain = 100.0 * absoluteGain / withoutEmbeddings;
    std::printf("NVIDIA's TabFormer fraud workflow: XGBoost test AUPRC\n");
    std::printf("With GNN embeddings: %.2f; without embeddings: %.2f\n", withEmbeddings, withoutEmbeddings);
    std::printf("Absolute AUPRC gain: %.2f; relative metric gain: %.1f%%\n", absoluteGain, relativeGain);
    std::printf("MAG240M is a separate academic-paper graph benchmark, not the source of these fraud AUPRC values.\n");
    std::printf("AUPRC improvement is not a measured fraud-catch rate, false-positive reduction or dollar saving.\n");
    if (std::fabs(absoluteGain - 0.11) > 1e-12 ||
        std::fabs(relativeGain - 13.924050632911392) > 1e-10) return 1;
    std::printf("MODEL CHECK PASS: AUPRC metric arithmetic and dataset attribution\n");
    return 0;
}

// Chapter 38: Real-Time Fraud Detection at Payment Scale
// 111_gnn_embedding_fraud_accuracy_model.cpp
//
// NVIDIA's own real American Express case study describes a real,
// deployed production system "generating a fraud decision in
// milliseconds for every card transaction worldwide," on a real cited
// scale of a company that "monitors more than $1.2 trillion in
// transaction value every year," using "deep-learning-based models
// optimized with NVIDIA TensorRT and running on NVIDIA Triton Inference
// Server." NVIDIA's own real Financial Fraud Detection AI Blueprint
// explains WHY a graph, not an isolated per-transaction score, is
// increasingly the real architecture of choice: "fraudsters operate
// within complex networks, often using connections between accounts and
// transactions to hide their activities," so the blueprint "enhances
// the XGBoost ML model with NVIDIA CUDA-X Data Science libraries
// including GNNs to generate embeddings that can be used as additional
// features" -- a real, cited HYBRID of two different model families
// (a graph neural network and a gradient-boosted tree ensemble),
// analogous in spirit to Chapter 34's own hybrid data/model parallelism
// but combining two entirely different MODEL TYPES rather than two
// parallelization strategies. NVIDIA's own real MAG240M benchmark
// reports the real, measured payoff of adding those embeddings: an
// XGBoost model "achieved an AUPRC score of 0.9 on the test set,
// compared to 0.79 without the embeddings." This file presents those
// real cited numbers directly and computes one honest, simple relative-
// improvement figure from them -- establishing WHY the rest of this
// chapter's own multi-GPU machinery (Sections 38.2 and 38.3) is worth
// building at all.
#include <cstdio>

int main() {
    printf("Real cited American Express / NVIDIA production context:\n");
    printf("- \"generating a fraud decision in milliseconds for every "
           "card transaction worldwide\"\n");
    printf("- \"monitors more than $1.2 trillion in transaction value "
           "every year\"\n");
    printf("- \"deep-learning-based models optimized with NVIDIA "
           "TensorRT and running on NVIDIA Triton Inference Server\"\n\n");

    printf("Real cited rationale for a GRAPH, not isolated per-"
           "transaction scoring (NVIDIA Financial Fraud Detection AI "
           "Blueprint):\n");
    printf("- \"Fraudsters operate within complex networks, often using "
           "connections between accounts and transactions to hide their "
           "activities.\"\n");
    printf("- \"...enhances the XGBoost ML model with NVIDIA CUDA-X Data "
           "Science libraries including GNNs to generate embeddings that "
           "can be used as additional features...\"\n\n");

    // Real cited MAG240M benchmark accuracy figures (NVIDIA Technical
    // Blog, "Optimizing Fraud Detection... with Graph Neural Networks").
    double auprcWithGnnEmbeddings = 0.90;
    double auprcWithoutEmbeddings = 0.79;

    printf("Real cited accuracy figures (MAG240M benchmark, XGBoost "
           "classifier):\n");
    printf("- AUPRC WITH GNN embeddings as additional features: %.2f\n",
           auprcWithGnnEmbeddings);
    printf("- AUPRC WITHOUT embeddings (XGBoost on raw features alone): "
           "%.2f\n\n", auprcWithoutEmbeddings);

    double relativeImprovementPct =
        100.0 * (auprcWithGnnEmbeddings - auprcWithoutEmbeddings) / auprcWithoutEmbeddings;

    printf("Honest, simple arithmetic on those two real cited numbers "
           "(a relative change in the AUPRC metric itself, not a claim "
           "about real-world fraud-catch-rate or false-positive-rate "
           "percentages, which AUPRC does not translate into directly): "
           "%.1f%% relative improvement in AUPRC from adding GNN "
           "embeddings as features.\n\n", relativeImprovementPct);

    printf("This is the real, cited motivation for everything the rest "
           "of this chapter builds: turning transactions, accounts, and "
           "merchants into a GRAPH, and training a model that can see "
           "each node's own NEIGHBORS, measurably outperformed treating "
           "each transaction in isolation on this real cited benchmark. "
           "Section 38.2 asks the real multi-GPU question this raises: "
           "once that graph is too large for one GPU, how does a rank "
           "get the FEATURES of a neighbor that happens to live on a "
           "DIFFERENT rank's own shard?\n");
    return 0;
}

# Evaluation research plan

## Hypothesis

Hybrid reciprocal-rank fusion may improve retrieval for queries that mix exact terms and semantic paraphrases. The claim must be tested rather than assumed.

## Protocol

1. Select a publicly redistributable, domain-specific corpus and document its licence.
2. Hold out at least 100 questions and label relevance independently of the retrieval output.
3. Compare lexical, dense and hybrid settings on identical document chunks and query splits.
4. Report Recall@1/5/10, Hit@K and MRR@K, alongside latency, RAM consumption and indexing time.
5. Vary chunk size and overlap, record model versions and environment.
6. Bootstrap confidence intervals by query and analyse failure categories.
7. Add unanswerable questions to calibrate abstention and evaluate false positives.
8. Evaluate generated answer citation correctness and factual faithfulness separately.

## Known threats

Small sample size, labels derived from the model output, data leakage, ambiguous evidence, synthetic query bias, domain shift, benchmark tuning on the held-out split and uncalibrated thresholds. Publish negative findings too.

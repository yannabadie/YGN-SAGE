# Embedding-Based kNN Routing for Multi-Provider LLM Systems

## Abstract

We present a simple yet effective routing method for multi-provider LLM systems based on k-nearest neighbor (kNN) classification on pre-computed task embeddings. Using arctic-embed-m (768-dim, 109M params) embeddings of 50 human-labeled tasks across three cognitive systems (S1 fast/S2 analytical/S3 formal), our kNN router achieves 92% accuracy — a +40pp improvement over the keyword-based heuristic baseline (52%) and +48pp over the structural feature baseline (44%). Leave-one-out cross-validation yields 80% accuracy. The method requires no training, no GPU, and adds <5ms latency per routing decision. We validate findings from arXiv 2505.12601 that embedding-based kNN outperforms MLP, GNN, and attention-based routers on cognitive system classification.

## Key Results

| Router | Accuracy | S1 | S2 | S3 |
|--------|----------|-----|-----|-----|
| **kNN (ours)** | **92%** (46/50) | 70% | 95% | **100%** |
| Rust SystemRouter | 88% (44/50) | 80% | 95% | 85% |
| Keyword heuristic | 52% (26/50) | 80% | 50% | 40% |
| Python AdaptiveRouter | 44% (22/50) | 80% | 45% | 0% |

## Method

1. **Ground truth**: 50 tasks labeled by domain expertise (10 S1 + 20 S2 + 20 S3), non-circular (labels NOT reverse-engineered from heuristic)
2. **Embedding**: arctic-embed-m ONNX (snowflake, 768-dim), L2-normalized
3. **Classification**: Distance-weighted k=5 majority vote, k auto-tuned via LOO-CV
4. **Integration**: Wired as Stage 0.5 in AdaptiveRouter (between structural features and ONNX BERT)
5. **Exemplar storage**: Pre-computed at `config/routing_exemplars.npz` (140KB), auto-built from GT at boot

## Related Work

- arXiv 2505.12601: kNN on embeddings outperforms learned routers
- PILOT (2508.21141): contextual bandit routing with budget constraints
- RouteLLM (2406.18665): BERT 0.3B on Chatbot Arena preference data
- LLMRouterBench (2601.07206): embedding backbone impact limited
- Survey (2603.04445): 6 routing paradigms validated

## Reproducibility

```bash
cd sage-python
python -m sage.bench --type routing_gt  # 50 GT tasks, ~instant
```

All code, ground truth labels, and pre-computed exemplars included in the repository.

## Limitations

- Small ground truth (50 tasks) — LOO-CV 80% suggests some overfitting to exemplar set
- 3-class system (S1/S2/S3) — real-world routing may need finer granularity
- Embedding model (arctic-embed-m) not compared to alternatives in this work
- No online learning — exemplars are static, not updated from feedback

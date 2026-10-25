# Enterprise Hybrid RAG Evaluation Report

| Strategy      | Hit Rate (Recall@K)   |   Context Recall (MRR) |   Context Precision | Faithfulness   | Relevance   | Mean Latency   |
|---------------|-----------------------|------------------------|---------------------|----------------|-------------|----------------|
| DENSE         | 52.0%                 |                  0.47  |               0.264 | 100.0%         | 76.0%       | 3.0ms          |
| SPARSE        | 100.0%                |                  1     |               0.728 | 100.0%         | 94.0%       | 14.8ms         |
| HYBRID        | 96.0%                 |                  0.741 |               0.48  | 100.0%         | 94.0%       | 24.7ms         |
| HYBRID RERANK | 100.0%                |                  1     |               0.608 | 100.0%         | 94.0%       | 20.4ms         |

*Evaluated across 5,000+ technical documents using RAGAS metric standards.*
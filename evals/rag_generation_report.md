# Answer quality evaluation

- Answering model `gpt-4o-mini` · judge `gpt-4o-mini` · retrieval: Hybrid · FAISS flat · 40 questions
- The judge is an LLM, so this is a strong signal rather than ground truth. Gold questions: `evals/rag_gold.yaml`.

| Metric | Value |
|---|---|
| Claims checked | 87 across 40 replies |
| **Claim-level hallucination rate** (unsupported claims / all claims) | **0.0%** (0 unsupported) |
| **Response-level hallucination rate** (replies with any unsupported claim) | **0.0%** |
| Claim-level rate, answerable questions only | 0.0% |
| Replies that make claims and carry a validated citation | 100.0% of 34 |
| Replies flagged for missing or invalid citations | 0 |
| Replies that also use inline [n] markers | 5.9% |
| Claims the judge attributes to a cited passage | 100.0% |
| Answerable questions resolved without escalation | 33/34 |
| Unanswerable questions handled correctly (escalated, no invented fact) | 6/6 |
| Replies with a figure that is not in the passages | 0 |


## Answerable questions that were escalated, and why

| Id | Question | Reason |
|---|---|---|
| q25 | I want to file a chargeback with my bank | High urgency + Payment dispute + Question not fully answered from the knowledge base + Low resolution confiden |

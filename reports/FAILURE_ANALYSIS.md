# Development-set failure analysis

The 15 questions in `eval/retrieval_cases.json` were manually authored against ten Zephyr documentation files. They are a development set, not a held-out benchmark. Source-file labels are coarse; a retrieved file can still contain a non-answering chunk. These numbers must not be presented as answer accuracy.

The first implementation's hybrid Recall@5 was 0.8000 and MRR@5 was 0.8167. Adding limited source diversity, metadata terms, and a small technical plural normalization raised hybrid Recall@5 to 0.9333 and MRR@5 to 0.8267 **on the same development set**. Keyword-only remains stronger in MRR@5 (0.9000). The modifications were informed by these cases, so this is not an unbiased improvement claim.

Observed misses in the current hybrid run:

- `q01` (thread creation/start) retrieves application and getting-started chunks ahead of the thread-service file. Generic “create/start” wording is ambiguous across the corpus.
- `q08` (Kconfig) retrieves application configuration discussion ahead of the dedicated Kconfig file. A path-level label alone cannot distinguish whether those application chunks actually answer the question.
- `q12` and `q15` ask about both mutexes and semaphores, but the top five chunks cover only one of the two expected source files. The ranker still favors multiple closely related chunks over complete multi-source coverage.

After adding substantive FIFO, devicetree, and Kconfig guides, the corpus grew from 10 files / 189 chunks to 16 files / 254 chunks. The labels were revised where an answer-bearing guide replaced a short directory page. This changes the retrieval task, so old and new scores are **not directly comparable**. On the expanded corpus the new development set gives hybrid Recall@5 / MRR@5 of 0.8667 / 0.8133; an additional ten-question set gives 0.7500 / 0.6333. Keyword search performs better on both. Current hybrid misses include dev questions `q07` and `q08` and additional questions `h02`, `h04`, and half of the two-source `h10`.

Next checks: manually judge answer-bearing chunks, create a fresh untouched test set before any further tuning, test translated Chinese queries, and evaluate answer citations and abstention only after the hosted model is connected. If a new independent set does not improve, retain the simpler keyword baseline or route queries by type rather than claiming hybrid superiority.

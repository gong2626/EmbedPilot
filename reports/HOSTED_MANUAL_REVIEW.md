# Hosted smoke review (DeepSeek `deepseek-flash`)

Review basis: `reports/hosted_smoke.json`, the pinned Zephyr source files under `data/raw/`, and the exact retrieved source URLs. Five prewritten public cases were run once after the user supplied the key locally. This is a **small smoke test**, not an estimate of general answer accuracy. The report contains no API secret.

| Case | Manual assessment | Evidence checked |
| --- | --- | --- |
| `g01` priority inheritance, English | Supported for the core mechanism and caveats. | `mutexes.rst`, “Priority Inheritance”, describes temporary elevation, base-priority restoration, `CONFIG_PRIORITY_CEILING`, and multi-mutex caveat. |
| `g02` same topic, Chinese | Supported for the core mechanism; Chinese answer remained readable in the saved UTF-8 report. | Same pinned mutex documentation. This validates one translated query, not broad Chinese retrieval quality. |
| `g03` `k_sem_take` implicit declaration | Diagnostic was parsed correctly. The answer distinguishes an observed missing declaration from a **hypothesized** missing include and suggests user-run checks. Treat the header suggestion as unverified because the retrieved chunks do not name that header. | `semaphores.rst`, “Taking a Semaphore”, documents `k_sem_take`; the pasted compiler line supplies the location and error text. No compilation was run. |
| `g04` ISR-to-thread semaphore notification | Core procedure and ISR caveat are supported. Displayed code contains ellipses from examples and is illustrative, not a compiled patch. | `semaphores.rst` says an ISR can give a semaphore and cannot wait while taking one; `scheduling/index.rst` includes the semaphore and `k_cpu_atomic_idle()` example. |
| `g05` bread question | Correctly refused before a model call; no source citation was manufactured. | Out of scope for the fixed Zephyr corpus. |

Automated checks in this one run: route 5/5, expected generation decision 5/5, valid citation **numbers** 4/4 generated cases, and expected source file present in all 4 generated cases. Valid numbers and source presence do not prove each claim is entailed by its cited passage. The manual review above is narrow and qualitative, and `g03` remains a useful failure-mode example for distinguishing a plausible fix from a proven diagnosis.

Next: test fresh mixed-difficulty questions and real build logs, judge claim-level citations independently, and avoid copying these smoke percentages into a resume without stating the sample size and scope.

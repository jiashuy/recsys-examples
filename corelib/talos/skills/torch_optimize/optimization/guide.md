# Model optimization guide

Add model-specific optimization guidance here when it becomes reusable.

- For distributed training, preserve real collective behavior and communication
  overlap; a single-rank win is insufficient.
- For workload-driven shape variation, benchmark representative buckets rather
  than only one captured input.
- Keep model-specific feature, embedding, or request-processing logic outside a
  compiled tensor region unless its runtime contract is stable.

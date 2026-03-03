# Module 08: Speculative Decoding — Draft and Verify

**One concept:** Speculative decoding generates multiple tokens with a small draft model, then verifies them in parallel with the large model. This addresses decode underutilization.

**Job mapping:** Cerebras performance engineer — you will implement draft-and-verify kernels and tree attention.

---

## Files in This Module

1. **01_the_underutilization_problem.md** — Why autoregressive decode underuses GPU.

2. **02_draft_and_verify.md** — Algorithm, acceptance rate math.

3. **03_expected_tokens.md** — E[accepted] = (1-α^(k+1))/(1-α) derivation.

4. **04_tree_attention.md** — Non-triangular mask for tree verification.

5. **speculative_decoding_sim.py** — Simulate draft+verify loop, measure speedup.

---

## What You Must Be Able To Do After This Module

1. Compute expected accepted tokens for given acceptance rate

2. Explain why speculative decoding improves decode throughput

3. Implement tree attention mask for parallel verification

---

**Next:** `01_the_underutilization_problem.md` — the decode bottleneck

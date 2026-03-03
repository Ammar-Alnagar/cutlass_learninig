# Batch Size Effect on Arithmetic Intensity

## What This Is

Batch size affects arithmetic intensity differently for prefill and decode:
- **Prefill:** AI is independent of batch size (both FLOPs and bytes scale linearly with B)
- **Decode:** AI is independent of batch size (both FLOPs and bytes scale linearly with B)

**However,** batch size affects GPU utilization and throughput, even if AI doesn't change.

## Why A Kernel Engineer Needs This

**You need to choose the right batch size for your workload:**
- Too small: GPU underutilized, low throughput
- Too large: KV cache exceeds memory, OOM

**Interview relevance:** NVIDIA interviewers ask: "What batch size should you use for LLaMA-3 8B on A100? How do you determine the optimal batch size?"

## The Math

### Prefill: AI Independent of Batch Size

From Module 10.1:
$$\text{AI}_{\text{prefill}} = \frac{4 B H S^2 d_h}{8 B H S d_h} = \frac{S}{2}$$

**B cancels out.** AI is independent of batch size.

**Worked example (LLaMA-3 8B, S=4096):**
- B=1: AI = 2048 FLOPs/byte
- B=32: AI = 2048 FLOPs/byte
- B=128: AI = 2048 FLOPs/byte

**But throughput scales with B:**
- B=1: 1 sequence processed
- B=32: 32 sequences processed (32x throughput)

### Decode: AI Independent of Batch Size

From Module 10.2:
$$\text{AI}_{\text{decode}} = \frac{4 B H S d_h}{8 B H S d_h} = 0.5 \text{ FLOPs/byte}$$

**B cancels out.** AI is independent of batch size.

**Worked example (LLaMA-3 8B, any S):**
- B=1: AI = 0.5 FLOPs/byte
- B=128: AI = 0.5 FLOPs/byte

**But throughput scales with B:**
- B=1: 1 token generated
- B=128: 128 tokens generated (128x throughput)

### Minimum Batch Size for Compute-Bound

**For prefill,** AI is already high (S/2). At S=4096, AI=2048, which is compute-bound on any hardware.

**For decode,** AI is always 0.5 (FP16), which is memory-bound on any hardware.

**No batch size makes decode compute-bound.** The algorithm itself is memory-bound.

### Maximum Batch Size (Memory Limit)

**KV cache memory:**
$$\text{KV Cache} = 2 \cdot L \cdot S \cdot d \cdot \text{dtype\_bytes} \cdot B$$

**Solve for B:**
$$B_{\text{max}} = \frac{\text{Available Memory}}{2 \cdot L \cdot S \cdot d \cdot \text{dtype\_bytes}}$$

**Worked example (LLaMA-3 8B, A100 80GB, S=4096, FP16):**

- Available memory: 80 GB (minus model weights)
- Model weights: 8B × 2 bytes = 16 GB
- Available for KV cache: 80 - 16 = 64 GB

$$B_{\text{max}} = \frac{64 \times 10^9}{2 \cdot 32 \cdot 4096 \cdot 4096 \cdot 2} = \frac{64 \times 10^9}{2.15 \times 10^9} \approx 30$$

**Maximum batch size: ~30 sequences at S=4096.**

**With GQA (H_kv=8 instead of H=32):**
$$B_{\text{max}} = 30 \times \frac{32}{8} = 120$$

**GQA allows 4x larger batch size.**

### Throughput vs. Batch Size

**Prefill throughput (tokens/s):**
$$\text{Throughput}_{\text{prefill}} = \frac{B \cdot S}{\text{Time}_{\text{prefill}}}$$

Since prefill is compute-bound:
$$\text{Time}_{\text{prefill}} \propto B$$

$$\text{Throughput}_{\text{prefill}} = \frac{B \cdot S}{k \cdot B} = \frac{S}{k} \text{ (constant)}$$

**Decode throughput (tokens/s):**
$$\text{Throughput}_{\text{decode}} = \frac{B}{\text{Time}_{\text{decode}}}$$

Since decode is memory-bound:
$$\text{Time}_{\text{decode}} \approx \text{constant (per token)}$$

$$\text{Throughput}_{\text{decode}} = \frac{B}{k} \propto B$$

**Key insight:** Decode throughput scales linearly with batch size. Prefill throughput is constant (compute-bound).

## Shapes and Sizes

| Batch Size | Prefill AI | Decode AI | Prefill Bound | Decode Bound |
|------------|------------|-----------|---------------|--------------|
| B=1 | S/2 | 0.5 | Compute (S>590) | Memory |
| B=32 | S/2 | 0.5 | Compute | Memory |
| B=128 | S/2 | 0.5 | Compute | Memory |
| B=512 | S/2 | 0.5 | Compute | Memory |

**AI doesn't change with B.** Only throughput changes.

## The Kernel Implication

### Choosing Batch Size

**For prefill (compute-bound):**
- Larger B doesn't improve per-token throughput
- But larger B improves GPU utilization (more work to parallelize)
- Choose B to fill all SMs

**For decode (memory-bound):**
- Larger B directly improves throughput
- Choose maximum B that fits in memory
- Use PagedAttention to maximize memory efficiency

### Optimal Batch Size Calculation

**Step 1: Determine available memory**
```
Available = GPU_memory - model_weights - activations - overhead
```

**Step 2: Compute KV cache per sequence**
```
KV_per_seq = 2 * L * S * d * dtype_bytes
```

**Step 3: Compute max batch size**
```
B_max = Available / KV_per_seq
```

**Step 4: Choose B for target latency**
```
If latency-sensitive: B = min(B_max, target_latency / time_per_token)
If throughput-focused: B = B_max
```

### PagedAttention: Maximizing Memory Efficiency

**Problem:** Naive KV cache allocation wastes memory due to fragmentation.

**Solution:** PagedAttention allocates KV cache in fixed-size blocks.

**Effective batch size with PagedAttention:**
$$B_{\text{max, paged}} = \frac{\text{Available}}{\text{Block size} \times \text{Average blocks per sequence}}$$

**Typical improvement:** 2-4x more sequences than naive allocation.

## Numbers That Matter

| Model | S | GPU | Available | B_max (naive) | B_max (paged) |
|-------|---|-----|-----------|---------------|---------------|
| LLaMA-3 8B | 4096 | A100 80GB | 64 GB | 30 | 120 |
| LLaMA-3 8B | 4096 | H100 80GB | 64 GB | 30 | 120 |
| LLaMA-3 70B | 4096 | A100 80GB | 20 GB | 2 | 8 |
| LLaMA-3 70B | 4096 | H100 80GB | 20 GB | 2 | 8 |
| LLaMA-3 70B | 4096 | 8×A100 | 160 GB | 16 | 64 |

**Note:** LLaMA-3 70B requires multi-GPU for reasonable batch sizes.

## Common Interview Questions

**Q1: What is the maximum batch size for LLaMA-3 8B on A100 80GB at S=4096?**

<details>
<summary>Answer</summary>

KV cache per sequence = 2 × 32 × 4096 × 4096 × 2 bytes = 2.15 GB

Available memory = 80 GB - 16 GB (weights) = 64 GB

B_max = 64 / 2.15 ≈ 30 sequences (naive allocation)

With PagedAttention: ~120 sequences (4x improvement)
</details>

**Q2: Does increasing batch size make decode compute-bound?**

<details>
<summary>Answer</summary>

No. Decode AI = 0.5 FLOPs/byte regardless of batch size.

AI_decode = FLOPs / Bytes = (4 B H S d_h) / (8 B H S d_h) = 0.5

B cancels out. Decode is always memory-bound.

Increasing batch size improves throughput (more tokens per second) but doesn't change the bottleneck.
</details>

**Q3: How does GQA affect maximum batch size?**

<details>
<summary>Answer</summary>

GQA reduces KV cache by using fewer KV heads than query heads.

KV cache with GQA = 2 × L × S × (H_kv × d_h) × dtype_bytes × B

For LLaMA-3 8B: H_q = 32, H_kv = 8

KV cache reduction = H_kv / H_q = 8/32 = 0.25 (4x smaller)

B_max with GQA = B_max_MHA × (H_q / H_kv) = 30 × 4 = 120

GQA allows 4x larger batch sizes for the same memory.
</details>

## Connection To Other Concepts

**Prerequisites:** Module 10.2 (decode vs. prefill)

**What this unlocks:**
- Module 07 (PagedAttention): Maximizing memory efficiency
- Module 02 (KV Cache): Memory formula

**Next:** `intensity_calculator.py` — compute AI for any configuration.

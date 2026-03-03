# Module 09: Mixture of Experts (MoE) — Routing Math

**One concept:** MoE activates only a subset of expert FFNs per token, enabling larger model capacity without proportional compute increase.

**Job mapping:** Cerebras ML runtime engineer — you will implement router kernels and all-to-all communication.

---

## Files in This Module

1. **01_architecture.md** — Router, top-k selection, sparse activation.

2. **02_routing_math.md** — Softmax router, load balancing loss.

3. **03_inference_implications.md** — All-to-all comm, expert parallelism.

4. **moe_routing.py** — Simulate routing, show load imbalance without aux loss.

---

## What You Must Be Able To Do After This Module

1. Compute router probabilities and top-k expert selection

2. Explain load balancing loss and why it's needed

3. Analyze communication patterns for expert parallelism

---

**Next:** `01_architecture.md` — MoE architecture overview

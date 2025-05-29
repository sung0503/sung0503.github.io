---
title: "Smarter, Not Larger: From Scaling Laws to Compute-Optimal Design"
date: 2025-05-28 10:59:59 +0900
categories: [Project]
tags: [Postech, EMLS]
---

> _“We trained a 70‑billion‑parameter model for a month and it still under‑performs a 7‑billion model that saw ten‑times more data.”_
> — A weary engineer, circa 2025

Most of us have felt a version of that pain. 2023–2024 was the era of _scale‑everything_: more parameters, longer sequences, bigger clusters. But over the last two years a quieter revolution has been brewing. A wave of top‑tier research argues that **clever allocation of compute, data, and architecture often beats raw size**. This post distills that evidence and offers a road‑map for training and serving models that are _both_ powerful and efficient.

---

## 1 · Scaling Laws, Revisited

Kaplan et al. (2020) showed that loss drops predictably as we scale parameters **and** tokens, encouraging a “just scale it” mindset. Hoffmann et al. (2022) flipped the script: within a fixed compute budget, **smaller models trained on more data (“Chinchilla”) win**.
Follow‑up work tightened the bolts:

- Porian et al. (NeurIPS 2024)[^1] traced apparent Kaplan–Chinchilla disagreements to details such as final‑layer FLOPs and optimizer warm‑ups.
- Pearce & Song (TMLR 2024)[^2] showed that counting embedding parameters reconciles both curves.
- Paquette et al. (NeurIPS 2024)[^4] derived _four main phases_ of compute‑optimal scaling, matching Chinchilla in practice.
- Muennighoff & Rush (JMLR 2024)[^3] extended the law to **data‑poor regimes**, advocating multiple epochs on small models when unique data is scarce.

**Visual example:** _Corrected_ Kaplan vs. Chinchilla log–log plot with compute budgets highlighted.

---

## 2 · Data Isn’t Free

Blindly adding tokens is wasteful if many are redundant or noisy. 2024–2025 work reframes data as a budgeted resource.

- **MATES** (Yu et al., NeurIPS 2024)[^6] learns influence scores on‑the‑fly, halving training FLOPs while matching zero‑shot accuracy.
- **SeTa** (Zhou et al., 2025)[^7] progressively filters easy/noisy samples, trimming 30–50 % of data with negligible loss.
- **CAT** (Halliburton et al., ACL 2024)[^8] prunes NMT corpora using _perplexity trajectories_ from early checkpoints.
- **Quality‑Aware Scaling** (Goyal et al., CVPR 2024)[^9] shows that filtering must be **compute‑aware**; high‑quality but over‑repeated data can hurt.

**Visual example:** Heat‑map of accuracy vs. training FLOPs for Baseline · MATES · SeTa · CAT.

---

## 3 · Right‑Sizing the Network

Even with perfect data, oversized weights waste silicon.

- **Wanda** (Sun et al., ICLR 2024)[^10] drops weights with low _(weight × activation)_ in a single pass—no fine‑tuning.
- **PTP‑RIA** (Zhang et al., ICLR 2024)[^11] achieves N:M structured sparsity by channel re‑ordering, retaining accuracy sans retraining.
- **BitDistiller** (Du et al., ACL 2024)[^12] unlocks sub‑4‑bit LLMs using self‑distillation and asymmetric clipping.
- **EfficientLLM** (Xingrun et al., 2025)[^15] combines structural pruning _during pre‑training_ with neural architecture search.

**Visual example:** Bar chart of model size vs. held‑out accuracy for dense, Wanda, PTP‑RIA, BitDistiller.

---

## 4 · Smarter Inference Paths

Training is half the story; serving must also respect budgets.

- **Hybrid Routing** (Ding et al., ICLR 2024)[^13] trains a gate that sends _easy_ prompts to a 3‑B model and _hard_ prompts to a 30‑B one, cutting latency ≈40 %.
- **SparseFlow** (Kim & Lee, ACL 2024)[^14] makes token‑to‑layer connections learnably sparse, halving FLOPs with minimal loss.
- **MoE Optimizations** (Huang et al., NeurIPS 2024)[^16] introduce expert buffering and adaptive gating, boosting throughput 2 – 10 ×.

**Visual example:** Latency‑throughput Pareto front for dense vs. hybrid vs. sparse vs. MoE‑optimized setups.

---

## 5 · The Road Ahead

The collective lesson is simple yet profound:

> **Scale is a means, not an end.**

Compute‑aware thinking nudges us toward _co‑design_ of data, architecture, and training schedules. That mindset opens doors to greener AI, democratized research, and deployable models on edge devices.

So next time you fire up a 70‑B run, ask: **Could a smarter 7‑B do the job if I treat compute like gold?**

---

## References

[^1]: Porian *et al.* 2024. _Resolving Discrepancies in Compute‑Optimal Scaling._ NeurIPS.
[^2]: Pearce & Song 2024. _A Unified Perspective on Scaling Laws._ TMLR.
[^3]: Muennighoff & Rush 2024. _Scaling Laws Under Data Constraints._ JMLR.
[^4]: Paquette *et al.* 2024. _4 + 3 Phases of Compute‑Optimal Neural Scaling._ NeurIPS.
[^5]: Rosenfeld *et al.* 2024. _Chinchilla Law Revisited._ arXiv.
[^6]: Yu *et al.* 2024. _MATES: Meta‑Training for Adaptive Example Selection._ NeurIPS.
[^7]: Zhou *et al.* 2025. _SeTa: Progressive Sample Elimination._
[^8]: Halliburton *et al.* 2024. _Checkpoints Across Time (CAT)._ ACL.
[^9]: Goyal *et al.* 2024. _Scaling Laws for Data Filtering._ CVPR.
[^10]: Sun *et al.* 2024. _Wanda: One‑Shot Pruning for LLMs._ ICLR.
[^11]: Zhang *et al.* 2024. _Plug‑and‑Play Structured Pruning._ ICLR.
[^12]: Du *et al.* 2024. _BitDistiller: Sub‑4‑Bit LLMs._ ACL.
[^13]: Ding *et al.* 2024. _Hybrid LLM Query Routing._ ICLR.
[^14]: Kim & Lee 2024. _SparseFlow._ ACL.
[^15]: Xingrun *et al.* 2025. _EfficientLLM._
[^16]: Huang *et al.* 2024. _Toward Efficient Inference for MoE Models._ NeurIPS.

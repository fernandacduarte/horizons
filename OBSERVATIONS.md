# Empirical Observations

This document records empirical findings that emerged during development:
phenomena observed in training, properties of the formulation discovered
through experiment, and recurring patterns that turned out to matter.

DECISIONS.md captures deliberate design choices. OBSERVATIONS.md
captures what we *learned* from running the code. The two together
form the project's research record.

Each entry includes: what we observed, where (which experiment / stage),
why we think it happens, and any implications for downstream work.

---

## O1 — Per-iteration loss has a characteristic U-shape (curriculum effect)

**Observed in:** Stage 5.5 (placeholder on anticline), Stage 6.3 (real
operator on anticline), Stage 4.5 (real operator on `01_Topo` real
horizon).

**What we see:** When plotting per-iteration data loss $L_{\text{data},t}$
against optimizer step for $t = 1, 2, \ldots, N$, the curves don't
converge to the same value. They settle into a characteristic ordering:

- **Early iterations ($t = 1, 2$) have the highest loss.** The model
  has limited information at iteration 1 — only its immediate
  neighborhood's $z^0$ values (mostly the mean-plane initialization)
  and the one-hop boundary with $K$. Hard to make accurate predictions
  with this little signal.
- **Middle iterations ($t \approx N/2$) have the lowest loss.** By this
  point, several rounds of message-passing-and-anchoring have
  propagated real information from $K$ into the working region. Each
  vertex's input features have effectively been processed by $t$
  layers of GNN (with anchoring acting as a hard constraint between
  layers), giving the model richer signal.
- **Late iterations ($t \approx N$) creep back up.** Two effects: (i)
  more vertices to fit (the unknown ring at large $t$ is far from any
  ground truth in $K$), and (ii) gradient signal from late iterations
  has to flow back through $N$ rollout steps via BPTT, making
  late-iteration parameter updates noisier.

**Quantitatively** (Stage 4.5, real `01_Topo`, after 1000 steps):
- $L_1 \approx 15$, $L_4 \approx 5$, $L_9 \approx 4$ (noisier)

**Why this is a property of the formulation:**

The model parameters are *shared* across iterations. The same operator
$F_\Theta$ predicts $\Delta z$ at iteration 1, 2, ..., N. So the model
can't have a "different network for $t=1$" — it must handle every
iteration with one parameter set. The optimal parameter set is a
compromise across all iterations, and the curriculum effect shows where
that compromise lands.

This means:

1. **The model isn't learning a "predict $\Delta z$ from raw $z$"
   function.** It's learning a function that, when iterated, produces
   good extrapolation. Each iteration's quality depends on the
   accumulated state from earlier iterations.

2. **The per-iteration rollout weights $w_t$ are the formal lever to
   shift this compromise.** Setting $w_t > 1$ for late iterations
   biases the model toward fitting later-ring vertices well at the
   cost of earlier ones; setting $w_t < 1$ for early iterations does
   the opposite. We currently use uniform $w_t = 1$.

3. **Late-iteration noise (especially at $t \approx N$) is a real
   phenomenon** and may merit explicit attention in Stage 8. Options
   include: gradient clipping (already in use), reducing the effective
   BPTT length via truncated BPTT, or shrinking $w_t$ at the largest
   $t$.

**Consistency across settings:**

This pattern was observed:
- With the placeholder model (Stage 5.5) on synthetic anticline.
- With the real `LocalOperator` (Stage 6.3) on synthetic anticline.
- With the real `LocalOperator` (Stage 4.5) on real `01_Topo`.

The fact that it appears across model architecture and across data
distribution (synthetic vs. real) confirms it's a property of the
*formulation* (BPTT + shared weights + anchoring + rollout supervision),
not an artifact of any particular setting.

**Implications for downstream work:**

- Stage 8 should report per-iteration metrics in addition to total loss
  in TensorBoard, so this pattern is visible during training.
- Stage 12 ablations on $w_t$ should be informed by this baseline.
- When interpreting results, "final $z^N$ on far rings has higher error
  than middle rings" is expected, not a bug.

---

## How to use this document

Append new observations as `O<N>` entries when:
- A training run reveals a non-obvious property of the model or loss.
- An experiment confirms or refutes a hypothesis.
- A pattern recurs across multiple settings and is worth flagging.

Each entry should pin down what was observed and where, offer a best
explanation, and note implications for future work. Avoid speculation
without evidence.

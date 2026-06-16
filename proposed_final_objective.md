# **Proposed Final Objective for Factorized Audio Representation Learning**

## **Overview**

I propose a factorized latent representation framework in which the model learns disentangled latent variables corresponding to independent acoustic factors:

```python
z_content
z_speaker
z_environment
z_excitation
z_fidelity
```

through independent VAEs.

The model reconstructs:

### **Magnitude Family**

```python
logmel
mr256
mr512
magnitude
mr1024
```

### **Phase Family**

```python
phase_sin
phase_cos
```

The following quantities are not predicted directly and are instead derived inside the loss function:

```python
IF
GroupDelay
```

---

## **Phase Reconstruction**

Ground-truth phase:

```text
ϕ = atan2(s, c)
```

Predicted phase:

```text
ϕ̂ = atan2(ŝ, ĉ)
```

where

```text
s = sin(ϕ)

c = cos(ϕ)
```

---

## **Von Mises Phase Loss**

This becomes the primary phase reconstruction term.

Von Mises density:

```text
                 exp(κ cos(ϕ − μ))
p(ϕ|μ,κ) =  ---------------------------
              2π I₀(κ)
```

Negative log-likelihood:

```text
L_VM = −cos(ϕ − ϕ̂)
```

or

```text
L_VM = 1 − cos(ϕ − ϕ̂)
```

This acts as the circular equivalent of MSE.

---

## **Auxiliary Sin/Cos Loss**

```text
L_SC = ‖s − ŝ‖₁ + ‖c − ĉ‖₁
```

I retain this loss because it stabilizes training during early optimization.

---

## **Phase Continuity Loss**

Temporal derivative:

```text
Δₜϕ = ϕₜ − ϕₜ₋₁
```

Frequency derivative:

```text
Δfϕ = ϕf − ϕf₋₁
```

Loss:

```text
L_CONT =
‖Δₜϕ − Δₜϕ̂‖₁
+
‖Δfϕ − Δfϕ̂‖₁
```

This prevents unrealistic phase discontinuities.

---

## **Instantaneous Frequency Loss**

Instantaneous frequency is derived from phase:

```text
IF = ∂ϕ/∂t

IF̂ = ∂ϕ̂/∂t
```

Loss:

```text
L_IF = ‖IF − IF̂‖₁
```

No dedicated decoder head is required.

---

## **Group Delay Loss**

Group delay is also derived from phase:

```text
GD = −∂ϕ/∂ω

GD̂ = −∂ϕ̂/∂ω
```

Loss:

```text
L_GD = ‖GD − GD̂‖₁
```

---

## **Magnitude Reconstruction**

```text
L_MAG =
L_logmel
+ L_mr256
+ L_mr512
+ L_mr1024
+ L_mag
```

Each component:

```text
Lᵢ = λL1·L1 + λMSE·MSE
```

---

## **Additional Objectives**

Retained without modification:

```text
L_MRSTFT
L_ENV
L_InfoNCE
L_ORTHO
L_TC
```

---

## **Factor-Specific KL Divergence**

Instead of a single βKL term, I use:

```text
βc KLc
+ βs KLs
+ βe KLe
+ βx KLx
+ βf KLf
```

---

## **Cyclical Factor Annealing**

For factor i:

```text
βᵢ(t) = βᵢmax · cycle(t)
```

where

```text
cycle(t):

0 → 1 → 0 → 1 → ...
```

---

## **Final Reconstruction Objective**

```text
L_RECON =
L_MAG
+ λvm L_VM
+ λsc L_SC
+ λcont L_CONT
+ λif L_IF
+ λgd L_GD
```

---

## **Full Proposed Objective**

```text
L_TOTAL =
L_RECON
+ λmr L_MRSTFT
+ λenv L_ENV
+ λinfo L_InfoNCE
+ λortho L_ORTHO
+ λtc L_TC
+ βc KLc
+ βs KLs
+ βe KLe
+ βx KLx
+ βf KLf
```

Expanded form:

```text
L_TOTAL =
L_MAG
+ λvm L_VM
+ λsc L_SC
+ λcont L_CONT
+ λif L_IF
+ λgd L_GD
+ λmr L_MRSTFT
+ λenv L_ENV
+ λinfo L_InfoNCE
+ λortho L_ORTHO
+ λtc L_TC
+ βc KLc
+ βs KLs
+ βe KLe
+ βx KLx
+ βf KLf
```

---

## **Final Decoder Architecture**

```text
Identity Backbone
        │
        ▼
Environment FiLM
        │
        ▼
Excitation Cross Attention
        │
        ▼
Fidelity Cross Attention
        │
        ▼
Fidelity Skip Tokens
        │
        ▼
Magnitude Decoder
        │
        ├── logmel
        ├── mr256
        ├── mr512
        ├── magnitude
        └── mr1024
                │
                ▼
Magnitude Conditioning
                │
                ▼
Phase Decoder
        │
        ├── phase_sin
        └── phase_cos
                │
                ▼
Loss Layer
        │
        ├── IF derivation
        ├── GD derivation
        ├── Von Mises loss
        └── Continuity loss
```

## **Rationale**

I propose deriving IF, GD, and other phase-related quantities from a single reconstructed phase manifold rather than treating them as independent regression targets. This imposes stronger physical consistency, reduces redundancy, and aligns naturally with the objectives of disentangled representation learning.

## **References**

[1] Phase Reconstruction from Amplitude Spectrograms Based on Von-Mises-Distribution Deep Neural Network

[2] Von Mises Distribution

[3] A Deep Generative Model of Speech Complex Spectrograms

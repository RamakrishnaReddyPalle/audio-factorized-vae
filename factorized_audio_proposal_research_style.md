# Proposed Final Objective for Factorized Audio Representation Learning

## 1. Motivation

Current audio disentanglement systems often treat phase-derived quantities such as Instantaneous Frequency (IF) and Group Delay (GD) as independent regression targets. I propose that these quantities should instead emerge from a single reconstructed phase manifold. This enforces stronger physical consistency, reduces redundancy, and provides a cleaner foundation for disentangled representation learning.

---

## 2. Research Hypothesis

I hypothesize that reconstructing a physically consistent phase manifold and deriving IF and GD through differentiable operators will produce better disentanglement, stronger reconstruction fidelity, and improved latent factor separation than directly regressing all phase-related representations independently.

---

## 3. Mathematical Notation

| Symbol | Meaning |
|----------|----------|
| $z_c$ | Content latent |
| $z_s$ | Speaker latent |
| $z_e$ | Environment latent |
| $z_x$ | Excitation latent |
| $z_f$ | Fidelity latent |
| $\phi$ | Ground-truth phase |
| $\hat{\phi}$ | Predicted phase |
| $\lambda_i$ | Loss weight |
| $\beta_i$ | KL annealing coefficient |

---

## 4. Latent Factorization Framework

I factorize the latent representation into five independent latent variables:

```python
z_content
z_speaker
z_environment
z_excitation
z_fidelity
```

Each latent variable is modeled through an independent VAE pathway.

---

## 5. Reconstruction Targets

### Magnitude Family

```python
logmel
mr256
mr512
magnitude
mr1024
```

### Phase Family

```python
phase_sin
phase_cos
```

The model does not directly predict:

```python
IF
GroupDelay
```

These quantities are derived inside the loss function.

---

## 6. Phase Reconstruction

Ground-truth phase:

$$
\phi
=
\operatorname{atan2}(s,c)
$$

Predicted phase:

$$
\hat{\phi}
=
\operatorname{atan2}(\hat{s},\hat{c})
$$

where

$$
s=\sin(\phi)
$$

$$
c=\cos(\phi)
$$

---

## 7. Von Mises Phase Objective

I use a Von Mises based reconstruction objective as the primary phase loss.

Von Mises density:

$$
p(\phi|\mu,\kappa)
=
\frac{
e^{\kappa \cos(\phi-\mu)}
}{
2\pi I_0(\kappa)
}
$$

Ignoring constant terms, the negative log-likelihood becomes

$$
L_{\mathrm{VM}}
=
-\cos(\phi-\hat{\phi})
\tag{1}
$$

or equivalently

$$
L_{\mathrm{VM}}
=
1-\cos(\phi-\hat{\phi})
\tag{2}
$$

This acts as the circular analogue of mean squared error.

---

## 8. Auxiliary Sin/Cos Reconstruction Loss

To stabilize early training, I retain a direct reconstruction objective on sine and cosine components.

$$
L_{\mathrm{SC}}
=
\|s-\hat{s}\|_1
+
\|c-\hat{c}\|_1
\tag{3}
$$

---

## 9. Phase Continuity Constraint

Temporal derivative:

$$
\Delta_t \phi
=
\phi_t-\phi_{t-1}
$$

Frequency derivative:

$$
\Delta_f \phi
=
\phi_f-\phi_{f-1}
$$

Continuity loss:

$$
L_{\mathrm{CONT}}
=
\|\Delta_t\phi-\Delta_t\hat{\phi}\|_1
+
\|\Delta_f\phi-\Delta_f\hat{\phi}\|_1
\tag{4}
$$

This discourages unrealistic phase discontinuities.

---

## 10. Instantaneous Frequency Constraint

Instantaneous frequency is derived directly from phase.

$$
IF
=
\frac{\partial \phi}{\partial t}
$$

$$
\widehat{IF}
=
\frac{\partial \hat{\phi}}{\partial t}
$$

Loss:

$$
L_{\mathrm{IF}}
=
\|IF-\widehat{IF}\|_1
\tag{5}
$$

No decoder head is required.

---

## 11. Group Delay Constraint

Group delay is similarly derived from phase.

$$
GD
=
-\frac{\partial \phi}{\partial \omega}
$$

$$
\widehat{GD}
=
-\frac{\partial \hat{\phi}}{\partial \omega}
$$

Loss:

$$
L_{\mathrm{GD}}
=
\|GD-\widehat{GD}\|_1
\tag{6}
$$

---

## 12. Magnitude Reconstruction

The magnitude objective is

$$
L_{\mathrm{MAG}}
=
L_{logmel}
+
L_{mr256}
+
L_{mr512}
+
L_{mr1024}
+
L_{mag}
\tag{7}
$$

Each component is defined as

$$
L_i
=
\lambda_{L1}L1
+
\lambda_{MSE}MSE
\tag{8}
$$

---

## 13. Disentanglement Objectives

I retain the following objectives without modification:

$$
L_{\mathrm{MRSTFT}}
$$

$$
L_{\mathrm{ENV}}
$$

$$
L_{\mathrm{InfoNCE}}
$$

$$
L_{\mathrm{ORTHO}}
$$

$$
L_{\mathrm{TC}}
$$

---

## 14. Factor-Wise KL Divergence

Instead of a single KL regularizer, I use factor-specific KL terms.

$$
\beta_c KL_c
+
\beta_s KL_s
+
\beta_e KL_e
+
\beta_x KL_x
+
\beta_f KL_f
\tag{9}
$$

---

## 15. Cyclical Factor Annealing

For factor $i$,

$$
\beta_i(t)
=
\beta_i^{max}
\cdot
cycle(t)
\tag{10}
$$

where

$$
cycle(t):
0
\rightarrow
1
\rightarrow
0
\rightarrow
1
\rightarrow \cdots
$$

---

## 16. Final Reconstruction Objective

$$
L_{\mathrm{RECON}}
=
L_{\mathrm{MAG}}
+
\lambda_{vm}L_{\mathrm{VM}}
+
\lambda_{sc}L_{\mathrm{SC}}
+
\lambda_{cont}L_{\mathrm{CONT}}
+
\lambda_{if}L_{\mathrm{IF}}
+
\lambda_{gd}L_{\mathrm{GD}}
\tag{11}
$$

---

## 17. Unified Training Objective

$$
\begin{aligned}
L_{\mathrm{TOTAL}}
=&\;
L_{\mathrm{MAG}}
+
\lambda_{vm}L_{\mathrm{VM}}
+
\lambda_{sc}L_{\mathrm{SC}}
+
\lambda_{cont}L_{\mathrm{CONT}}
+
\lambda_{if}L_{\mathrm{IF}}
+
\lambda_{gd}L_{\mathrm{GD}}
\\
&+
\lambda_{mr}L_{\mathrm{MRSTFT}}
+
\lambda_{env}L_{\mathrm{ENV}}
+
\lambda_{info}L_{\mathrm{InfoNCE}}
+
\lambda_{ortho}L_{\mathrm{ORTHO}}
+
\lambda_{tc}L_{\mathrm{TC}}
\\
&+
\beta_c KL_c
+
\beta_s KL_s
+
\beta_e KL_e
+
\beta_x KL_x
+
\beta_f KL_f
\end{aligned}
\tag{12}
$$

---

## 18. Final Decoder Architecture

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

---

## 19. Expected Contributions

1. Physically consistent phase modeling through a unified phase manifold.
2. Derived IF and GD constraints instead of direct regression.
3. Factor-specific KL annealing for stronger disentanglement.
4. Joint magnitude-phase reconstruction framework.
5. Improved interpretability of latent acoustic factors.

---

## References

[1] Takaki et al., Phase Reconstruction from Amplitude Spectrograms Based on Von-Mises-Distribution Deep Neural Network.

[2] Von Mises Distribution.

[3] A Deep Generative Model of Speech Complex Spectrograms.

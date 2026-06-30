This is actually the right point to pause implementation details and understand the entire system as one probabilistic graphical model. Once you understand this diagram, every script becomes obvious.

---

# Entire Architecture (High Level)

```
                 Stage 1
         Raw waveform fragments
                │
                ▼
      9 Acoustic Feature Maps
                │
                ▼
     Feature Group Manager
                │
      ┌─────────┼──────────┬──────────┬──────────┐
      ▼         ▼          ▼          ▼          ▼
  Content    Speaker   Environment Excitation Fidelity
  Encoder     Encoder    Encoder     Encoder     Encoder
      │         │          │            │          │
      ▼         ▼          ▼            ▼          ▼
 (μc,σc)   (μs,σs)   (μe,σe)    (μx,σx)   (μf,σf)
      │         │          │            │          │
      ▼         ▼          ▼            ▼          ▼
     zc        zs         ze           zx         zf
      └─────────┴──────────┴────────────┴──────────┘
                       │
                Joint Latent
      z=[zc,zs,ze,zx,zf]
                       │
               FactorVAE Discriminator
                       │
         Independence / TC Regularization
                       │
                       ▼
                 Decoder Core
                       │
       Identity Backbone (zc,zs)
                       │
       Environment FiLM (ze)
                       │
 Excitation Cross Attention (zx)
                       │
 Fidelity Cross Attention (zf)
                       │
          Hidden Sequence H
                       │
          Reconstruction Heads
                       │
                       ▼
      9 reconstructed acoustic features
```

This is the complete information flow during every forward pass.

---

# Fundamental Idea

Your model assumes

[
x
=

g
(z_c,
z_s,
z_e,
z_x,
z_f)
]

where

* (z_c): linguistic content
* (z_s): speaker identity
* (z_e): recording environment
* (z_x): excitation / glottal behaviour
* (z_f): signal fidelity

Instead of one latent vector,

[
z\in\mathbb R^{384},
]

you explicitly decompose

[
z=
[z_c,z_s,z_e,z_x,z_f].
]

---

# Why Five Encoders?

Each encoder is intentionally biased toward one acoustic factor.

---

## Content Encoder

Input

[
{
LogMel,
MR256,
IF
}
]

Idea

```
phonemes
syllables
prosody
temporal articulation
```

Expected to ignore

* microphone
* room
* speaker

Produces

[
z_c\in\mathbb R^{64}
]

---

## Speaker Encoder

Input

```
MR512
MR256
LogMel
```

Expected to capture

* vocal tract length

* formant structure

* habitual articulation

Produces

[
z_s
\in
\mathbb R^{64}
]

---

## Environment Encoder

Input

```
Magnitude
MR1024
IF
```

Expected to capture

* room impulse response

* reverberation

* microphone coloration

* recording chain

Produces

[
z_e
\in
\mathbb R^{96}
]

Notice

environment gets the largest latent.

That is because environment is expected to contain the richest nuisance information.

---

## Excitation Encoder

Input

```
MODGD
```

Expected to learn

* glottal pulses

* excitation

* voiced/unvoiced structure

Produces

[
z_x
\in
\mathbb R^{32}
]

---

## Fidelity Encoder

Input

```
phase
phase_cos
MR512
MR1024
Magnitude
MODGD
```

Expected to model

small reconstruction details

such as

* phase consistency

* fine spectral detail

* reconstruction sharpness

Produces

[
z_f
\in
\mathbb R^{128}
]

Largest latent besides environment because fidelity reconstruction is information-heavy.

---

# Variational Projection

Each encoder finally outputs one feature vector

[
h_i
]

For example

[
h_c
\in
\mathbb R^{256}
]

Instead of directly using

[
h_c,
]

you learn

[
\mu_c
=====

W_\mu h_c
]

[
\log\sigma_c^2
==============

W_\sigma h_c
]

Then

[
\boxed{
z_c
===

\mu_c
+
\sigma_c\odot\epsilon
}
]

where

[
\epsilon
\sim
\mathcal N(0,I)
]

Purpose

Learn

probability distributions

instead of deterministic embeddings.

---

# Joint Latent

All factors are concatenated

[
z
=

[
z_c,
z_s,
z_e,
z_x,
z_f
]
]

Dimension

[
64
+
64
+
96
+
32
+
128
===

384
]

This joint latent is only used

for FactorVAE.

Decoder never directly consumes this concatenation.

---

# What is FactorVAE?

This is the most important conceptual component.

Normally,

VAE only minimizes

[
KL(q(z|x)|p(z)).
]

This only encourages

each latent individually

to resemble

[
\mathcal N(0,I).
]

It does **not** enforce

independence

between dimensions.

FactorVAE adds

Total Correlation

[
TC(z)
=====

KL
(
q(z)
|
\prod_i q(z_i)
)
]

If

[
TC=0
]

then

all latent dimensions become statistically independent.

That is exactly what disentanglement requires.

---

# FactorVAE Discriminator

How do we estimate

Total Correlation?

We train another network.

Input

real latent

[
z
]

and

dimension-wise shuffled latent

[
\tilde z
]

where

each latent dimension is independently permuted across the batch.

Example

Original

```
Sample1

zc zs ze zx zf

Sample2

zc zs ze zx zf
```

Permuted

```
zc ← sample2

zs ← sample1

ze ← sample5

...
```

Marginals stay identical

but

joint correlations disappear.

---

Discriminator learns

Real

[
q(z)
]

vs

Permuted

[
\prod_j q(z_j)
]

If discriminator cannot distinguish

them,

then

latent factors have become independent.

---

# Decoder Philosophy

Instead of simply concatenating all latents,

decoder reconstructs audio

hierarchically.

---

## Identity Backbone

Uses

only

[
z_c
]

and

[
z_s
]

Identity means

"What is being spoken?"

and

"Who is speaking?"

These are expanded into

a sequence

[
H
\in
\mathbb R^{B\times T\times256}
]

using a Transformer.

Conceptually

[
H_0
===

f(z_c,z_s)
]

---

# Why Expand to Sequence?

Latents are

global

vectors.

Decoder needs

time-varying

features.

So

[
[B,D]
\rightarrow
[B,T,D]
]

via

sequence expansion.

Target length

is

taken from

LogMel.

---

# Environment FiLM

Environment should not overwrite identity.

Instead,

it should

modulate

it.

FiLM computes

[
\gamma
======

W_\gamma z_e
]

[
\beta
=====

W_\beta z_e
]

Then

[
\boxed{
H'
==

\gamma
\odot
H
+
\beta
}
]

Meaning

same speaker

same sentence

different room.

Environment changes

representation

without replacing it.

---

# Excitation Cross Attention

Excitation is

not constant.

It selectively influences

certain frames.

Instead of addition,

use attention.

Memory

comes from

[
z_x
]

Decoder sequence queries it

[
H
\leftarrow
Attention
(
H,
z_x,
z_x
)
]

Meaning

inject excitation only where needed.

---

# Fidelity Cross Attention

Fidelity behaves similarly,

but reconstruction quality is fragile.

Therefore

you use

two paths.

Attention

*

Skip pathway

[
H
=

H
+
Attention
+
Skip(z_f)
]

The skip avoids losing fidelity information through attention bottlenecks.

---

# Decoder Hidden Representation

Finally

[
H
\in
\mathbb R^{B\times T\times256}
]

contains

```
content

+

speaker

+

environment

+

excitation

+

fidelity
```

already fused.

---

# Reconstruction Heads

Instead of reconstructing

waveform,

decoder predicts

all feature representations.

Each head is specialized.

Example

Magnitude

[
H
\rightarrow
\hat M
]

LogMel

[
H
\rightarrow
\widehat{Mel}
]

Phase

[
H
\rightarrow
\sin\phi,
\cos\phi
]

IF

[
H
\rightarrow
\widehat{IF}
]

etc.

Each head simply learns

[
f_i(H)
]

for feature

(i).

---

# Handling Different Decoder Lengths

Hidden sequence initially has

[
T
=

T_{LogMel}
]

Other features have

different frame counts

For example

```
LogMel

49

MR256

97

MR512

49

MR1024

25
```

Before each reconstruction head

hidden sequence is resized

[
H
\in
\mathbb R^{T\times256}
]

↓

linear interpolation

↓

[
H_i
\in
\mathbb R^{T_i\times256}
]

Mathematically

[
H_i
===

Interp(H,T_i)
]

Each head therefore always receives

its own correct temporal resolution.

---

# Training During One Epoch

For every mini-batch,

the following occurs:

1. **Feature grouping:** The batch of feature maps is partitioned into five semantically motivated groups (content, speaker, environment, excitation, fidelity).

2. **Encoding:** Each group passes through its dedicated encoder to produce
   [
   (\mu_i,\log\sigma_i^2,z_i),\qquad i\in{c,s,e,x,f}.
   ]

3. **Latent regularization:** The variational sampling and KL terms encourage each latent distribution to remain close to a standard Gaussian, while the FactorVAE discriminator encourages statistical independence between the different latent dimensions.

4. **Decoding:** The latent factors are fused hierarchically:

   * (z_c,z_s) establish the identity sequence.
   * (z_e) modulates it through FiLM.
   * (z_x) injects excitation information via cross-attention.
   * (z_f) restores fine spectral detail through cross-attention and a gated skip connection.

5. **Feature reconstruction:** The decoder predicts all nine acoustic feature maps at their respective temporal resolutions.

6. **Loss computation:** Reconstruction, KL divergence, Total Correlation (FactorVAE), and any additional regularization losses are computed and summed.

7. **Backpropagation:** Gradients from the combined objective update all trainable modules jointly (encoders, decoder, discriminator, and auxiliary components according to the active training stage).

---

# Staged Activation (Training Strategy)

Although the exact scheduling logic is outside the code shown, staged activation generally means that different objectives or modules become trainable progressively rather than all at once.

Conceptually,

**Stage 1**
[
\text{Encoders} + \text{Decoder}
]

learn stable feature reconstruction.

↓

**Stage 2**

Variational regularization (KL) is increased so the latent spaces become meaningful.

↓

**Stage 3**

FactorVAE discrimination and disentanglement losses are activated, encouraging the latent factors to become statistically independent.

↓

**Final stage**

All objectives are optimized jointly, producing latent spaces that are simultaneously reconstructive, structured, and disentangled.

This progressive schedule is intended to prevent the strong disentanglement objectives from destabilizing feature reconstruction before the encoder–decoder pair has learned a reasonable acoustic representation.

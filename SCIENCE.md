# ConformerFlow: Scientific Background and Design Rationale

This document explains the biological motivation, architectural decisions, and
training choices behind ConformerFlow. It is written for readers who understand
protein structure and biology but may not have deep machine learning background.

---

## Scope and Purpose

ConformerFlow is designed to do one specific thing: given a single input
protein structure (typically from X-ray crystallography), generate a diverse
set of conformations that samples the protein's natural flexibility distribution
as measured by NMR.

**What it is designed for:**

- Drug discovery: sampling cryptic binding pockets that appear only in
  specific conformations. A single crystal structure may not reveal the
  pocket because crystal packing forces the protein into an atypical state.
- NMR structure validation: comparing a generated ensemble against a deposited
  NMR ensemble to check whether the predicted flexibility pattern is realistic.
- Allostery prediction: identifying whether a perturbation at one site
  (mutation, ligand binding) changes the flexibility distribution at a
  distal site.

**What it is not designed for:**

- Protein folding from sequence alone. ConformerFlow requires a starting
  structure and learns conformational variation around it, not ab initio
  structure prediction.
- Predicting absolute free energies or population weights of specific states.
  The model generates conformers that collectively span the likely ensemble,
  but it does not assign thermodynamic probabilities.
- Replacing molecular dynamics for timescale-resolved dynamics. MD samples
  a physical trajectory with explicit solvent and force fields at nanosecond
  to microsecond timescales. ConformerFlow generates an approximate ensemble
  without temporal information.

---

## Biological Motivation

### Why protein conformational ensembles matter

Textbook depictions of proteins as rigid, single structures are a useful
simplification but misleading in practice. Proteins are not static objects:
they continuously fluctuate among many slightly different arrangements, and
these fluctuations are often functionally essential.

Three classes of phenomenon depend directly on conformational diversity:

**Intrinsic disorder**: Many proteins contain regions—sometimes their entire
length—that have no single stable fold. These intrinsically disordered regions
(IDRs) adopt many conformations in solution and can only be described by an
ensemble, not a single structure. IDRs are disproportionately represented in
signalling and transcriptional regulation.

**Allostery**: When a ligand binds at one site and changes the activity at a
spatially distant site, the mechanism is often a shift in the conformational
ensemble rather than a simple rigid-body movement. The bound state is not
created by the ligand; it was already sampled at low frequency before binding.
The ligand shifts the population. This means that computing allostery
requires access to the ensemble, not just the minimum-energy structure.

**Cryptic pockets**: Many drug targets have binding sites that are invisible
in their crystal structures because the pocket only opens in a minority
conformation. High-throughput conformer sampling is now a standard step in
computational drug discovery screens to expose these sites.

### Why NMR is the ground truth for conformational diversity

X-ray crystallography is the dominant method for protein structure
determination because it produces very high-resolution models. However,
crystal structures represent an average over trillions of molecules packed
into a lattice, and the crystal contacts themselves restrict conformational
freedom. The result is a structure that is more rigid than the protein in
solution.

NMR spectroscopy measures proteins in solution, where they move freely.
An NMR structure deposition typically contains 10–40 models (conformers),
each a complete structure consistent with the experimental restraints
(nuclear Overhauser effects, coupling constants, chemical shifts). The spread
across these models directly reflects the conformational heterogeneity that
exists in solution.

This makes NMR ensembles the most appropriate ground truth for training a
model of conformational diversity. ConformerFlow is trained entirely on NMR
ensembles: it learns what flexible and rigid regions look like across thousands
of proteins, and uses this knowledge to predict conformational distributions
for new structures.

### Why CA-only representation is chosen over all-atom

The Cα (alpha carbon) is the backbone atom common to all standard amino acids.
It connects the N-terminal backbone N atom to the C-terminal backbone C atom
and sits at the branch point leading to the side chain.

Representing protein conformations by Cα coordinates (one point per residue)
offers four practical advantages:

1. **Completeness**: CA is present in every standard residue and is never
   disordered in well-resolved structures. Other backbone atoms (N, C) and
   side chain atoms are sometimes absent in lower-resolution structures.

2. **Sufficient resolution for ensemble comparison**: The metrics that matter
   most for conformational ensemble quality—pairwise RMSD, RMSF per residue,
   covariance of motions—are all computable from CA alone. Side chain
   conformations are determined primarily by backbone geometry via rotamer
   libraries; they are a downstream consequence, not an independent degree
   of freedom at the coarse scale the model operates on.

3. **Memory and compute**: An all-atom representation with side chains
   adds roughly 7–15 atoms per residue depending on side chain size, plus
   the associated distance matrices grow as L squared. CA-only keeps
   the representation tractable for proteins up to 800 residues.

4. **Consistency with training data**: NMR models are internally consistent
   for backbone atoms but can differ in reported side chain positions.
   Using CA sidesteps the noise introduced by modelling ambiguous side chains.

During parsing, N, CA, C, and CB coordinates are all extracted and used for
frame construction and the chirality loss, but the generative model predicts
CA positions, and all evaluation metrics are computed on CA coordinates.

---

## Model Architecture

### SE(3) invariance and why it is required

SE(3) is the group of rigid body motions in three dimensions: rotations and
translations. SE(3) invariance means that if you rotate or translate the
entire molecule, the model's output does not change (for scalar outputs like
energies or RMSD) or rotates/translates in the same way (for vector outputs
like coordinates).

This is a physical requirement, not a design choice. Protein conformation is
defined by the relative positions of atoms, not by where the molecule happens
to be placed in the laboratory coordinate system. If you rotate a protein by
90 degrees, its flexibility is unchanged. A model that is not SE(3) invariant
would predict different flexibility depending on the orientation of the input
structure, which is nonsensical.

There are two general strategies for achieving SE(3) invariance in neural
networks. The first is to represent geometry only through invariant quantities:
pairwise distances, angles, and dihedral angles. The second is to build
equivariant networks that explicitly track how features transform under
rotation, and then extract invariants at the output. ConformerFlow uses the
second approach, specifically Invariant Point Attention (IPA) from AlphaFold2.

### How IPA achieves SE(3) invariance

IPA operates on backbone frames: each residue is assigned a local coordinate
system (a 3x3 rotation matrix R and a translation vector t corresponding to
the CA position). Attention is computed using both scalar features and the
positions of learned query/key points in 3D space. These 3D points are placed
in the local frame of each residue and then moved to the global frame for
distance computation:

```
p_global = R^T @ p_local + t
```

Because both query and key points are constructed this way, and pairwise
distances are invariant under global rigid body motion, the attention weights
are SE(3)-invariant. The output aggregated from the attention is then brought
back to each residue's local frame, making the whole operation equivariant.

### Backbone frames: the rows-as-axes convention

ConformerFlow's frame builder constructs a right-handed orthonormal frame at
each residue using Gram-Schmidt orthonormalization on the CA→C and CA→N
vectors:

```
e1 = normalize(CA → C)              # x-axis
e2 = normalize(CA→N projected ⊥ e1) # y-axis
e3 = e1 × e2                        # z-axis (right-handed)
R  = stack([e1, e2, e3], dim=-2)    # rows are frame axes
```

This is the "rows-as-axes" convention (convention B): row 0 of R is the x-axis,
row 1 is the y-axis, row 2 is the z-axis.

The relative frame between residues i and j is computed as:

```
R_rel = R_i @ R_j^T
t_rel = R_i @ (t_j - t_i)
```

To verify that this is SE(3)-invariant, consider a global rotation G applied
to all frames. After the rotation, R_i becomes R_i @ G^T (rotating the
frame axes). Then:

```
R_rel_new = (R_i @ G^T) @ (R_j @ G^T)^T
          = R_i @ G^T @ G @ R_j^T
          = R_i @ R_j^T     (since G^T @ G = I)
```

The global rotation cancels exactly. The same argument applies to t_rel.

An earlier version of the code used `R_i^T @ R_j` instead of `R_i @ R_j^T`.
For the columns-as-axes convention, R_i^T @ R_j is correct. For the
rows-as-axes convention used here, it is not invariant: the global rotation
does not cancel. This was a bug in the frame module; the current implementation
uses the correct `R_i @ R_j^T`.

### d_latent = 16 vs. 128

The latent dimension controls the information bottleneck between the encoder
(which sees all M conformers) and the flow matching head (which generates
new conformers). A smaller latent forces the model to compress the conformational
distribution into fewer numbers.

NMR ensembles of typical proteins have 10–20 meaningful collective motions
(principal components) that account for most of the variance. The remaining
components are noise from inadequate NMR restraints or refinement artifacts.
A latent dimension of 128 would be large enough for the encoder to memorise
each conformer individually rather than learning the underlying distribution,
a failure mode called posterior collapse: the decoder ignores the latent z
because the encoder encodes nothing useful.

d_latent = 16 is chosen because:

- It matches the approximate number of slow collective modes in a typical
  NMR ensemble (10–20 modes contain >90% of variance for most proteins).
- It is too small to memorise individual conformers, forcing genuine
  compression of the flexibility pattern.
- It prevents posterior collapse without requiring extreme KL annealing
  schedules.

The config validator sets d_latent = 16 as the default. Increasing it is
possible but should be accompanied by an increase in lambda_kl or a stronger
annealing schedule to prevent collapse.

### Chirality loss: L-amino acids and the triple product

All 20 standard amino acids in proteins are L-amino acids, defined by the
handedness of the arrangement of substituents around the CA carbon. The CA
carbon is a stereocenter bonded to four distinct groups: the amino group (N),
the carbonyl group (C), the side chain (CB), and the hydrogen. L-amino acids
have a specific spatial arrangement of these groups.

The chirality of the CA carbon can be measured by the scalar triple product
of three vectors emanating from CA:

```
triple = (N - CA) × (C - CA) · (CB - CA)
```

For an L-amino acid, this triple product is positive. For a D-amino acid
(mirror image), it is negative.

The chirality loss penalises any residue where the triple product is negative:

```
L_chirality = mean(relu(-triple)) over all residues with N, CA, C, CB present
```

This loss is zero for clean NMR data (all L-amino acids), so it adds no
gradient noise during normal training. It serves two purposes:

1. **Data quality guard**: If parsing produces coordinates with incorrect
   atom assignments or inverted geometry, the chirality loss will produce
   a non-zero gradient that pushes the model away from those configurations.

2. **Regulariser for non-SE3 encoders**: When using the Cartesian or
   distance matrix encoders (not the default SE3 encoder), the encoder is
   insensitive to the mirror image of a structure. The generative model can
   then produce D-amino acid configurations without incurring any structural
   cost. The chirality loss prevents this.

The weight lambda_chirality = 0.1 is intentionally light. It is a soft
constraint, not a hard one. The model should primarily learn correct geometry
from the data; the chirality loss acts as a fallback.

### EMA decay = 0.9999

During training, the optimizer updates model weights at each step. These
live weights are optimised for the training objective but can be noisy,
especially during the early stages when gradients are large and the learning
rate is high.

Exponential Moving Average (EMA) maintains a shadow copy of the weights that
is updated at each step as:

```
ema_w = 0.9999 * ema_w + 0.0001 * live_w
```

The EMA weights change much more slowly than the live weights. They represent
a smoothed average over roughly the last 10,000 steps (1 / (1 - 0.9999)).

Validation and inference always use EMA weights, not live weights. This is
standard practice for flow matching and diffusion models because:

- EMA weights produce lower variance in outputs: individual training steps
  may push weights into bad configurations that are corrected by subsequent
  steps. The EMA bypasses these transient bad states.
- EMA weights generalise better: they are an average over many training
  states rather than the endpoint of a single optimisation trajectory.
- The benefit is especially large for flow matching, where the learned
  vector field must be smooth and self-consistent. Noisy live weights
  can produce discontinuous vector fields that lead to poor ODE integration.

---

## Generative Model

### Flow matching vs. diffusion

Diffusion models (DDPM) corrupt data by adding Gaussian noise over many steps
and then learn to reverse this process. The forward process follows a
stochastic differential equation (SDE), and the reverse process is also
stochastic (Langevin dynamics). The learned score function approximates the
gradient of the log-density. Because the SDE adds noise at every step, the
trajectory from noise to data is curved and often takes hundreds of steps to
denoise cleanly.

Flow matching (Lipman et al., 2023) learns a vector field that defines an
ordinary differential equation (ODE) instead of an SDE:

```
dx/dt = v_θ(x_t, t)
```

The training target is defined by a simple interpolation: at time t, the
training trajectory is a straight line between a noise sample x_0 and a
data sample x_1:

```
x_t = (1 - t) * x_0 + t * x_1
u_t = x_1 - (1 - σ_min) * x_0  (target vector field)
```

Because the trajectories are straight lines, the vector field is simpler
to learn, fewer integration steps are needed, and the ODE can be integrated
with higher-order methods that exploit smoothness. ConformerFlow achieves
accurate ensembles with 20 ODE steps. DDPM typically needs 100–1000 steps
for equivalent quality.

A further advantage is that flow matching on SE(3) has a clean geometric
interpretation: the model learns to move backbone frames from random noise
configurations to configurations consistent with the protein's conformational
distribution.

### Heun ODE solver

The Euler method integrates the ODE by taking a single step in the direction
of the current vector field:

```
x_{t+dt} = x_t + dt * v_θ(x_t, t)
```

This is first-order accurate: the error per step is proportional to dt.

Heun's method (second-order Runge-Kutta) takes a predictor step, evaluates
the vector field at the predicted point, then uses the average of both
evaluations:

```
x_pred = x_t + dt * v_θ(x_t, t)          # predictor
x_{t+dt} = x_t + 0.5 * dt * (v_θ(x_t, t) + v_θ(x_pred, t+dt))  # corrector
```

This requires two model evaluations per step (twice the compute of Euler)
but achieves second-order accuracy. For smooth vector fields, the error is
proportional to dt squared rather than dt, meaning that halving the step
size reduces error four-fold instead of two-fold. In practice, Heun with
20 steps produces better ensembles than Euler with 40 steps for the same
wall-clock time, because 20 Heun steps (40 evaluations) are competitive with
40 Euler steps (40 evaluations) but much more accurate.

### sigma_min = 0.01

In the flow matching training objective, sigma_min appears in the target
vector field formula:

```
u_t = x_1 - (1 - sigma_min) * x_0
```

Without sigma_min (i.e., sigma_min = 0), the vector field at t=1 would point
exactly to x_1. This means the ODE trajectory collapses to a single point
at t=1, reproducing the training example exactly. During inference this is
a problem because x_0 is a random noise sample, and two different noise
samples would integrate to two different, but potentially very similar, output
structures.

Setting sigma_min = 0.01 keeps a small amount of noise in the target
specification, ensuring that the ODE trajectory at t=1 does not pin
exactly to x_1. The practical effect is that structures generated from
different noise samples remain distinct even at the end of integration,
which is precisely what is needed to produce a diverse ensemble.

---

## Data Pipeline Decisions

### Temporal split with 2020 cutoff

A naive random split of the NMR dataset into train and test would group
proteins deposited at different times but with the same sequence into
different splits. An NMR structure of ubiquitin deposited in 2005 would
train on, and a different NMR structure of ubiquitin deposited in 2018 might
end up in the test set. The model would trivially score well on the test
protein because it already learned its flexibility from the near-identical
training structure.

This is data leakage: the test set is not genuinely unseen.

The temporal split solves this by placing all structures deposited from
2020 onwards into the test set. The model has never seen any structure
deposited after the cutoff date, regardless of sequence similarity. This
tests whether the model generalises to proteins whose structures became
available after its knowledge cutoff, which is the appropriate evaluation
for a generative model intended to be used on new proteins.

The cutoff year of 2020 was chosen because it provides a large enough test
set (the PDB grows by roughly 10,000 entries per year) while leaving the
majority of depositions for training.

### MMseqs2 at 30% sequence identity

Even with a temporal split, proteins with very similar sequences in train and
test would not constitute a fair evaluation. If 95% of the training set
consists of G-protein coupled receptors and the test set contains several
GPCRs from 2020 onwards, the model is effectively tested on sequences it has
seen.

Sequence redundancy removal using MMseqs2 cluster all-vs-all at 30% identity
removes any training entry that shares 30% or more sequence identity with any
validation or test entry. This is the standard cutoff used in structural
bioinformatics benchmarks (CASP, CAMEO) and corresponds roughly to the
boundary beyond which homologous proteins share similar folds.

Note: redundancy removal is applied only to train entries. Val and test
entries are never removed.

### Outlier conformer removal (3-sigma)

NMR structure determination involves computational refinement with restraints
derived from experimental measurements. Most conformers in a deposited
ensemble are physically reasonable, but some may represent poorly converged
structures, residues with unresolved long-range contacts, or computational
artifacts from the refinement algorithm.

The parse step removes conformers whose Cα RMSD from the ensemble mean
exceeds 3 standard deviations. If the mean pairwise RMSD of the ensemble
is 2 Å and the standard deviation is 0.5 Å, a conformer with RMSD 4 Å
from the mean (z-score = 4) is almost certainly an artifact. Keeping such
outliers would bias the model: the loss computed against an outlier conformer
provides no useful gradient because the outlier does not represent the true
conformational distribution of the protein.

At least 2 conformers are always retained regardless of z-scores, to avoid
reducing legitimate small ensembles to a single structure.

### Backbone imputation: 1.46 Å and 1.52 Å bond lengths

Some NMR PDB files contain residues where the N or C backbone atoms are
not listed (missing coordinates). The CA atom is always present when a
residue is included. Missing N and C are common in low-resolution structures
at chain termini or at positions with unresolved density.

Rather than discarding these residues, the parser imputes approximate positions
using ideal backbone bond lengths:

```
N_i  ≈ CA_i - 1.46 Å * normalize(CA_{i-1} → CA_i)
C_i  ≈ CA_i + 1.52 Å * normalize(CA_i → CA_{i+1})
```

The values 1.46 Å (N-CA bond) and 1.52 Å (CA-C bond) are standard ideal
backbone bond lengths from the Engh and Huber parameters used in crystallographic
refinement. The direction is approximated from the CA-CA virtual bond vector,
which deviates from the true bond direction by roughly 15–20 degrees but is
sufficient for frame construction.

Imputed atoms are used only for building the local backbone frame. They do
not contribute to the loss because the atom mask records which atoms were
actually observed.

### Chain selection: longest chain, chain A on tie

Multimeric PDB entries can contain multiple protein chains. ConformerFlow
is designed for monomers: it learns from a single polypeptide chain. The
parser applies a deterministic selection policy: choose the chain with the
most standard amino acids that have a CA atom; on a tie in length, prefer
chain A; on a further tie, take the chain that appears first in the file
(BioPython iteration order).

This policy is logged explicitly in the code because reproducibility requires
that the same PDB entry always produces the same training example. Arbitrary
or random chain selection would make the dataset irreproducible.

The selection of the longest chain gives the biologically most informative
polypeptide in most cases. For a complex of a 200-residue enzyme and a
15-residue inhibitor peptide, the enzyme chain is selected. Modelling the
full flexibility of the small peptide alone would not be scientifically useful.

---

## Training Decisions

### lambda_kl = 0.01

The KL divergence loss penalises the learned posterior q(z|x) for deviating
from the prior p(z) = N(0, I). Without this penalty, the encoder would learn
to encode each conformer in a distinct, non-overlapping region of latent space,
providing no generalisation at inference time (where z is sampled from the
prior, not from a specific conformer).

However, a strong KL penalty early in training prevents the encoder from
learning anything useful: if the KL penalty dominates, the model collapses
to q(z|x) = N(0, I) regardless of x, which means the latent z carries no
information about the conformational distribution. The decoder then learns
to ignore z.

A weight of lambda_kl = 0.01 is a deliberately small value. It allows the
encoder to learn to separate different conformational distributions in latent
space during the early epochs, when the flow matching loss is also learning
what valid backbone geometry looks like. Once the model has learned to
generate reasonable conformers, the KL term provides mild pressure toward
a well-structured prior without collapsing the posterior.

### lambda_chirality = 0.1

The chirality loss is a soft constraint. A hard constraint (very large lambda)
would force every generated backbone to have perfect L-amino acid chirality,
but this would interfere with early training when the model is also learning
what correct backbone geometry looks like more broadly. The model needs space
to explore geometry before chirality is tightly enforced.

A weight of 0.1 makes the chirality penalty comparable in magnitude to the
geometry loss (lambda_geometry = 0.1) and smaller than the flow and ensemble
losses. In practice, well-trained models have near-zero chirality loss because
they learn L-amino acid geometry from the data directly; the chirality term
is an additional signal that penalises the rare cases where the generative
model proposes a mirror-image configuration.

### free_bits = 0.5

The free bits technique prevents over-regularisation of latent dimensions
that the model has already learned well. The KL loss for a single latent
dimension d is:

```
KL_d = -0.5 * (1 + log_var_d - mu_d^2 - exp(log_var_d))
```

Free bits sets a minimum threshold: if KL_d < 0.5 nats, it is treated as
already compressed and no additional penalty is applied. Only dimensions
where KL_d > 0.5 are penalised.

Without free bits, the KL loss would push every latent dimension toward
exactly KL_d = 0 (posterior = prior). But some latent dimensions encode
genuinely useful conformational information, and forcing them to zero
destroys that information. Free bits allows the model to use latent dimensions
freely up to 0.5 nats each, after which the KL penalty kicks in.

The value 0.5 nats corresponds to a latent posterior that has half the
variance of the prior (sigma ≈ 0.7). This is a moderate compression; the
dimension still carries information but is constrained.

### Warmup + cosine decay

The learning rate schedule uses linear warmup for the first 1000 steps,
followed by cosine decay to a floor of 1e-6 over 100,000 steps.

Linear warmup is necessary because, at the start of training, the backbone
frame module has random weights. Gradients of the flow matching loss are
large and unstructured because the model predicts essentially random vector
fields. A large learning rate during this phase would cause catastrophic
updates that destroy the initialisation of the IPA layers, which rely on
careful attention weight initialisation. Warmup ramps the effective learning
rate up slowly so that the frame module has time to stabilise before large
updates are applied.

After warmup, cosine decay gradually reduces the learning rate. This has
been empirically shown to produce better generalisation than constant learning
rate schedules in transformer models because it allows the model to converge
more gently in the later stages of training, when the main task is fine-tuning
the distribution rather than learning broad patterns.

---

## Evaluation Choices

### Coverage RMSD and precision RMSD together

Coverage RMSD measures, for each NMR conformer, the distance to the closest
generated conformer. Low coverage RMSD means the model produces at least one
generated conformer near every NMR conformer.

Precision RMSD measures, for each generated conformer, the distance to the
closest NMR conformer. Low precision RMSD means the model does not produce
many conformers that are far from any NMR conformer.

Neither metric alone is sufficient:

- A model that generates a single conformer close to every NMR conformer
  would have excellent coverage (it covers the entire NMR space) but
  terrible precision for all other generated conformers that land far from
  the NMR ensemble.
- A model that generates many conformers that are all near one NMR conformer
  (mode collapse to the most common conformation) would have good precision
  but poor coverage of the full ensemble.

Reporting both metrics forces the model to simultaneously cover the NMR
ensemble and avoid generating physically implausible structures. They are
the protein conformer equivalents of recall and precision in classification.

### RMSF correlation vs. mean RMSD

Mean RMSD between predicted and NMR ensembles is easy to compute but does
not capture whether the model gets the flexibility pattern right. Two
structures could have the same mean RMSD while one places the flexible
region in loops and the other in helices.

Root Mean Square Fluctuation (RMSF) is the per-residue standard deviation
of Cα position across an ensemble. High RMSF at a given residue means that
position is flexible across conformers; low RMSF means it is rigid.

The RMSF profile of a protein is its fingerprint of flexibility. Two
ensembles can be compared not by their overall RMSD but by how well their
RMSF profiles correlate. A Pearson correlation of 0.9 between predicted
and NMR RMSF profiles means that the model correctly identifies which
residues are flexible and which are rigid, even if the absolute magnitudes
differ. This is more scientifically meaningful than a single RMSD number.

The Spearman correlation is also reported, which measures only the rank order
of flexibility across residues and is insensitive to the scale of RMSF values.
High Spearman correlation means the model correctly orders residues from most
to least flexible.

### Ramachandran favored fraction

The Ramachandran plot maps the phi (φ) and psi (ψ) backbone dihedral angles
for each residue. These angles are not arbitrary: the local geometry of the
polypeptide chain restricts which combinations of φ and ψ are physically
accessible without steric clashes between backbone atoms.

Approximately 98% of residues in high-quality crystal structures fall in the
favoured Ramachandran regions (alpha-helix core, beta-sheet, PPII helix, and
left-handed helix). Structures with many residues outside the allowed regions
are physically unlikely and typically indicate a modelling error.

The Ramachandran favoured fraction is therefore a direct measure of whether
the generated backbone geometries are physically reasonable. A model that
generates ensembles with 90% Ramachandran-favoured residues is producing
broadly plausible structures. A model with 60% favoured residues is generating
many physically impossible conformations.

### Clash score per 100 residues (MolProbity convention)

Steric clashes occur when two non-bonded atoms overlap. In a protein structure,
a clash between two CA atoms means their Cα-Cα distance falls below the van
der Waals contact distance (approximately 3.5 Å for non-bonded Cα pairs).

The clash score is the number of such clashes per 100 residues, averaged
across all conformers:

```
clash_score_per100 = total_clashes / N_conformers / L_residues * 100
```

Dividing by the number of residues (and multiplying by 100) makes proteins
of different lengths directly comparable. A 50-residue protein might have
3 clashes and a 500-residue protein might have 30 clashes; both have a
clash score of 6 per 100 residues, indicating the same density of problematic
contacts. This is the convention used by MolProbity (the standard structure
quality assessment server in the field) and is adopted here for consistency
with published benchmarks.

The validation pipeline computes clash scores on the generated Cα ensemble.
Because ConformerFlow generates only Cα positions (not all-atom structures),
the clash score specifically measures backbone-level clashes, not side chain
clashes. A score near zero is expected for a well-trained model; a high score
indicates that the ODE integration produced physically unreasonable geometries.

---

## Summary of Key Design Parameters

| Parameter | Value | Rationale |
|---|---|---|
| d_latent | 16 | Matches ~10-20 slow collective modes of NMR ensembles |
| sigma_min | 0.01 | Prevents ODE trajectory collapse at t=1 |
| ema_decay | 0.9999 | Smooths inference over ~10,000 training steps |
| lambda_kl | 0.01 | Prevents posterior collapse without destroying information |
| lambda_chirality | 0.1 | Light soft constraint; model learns geometry from data first |
| free_bits | 0.5 nats | Allows latent dims to carry information before KL kicks in |
| test_cutoff_year | 2020 | Temporal split prevents data leakage across time |
| cluster_identity | 0.30 | Standard homology cutoff for unbiased benchmarking |
| outlier_z | 3.0 | Removes >3-sigma NMR conformers (artifacts, not biology) |
| N_bond (imputed) | 1.46 Å | Engh-Huber ideal N-CA bond length |
| C_bond (imputed) | 1.52 Å | Engh-Huber ideal CA-C bond length |
| warmup_steps | 1000 | Allows frame module to stabilise before large gradient updates |
| ODE method | Heun | Second-order accuracy with 2x evaluations per step |
| ODE steps | 20 | Sufficient for Heun; DDPM would need 100-1000 |

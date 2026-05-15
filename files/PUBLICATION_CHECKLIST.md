# ConformerFlow — Publication Blockers Checklist

Priority tiers:
  P0 = results are meaningless without this fix
  P1 = core claims cannot be made without this
  P2 = reviewers will reject without this
  P3 = required for submission, not blocking analysis
  P4 = strengthens paper, not strictly required

─────────────────────────────────────────────────────────────────
P0 — Results Are Meaningless Without This
─────────────────────────────────────────────────────────────────

[ ] P0-1  Sequence redundancy removal
          Problem:  No MMseqs2 clustering between train and test.
                    Model memorises seen folds. All eval metrics
                    are inflated.
          Fix:      Run MMseqs2 at 30% sequence identity. Remove
                    test/val proteins that share a cluster with any
                    training protein. Re-split.
          Files:    data/filter.py — add --cluster_identity flag
                    and MMseqs2 subprocess call.

[ ] P0-2  Diversity loss always zero during training
          Problem:  Every training run logs diversity_loss=0.0000.
                    If generated conformers are all identical the
                    model has no scientific utility, and the loss
                    weight is doing nothing.
          Fix:      Diagnose: (a) log pairwise RMSD of generated
                    conformers at each val step; (b) check
                    min_spread threshold logic in losses.py;
                    (c) check whether generate() is actually
                    producing different samples or returning the
                    same tensor N times.
          Files:    training/losses.py  diversity_loss()
                    model/model_factory.py  ConformerFlowModel.forward()

[ ] P0-3  EMA (exponential moving average) of weights missing
          Problem:  Flow matching and diffusion papers universally
                    use EMA for inference. Training weights
                    oscillate — EMA weights give ~5-15% better
                    metrics. Without EMA, reported numbers are
                    systematically below what the method can do.
          Fix:      Add EMA wrapper in Trainer.__init__ and update
                    EMA after every optimizer step. Save EMA weights
                    separately in checkpoint. Use EMA weights for
                    all validation and inference.
          Files:    training/trainer.py

─────────────────────────────────────────────────────────────────
P1 — Core Claims Cannot Be Made Without This
─────────────────────────────────────────────────────────────────

[ ] P1-1  SE(3) equivariance not numerically verified
          Problem:  Equivariance is assumed but untested. One subtle
                    bug in IPA or Gram-Schmidt breaks the whole claim.
          Fix:      Add equivariance unit test: rotate input coords
                    by random R ∈ SO(3), verify encoder output
                    transforms by the same R. Run for all 4 encoders.
                    SE(3) encoder must pass; others are expected to fail
                    (document this explicitly).
          Files:    New file: tests/test_equivariance.py

[ ] P1-2  Per-residue latent design unresolved
          Problem:  z has shape (B, L, d_latent) — a per-residue
                    latent. This is unusual. With d_latent=128 and
                    L=200 that is 25,600 latent dimensions. KL
                    regularisation is extremely hard at this scale.
                    Model will either ignore the latent (posterior
                    collapse) or overfit to it. Neither is publishable.
          Fix:      Make an explicit design decision with ablation:
                    Option A — keep per-residue latent but reduce
                              d_latent to 8-16 and add strong KL weight.
                    Option B — use a global latent z ∈ R^d (standard
                              for conformer VAE/flow papers).
                    Report both in ablation table.
          Files:    model/ensemble_stats.py  DistributionSampler
                    model/generative_models.py  _DenoisingTransformer

[ ] P1-3  Chirality blind for non-SE3 encoders
          Problem:  Cartesian and distance matrix encoders cannot
                    distinguish L from D amino acids. Model may
                    generate mirror-image proteins. This would invalidate
                    any claim about those encoder variants.
          Fix:      Add chirality loss: cross product of N→CA and
                    CA→C vectors must have consistent sign per residue.
                    OR restrict non-SE3 encoders to paired with SE3
                    flow matching decoder which preserves chirality.
                    OR document the limitation explicitly if only
                    reporting SE3 frames results.
          Files:    training/losses.py — add chirality_loss()

[ ] P1-4  Outlier conformer removal missing
          Problem:  Individual NMR models can be misfolded artifacts.
                    Including them corrupts the ensemble statistics
                    the model learns from.
          Fix:      Per-conformer RMSD vs ensemble mean. Remove
                    conformers > 2 standard deviations from mean RMSD.
                    Log how many conformers are removed per structure.
          Files:    data/parse_nmr.py — add outlier filtering step

[ ] P1-5  Missing backbone atom handling is silent
          Problem:  Some NMR conformers have missing N, CA, or C atoms
                    mid-chain. The mask may be wrong and coordinates
                    may be zero, silently corrupting training signal.
          Fix:      Add per-residue atom completeness check. Exclude
                    any residue where backbone N, CA, or C is missing
                    from the mask. Log missing-atom rates per structure.
          Files:    data/parse_nmr.py

─────────────────────────────────────────────────────────────────
P2 — Reviewers Will Reject Without This
─────────────────────────────────────────────────────────────────

[ ] P2-1  No temporal split
          Problem:  Random train/val/test split. Structures deposited
                    after a date cutoff must be the test set to simulate
                    real prospective use. Standard requirement for all
                    structural prediction papers post-AlphaFold2.
          Fix:      Add deposition date filter to fetch_pdb.py.
                    Cutoff suggestion: train on structures deposited
                    before 2020-01-01, test on 2020 onwards.
          Files:    data/fetch_pdb.py — query RCSB deposition_date
                    data/filter.py — --date_cutoff flag

[ ] P2-2  Ensemble RMSD coverage metric not implemented
          Problem:  This is the primary metric in every competitor
                    paper (EigenFold, Str2Str, FoldFlow). Fraction of
                    NMR conformers within 2 Å RMSD of at least one
                    generated conformer.
          Fix:      Implement in metrics.py. Report mean ± std across
                    test proteins. Also report at 1 Å, 2 Å, 3 Å
                    thresholds.
          Files:    evaluation/metrics.py — add coverage_rmsd()

[ ] P2-3  Ramachandran validation missing
          Problem:  No check that generated backbone torsions are in
                    physically allowed regions. Papers report %
                    Ramachandran favoured/allowed/outlier from MolProbity.
          Fix:      Compute φ/ψ for all generated conformers using the
                    vectorised dihedral code already in encoders.py.
                    Report % in favoured (>98% expected for good models),
                    allowed, and outlier regions.
          Files:    evaluation/metrics.py — add ramachandran_stats()

[ ] P2-4  Clash score missing
          Problem:  No check for atomic overlaps in generated structures.
                    Standard MolProbity clash score required by
                    structural biology journals.
          Fix:      Compute all-atom pairwise distances. Count pairs
                    < (sum of van der Waals radii - 0.4 Å). Report
                    clashes per 1000 atoms. Target < 20 (good model)
                    or < 10 (excellent).
          Files:    evaluation/metrics.py — add clash_score()

[ ] P2-5  Random seed not fixed
          Problem:  Results not reproducible run-to-run. Reviewers and
                    replicators cannot reproduce numbers from the paper.
          Fix:      Set torch.manual_seed, numpy.random.seed, random.seed
                    at training start. Add --seed argument to train.py.
                    Hash config + seed into checkpoint filename.
          Files:    scripts/train.py
                    training/trainer.py

[ ] P2-6  Chain selection undefined for multi-chain NMR
          Problem:  Multi-chain NMR structures — which chain is used?
                    Not documented. Different choices give different
                    residue counts and flexibility profiles.
          Fix:      Define explicit policy: longest chain, or chain
                    matching a minimum residue count, or all chains
                    concatenated with chain break masking.
                    Document in paper methods section.
          Files:    data/parse_nmr.py

─────────────────────────────────────────────────────────────────
P3 — Required for Submission
─────────────────────────────────────────────────────────────────

[ ] P3-1  Baseline implementations
          Required comparisons:
            - EigenFold  (Jing et al., 2023)
            - Str2Str    (Lu et al., 2023)
            - FoldFlow   (Stark et al., 2023)
            - AlphaFold2 B-factors as trivial flexibility predictor
            - Gaussian noise scaled to local B-factor (trivial baseline)
          Fix:      Run inference from each baseline's public code on
                    the same test set. Use identical evaluation code.

[ ] P3-2  Ablation table
          Must ablate each component independently:
            - SE3 frames vs cartesian vs distance vs torsion encoder
            - Flow matching vs DDPM vs VAE vs score matching
            - Full covariance vs diagonal vs PCA latent
            - With vs without diversity loss
            - With vs without KL regularisation
            - With vs without geometry loss
          Fix:      Standardise a single evaluation script that takes
                    a checkpoint and returns all metrics. Run for each
                    ablation variant.

[ ] P3-3  Gradient monitoring per component
          Problem:  If flow loss gradient is 1000x geometry gradient,
                    one dominates and the others do nothing. Currently
                    not tracked.
          Fix:      Log per-component gradient norm in Trainer._train_step.
                    Useful for diagnosing instability and justifying
                    loss weight choices in the paper.
          Files:    training/trainer.py

[ ] P3-4  Post-generation energy minimisation
          Problem:  Raw model output will have bad bond lengths and
                    angles. All published structure prediction models
                    run at least one round of energy minimisation
                    before reporting metrics.
          Fix:      Add optional OpenMM relaxation step in
                    inference/sample.py. Report metrics before and
                    after relaxation separately.
          Files:    inference/sample.py — add --relax flag

[ ] P3-5  RCSB search API fix
          Problem:  fetch_pdb.py search query returns 400 Bad Request.
                    Full dataset cannot be fetched automatically.
          Fix:      Update RCSB search query to current API format.
                    Test with curl before committing.
          Files:    data/fetch_pdb.py  query_nmr_entries()
                    data/fetch_pdb.py  query_xray_entries()

─────────────────────────────────────────────────────────────────
P4 — Strengthens Paper
─────────────────────────────────────────────────────────────────

[ ] P4-1  MD simulation comparison
          Run 100 ns GROMACS simulation on 5-10 test proteins.
          Compare ensemble statistics. Shows model captures
          physics without explicit force field.

[ ] P4-2  Diversity vs accuracy tradeoff curve
          Vary N (number of generated conformers) from 1 to 100.
          Plot coverage vs self-diversity. Shows model generates
          meaningful new conformers not just copies.

[ ] P4-3  Latent space interpolation demo
          For VAE variant: interpolate in latent space between two
          known conformations. Should produce physically reasonable
          intermediate states. Strong visual for paper figure.

[ ] P4-4  Conditional generation (known binding site)
          Condition generation on a specific residue range being
          in a particular state. Demonstrates controllability.

[ ] P4-5  Uncertainty calibration
          Does the model's ensemble spread correlate with actual
          experimental B-factors across proteins? Plot scatter.
          Well-calibrated model = scatter follows y=x line.

[ ] P4-6  Hyperparameter sensitivity report
          Vary lambda_kl (0.001 to 0.1), lambda_diversity (0.1 to 1.0),
          d_latent (16 to 256), n_flow_layers (2 to 12).
          Show results are stable across a reasonable range.

─────────────────────────────────────────────────────────────────
Summary — order to address
─────────────────────────────────────────────────────────────────

1.  P0-2   Diversity loss = 0  (diagnose first, model may be broken)
2.  P0-3   Add EMA weights
3.  P1-1   Equivariance test
4.  P1-2   Per-residue latent redesign decision
5.  P0-1   Sequence redundancy removal (MMseqs2)
6.  P2-1   Temporal split
7.  P1-4   Outlier conformer removal
8.  P1-5   Missing backbone atom handling
9.  P2-2   Ensemble RMSD coverage metric
10. P2-3   Ramachandran validation
11. P2-4   Clash score
12. P1-3   Chirality loss for non-SE3 encoders
13. P2-5   Random seed control
14. P2-6   Chain selection policy
15. P3-5   RCSB search API fix
16. P3-3   Gradient monitoring
17. P3-1   Baseline comparisons
18. P3-2   Ablation table
19. P3-4   Post-generation energy minimisation
20. P4-*   Strengthening experiments (as time allows)

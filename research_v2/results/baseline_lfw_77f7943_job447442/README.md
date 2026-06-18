# Baseline snapshot — LFW (job 447442, git sha 77f7943)

Frozen copy of the metric artefacts from the **pre-scale-up** PARAM A100 run.
Kept so the upcoming CASIA-WebFace / IR-100 / DDP scale-up can overwrite the
live `research_v2/results/*` files without losing this reference point.

- Dataset: LFW (`min_faces_per_person=8`, ~13k images, 173 train identities)
- Hardware: 1x NVIDIA A100-SXM4-40GB, torch 2.5.1+cu121, python 3.11.15
- Git sha: `77f7943`, SLURM job 447442

## Headline numbers (this snapshot)
- MDIE-full pooled AUC **0.979** (occlusion 0.975, lighting 0.980); best
  comparably-trained baseline 0.749.
- Ablation ordering full > noICCL = noAMD > noRATA (every component helps).
- Real transfer (AUC): MeGlass 0.824, MFR2 0.734, CALFW 0.557, AgeDB-30 0.594 —
  MDIE wins every benchmark.
- ArcFace-compat proof: `ALL_PASS = true` (512-d, unit-norm, single forward,
  deterministic, masked-self 0.917 > imposter 0.750).

## NOTE on paper.tex
`research_v2/paper/paper.tex` Table `tab:real` currently cites numbers from an
earlier/stronger run (MeGlass 0.926, MFR2 0.826, CALFW 0.599, AgeDB-30 0.683).
Those do NOT match this snapshot; the qualitative claim ("MDIE wins every
benchmark") still holds, but the magnitudes differ. Reconcile before submission.

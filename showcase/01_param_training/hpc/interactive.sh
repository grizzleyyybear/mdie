#!/bin/bash
# ============================================================================
#  Interactive A100 shell on PARAM Siddhi-AI (per user-guide §8.3.2).
#  Drops you onto a compute node with 1 A100-SXM4. Max walltime 4 h
#  (advisory 3: interactive cap reduced from 7 d → 4 h).
#
#  Usage:  bash hpc/interactive.sh
#  Then inside the shell:
#      source $HOME/Conda/bin/activate mdie
#      cd $HOME/mdie
#      python -m research_v2.src.preflight
# ============================================================================
exec srun --partition=dgxnp \
          --nodes=1 --ntasks-per-node=16 \
          --gres=gpu:A100-SXM4:1 \
          --time=01:00:00 \
          --pty /bin/bash

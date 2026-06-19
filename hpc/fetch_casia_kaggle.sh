#!/bin/bash
# ============================================================================
#  Fetch CASIA-WebFace from Kaggle DIRECTLY ON THE PARAM LOGIN NODE.
#
#  PARAM *compute* nodes have no internet, but the *login* node does (the same
#  way `git pull` works there). This script downloads the Kaggle mirror
#  `debarghamitraroy/casia-webface` (the genuine CASIA-WebFace: Institute of
#  Automation, Chinese Academy of Sciences) with kagglehub, locates the
#  per-identity ImageFolder inside it, and symlinks it into the exact path the
#  trainer reads:  research_v2/datasets_cache/casia/<identity>/<img>.jpg
#
#  Run this on the LOGIN NODE (NOT inside an sbatch job):
#       cd ~/mrinal/projects/mdie
#       source ~/Conda/bin/activate mdie          # or your env
#       bash hpc/fetch_casia_kaggle.sh
#
#  One-time Kaggle auth (kagglehub needs an API token):
#       # On kaggle.com -> Settings -> "Create New API Token" downloads kaggle.json
#       mkdir -p ~/.kaggle
#       # paste/scp kaggle.json to ~/.kaggle/kaggle.json, then:
#       chmod 600 ~/.kaggle/kaggle.json
#       # (alternatively export KAGGLE_USERNAME=... KAGGLE_KEY=... )
#
#  Nothing about the existing pipeline changes: this only populates the staged
#  CASIA dir that `--dataset casia` already expects. LFW stays the default.
# ============================================================================
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
CACHE_ROOT="$REPO_ROOT/research_v2/datasets_cache"
DST="$CACHE_ROOT/casia"
SLUG="debarghamitraroy/casia-webface"
mkdir -p "$CACHE_ROOT"

# Keep kagglehub's cache on the big project filesystem (not a quota'd ~/.cache).
export KAGGLEHUB_CACHE="${KAGGLEHUB_CACHE:-$CACHE_ROOT/_kaggle}"
mkdir -p "$KAGGLEHUB_CACHE"

# ---- Already staged? -------------------------------------------------------
if [[ -e "$DST" && ! -L "$DST" ]]; then
    if find "$DST" -mindepth 2 \( -iname '*.jpg' -o -iname '*.png' \) -print -quit 2>/dev/null | grep -q .; then
        echo "[casia] already staged at $DST (real dir with images) — nothing to do."
        exit 0
    fi
fi

# ---- Credentials check -----------------------------------------------------
if [[ ! -f "$HOME/.kaggle/kaggle.json" && ( -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ) ]]; then
    echo "[casia] ERROR: no Kaggle credentials found."
    echo "        Put your token at ~/.kaggle/kaggle.json (chmod 600), OR export"
    echo "        KAGGLE_USERNAME and KAGGLE_KEY. Get the token from kaggle.com ->"
    echo "        Settings -> 'Create New API Token'."
    exit 1
fi
if [[ -f "$HOME/.kaggle/kaggle.json" ]]; then chmod 600 "$HOME/.kaggle/kaggle.json" 2>/dev/null || true; fi

# ---- Ensure kagglehub is importable ---------------------------------------
if ! python -c "import kagglehub" 2>/dev/null; then
    echo "[casia] installing kagglehub (login node has internet) ..."
    pip install -q kagglehub || { echo "[casia] ERROR: pip install kagglehub failed."; exit 1; }
fi

# ---- Download + locate the ImageFolder root -------------------------------
echo "[casia] downloading '$SLUG' via kagglehub into $KAGGLEHUB_CACHE ..."
echo "        (~5 GB; resumes if interrupted — just re-run this script)"
IMAGEFOLDER_ROOT="$(python - "$SLUG" <<'PY'
import os, sys
import kagglehub

slug = sys.argv[1]
path = kagglehub.dataset_download(slug)   # auto-resumes, returns extracted dir

IMG_EXT = (".jpg", ".jpeg", ".png")

def has_direct_image(d):
    try:
        for f in os.scandir(d):
            if f.is_file() and f.name.lower().endswith(IMG_EXT):
                return True
    except OSError:
        pass
    return False

def identity_score(d, cap=40):
    """How many immediate subdirs of d directly contain an image (capped)."""
    c = 0
    try:
        for sub in os.scandir(d):
            if sub.is_dir() and has_direct_image(sub.path):
                c += 1
                if c >= cap:
                    break
    except OSError:
        pass
    return c

# BFS to depth 3 from the download path; pick the dir whose immediate children
# look most like per-identity image folders.
best_dir, best_score = path, identity_score(path)
frontier, depth = [path], 0
while frontier and depth < 3:
    nxt = []
    for d in frontier:
        try:
            for sub in os.scandir(d):
                if sub.is_dir():
                    s = identity_score(sub.path)
                    if s > best_score:
                        best_dir, best_score = sub.path, s
                    nxt.append(sub.path)
        except OSError:
            pass
    frontier, depth = nxt, depth + 1

if best_score < 5:
    sys.stderr.write(
        f"[casia] could not find a per-identity ImageFolder under {path} "
        f"(best score {best_score}). Inspect it manually.\n")
    sys.exit(2)

print(os.path.realpath(best_dir))
PY
)"

if [[ -z "$IMAGEFOLDER_ROOT" || ! -d "$IMAGEFOLDER_ROOT" ]]; then
    echo "[casia] ERROR: failed to resolve the ImageFolder root."
    exit 1
fi
echo "[casia] ImageFolder root: $IMAGEFOLDER_ROOT"

# ---- Symlink into the path the trainer reads ------------------------------
[[ -L "$DST" ]] && rm -f "$DST"
if [[ -e "$DST" ]]; then
    # An empty real dir from a prior partial run — safe to replace.
    rmdir "$DST" 2>/dev/null || {
        echo "[casia] ERROR: $DST exists and is not empty/symlink. Move it aside first."; exit 1; }
fi
ln -s "$IMAGEFOLDER_ROOT" "$DST"
echo "[casia] linked $DST -> $IMAGEFOLDER_ROOT"

# ---- Validate via the project loader --------------------------------------
echo "[casia] verifying with the project loader ..."
PYTHONPATH="$REPO_ROOT" python - <<'PY'
from pathlib import Path
from research_v2.src.config import DATA_DIR
from research_v2.src.data import build_train_dataset
paths, labels, names = build_train_dataset("casia", DATA_DIR, min_imgs=4)
print(f"  [casia] loader sees {len(paths)} images across {len(names)} identities")
assert len(names) > 1000, "expected thousands of identities — check the dataset"
PY

cat <<MSG

[casia] OK. CASIA is staged at $DST
[next] Smoke, then full run (all on the login node; they sbatch):
   DATASET=casia EPOCHS_S2=2 sbatch hpc/slurm_fanout_train.sh   # 2-epoch smoke
   DATASET=casia bash hpc/submit_fanout.sh                      # full fan-out
   # or one big DDP IR-100 run:
   GPUS=4 DATASET=casia BACKBONE=ir100 sbatch hpc/slurm_ddp_full.sh
MSG

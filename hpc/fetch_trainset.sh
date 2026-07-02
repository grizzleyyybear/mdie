#!/bin/bash
# ============================================================================
#  Fetch a LARGE face-recognition training set on the PARAM LOGIN NODE and
#  stage it as the ImageFolder the trainer reads:
#       research_v2/datasets_cache/<DATASET>/<identity>/<img>.jpg
#
#  PARAM *compute* nodes have no internet; the *login* node does. Run this on
#  the LOGIN NODE (not in an sbatch job). It auto-activates the mdie conda env,
#  so you can just:
#       cd ~/mrinal/projects/mdie
#       bash hpc/fetch_trainset.sh ms1mv3          # dataset name is positional
#  or the env-var form:
#       DATASET=glint360k bash hpc/fetch_trainset.sh
#
#  Pick ONE source (checked in this order):
#    1. SRC=/abs/path        an already-downloaded RecordIO dir (train.rec +
#                            train.idx) OR a per-identity ImageFolder.
#    2. KAGGLE_SLUG=owner/ds  download via kagglehub (needs ~/.kaggle/kaggle.json
#                            or KAGGLE_USERNAME/KAGGLE_KEY).
#    3. HF_REPO=org/name     download a HuggingFace *dataset* repo snapshot
#                            (needs `pip install huggingface_hub`; set HF_TOKEN
#                            for gated repos).
#
#  Known public large sets (verify the exact slug/repo before using — mirrors
#  move around). All ship as InsightFace RecordIO, which this script converts
#  to ImageFolder with the bundled pure-python extractor (no mxnet):
#    * Glint360K  ~17M imgs / 360k IDs   (largest; best identity diversity)
#    * MS1M-V3    ~5.1M imgs / 93k IDs    (the standard ArcFace training set)
#    * WebFace4M  ~4.2M imgs / 205k IDs   (most identities, very clean)
#
#  With 1 TB of project storage and A100s, Glint360K is the recommended set:
#  big enough that you may even unfreeze the backbone (UNFREEZE=1 in
#  hpc/slurm_train_specialist.sh) at a small LR without catastrophic forgetting.
#
#  Nothing else in the pipeline changes: this only populates
#  datasets_cache/<DATASET>, which `--dataset <DATASET>` already resolves.
# ============================================================================
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
CACHE_ROOT="$REPO_ROOT/research_v2/datasets_cache"
# Dataset name may be given positionally ($1) or via the DATASET env var.
DATASET="${1:-${DATASET:-glint360k}}"
DST="$CACHE_ROOT/$DATASET"
mkdir -p "$CACHE_ROOT"

# ---- Ensure the mdie conda env is active -----------------------------------
# This script needs the PROJECT's Python (huggingface_hub / kagglehub / the
# loader). The PARAM login node's default `python` is Python 2, so if we're not
# already inside the mdie env, activate it the same workspace-aware way the
# SLURM prelude does. Idempotent: a no-op when the env is already active.
ENV_NAME="${ENV_NAME:-mdie}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
    _grand="$(cd "$REPO_ROOT/../.." 2>/dev/null && pwd || echo "")"
    _parent="$(cd "$REPO_ROOT/.." 2>/dev/null && pwd || echo "")"
    if [[ -n "$_grand" && "$_grand" == "$HOME"/* \
          && "$(basename "$_parent")" == "projects" ]]; then
        _PREFIX_DEFAULT="$_grand"
    else
        _PREFIX_DEFAULT="$HOME"
    fi
    CONDA_PREFIX_DIR="${CONDA_PREFIX_DIR:-$_PREFIX_DEFAULT/Conda}"
    if [[ -f "$CONDA_PREFIX_DIR/bin/activate" ]]; then
        echo "[fetch] activating conda env '$ENV_NAME' from $CONDA_PREFIX_DIR"
        # shellcheck disable=SC1091
        source "$CONDA_PREFIX_DIR/bin/activate" "$ENV_NAME"
    else
        echo "[fetch] WARNING: conda not found at $CONDA_PREFIX_DIR/bin/activate"
        echo "        Falling back to PATH python: $(command -v python) ($(python --version 2>&1))"
        echo "        If import errors follow, activate the env first:"
        echo "            source <your-Conda>/bin/activate mdie"
    fi
fi
echo "[fetch] python: $(command -v python)  ($(python --version 2>&1))"
if ! python -c 'import sys; raise SystemExit(0 if sys.version_info[0] >= 3 else 1)' 2>/dev/null; then
    echo "[fetch] ERROR: 'python' is Python 2 and no mdie conda env was found." >&2
    echo "        Activate your env, then re-run, e.g.:" >&2
    echo "            source ~/mrinal/Conda/bin/activate mdie" >&2
    echo "            bash hpc/fetch_trainset.sh $DATASET" >&2
    exit 1
fi

# ---- CDAC proxy for outbound HTTP (login node reaches internet via proxy) ---
# Auto mode only sets the proxy if a probe through it actually works, so this is
# harmless if the node has direct internet. Disable with PARAM_USE_PROXY=0.
PARAM_PROXY="${PARAM_PROXY:-http://proxy-10g.10g.siddhi.param:9090}"
_use_proxy="${PARAM_USE_PROXY:-auto}"
if [[ "$_use_proxy" != "0" ]]; then
    if [[ "$_use_proxy" == "1" ]] || \
       curl -fsS --max-time 5 --proxy "$PARAM_PROXY" -o /dev/null https://huggingface.co 2>/dev/null; then
        export http_proxy="$PARAM_PROXY" https_proxy="$PARAM_PROXY"
        export HTTP_PROXY="$PARAM_PROXY" HTTPS_PROXY="$PARAM_PROXY"
        echo "[fetch] using CDAC proxy $PARAM_PROXY"
    fi
fi

# --- Validated public presets ----------------------------------------------
# If no explicit SRC/KAGGLE_SLUG/HF_REPO is given, map the well-known DATASET
# names to verified, non-gated HuggingFace mirrors (InsightFace data, aligned
# 112x112). Override any of these by exporting HF_REPO/KAGGLE_SLUG/SRC yourself.
#   glint360k : gaunernst/glint360k-wds-gz  17.09M imgs / 360k IDs  (WebDataset)
#   ms1mv3    : gaunernst/ms1mv3-recordio    5.18M imgs /  93k IDs  (RecordIO)
#   ms1mv3wds : gaunernst/ms1mv3-wds         5.18M imgs /  93k IDs  (WebDataset)
if [[ -z "${SRC:-}" && -z "${KAGGLE_SLUG:-}" && -z "${HF_REPO:-}" ]]; then
    case "$DATASET" in
        glint360k) HF_REPO="gaunernst/glint360k-wds-gz" ;;
        ms1mv3)    HF_REPO="gaunernst/ms1mv3-recordio" ;;
        ms1mv3wds) HF_REPO="gaunernst/ms1mv3-wds" ;;
    esac
fi

# Keep download caches on the big project filesystem (not quota'd ~/.cache).
export KAGGLEHUB_CACHE="${KAGGLEHUB_CACHE:-$CACHE_ROOT/_kaggle}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/_hf}"
mkdir -p "$KAGGLEHUB_CACHE" "$HF_HOME"

# ---- Already staged? -------------------------------------------------------
if [[ -e "$DST" && ! -L "$DST" ]]; then
    if find "$DST" -mindepth 2 \( -iname '*.jpg' -o -iname '*.png' \) -print -quit 2>/dev/null | grep -q .; then
        echo "[fetch] $DATASET already staged at $DST — nothing to do."
        exit 0
    fi
fi

# ---- Resolve a download root from SRC / KAGGLE_SLUG / HF_REPO ---------------
DL_ROOT=""
if [[ -n "${SRC:-}" ]]; then
    [[ -d "$SRC" ]] || { echo "[fetch] ERROR: SRC=$SRC is not a directory."; exit 1; }
    echo "[fetch] using local SRC=$SRC"
    DL_ROOT="$SRC"
elif [[ -n "${KAGGLE_SLUG:-}" ]]; then
    if [[ ! -f "$HOME/.kaggle/kaggle.json" && ( -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ) ]]; then
        echo "[fetch] ERROR: no Kaggle credentials. Put ~/.kaggle/kaggle.json (chmod 600)"
        echo "        or export KAGGLE_USERNAME and KAGGLE_KEY."
        exit 1
    fi
    [[ -f "$HOME/.kaggle/kaggle.json" ]] && chmod 600 "$HOME/.kaggle/kaggle.json" 2>/dev/null || true
    python -c "import kagglehub" 2>/dev/null || { echo "[fetch] installing kagglehub ..."; pip install -q kagglehub; }
    echo "[fetch] downloading Kaggle '$KAGGLE_SLUG' into $KAGGLEHUB_CACHE (resumes if interrupted) ..."
    DL_ROOT="$(python - "$KAGGLE_SLUG" <<'PY'
import sys, kagglehub
print(kagglehub.dataset_download(sys.argv[1]))
PY
)"
elif [[ -n "${HF_REPO:-}" ]]; then
    python -c "import huggingface_hub" 2>/dev/null || { echo "[fetch] installing huggingface_hub ..."; pip install -q huggingface_hub; }
    echo "[fetch] downloading HuggingFace dataset '$HF_REPO' into $HF_HOME (resumes if interrupted) ..."
    DL_ROOT="$(python - "$HF_REPO" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=sys.argv[1], repo_type="dataset",
                        token=os.environ.get("HF_TOKEN")))
PY
)"
else
    echo "[fetch] ERROR: set exactly one of SRC=, KAGGLE_SLUG=, or HF_REPO=."
    echo "        e.g. DATASET=glint360k KAGGLE_SLUG=owner/glint360k bash hpc/fetch_trainset.sh"
    exit 1
fi

[[ -n "$DL_ROOT" && -d "$DL_ROOT" ]] || { echo "[fetch] ERROR: download root not resolved: '$DL_ROOT'"; exit 1; }

# ---- Resolve to a RecordIO dir or a per-identity ImageFolder ----------------
RESOLVED="$(python - "$DL_ROOT" <<'PY'
import os, sys
root = sys.argv[1]
IMG_EXT = (".jpg", ".jpeg", ".png")

def find_recordio(r):
    frontier, depth = [r], 0
    while frontier and depth <= 5:
        nxt = []
        for d in frontier:
            try:
                entries = list(os.scandir(d))
            except OSError:
                continue
            names = {e.name for e in entries if e.is_file()}
            if "train.rec" in names and "train.idx" in names:
                return d
            nxt += [e.path for e in entries if e.is_dir()]
        frontier, depth = nxt, depth + 1
    return None

def has_direct_image(d):
    try:
        for f in os.scandir(d):
            if f.is_file() and f.name.lower().endswith(IMG_EXT):
                return True
    except OSError:
        pass
    return False

def identity_score(d, cap=40):
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

rec = find_recordio(root)
if rec:
    print("REC\t" + os.path.realpath(rec)); sys.exit(0)

def find_webdataset(r):
    """Return the dir holding *.tar / *.tar.gz shards, if any (BFS, depth<=5)."""
    frontier, depth = [r], 0
    while frontier and depth <= 5:
        nxt = []
        for d in frontier:
            try:
                entries = list(os.scandir(d))
            except OSError:
                continue
            if any(e.is_file() and (e.name.endswith(".tar") or e.name.endswith(".tar.gz"))
                   for e in entries):
                return d
            nxt += [e.path for e in entries if e.is_dir()]
        frontier, depth = nxt, depth + 1
    return None

wds = find_webdataset(root)
if wds:
    print("WDS\t" + os.path.realpath(wds)); sys.exit(0)

best_dir, best_score = root, identity_score(root)
frontier, depth = [root], 0
while frontier and depth < 4:
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
    sys.stderr.write(f"[fetch] neither RecordIO nor ImageFolder under {root} "
                     f"(best score {best_score}). Inspect manually.\n")
    sys.exit(2)
print("IMG\t" + os.path.realpath(best_dir))
PY
)"

KIND="${RESOLVED%%$'\t'*}"
PAYLOAD="${RESOLVED#*$'\t'}"
[[ -n "$KIND" && -n "$PAYLOAD" && -d "$PAYLOAD" ]] || { echo "[fetch] ERROR: could not resolve payload."; exit 1; }

if [[ "$KIND" == "REC" ]]; then
    echo "[fetch] InsightFace RecordIO at $PAYLOAD"
    echo "[fetch] extracting to ImageFolder at $DST (pure-python, no mxnet; this is the slow step) ..."
    [[ -L "$DST" ]] && rm -f "$DST"
    TMP_OUT="$DST.partial"
    rm -rf "$TMP_OUT"
    PYTHONPATH="$REPO_ROOT" python "$REPO_ROOT/hpc/recordio_to_imagefolder.py" "$PAYLOAD" "$TMP_OUT"
    rm -rf "$DST" 2>/dev/null || true
    mv "$TMP_OUT" "$DST"
elif [[ "$KIND" == "WDS" ]]; then
    echo "[fetch] WebDataset shards at $PAYLOAD"
    echo "[fetch] extracting to ImageFolder at $DST (pure-python tar/gzip; slow on millions of images) ..."
    [[ -L "$DST" ]] && rm -f "$DST"
    TMP_OUT="$DST.partial"
    rm -rf "$TMP_OUT"
    PYTHONPATH="$REPO_ROOT" python "$REPO_ROOT/hpc/webdataset_to_imagefolder.py" "$PAYLOAD" "$TMP_OUT"
    rm -rf "$DST" 2>/dev/null || true
    mv "$TMP_OUT" "$DST"
else
    echo "[fetch] ImageFolder root: $PAYLOAD"
    [[ -L "$DST" ]] && rm -f "$DST"
    if [[ -e "$DST" ]]; then
        rmdir "$DST" 2>/dev/null || { echo "[fetch] ERROR: $DST exists and is not empty/symlink."; exit 1; }
    fi
    ln -s "$PAYLOAD" "$DST"
    echo "[fetch] linked $DST -> $PAYLOAD"
fi

# ---- Validate via the project loader --------------------------------------
echo "[fetch] verifying with the project loader ..."
PYTHONPATH="$REPO_ROOT" DATASET="$DATASET" python - <<'PY'
import os
from research_v2.src.config import DATA_DIR
from research_v2.src.data import build_train_dataset
ds = os.environ["DATASET"]
paths, labels, names = build_train_dataset(ds, DATA_DIR, min_imgs=4)
print(f"  [fetch] loader sees {len(paths)} images across {len(names)} identities")
assert len(names) > 1000, "expected many identities — check the dataset"
PY

cat <<MSG

[fetch] OK. $DATASET is staged at $DST
[next] (all on the login node; the sbatch jobs run on compute nodes)
   # smoke test (2 epochs):
   DATASET=$DATASET EPOCHS_S2=2 sbatch hpc/slurm_train_specialist.sh
   # full specialist run (frozen backbone + residual fusion):
   DATASET=$DATASET sbatch hpc/slurm_train_specialist.sh
   # ambitious "beat production on clean too" run (unfreeze backbone, tiny LR):
   DATASET=$DATASET UNFREEZE=1 sbatch hpc/slurm_train_specialist.sh
MSG

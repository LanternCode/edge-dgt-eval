#!/bin/bash
#SBATCH --job-name=minfill_launcher
#SBATCH --account=...
#SBATCH --partition=...
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

SUBMIT_DIR="$SLURM_SUBMIT_DIR"
REPO_ROOT="$(realpath "$SUBMIT_DIR/..")"

# Builder path: one directory above submit dir (works if you submit from ./sbatch)
BUILDER_MODULE="scripts.true_minfill_dataset_builder"
if [[ ! -f "$REPO_ROOT/scripts/true_minfill_dataset_builder.py" ]]; then
  echo "ERROR: builder script not found at: $REPO_ROOT/scripts/true_minfill_dataset_builder.py"
  exit 1
fi

OUT_DIR="$REPO_ROOT/data/minfill_true_1000"
LOG_DIR="$REPO_ROOT/sbatch/logs/minfill_true_1000"
mkdir -p "$OUT_DIR" "$LOG_DIR"

CONDA_ENV="bench"

GRAPHS_TOTAL=1000
MIN_NODES=6
MAX_NODES=140
ER_P=0.30

SOLVER="cbc"
SOLVER_TIME_LIMIT_S=30
MAX_CUTS=400
MAX_ATTEMPTS_PER_GRAPH=80

NUM_SHARDS=100
MAX_IN_FLIGHT=25

GEN_CPUS_PER_TASK=16
GEN_MEM=96G
GEN_TIME=06:00:00

FEAT_CPUS_PER_TASK=8
FEAT_MEM=64G
FEAT_TIME=03:00:00

MERGE_CPUS_PER_TASK=4
MERGE_MEM=32G
MERGE_TIME=01:00:00

GEN_JOBID=$(sbatch --parsable \
  --account=... \
  --job-name=minfill_gen \
  --partition=... \
  --nodes=1 --ntasks=1 \
  --cpus-per-task="$GEN_CPUS_PER_TASK" \
  --mem="$GEN_MEM" \
  --time="$GEN_TIME" \
  --array="0-$((NUM_SHARDS-1))%${MAX_IN_FLIGHT}" \
  --output="${LOG_DIR}/%x-%A_%a.out" \
  --error="${LOG_DIR}/%x-%A_%a.err" \
  --export=NONE \
  --wrap="
    set -euo pipefail
    cd \"${REPO_ROOT}\"
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate ${CONDA_ENV}
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    python -m \"${BUILDER_MODULE}\" generate \
      --out_dir \"${OUT_DIR}\" \
      --graphs ${GRAPHS_TOTAL} \
      --min_nodes ${MIN_NODES} --max_nodes ${MAX_NODES} \
      --er_p ${ER_P} \
      --solver ${SOLVER} \
      --solver_time_limit_s ${SOLVER_TIME_LIMIT_S} \
      --max_cuts ${MAX_CUTS} \
      --max_attempts_per_graph ${MAX_ATTEMPTS_PER_GRAPH} \
      --workers ${GEN_CPUS_PER_TASK}
  ")

echo "Submitted generate array job: ${GEN_JOBID}"

FEAT_JOBID=$(sbatch --parsable \
  --account=... \
  --job-name=minfill_feat \
  --partition=... \
  --nodes=1 --ntasks=1 \
  --cpus-per-task="$FEAT_CPUS_PER_TASK" \
  --mem="$FEAT_MEM" \
  --time="$FEAT_TIME" \
  --dependency="afterok:${GEN_JOBID}" \
  --array="0-$((NUM_SHARDS-1))%${MAX_IN_FLIGHT}" \
  --output="${LOG_DIR}/%x-%A_%a.out" \
  --error="${LOG_DIR}/%x-%A_%a.err" \
  --export=NONE \
  --wrap="
    set -euo pipefail
    cd \"${REPO_ROOT}\"
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate ${CONDA_ENV}
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    python -m \"${BUILDER_MODULE}\" features \
      --out_dir \"${OUT_DIR}\" \
      --feature_set canonical
  ")

echo "Submitted features array job: ${FEAT_JOBID}"

MERGE_JOBID=$(sbatch --parsable \
  --account=... \
  --job-name=minfill_merge \
  --partition=... \
  --nodes=1 --ntasks=1 \
  --cpus-per-task="$MERGE_CPUS_PER_TASK" \
  --mem="$MERGE_MEM" \
  --time="$MERGE_TIME" \
  --dependency="afterok:${FEAT_JOBID}" \
  --output="${LOG_DIR}/%x-%j.out" \
  --error="${LOG_DIR}/%x-%j.err" \
  --export=NONE \
  --wrap="
    set -euo pipefail
    cd \"${REPO_ROOT}\"
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate ${CONDA_ENV}
    python -m \"${BUILDER_MODULE}\" merge \
      --out_dir \"${OUT_DIR}\" \
      --out_jsonl min_fill_dataset.jsonl \
      --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
  ")

echo "Submitted merge job: ${MERGE_JOBID}"

echo ""
echo "All submitted:"
echo "  Generate: ${GEN_JOBID}"
echo "  Features: ${FEAT_JOBID}"
echo "  Merge:    ${MERGE_JOBID}"
echo ""
echo "Final outputs will be in:"
echo "  ${OUT_DIR}/min_fill_dataset.jsonl"
echo "  ${OUT_DIR}/min_fill_dataset.jsonl.features.pt"

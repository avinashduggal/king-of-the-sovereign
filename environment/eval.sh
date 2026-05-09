#!/usr/bin/env bash
# Evaluate every trained model in checkpoints/ and save results to eval.log
# alongside model.zip and train.log in each checkpoint directory.
#
# Usage:
#   bash eval.sh [--force]
#
# Options:
#   --force   Re-evaluate models that already have an eval.log
#
# Environment variables:
#   EPISODES  Number of evaluation episodes per model (default: 50)
#   SEED      Random seed (default: 2025)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINTS_DIR="$(cd "$SCRIPT_DIR/../checkpoints" && pwd)"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"

EPISODES="${EPISODES:-50}"
SEED="${SEED:-2025}"
FORCE=0

for arg in "$@"; do
    [[ "$arg" == "--force" ]] && FORCE=1
done

# ---------------------------------------------------------------------------
# extract_preset <dir_name> <algo_prefix>
# Strips the algo prefix then greedily matches the longest known preset.
# ---------------------------------------------------------------------------
extract_preset() {
    local name="$1" prefix="$2"
    local rest="${name#${prefix}_}"
    for p in no_occupation_cost no_neutral_posture no_legitimacy baseline full; do
        if [[ "$rest" == "${p}_"* || "$rest" == "$p" ]]; then
            echo "$p"
            return
        fi
    done
    echo "full"  # fallback
}

# ---------------------------------------------------------------------------
# skip_ts <dir_name>  — returns 0 (skip) if timesteps < 10000
# ---------------------------------------------------------------------------
skip_ts() {
    local name="$1"
    if [[ "$name" =~ _([0-9]+)ts$ ]]; then
        local ts="${BASH_REMATCH[1]}"
        (( ts < 10000 )) && return 0
    fi
    return 1
}

echo "============================================================"
echo " Sovereign model evaluation"
echo " Checkpoints : $CHECKPOINTS_DIR"
echo " Episodes    : $EPISODES   Seed: $SEED"
[[ $FORCE -eq 1 ]] && echo " Mode        : --force (overwriting existing eval.log)"
echo "============================================================"
echo ""

passed=0
skipped=0
failed=0

for checkpoint_dir in "$CHECKPOINTS_DIR"/*/; do
    [[ -d "$checkpoint_dir" ]] || continue
    model_zip="$checkpoint_dir/model.zip"
    [[ -f "$model_zip" ]] || continue

    dir_name="$(basename "$checkpoint_dir")"

    # Skip tiny test runs
    if skip_ts "$dir_name"; then
        echo "SKIP  $dir_name  (< 10 000 timesteps)"
        (( skipped++ )) || true
        continue
    fi

    eval_log="$checkpoint_dir/eval.log"

    # Skip already-evaluated unless --force
    if [[ -f "$eval_log" && $FORCE -eq 0 ]]; then
        echo "SKIP  $dir_name  (eval.log exists; use --force to re-run)"
        (( skipped++ )) || true
        continue
    fi

    echo "--> $dir_name"

    # Route to the right evaluator based on directory-name prefix
    set +e
    case "$dir_name" in
        ppo_*|a2c_*|dqn_*|qrdqn_*|recppo_*)
            (cd "$SCRIPTS_DIR" && python evaluate.py "$checkpoint_dir" \
                --episodes "$EPISODES" --seed "$SEED") \
                2>&1 | tee "$eval_log"
            exit_code="${PIPESTATUS[0]}"
            ;;
        gnn_ppo_*)
            preset="$(extract_preset "$dir_name" "gnn_ppo")"
            (cd "$SCRIPTS_DIR" && python evaluate_ppo_gnn.py \
                --checkpoint "$model_zip" \
                --preset "$preset" \
                --episodes "$EPISODES" \
                --seed "$SEED") \
                2>&1 | tee "$eval_log"
            exit_code="${PIPESTATUS[0]}"
            ;;
        gat_ppo_*)
            preset="$(extract_preset "$dir_name" "gat_ppo")"
            (cd "$SCRIPTS_DIR" && python evaluate_ppo_gat.py \
                --checkpoint "$model_zip" \
                --preset "$preset" \
                --episodes "$EPISODES" \
                --seed "$SEED") \
                2>&1 | tee "$eval_log"
            exit_code="${PIPESTATUS[0]}"
            ;;
        recgat_ppo_*)
            preset="$(extract_preset "$dir_name" "recgat_ppo")"
            (cd "$SCRIPTS_DIR" && python evaluate_recgat_ppo.py \
                --checkpoint "$model_zip" \
                --preset "$preset" \
                --episodes "$EPISODES" \
                --seed "$SEED") \
                2>&1 | tee "$eval_log"
            exit_code="${PIPESTATUS[0]}"
            ;;
        *)
            echo "WARNING: unknown algo prefix in '$dir_name', skipping"
            (( skipped++ )) || true
            set -e
            continue
            ;;
    esac
    set -e

    if [[ "$exit_code" -eq 0 ]]; then
        (( passed++ )) || true
    else
        echo "ERROR: evaluation failed for $dir_name (exit $exit_code)"
        # Remove partial eval.log so a re-run will retry this model
        rm -f "$eval_log"
        (( failed++ )) || true
    fi
    echo ""
done

echo "============================================================"
echo " Done.  passed=$passed  skipped=$skipped  failed=$failed"
echo "============================================================"

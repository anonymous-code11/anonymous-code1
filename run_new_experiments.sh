
set -e
cd "$(dirname "$0")"

CONDA_ENV="hniti"
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

run_step() {
    local step="$1"
    local script="$2"
    local logfile="$LOG_DIR/${script%.py}.log"
    echo ""
    echo "=================================================="
    echo "  STEP $step: $script"
    echo "  Log: $logfile"
    echo "=================================================="
    conda run -n "$CONDA_ENV" python "$script" 2>&1 | tee "$logfile"
    echo "  [DONE] $script"
}

echo "Starting new experiments at $(date)"
echo "Conda env: $CONDA_ENV"

run_step 1 cross_domain_matrix.py
run_step 2 latency_benchmark.py
run_step 3 intervention_v2.py

echo ""
echo "=================================================="
echo "All experiments completed at $(date)"
echo "Results in:"
echo "  ./results/cross_domain/"
echo "  ./results/latency/"
echo "  ./results/intervention_v2/"
echo "Figures in:"
echo "  ./figures/cross_domain_matrix.png"
echo "  ./figures/latency_benchmark.png"
echo "  ./figures/intervention_v2.png"
echo "=================================================="

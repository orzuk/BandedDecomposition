#!/bin/bash
#
# Submit multiple SLURM jobs for finance_example.py with different parameters
# Usage: ./submit_jobs.sh [--dry-run]
#
# Parameters to sweep:
#   - model: mixed_fbm (fbm optional)
#   - n: 200, 500, 1000
#   - alpha: 0.2, 1.0, 5.0
#   - H: 0.01 to 0.99 with step 0.01
#

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORK_DIR="/sci/labs/orzuk/orzuk/github/BandedDecomposition"
CONDA_ENV="pymol-env"

# Parameter arrays
N_VALUES=(200 500 1000)
ALPHA_VALUES=(0.2 1.0 5.0)
MODELS=("mixed_fbm" "fbm")  # Both models

# H range
HMIN=0.01
HMAX=0.99
HRES=0.01

# SLURM settings (adjust based on n)
get_slurm_settings() {
    local n=$1
    local model=$2

    if [ "$n" -le 200 ]; then
        echo "02:00:00 8G 4"  # time mem cpus
    elif [ "$n" -le 500 ]; then
        echo "06:00:00 16G 4"
    else
        echo "12:00:00 32G 4"
    fi
}

# Check for dry run
DRY_RUN=false
if [ "$1" == "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - No jobs will be submitted ==="
    echo ""
fi

# Create jobs directory
JOBS_DIR="${SCRIPT_DIR}/slurm_jobs"
mkdir -p "$JOBS_DIR"

# Counter for jobs
job_count=0

echo "=============================================="
echo "Submitting finance_example.py parameter sweep"
echo "=============================================="
echo ""
echo "Models: ${MODELS[*]}"
echo "N values: ${N_VALUES[*]}"
echo "Alpha values: ${ALPHA_VALUES[*]}"
echo "H range: ${HMIN} to ${HMAX} (step ${HRES})"
echo ""

for model in "${MODELS[@]}"; do
    for n in "${N_VALUES[@]}"; do
        for alpha in "${ALPHA_VALUES[@]}"; do
            # Get SLURM settings based on n
            read -r time mem cpus <<< $(get_slurm_settings $n $model)

            # Create job name
            job_name="${model}_n${n}_a${alpha}"
            job_script="${JOBS_DIR}/${job_name}.sh"

            # Determine strategy based on model
            if [ "$model" == "fbm" ]; then
                strategy="both"  # markovian + full
            else
                strategy="both"  # sum + markovian + full (sum always computed)
            fi

            # Create SLURM job script
            cat > "$job_script" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${JOBS_DIR}/${job_name}_%j.out
#SBATCH --error=${JOBS_DIR}/${job_name}_%j.err
#SBATCH --time=${time}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${mem}

cd ${WORK_DIR}
source ~/.bashrc
conda activate ${CONDA_ENV}

echo "Starting job: ${job_name}"
echo "Parameters: model=${model}, n=${n}, alpha=${alpha}, strategy=${strategy}"
echo "H range: ${HMIN} to ${HMAX} (step ${HRES})"
echo "Time: \$(date)"
echo ""

python finance_example.py \\
    --model ${model} \\
    --n ${n} \\
    --alpha ${alpha} \\
    --strategy ${strategy} \\
    --method lbfgs \\
    --hmin ${HMIN} \\
    --hmax ${HMAX} \\
    --hres ${HRES} \\
    --incremental \\
    --max-cond 1e8

echo ""
echo "Job completed at: \$(date)"
EOF

            chmod +x "$job_script"

            # Submit or print
            if [ "$DRY_RUN" == "true" ]; then
                echo "[DRY RUN] Would submit: $job_name (time=$time, mem=$mem, cpus=$cpus)"
            else
                sbatch "$job_script"
                echo "Submitted: $job_name (time=$time, mem=$mem, cpus=$cpus)"
            fi

            ((job_count++))
        done
    done
done

echo ""
echo "=============================================="
echo "Total jobs: $job_count"
if [ "$DRY_RUN" == "true" ]; then
    echo "Run without --dry-run to submit jobs"
fi
echo "=============================================="
echo ""
echo "Job scripts saved to: $JOBS_DIR"
echo "Results will be saved to: ${WORK_DIR}/results/all_results.csv"
echo ""
echo "To monitor jobs: squeue -u \$USER"
echo "To cancel all: scancel -u \$USER"

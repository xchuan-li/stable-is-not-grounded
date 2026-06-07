#!/bin/bash -l
# run_agreement.sbatch — exp09 cross-lingual agreement grid on FAU NHR (Alex).
# Eval is light (small causal LMs, minimal-pair surprisal); one MIG slice is plenty.
#   ssh v123be54@alex.nhr.fau.de
#   cd .../experiments/paper1/09_crosslingual_agreement
#   sbatch run_agreement.sbatch
# Adjust --partition / module / venv to your environment before submitting.
#SBATCH --job-name=xling-agrmt
#SBATCH --gres=gpu:a40:1          # or: --partition=a100mig --gres=gpu:a100:1g.10gb
#SBATCH --time=02:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

set -euo pipefail
cd "$(dirname "$0")"

# --- environment (edit to match your setup) ---------------------------------
# module load python/3.12-conda && conda activate pytorch    # example
# or: source ~/venvs/agrmt/bin/activate
python -c "import torch; print('cuda', torch.cuda.is_available())"

export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error

# --- 0) sanity: metric logic ------------------------------------------------
python run_crosslingual_agreement.py --selftest

# --- 1) build normalized data (expects CLAMS/Marvin-Linzen TSVs present) -----
# Convert your suites first, e.g.:
#   python prepare_clams.py --tsv clams_en.tsv --lang en --out data/en.jsonl
#   python prepare_clams.py --tsv clams_de.tsv --lang de --out data/de.jsonl
# Falls back to the demo set so the job never dies on missing data:
[ -f data/en.jsonl ] || { python prepare_clams.py --demo; cp data/en_demo.jsonl data/en.jsonl; cp data/de_demo.jsonl data/de.jsonl; echo "WARN: using DEMO data — replace with real suites"; }

# --- 2) score the en+de x {>=2 architectures} grid --------------------------
declare -A MODELS=(
  [en]="gpt2 distilgpt2 EleutherAI/pythia-160m"
  [de]="dbmdz/german-gpt2 ai-forever/mGPT"
)
for lang in en de; do
  for model in ${MODELS[$lang]}; do
    echo "=== $lang :: $model ==="
    python run_crosslingual_agreement.py --pairs "data/${lang}.jsonl" \
        --model "$model" --lang "$lang" --tag zeroshot || echo "FAILED: $lang $model"
  done
done

echo "done. summaries in results/agreement_*_zeroshot.json"

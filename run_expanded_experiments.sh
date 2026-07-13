
set -e
cd /home/pzh/analysis/origin
mkdir -p logs

echo "============================================"
echo "Step 1: expand_domains.py (7域矩阵)"
echo "Started: $(date)"
echo "============================================"
conda run -n hniti python expand_domains.py > logs/expand_domains.log 2>&1
echo "expand_domains done: $(date)"

echo "============================================"
echo "Step 2: subspace_analysis.py"
echo "Started: $(date)"
echo "============================================"
conda run -n hniti python subspace_analysis.py > logs/subspace_analysis.log 2>&1
echo "subspace_analysis done: $(date)"

echo "============================================"
echo "Step 3: fewshot_adaptation.py"
echo "Started: $(date)"
echo "============================================"
conda run -n hniti python fewshot_adaptation.py > logs/fewshot_adaptation.log 2>&1
echo "fewshot_adaptation done: $(date)"

echo "============================================"
echo "ALL DONE: $(date)"
echo "============================================"

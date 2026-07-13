
set -e
cd "$(dirname "$0")"

echo "=== Step 1: Extract hidden states ==="
python extract_hidden.py

echo ""
echo "=== Step 2: Analyze subspace ==="
python analyze_subspace.py

echo ""
echo "=== Done. Check ./figures/ ==="
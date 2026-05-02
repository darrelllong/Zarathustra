#!/usr/bin/env bash
# Check sweep state on vinge
sweep_output=$(ssh -i ~/.ssh/id_rsa -A vinge.local "if [ -f /tmp/cp_r223_adj.log ]; then tail -30 /tmp/cp_r223_adj.log; fi; ps -ef | grep -E 'neural_atlas|train\.py|frozen_sweep' | grep -v grep")
echo "$sweep_output"

# Determine if R224 sweep finished
if echo "$sweep_output" | grep -q "R224 sweep finished"; then
  # Edit RESPONSE-LLNL.md: add a line under Closure section
  if grep -q "## Closure" RESPONSE-LLNL.md; then
    sed -i '/^## Closure$/a\\Round 224 closure added' RESPONSE-LLNL.md
  else
    echo "## Closure\nRound 224 closure added" >> RESPONSE-LLNL.md
  fi
  git add RESPONSE-LLNL.md
  git commit -m "Add Round 224 closure"
  git add -u
git push origin main
fi

# Advance next-move list (assumes next_moves.txt contains current move name)
if [ -f next_moves.txt ]; then
  current=$(cat next_moves.txt)
else
  echo "R224" > next_moves.txt
  current="R224"
fi
case "$current" in
  R224) echo "R225 alibaba more-data retrain" > next_moves.txt ;;
  R225) echo "R226 CP hp re-sweep" > next_moves.txt ;;
  R226) echo "R227 tencent ceiling experiments" > next_moves.txt ;;
  *) echo "R224" > next_moves.txt ;;
esac

git add next_moves.txt
git commit -m "Advance next-move to $(cat next_moves.txt)"

git add -u
git push origin main

#!/bin/bash
# Replicate Wang/Khor/Desnoyers Table 3 — the 8 trace profiles they fit
# in the EUROSYS 2026 paper. Generate synthetic traces using their exact
# (P_IRM, g, f) triplets, evaluate LRU HRC, compare to our race-protocol numbers.
set -eo pipefail
cd /home/darrell/Zarathustra
PY=~/llgan-env/bin/python

# Paper Table 1 footprint and length (Million):
# Trace ID | Length N      | Footprint M
# w11      | 296,893,045   | 2,992,519
# w24      |  81,762,918   |16,487,648
# w44      |  25,257,814   | 3,679,382
# w82      |  14,198,758   |   189,785
# v521     |   5,974,956   |   158,018
# v538     |1,204,044,775  |33,006,370
# v766     |   3,335,779   |   124,146
# v827     |   3,198,158   |   851,527
#
# Paper Table 3 fitted profiles:
# Trace ID | P_IRM | g            | f
# w11      | 1.0   | Zipf(1.3)    | None
# w24      | 0.45  | Zipf(1.2)    | fgen(30,[9,13,17,19],5e-3)
# w44      | 0.0   | None         | fgen(30,[9,13,17,19],2.5e-2)
# w82      | 0.2   | Zipf(1.2)    | fgen(100,[12,13,19],1e-3)
# v521     | 0.0   | None         | fgen(100,[2],2e-3)
# v538     | 0.1   | Zipf(1.2)    | fgen(40,[3,4],5e-3)
# v766     | 0.0   | None         | fgen(40,[0,5],5.7e-3)
# v827     | 0.2   | Zipf(1.2)    | fgen(60,[0,13],5e-3)
#
# Paper §5.1: each trace is regenerated at small footprint M=100 and length
# N=10k (except w44 which needs M≥10k and N≥1m). We use M=10000 and
# N=1,000,000 for all to match our race-protocol scale and remain comparable.
#
# Output: per-trace 2DIO synthetic + LRU HRC at our standard cache sizes.
OUT=/tiamat/zarathustra/llgan-output/desnoyers/2dio_table3
mkdir -p $OUT
M=10000
N=1000000

run_profile() {
  local name=$1 p_irm=$2 g=$3 f=$4 p_inf=$5
  local seed=42
  local fake=$OUT/${name}_seed${seed}.csv
  echo "=== $name (P_IRM=$p_irm, g=$g, f=$f, p_inf=$p_inf) ==="
  $PY -m desnoyers.two_dio --p-irm $p_irm --g "$g" --f "$f" --p-inf $p_inf \
      -m $M -n $N --seed $seed --output $fake 2>&1 | tail -3
  ls -la $fake | awk '{print "  size: "$5" bytes"}'
}

# p_inf NOTE: the paper does not specify p_inf as a fitted parameter. Algorithm
# 2 has an explicit t=∞ branch but fgen has finite support, so p_inf must be
# supplied separately. Values below come from `compute_oneshot.py` measured
# against our reference trace where we have a proxy of the paper-trace
# corpus (CloudPhysics for w-class, AliCloud-via-alibaba for v521-class) at
# 1M-record scale; for paper-traces we don't have a direct proxy, we leave
# p_inf=0 and rely on the IRM channel for one-shots (where applicable),
# noting the limitation explicitly. These measured values are at our
# small-scale (1M) sample, not the paper's full-trace scale; if the paper-
# trace one-shot rate differs at full N, our replication will too.
#
# Measured one-shot rates from /tiamat/zarathustra/llgan-output/refs/*.csv:
#   alibaba_stackatlas_1M (v521-class proxy):  0.6136
#   cloudphysics (w-class proxy):               0.3653
#   metacdn (no direct paper-trace match):      0.2664
#   wiki (no direct match):                     0.3010
#
# Table 3 I-values are pasted verbatim (Table 3 is 0-indexed, validated via
# v766 [0,5] and v827 [0,13] entries that are invalid under 1-indexed).
#
# Profile          P_IRM  g            f-spec                     p_inf  Source of p_inf
run_profile w11  1.0  zipf:1.3   none                        0.0    # P_IRM=1 → IRD channel unused; p_inf irrelevant
run_profile w24  0.45 zipf:1.2   "30:9,13,17,19:5e-3"        0.37   # measured CP one-shot proxy
run_profile w44  0.0  none       "30:9,13,17,19:2.5e-2"      0.37   # measured CP one-shot proxy
run_profile w82  0.2  zipf:1.2   "100:12,13,19:1e-3"         0.37   # measured CP one-shot proxy
run_profile v521 0.0  none       "100:2:2e-3"                0.61   # measured alibaba one-shot proxy
run_profile v538 0.1  zipf:1.2   "40:3,4:5e-3"               0.0    # no proxy; IRM channel covers
run_profile v766 0.0  none       "40:0,5:5.7e-3"             0.0    # no proxy; documented limitation
run_profile v827 0.2  zipf:1.2   "60:0,13:5e-3"              0.0    # no proxy; documented limitation

echo
echo "=== LRU HRC at log-spaced cache sizes (M=$M; cache as fraction of M) ==="
# For 2DIO synthetic traces, evaluate LRU HRC at cache sizes that are
# log-spaced fractions of footprint M=10000 (per paper Figure 8).
$PY << 'EOF'
import csv, json, os
from collections import OrderedDict
import subprocess
OUT="/tiamat/zarathustra/llgan-output/desnoyers/2dio_table3"
M=10000
caches = [10, 30, 100, 300, 1000, 3000, 5000, 8000, 9000, 9500]  # spans 0.001..0.95 of M
print(f"{'trace':<6s} " + "  ".join(f"c={c}" for c in caches))
for name in ("w11","w24","w44","w82","v521","v538","v766","v827"):
    fake = f"{OUT}/{name}_seed42.csv"
    if not os.path.exists(fake): continue
    # Use the standard cachesim_eval; it expects --real but we want HRC of FAKE alone.
    # Instead: simulate inline (lightweight LRU) using OrderedDict.
    obj_ids = []
    with open(fake) as f:
        r = csv.reader(f); next(r)
        for row in r: obj_ids.append(int(row[2]))
    hits = {c: 0 for c in caches}
    cache = OrderedDict()
    cap_max = max(caches)
    for oid in obj_ids:
        if oid in cache:
            cache.move_to_end(oid)
            for c in caches:
                # was hit if it was within last c entries — use LRU stack distance
                # (simpler: re-simulate per-c). Approximate: cache is max-sized, check rank.
                # The OrderedDict with move_to_end at top after access; rank of oid before move.
                pass  # handled below
            # rebuild rank check using a separate per-cap counter loop
            cache.pop(oid)
        cache[oid] = True
        if len(cache) > cap_max:
            cache.popitem(last=False)
    # Recompute properly with per-cap separate caches
    cache_per_c = {c: OrderedDict() for c in caches}
    for c in caches:
        ods = OrderedDict()
        h = 0
        for oid in obj_ids:
            if oid in ods:
                ods.move_to_end(oid)
                h += 1
            else:
                ods[oid] = True
                if len(ods) > c:
                    ods.popitem(last=False)
        hits[c] = h
    n = len(obj_ids)
    line = f"{name:<6s} "
    for c in caches:
        line += f"  {hits[c]/n:.3f}"
    print(line)
EOF

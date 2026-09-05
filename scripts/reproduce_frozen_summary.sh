#!/usr/bin/env bash
# Regenerate paper/figures/frozen_summary.png from scratch, on the current
# track geometry. Every row of the table, in dependency order.
#
#   bash scripts/reproduce_frozen_summary.sh            # everything, ~6 h on 16 cores
#   bash scripts/reproduce_frozen_summary.sh --from 4   # resume at step 4
#
# The result this reproduces (ICRA T2, smoothed corridor, three physically
# different starts, every policy driven FROZEN -- learning off, noise off):
#
#   START (constant, k_v 0.40)                         1.87  2.01  2.00
#   best constant by search (k_v 0.50)                 2.32  2.33  2.40
#   grid-fitted network, k_v capped 0.50               2.48  2.49  2.76
#   online from fitted, MPCC critic, validated  <<<    2.72  2.78  2.99   BEST RESULT SO FAR
#   online from fitted, RETURN critic, validated       2.34  2.30  2.57
#
# Runs are deterministic given the plant and the acados build; small drift
# between acados versions is possible. Everything is on the geometry of
# mpcc_tuning/track.py at the commit tagged paper-frozen-summary-v1. If
# track.py changes, EVERY row must be regenerated (TODO 2p): a policy learned
# on one boundary does not transfer to another.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
export ACADOS_SOURCE_DIR="${ACADOS_SOURCE_DIR:-$HOME/acados-v060}"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ACADOS_SOURCE_DIR/interfaces/acados_template:$HOME/github/scuderia_gym_jax:.${PYTHONPATH:+:$PYTHONPATH}"
FROM=1; [[ "${1:-}" == "--from" ]] && FROM=$2
step() { [[ $1 -ge $FROM ]]; }
T=icra_t2_raceline
COMMON="--tracks $T --clock progress --keep-best --validate --eval-episodes 1 --conditions tuner --seeds 3 --episodes 10 --jobs 3"

step 1 && echo "== 1. sectors of the loaded track (automatic) ==" && python3 scripts/show_sectors.py --track $T --png
step 2 && echo "== 2. situation grid, 27 (q_v,k_v,q_c) x 8 (sector, entry speed) cells, ~25 min ==" && \
  python3 experiments/situation_demands_acados.py --tracks $T --steps 400 --jobs 1 --with-qc --out situation_demands_qc.json
step 3 && echo "== 3. fit the weight policy to the grid: uncapped (crashes) and k_v capped at 0.50 (holds), each with 3 frozen starts ==" && \
  python3 scripts/fit_policy_to_grid.py --grid situation_demands_qc.json --epochs 10 --lr 0.1 --eval && \
  python3 scripts/fit_policy_to_grid.py --grid situation_demands_qc.json --epochs 10 --lr 0.1 --kv-cap 0.50 --eval
step 4 && echo "== 4. controls from START: fixed and fixed+noise, factor-2 box, ~1 h ==" && \
  python3 experiments/online_from_baseline.py --tracks $T --box adapt --factor 2.0 --conditions fixed fixed_noise --seeds 3 --episodes 10 --jobs 6 --out online_t2_adapt.json
step 5 && echo "== 5. online from START, MPCC critic, validated keep-best, ~1.5 h ==" && \
  python3 experiments/online_from_baseline.py $COMMON --box adapt --factor 2.0 --critic mpcc --explore 0.05 --out online_t2_mpcc_val.json
step 6 && echo "== 6. online from START, RETURN critic, validated keep-best, ~1.5 h ==" && \
  python3 experiments/online_from_baseline.py $COMMON --box adapt --factor 2.0 --critic return --theta-explore 0.1 --explore 0.0 --out online_t2_fitted_val.json
step 7 && echo "== 7. online from the FITTED network, MPCC critic -- the best result, ~1.5 h ==" && \
  python3 experiments/online_from_baseline.py $COMMON --init-policy results/fitted_policy_${T}_kv0.50.npz --critic mpcc --explore 0.0 --out online_t2_init_mpcc.json
step 8 && echo "== 8. online from the FITTED network, RETURN critic, ~1.5 h ==" && \
  python3 experiments/online_from_baseline.py $COMMON --init-policy results/fitted_policy_${T}_kv0.50.npz --critic return --theta-explore 0.05 --explore 0.0 --out online_t2_init_return.json
step 9 && echo "== 9. merge and draw ==" && \
  python3 scripts/merge_online_results.py online_t2_all.json online_t2_adapt.json \
    "online_t2_mpcc_val.json:tuner=tuner_mpcc_val" "online_t2_fitted_val.json:tuner=tuner_fitted_val" \
    "online_t2_init_mpcc.json:tuner=tuner_init_mpcc" "online_t2_init_return.json:tuner=tuner_init_return" && \
  python3 scripts/make_online_figures.py online_t2_all.json && echo "done: paper/figures/frozen_summary.png"
echo
echo "NOTE: steps 4-6 in the archived v1 table were measured on the corridor BEFORE smoothing"
echo "(marked on the figure). Running this script puts every row on the current geometry."
echo "results/frozen_summary_refs.json holds the reference rows (START, best constant, fitted)"
echo "and must be refreshed from the printed frozen values of steps 3 and the baselines re-measure."

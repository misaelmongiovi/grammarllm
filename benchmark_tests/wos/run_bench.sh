#!/usr/bin/env bash
#
# run_bench.sh — coda dei benchmark WoS con separatore PIPE e beam=3.
#
#   modelli : Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Meta-Llama-3-8B-Instruct
#   task    : 0-shot, 1-shot, 10-shot   (esempi da few_shot{1,10}.py, convertiti a pipe)
#   decoding: num_beams=3, do_sample=True
#   righe   : tutte (2000)
#
# I job vengono distribuiti sulle GPU disponibili, uno per GPU, e la coda avanza
# man mano che una GPU si libera. Al termine stampa la tabella dei risultati.
#
#   ./run_bench.sh                             # tutto
#   GPUS="0 1"                 ./run_bench.sh  # solo su 2 GPU
#   ROWS=200                   ./run_bench.sh  # smoke test veloce
#   LOOKAHEAD=off              ./run_bench.sh  # baseline boundary-strict
#   MODELS="Llama-3.2-3B-Instruct" ./run_bench.sh   # un solo modello
#   SHOTS="0 1 10"             ./run_bench.sh  # sottoinsieme dei task
#
set -uo pipefail

cd "$(dirname "$0")"
HERE="$PWD"
ROOT="$(cd ../.. && pwd)"

MODELS_DIR="${MODELS_DIR:-/home/stlab/models}"
GPUS="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}"
ROWS="${ROWS:-0}"                    # 0 = tutte le righe
BEAMS="${BEAMS:-3}"
LOOKAHEAD="${LOOKAHEAD:-on}"         # on | off
SAMPLE="${SAMPLE:-on}"               # on | off  (off = deterministico: greedy con BEAMS=1)
OUT="${OUT:-$HERE/out_pipe_bench}"
LOGS="$OUT/logs"

LA_FLAG=""
[ "$LOOKAHEAD" = "off" ] && LA_FLAG="--no-lookahead"

SAMPLE_FLAG="--sample"
[ "$SAMPLE" = "off" ] && SAMPLE_FLAG="--no-sample"

mkdir -p "$OUT" "$LOGS"

# ── la coda: un job per riga, "modello:nshot" ────────────────────────────────
MODELS="${MODELS:-Llama-3.2-1B-Instruct Llama-3.2-3B-Instruct Meta-Llama-3-8B-Instruct}"
SHOTS="${SHOTS:-0 1 10}"

QUEUE=()
for model in $MODELS; do
  for nshot in $SHOTS; do
    QUEUE+=("$model:$nshot")
  done
done

echo "=============================================================="
echo " WoS pipe benchmark — beam=$BEAMS, sample=$SAMPLE, lookahead=$LOOKAHEAD"
echo " job in coda : ${#QUEUE[@]}   ($(printf '%s ' "${QUEUE[@]}"))"
echo " GPU         : $GPUS"
echo " righe       : $([ "$ROWS" = 0 ] && echo 'tutte (2000)' || echo "$ROWS")"
echo " output      : $OUT"
echo "=============================================================="

# ── dispatcher: tiene occupata ogni GPU con al piu' un job ───────────────────
declare -A PID_ON_GPU=()   # gpu -> pid
declare -A JOB_ON_GPU=()   # gpu -> nome job
FAILED=0

launch() {   # launch <gpu> <model> <nshot>
  local gpu="$1" model="$2" nshot="$3"
  local tag="${model}_${nshot}shot"
  local out="$OUT/${tag}.csv"
  local log="$LOGS/${tag}.log"

  if [ -s "$out" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[skip] $tag  (esiste gia': $out — FORCE=1 per rifare)"
    return 1
  fi

  local rows_arg=""
  [ "$ROWS" != "0" ] && rows_arg="--rows $ROWS"

  echo "[gpu $gpu] avvio $tag"
  CUDA_VISIBLE_DEVICES="$gpu" nohup uv run --project "$ROOT" python "$HERE/pipe_bench.py" \
      --model "$MODELS_DIR/$model" --nshot "$nshot" --out "$out" \
      --beams "$BEAMS" $SAMPLE_FLAG $LA_FLAG $rows_arg > "$log" 2>&1 &
  PID_ON_GPU[$gpu]=$!
  JOB_ON_GPU[$gpu]="$tag"
  return 0
}

reap() {     # attende che ALMENO una gpu si liberi; ne stampa l'esito
  while true; do
    for gpu in $GPUS; do
      local pid="${PID_ON_GPU[$gpu]:-}"
      [ -z "$pid" ] && return 0                       # gpu gia' libera
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid"; local rc=$?
        local tag="${JOB_ON_GPU[$gpu]}"
        if [ $rc -eq 0 ]; then
          echo "[gpu $gpu] OK   $tag"
        else
          echo "[gpu $gpu] FALLITO $tag (exit $rc) — vedi $LOGS/${tag}.log"
          FAILED=$((FAILED + 1))
        fi
        unset 'PID_ON_GPU[$gpu]' 'JOB_ON_GPU[$gpu]'
        return 0
      fi
    done
    sleep 20
  done
}

for job in "${QUEUE[@]}"; do
  model="${job%%:*}"; nshot="${job##*:}"
  while true; do
    reap                                              # garantisce una gpu libera
    placed=0
    for gpu in $GPUS; do
      if [ -z "${PID_ON_GPU[$gpu]:-}" ]; then
        launch "$gpu" "$model" "$nshot" && placed=1
        break                                         # job piazzato (o saltato)
      fi
    done
    break
  done
done

# ── attende i job ancora in volo ─────────────────────────────────────────────
for gpu in $GPUS; do
  pid="${PID_ON_GPU[$gpu]:-}"
  [ -z "$pid" ] && continue
  wait "$pid"; rc=$?
  tag="${JOB_ON_GPU[$gpu]}"
  if [ $rc -eq 0 ]; then echo "[gpu $gpu] OK   $tag"
  else echo "[gpu $gpu] FALLITO $tag (exit $rc) — vedi $LOGS/${tag}.log"; FAILED=$((FAILED+1)); fi
done

echo
echo "=============================================================="
echo " RISULTATI   (invalid = parse non validi, devono essere 0)"
echo "=============================================================="
uv run --project "$ROOT" --with scikit-learn python "$HERE/score_bench.py" "$OUT"/*.csv

[ "$FAILED" -gt 0 ] && { echo; echo "$FAILED job falliti."; exit 1; }
echo
echo "fatto."

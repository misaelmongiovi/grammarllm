#!/usr/bin/env bash
# Full gloss-translation benchmark: 2000 test rows, greedy + beam3 vanilla
# (default HF, nessun parametro extra) in parallelo su 2 GPU, prima Llama-1B
# poi Llama-8B.
#
# Uso:
#   ./run_full_benchmark.sh                 # tutte le coppie (1b, 3b, 8b), GPU 0 e 1
#   ./run_full_benchmark.sh 3b              # solo la coppia indicata
#   ./run_full_benchmark.sh 1b 8b           # più coppie in sequenza
#   GPU_A=2 GPU_B=3 ./run_full_benchmark.sh # override GPU
#
# Esempio completo (background, sopravvive al logout):
#   cd ~/grammarllm/benchmark_tests/gloss_translation
#   nohup ./run_full_benchmark.sh 3b > run_full_3b.log 2>&1 &
#   tail -f run_full_3b.log                 # per seguire il progresso
#
# I run con output già presente vengono SALTATI (mai sovrascritti).
# Output: output/<name>/predictions_<name>.csv + scores_per_line.txt
# Se un run muore a metà: riprendi con
#   python gloss_eval.py --name <name> --model <path> [--num-beams 3] \
#       --start N --resume-csv output/<name>/checkpoint.csv
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=/home/gtuccio/grammarllm/.venv/bin/python
GPU_A="${GPU_A:-0}"   # greedy
GPU_B="${GPU_B:-1}"   # beam3

MODEL_1B=/home/stlab/models/Llama-3.2-1B-Instruct
MODEL_3B=/home/stlab/models/Llama-3.2-3B-Instruct
MODEL_8B=/home/stlab/models/Meta-Llama-3-8B-Instruct

already_done() {
    [ -f "output/$1/predictions_$1.csv" ]
}

run_pair() {
    local tag="$1" model="$2"
    if already_done "${tag}_greedy_full" && already_done "${tag}_beam3_full"; then
        echo "=== ${tag}: output già presenti, salto (nessuna sovrascrittura) ==="
        return
    fi
    echo "=== ${tag}: greedy (GPU ${GPU_A}) + beam3 vanilla (GPU ${GPU_B}) ==="

    local pid_greedy="" pid_beam=""

    if already_done "${tag}_greedy_full"; then
        echo "${tag} greedy: già presente, salto"
    else
        CUDA_VISIBLE_DEVICES="${GPU_A}" "$PYTHON" gloss_eval.py \
            --name "${tag}_greedy_full" --model "$model" \
            > "output_${tag}_greedy_full.log" 2>&1 &
        pid_greedy=$!
    fi

    if already_done "${tag}_beam3_full"; then
        echo "${tag} beam3: già presente, salto"
    else
        CUDA_VISIBLE_DEVICES="${GPU_B}" "$PYTHON" gloss_eval.py \
            --name "${tag}_beam3_full" --model "$model" --num-beams 3 \
            > "output_${tag}_beam3_full.log" 2>&1 &
        pid_beam=$!
    fi

    [ -n "$pid_greedy" ] && { wait "$pid_greedy"; echo "${tag} greedy done"; }
    [ -n "$pid_beam" ]   && { wait "$pid_beam";   echo "${tag} beam3 done"; }

    for name in "${tag}_greedy_full" "${tag}_beam3_full"; do
        echo "--- metrics ${name} ---"
        "$PYTHON" metrics.py "output/${name}/predictions_${name}.csv"
    done
}

# Coppie richieste come argomenti (default: tutte)
PAIRS=("$@")
[ ${#PAIRS[@]} -eq 0 ] && PAIRS=(1b 3b 8b)

for pair in "${PAIRS[@]}"; do
    case "$pair" in
        1b) run_pair "1b" "$MODEL_1B" ;;
        3b) run_pair "3b" "$MODEL_3B" ;;
        8b) run_pair "8b" "$MODEL_8B" ;;
        *)  echo "Coppia sconosciuta: $pair (usa 1b|3b|8b)"; exit 1 ;;
    esac
done

echo "=== Benchmark completo ==="

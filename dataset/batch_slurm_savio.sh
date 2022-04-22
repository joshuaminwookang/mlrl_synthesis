#!/bin/bash

SCRIPT_DIR="$( dirname "$( readlink -f "${BASH_SOURCE[0]}" )" )"
RUN_DIR="${SCRIPT_DIR}/../runs"
TEST_SCRIPT="${SCRIPT_DIR}/single_run_yosys-abc.sh"
STATIC_TEST_ARGS="-s 5000"
INPUT_DIR=  # $(readlink -f "${1:-vtr/verilog}")
BATCH_SIZE=8 # Actually spawns 3x this many jobs, one for each synth method (below)
USE_LSF=false
USE_SLURM=false
DEVICE="xc7a200"
# SYNTH_METHODS "yosys yosys-abc9"
SYNTH_METHODS="yosys-abc9"
LUT_LIB=0
IS_BATCH_MODE=false

STOCHASTIC=0
RANDOM_SEQ_LEN=0
NUM_OPTS=14
PERMUTATIONS=$(( 1 * 1))


# NOTE(aryap): 'realpath' is a nice tool to do 'readlink -f' which is itself a
# nice too to recursively expand symlinks, but it isn't available on BWRC
# servers, and we have a more portable solution so I'm not installing it.

while [ "$1" != "" ]; do
  case $1 in
    -o | --output )         shift
                            RUN_DIR="$(readlink -f "$1")"
                            ;;
    -i | --input )          shift
                            INPUT_DIR="$(readlink -f "$1")"
                            ;;
    -r | --random )         shift
                            RANDOM_SEQ_LEN=$1
                            ;;
    -t | --stoch )          shift
                            STOCHASTIC=$1
                            ;;
    -j | --batch_size)      shift
                            BATCH_SIZE="$1"
                            ;;
    -l | --lut_lib)         shift
                            LUT_LIB="$1"
                            ;;
    -s | --slurm)           USE_SLURM=true
                            ;;
    -f | --lsf)             USE_LSF=true
                            ;;
    -d | --device )         shift
                            DEVICE="$1"
                            ;;
    -m | --synth_method )   shift
                            SYNTH_METHODS="$1"
                            ;;
    -b | --batch )          shift
                            IS_BATCH_MODE=true
                            ;;
    * )                     echo "computer says no: ${1}"
                            exit 1
  esac
  shift
done

if [ -z "${RUN_DIR}" ]; then
  echo "Output directory must be specified!"
  exit 2
fi

# Setup run output directory
if ! [ -d "${RUN_DIR}" ]; then
  mkdir -p "${RUN_DIR}"
fi
pushd ${RUN_DIR}

if [ -d "${INPUT_DIR}" ]; then
    shopt -s nullglob
    slurm_scripts=( ${INPUT_DIR}/*.sh )
    num_jobs=${#slurm_scripts[@]}
    echo "Found ${num_jobs} runs"
    shopt -u nullglob
else
  echo "Unsuitable input source: ${INPUT_DIR}"
  exit 3
fi
echo "Input is ${INPUT_DIR}: ${#slrum_scripts[@]} files"
echo "Output is: ${RUN_DIR}"
let "i=0"
while [ ${i} -lt ${num_jobs} ]; do
    unset pids
    pids=()
    for ((j=0;j<${BATCH_SIZE} && i < ${num_jobs};j++)); do
        sbatch "${slurm_scripts[i]}"
	pids["j"]=$!
        let "i=i+1"
    done
    echo "Dispatched ${#pids[@]} jobs"
    for pid in ${pids[*]}; do
	wait ${pid}
    done
    unset pids
  done
popd  
# # Dispatch ${BATCH_SIZE}-many groups of jobs in parallel and wait for them to
# # complete, then continue, until all jobs are complete.
# batch_controlled_launch() {
#   let "i=0"
#   while [ ${i} -lt ${num_jobs} ]; do
#     unset pids
#     pids=()
#     for ((j=0;j<${BATCH_SIZE} && i < ${num_jobs};j++)); do
# 	sbatch "${slurm_scripts[i]"
#         let "i=i+1"
#     done
#     echo "Dispatched ${#pids[@]} jobs"
#     for pid in ${pids[*]}; do
#       wait ${pid}
#     done
#     unset pids
#   done
# }

# # Launch up to #{BATCH_SIZE}-many jobs. As soon as one terminates, launch the next.
# # REQUIRES Bash 4.3.
# # https://mywiki.wooledge.org/ProcessManagement#advanced
# token_controlled_launch() {
#   i=0
#   tokens=0
#   while [ ${i} -lt ${MAX_NUM_RUNS} ]; do
#       for ((j=0;j<${BATCH_SIZE} && i < ${num_jobs};j++)); do
#         sbatch "${slurm_scripts[i]"
# 	if (( tokens++ >= BATCH_SIZE )); then
#             wait -n
#             let "tokens=tokens-1"
#         fi
#         let "i=i+1"
#       done
#   done
# }

# bash -c "wait -n" 2>/dev/null
# return_value=$?
# if (( return_value == 2 )); then
#     batch_controlled_launch
# else
#     token_controlled_launch
# fi
# popd

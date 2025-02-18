#!/bin/bash

DEVICE=${1:-"cuda"}
DEVICE_MODEL=${2:-"v100"}
OUTDIR_ROOT=${3:-"benchmarks/hfht"}

source benchmarks/_workflow_pointnet.sh
source benchmarks/_workflow_mobilenet.sh

_get_modes() {
  if [ "${DEVICE}" == "cuda" ]; then
    modes=("serial" "concurrent" "mps" "hfta")
    if [ "${DEVICE_MODEL}" == "a100" ]; then
	modes+=("mig")
    fi 
  elif [ "${DEVICE}" == "xla" ]; then
    modes=("serial" "hfta")
  elif [ "${DEVICE}" == "cpu" ]; then
    modes=("serial" "concurrent" "hfta")
  else
    echo "Unknown DEVICE ${DEVICE} !"
    return -1
  fi
}

_get_precs() {
  if [ "${DEVICE}" == "cuda" ]; then
    precs=("fp32" "amp")
  elif [ "${DEVICE}" == "xla" ]; then
    precs=("bf16")
  elif [ "${DEVICE}" == "cpu" ]; then
    precs=("fp32")
  else
    echo "Unknown DEVICE ${DEVICE} !"
    return -1
  fi
}

_get_dry_run_args() {
  local mode=$1
  if [ "$mode" == "serial" ]; then
    cmd_dry_run_opts=" "
    return 0
  fi  

  # used for concurrent, mps, hfta, mig
  local hfht_dry_run_epochs=3
  local hfht_dry_run_iters_per_epochs
  local hfht_dry_run_repeats
  if [ "$mode" == "concurrent" ] || [ "$mode" == "mps" ] || [ "$mode" == "mig" ]; then
    hfht_dry_run_repeats=10
    hfht_dry_run_iters_per_epochs=10
  elif [ $mode == "hfta" ]; then
    hfht_dry_run_repeats=3
    hfht_dry_run_iters_per_epochs=2
  else
    echo "Unknown mode ${mode}"
    return -1
  fi
  cmd_dry_run_opts="\
  	--dry-run-repeats $hfht_dry_run_repeats\
  	--dry-run-epochs $hfht_dry_run_epochs\
  	--dry-run-iters-per-epoch $hfht_dry_run_iters_per_epochs"
  return 0

}

_sweep() {
  local base_cmd=$1
  local outdir_root=$2
  local repeats=$3
  local modes

  _get_modes
  local precs
  _get_precs
  for ((i=0; i<${repeats}; i++)); do
    for algorithm in random hyperband
    do
      local cmd_algo="${base_cmd} --algorithm ${algorithm}"
      for mode in "${modes[@]}"
      do
        local cmd_mode="${cmd_algo} --mode ${mode}"
	local cmd_dry_run_opts
	_get_dry_run_args $mode
        for prec in "${precs[@]}"
        do
          local cmd="${cmd_mode} $cmd_dry_run_opts"
          if [ "${prec}" == "amp" ]; then
            cmd+=" --amp"
          fi
          if [ "${prec}" == "fp32" ] && [ "${DEVICE}" == "cuda" ] \
              && [ "${DEVICE_MODEL}" == "a100" ]; then
            cmd="NVIDIA_TF32_OVERRIDE=0 ${cmd}"
          fi
          cmd+=" --outdir ${outdir_root}/run${i}/${algorithm}/${DEVICE}/${DEVICE_MODEL}/${prec}/${mode}"
          echo "Running ${cmd} ..."
          eval ${cmd}
        done
      done
    done
  done
}


hfht_workflow_pointnet_cls() {
  local repeats=${1:-"3"}
  local base_cmd="\
    python examples/hfht/pointnet_classification.py \
    --dataset datasets/shapenetcore_partanno_segmentation_benchmark_v0/ \
    --device ${DEVICE}"
  echo "Warmup ..."
  _pointnet_warmup_data cls
  _sweep "${base_cmd}" ${OUTDIR_ROOT}/pointnet_cls ${repeats}
}


hfht_workflow_mobilenet_cifar10() {
  local repeats=${1:-"3"}
  local base_cmd="\
    python examples/hfht/mobilenet.py \
    --dataset cifar10 \
    --dataroot datasets/cifar10 \
    --device ${DEVICE}"
  
  echo "Warmup ..."
  _mobilenet_warmup_data cifar10
  _sweep "${base_cmd}" ${OUTDIR_ROOT}/mobilenet_cifar10 ${repeats}
}

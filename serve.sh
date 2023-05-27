#!/bin/bash
set -e
cd "${0%/*}"
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

if [[ ! -z "$CONDA_ACTIVATE" ]]; then
  conda activate $CONDA_ACTIVATE
fi

export MODEL_BASE=`echo $MODEL | awk -F '-tp' '{print $1}'`
export MODEL_SHARDS=`echo $MODEL | awk -F '-tp' '{print $2}'`
if [[ ! "$MODEL_SHARDS" -gt 0 ]]; then
  echo "Couldn't parse tensor parallelism"
  exit 1
fi

NUM_WORKERS=${NUM_WORKERS-auto}
if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
  unset CUDA_VISIBLE_DEVICES
  if [[ "$NUM_WORKERS" == "auto" ]]; then
    NUM_WORKERS=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
    echo NUM_WORKERS=auto resolved to NUM_WORKERS=${NUM_WORKERS}
  fi
  DEVICES=
  for ((i = 0; i < $NUM_WORKERS; i++)); do
    DEVICES=$DEVICES,$i
  done
else
  DEVICES=$CUDA_VISIBLE_DEVICES
  if [[ "$NUM_WORKERS" == "auto" ]]; then
    NUM_WORKERS=`echo $DEVICES | sed 's/[^,]//g' | wc -c`
    echo NUM_WORKERS=auto resolved to NUM_WORKERS=${NUM_WORKERS}
  fi
fi
if [ "$NUM_WORKERS" -gt "$MODEL_SHARDS" ]; then
  NUM_WORKERS=$MODEL_SHARDS
  echo Reduced NUM_WORKERS to ${NUM_WORKERS} to match tensor parallelism
fi

if [ "$MODEL_SHARDS" -gt 1 ]; then
  case ${MODEL_TYPE-gptj} in
    gpt)
      env GROUP=${GROUP-group$i} /bin/bash -c 'mpirun -n $MODEL_SHARDS --allow-run-as-root python examples/pytorch/gpt/app/serving_opt_multi_gpu.py --hf_model_name facebook/$MODEL_BASE --tensor_para_size $MODEL_SHARDS --ckpt_path /home/user/.together/models/$MODEL'
    ;;
    gptneox)
      env GROUP=${GROUP-group$i} /bin/bash -c 'mpirun -n $MODEL_SHARDS --allow-run-as-root python examples/pytorch/gptneox/app/serving_multi_gpu.py --hf_model_name --ckpt_path /home/user/.together/models/$MODEL_BASE'
    ;;
    *)
      echo Unknown MODEL_TYPE
      exit 1
    ;;
  esac
  exit 0
fi

count=0
for i in ${DEVICES//,/$IFS}; do
  case ${MODEL_TYPE-gptj} in
    gpt)
      env DEVICE=${DEVICE-cuda:$i} GROUP=${GROUP-group$i} /bin/bash -c 'python examples/pytorch/gpt/app/serving_opt_single_gpu.py --hf_model_name facebook/$MODEL_BASE --ckpt_path /home/user/.together/models/$MODEL_BASE' &
    ;;
    gptj)
      env DEVICE=${DEVICE-cuda:$i} GROUP=${GROUP-group$i} /bin/bash -c 'python examples/pytorch/gptj/app/serving.py --ckpt_path /home/user/.together/models/$MODEL_BASE' &
    ;;
    gptneox)
      env DEVICE=${DEVICE-cuda:$i} GROUP=${GROUP-group$i} /bin/bash -c 'python examples/pytorch/gptneox/app/serving_single_gpu.py --ckpt_path /home/user/.together/models/$MODEL_BASE' &
    ;;
    *)
      echo Unknown MODEL_TYPE
      exit 1
    ;;
  esac
  count=`expr $count + 1`
  if [ $count -eq $NUM_WORKERS ]; then
    break
  fi
done

wait -n
exit $?

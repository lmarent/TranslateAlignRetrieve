# Average at most three model checkpoints to obtain the best model
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR=${SCRIPT_DIR}/../../env/bin
source $ENV_DIR/activate

MODEL1=$1
MODEL2=$2
MODEL3=$3
LANG_SRC=$4
LANG_TGT=$5

if [[ $# -eq 0 ]]
  then
  MODEL1=/content/gdrive/MyDrive/NMT_Test/nmt/data/exp/model.en-es_step_100000.pt
  MODEL2=/content/gdrive/MyDrive/NMT_Test/nmt/data/exp/model.en-es_step_99000.pt
  MODEL3=/content/gdrive/MyDrive/NMT_Test/nmt/data/exp/model.en-es_step_98000.pt
  LANG_SRC=en
  LANG_TGT=es
fi

echo -e "Averaging models...\n${MODEL1}\n${MODEL2}\n${MODEL3}"

MODEL_DIR=/content/gdrive/MyDrive/data/exp
AVERAGE_MODEL=${MODEL_DIR}/${LANG_SRC}"2"${LANG_TGT}_average_model.pt

python ${SCRIPT_DIR}/../../tools/OpenNMT-py/tools/average_models.py \
    -models ${MODEL1} ${MODEL2} ${MODEL3} \
    -output ${MODEL_DIR}/en2es_average_model.pt

echo -e "Average model:${AVERAGE_MODEL}"

# This script create the Train/Dev/Test datasets.
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR=${SCRIPT_DIR}/../../../../venv/bin
source $ENV_DIR/activate

SRC_FILE=$1
SRC_LANG=$2
TGT_FILE=$3
TGT_LANG=$4

if [[ $# -eq 0 ]]
  then
  SRC_FILE=${SCRIPT_DIR}/corpora/en-es/corpora.en
  SRC_LANG=en
  TGT_FILE=${SCRIPT_DIR}/corpora/en-es/corpora.es
  TGT_LANG=es
fi
echo "source_file", ${SRC_FILE}
echo "SRC_LANG", ${SRC_LANG}
echo "TGT_FILE", ${TGT_FILE}
echo "TGT_LANG", ${TGT_LANG}
echo "DATASETS_DIR", ${DATASETS_DIR}


# Create datasets dir
SRC_TO_TGT=${SRC_LANG}'2'${TGT_LANG}
DATASETS_DIR=${SCRIPT_DIR}/data/${SRC_TO_TGT}/datasets
echo ${DATASETS_DIR}

mkdir -p ${DATASETS_DIR}
python ${SCRIPT_DIR}/create_datasets.py \
    --source_file ${SRC_FILE} \
    --source_lang ${SRC_LANG} \
    --target_file ${TGT_FILE} \
    --target_lang ${TGT_LANG} \
    --output_dir ${DATASETS_DIR} \
    --test_size 1000 \
    --valid_size 5000

# python create_datasets.py --source_file /home/advicetec/PycharmProjects/TranslateAlignRetrieve/src/tar/src/nmt/corpora/en-es/corpora.en \
         --source_lang en \
         --target_file /home/advicetec/PycharmProjects/TranslateAlignRetrieve/src/tar/src/nmt/corpora/en-es/corpora.es \
         --target_lang es \
         --output_dir /home/advicetec/PycharmProjects/TranslateAlignRetrieve/src/tar/src/nmt/data/en2es/datasets \
         --test_size 1000 \
         --valid_size 5000



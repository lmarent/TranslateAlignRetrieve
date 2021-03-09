#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR=${SCRIPT_DIR}/env
source ${ENV_DIR}/bin/activate

STS_FILE=$1
OUTPUT_DIR=$2

LANG_SRC=en
LANG_TGT=es

TRANSLATE_RETRIEVE_DIR=${SCRIPT_DIR}/src/retrieve

python ${TRANSLATE_RETRIEVE_DIR}/translate_retrieve_sts_benchmark.py \
           -sts_benchmark_file   ${STS_FILE} \
           -lang_source ${LANG_SRC} \
           -lang_target ${LANG_TGT} \
           -output_dir ${OUTPUT_DIR}

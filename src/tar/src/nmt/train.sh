#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR=${SCRIPT_DIR}/../../env
source $ENV_DIR/bin/activate

LANG_SRC=$1
LANG_TGT=$2

if [[ $# -eq 0 ]]
  then
  LANG_SRC=en
  LANG_TGT=es
fi


onmt_train -config ${SCRIPT_DIR}/en_es.yaml

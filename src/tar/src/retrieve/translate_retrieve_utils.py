import requests
import subprocess
import json
import os
import tempfile
from sacremoses import MosesTokenizer, MosesDetokenizer

from nltk import sent_tokenize

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# PROCESSING TEXT
tokenizer_en = MosesTokenizer(lang='en')
detokenizer_en = MosesDetokenizer(lang='en')
tokenizer_es = MosesTokenizer(lang='es')
detokenizer_es = MosesDetokenizer(lang='es')

MAX_NUM_TOKENS = 10
SPLIT_DELIMITER = ';'
LANGUAGE_ISO_MAP = {'en': 'english', 'es': 'spanish'}

def tokenize(text, lang, return_str=True):
    if lang == 'en':
        text_tok = tokenizer_en.tokenize(text, return_str=return_str, escape=False)
        return text_tok
    elif lang == 'es':
        text_tok = tokenizer_es.tokenize(text, return_str=return_str, escape=False)
        return text_tok


def de_tokenize(text, lang):
    if not isinstance(text, list):
        text = text.split()

    if lang == 'en':
        text_detok = detokenizer_en.detokenize(text, return_str=True)
        return text_detok
    elif lang == 'es':
        text_detok = detokenizer_es.detokenize(text, return_str=True)
        return text_detok
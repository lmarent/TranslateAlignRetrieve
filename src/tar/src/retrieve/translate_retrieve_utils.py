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


def translate(source_sentences, file, output_dir, batch_size):
    """
    Translate via the OpenNMT-py script
    :param source_sentences: list of sentences to translate
    :param file: file name to use for translation
    :param output_dir: output directory to use for translation
    :param batch_size: number of sentence to translate in an execution.
    :return:
    """

    filename = os.path.basename(file)
    source_filename = os.path.join(output_dir, '{}_source_translate'.format(filename))
    with open(source_filename, 'w') as sf:
        sf.writelines('\n'.join(s for s in source_sentences))

    translation_filename = os.path.join(output_dir, '{}_target_translated'.format(filename))
    en2es_translate_cmd = SCRIPT_DIR + '/../nmt/en2es_translate.sh {} {} {}'.format(source_filename,
                                                                                    translation_filename,
                                                                                    batch_size)
    subprocess.run(en2es_translate_cmd.split())

    with open(translation_filename) as tf:
        translated_sentences = [s.strip() for s in tf.readlines()]

    os.remove(source_filename)
    os.remove(translation_filename)

    return translated_sentences
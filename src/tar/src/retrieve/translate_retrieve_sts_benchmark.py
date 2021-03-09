import json
import time
import subprocess
import csv
from tqdm import tqdm
import os
from collections import defaultdict
import pickle
import argparse
import translate_retrieve_utils as utils
from nltk import sent_tokenize
import logging
import stanza

logging.basicConfig(level=logging.INFO)


def split_sentences(text, lang, delimiter=utils.SPLIT_DELIMITER, max_size=utils.MAX_NUM_TOKENS, tokenized=True):
    """
       Chunk sentences longer than a maximum number of words/tokens based on a delimiter character.
       This option is used only for very long sentences to avoid shorter translation than the
       original source length.
       Note that the delimiter can't be a trailing character
    """
    text_len = len(utils.tokenize(text, lang, return_str=True).split()) if tokenized else len(text.split())
    if text_len >= max_size:
        delimiter_match = delimiter + ' '
        text_chunks = [chunk.strip() for chunk in text.split(delimiter_match) if chunk]
        # Add the delimiter lost during chunking
        text_chunks = [chunk + delimiter for chunk in text_chunks[:-1]] + [text_chunks[-1]]
        return text_chunks
    return [text]


def tokenize_sentences(text, lang):
    sentences = [chunk
                 for sentence in sent_tokenize(text, utils.LANGUAGE_ISO_MAP[lang])
                 for chunk in split_sentences(sentence, lang, '|')]
    return sentences

def tokenize_sentences_unlimited_size(text, lang):
    sentences = [chunk
                 for chunk in split_sentences(text, lang, '|', 100000000000)]
    return sentences

class STSBenchmarkTranslator:
    def __init__(self,
                 sts_benchmark_file,
                 lang_source,
                 lang_target,
                 output_dir,
                 alignment_type,
                 answers_from_alignment,
                 batch_size):

        self.sts_benchmark_file = sts_benchmark_file
        self.lang_source = lang_source
        self.lang_target = lang_target
        self.output_dir = output_dir
        self.alignment_type = alignment_type
        self.batch_size = batch_size

        # initialize content_translations_alignments
        self.content_translations_alignments = defaultdict()

        # initialize SNLI version
        self.sts_benchmark_version = '2017'

    def translate_align_content(self):
        # Translate all the textual content in the SNLI dataset,
        # that are, sentences and gold classification.
        # The alignment between context and its translation is then computed.
        # The output is a dictionary with sentence pairs, sentences and classification
        # and their translation/alignment as values

        stanza.download('es', processors='tokenize,mwt,pos,lemma,depparse')
        nlp = stanza.Pipeline('es', processors='tokenize,mwt,pos,lemma,depparse')

        # Load snli content and get snli contexts
        headers = ['genre' , 'filename', 'year', 'captionID', 'score', 'sentence1', 'sentence2']
        content_lines = []
        with open(self.sts_benchmark_file) as hn:
            csv = csv.reader(hn, delimiter='\t')
            for row in csvFile:
                line = {}
                for i in range(len(headers)): 
                    line[header[i]] = row[i]
                content_lines.append(line)

        # Check if the content of STS Benchmark has been translated and aligned already
        content_translations_alignments_file = os.path.join(self.output_dir,
                                                    '{}_content_translations_alignments.{}'.format(
                                                        os.path.basename(self.snli_file),
                                                        self.lang_target))
        if not os.path.isfile(content_translations_alignments_file):
            # Extract sentence one and two.             
            sentences_one = []
            sentences_two = []
            max_len_sentence_1 = 0
            max_len_sentence_2 = 0
            for content in tqdm(content_lines):
                if len(content[5]) > max_len_sentence_1:
                    max_len_sentence_1 = len(content[5])
                sentences_one.extend(tokenize_sentences(content[5],
                                                              lang=self.lang_source))

                if len(content[6]) > max_len_sentence_2:
                    max_len_sentence_2 = len(content[6])
                sentences_two.extend(tokenize_sentences(content[6],
                                                              lang=self.lang_source))

            sentence_one_translated = utils.translate(sentences_one, self.sts_benchmark_file, self.output_dir, self.batch_size)
            sentence_two_translated = utils.translate(sentences_two, self.sts_benchmark_file,
                                                      self.output_dir, self.batch_size)

            logging.info('Collected {} sentence to translate'.format(len(sentences_one)))

            translated_file = os.path.join(self.output_dir,
                                       os.path.basename(self.snli_file).replace(
                                           '.json',
                                           '-{}_small.json'.format(self.lang_target)))
            with open(translated_file, 'w') as fn:
                i = 0
                for content in tqdm(content_lines):
                    content_line = {}
                    content_line['sentence1'] = sentence_one_translated[i]
                    content_line['sentence2'] = sentence_two_translated[i]
                    sentence_one_parsed = nlp(sentence_one_translated[i])
                    sentence_two_parsed = nlp(sentence_two_translated[i])

                    content_line['sentence1_parse'] = sentence_one_parsed.to_dict()
                    content_line['sentence2_parse'] = sentence_two_parsed.to_dict()
                    content_line['genre'] = content['genre']
                    content_line['captionID'] = content['captionID']
                    content_line['score'] = content['score']
                    content_line['year'] = content['year']
                    content_line['filename'] = content['filename']
                    json.dump(content_line, fn)
                    fn.write('\n')
                    i = i + 1

        # Load content translated and aligned from file
        else:
            logging.info('Use previously content translations and alignments')
            with open(content_translations_alignments_file, 'rb') as fn:
                self.content_translations_alignments = pickle.load(fn)


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-sts_benchmark_file', type=str, help='STS Benchmark dataset to translate')
    parser.add_argument('-lang_source', type=str, default='en',
                        help='language of the STS Benchmark dataset to translate (the default value is set to English)')
    parser.add_argument('-lang_target', type=str, help='translation language')
    parser.add_argument('-output_dir', type=str, help='directory where all the generated files are stored')
    parser.add_argument('-alignment_type', type=str,
                        default='forward', help='use a given translation service')
    parser.add_argument('-batch_size', type=int, default='32', help='batch_size for the translation script '
                                                                    '(change this value in case of CUDA out-of-memory')
    args = parser.parse_args()
    # Create output directory if doesn't exist already
    try:
        os.mkdir(args.output_dir)
    except FileExistsError:
        pass

    translator = STSBenchmarkTranslator(args.sts_benchmark_file,
                                 args.lang_source,
                                 args.lang_target,
                                 args.output_dir,
                                 args.alignment_type,
                                 args.batch_size)

    logging.info('Translate STS Benchmark textual content')
    translator.translate_align_content()


    end = time.time()
    logging.info('Total execution time: {} s'.format(round(end - start)))


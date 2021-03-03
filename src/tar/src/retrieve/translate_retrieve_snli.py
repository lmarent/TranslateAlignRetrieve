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
    print('initial text:', text)
    sentences = [chunk
                 for sentence in sent_tokenize(text, utils.LANGUAGE_ISO_MAP[lang])
                 for chunk in split_sentences(sentence, lang, '|', 100000000000)]
    print('sentences:', sentences)
    return sentences

class SNLITranslator:
    def __init__(self,
                 snli_file,
                 lang_source,
                 lang_target,
                 output_dir,
                 alignment_type,
                 answers_from_alignment,
                 batch_size):

        self.snli_file = snli_file
        self.lang_source = lang_source
        self.lang_target = lang_target
        self.output_dir = output_dir
        self.alignment_type = alignment_type
        self.answers_from_alignment = answers_from_alignment
        self.batch_size = batch_size

        # initialize content_translations_alignments
        self.content_translations_alignments = defaultdict()

        # initialize SNLI version
        self.snli_version = '1.0'

    # Translate all the textual content in the SNLI dataset,
    # that are, sentences and gold classification.
    # The alignment between context and its translation is then computed.
    # The output is a dictionary with sentence pairs, sentences and classification
    # and their translation/alignment as values
    def translate_align_content(self):
        # Load snli content and get snli contexts
        with open(self.snli_file) as hn:
            lines = hn.readlines()
        content_lines = []
        for line in lines:
            content_lines.append(json.loads(line))

        # Check if the content of SNLI has been translated and aligned already
        content_translations_alignments_file = os.path.join(self.output_dir,
                                                    '{}_content_translations_alignments.{}'.format(
                                                        os.path.basename(self.snli_file),
                                                        self.lang_target))
        if not os.path.isfile(content_translations_alignments_file):
            # Extract contexts, questions and answers. The context is further
            # divided into sentence in order to translate and compute the alignment.
            sentences_one = []
            sentences_two = []
            sentences_one_parse = []
            sentences_two_parse = []
            sentences_one_binary_parse = []
            sentences_two_binary_parse = []
            max_len_sentence_1 = 0
            max_len_sentence_2 = 0
            for content in tqdm(content_lines):
                if len(content['sentence1']) > max_len_sentence_1:
                    max_len_sentence_1 = len(content['sentence1'])
                sentences_one.extend(tokenize_sentences(content['sentence1'],
                                                              lang=self.lang_source))

                if len(content['sentence2']) > max_len_sentence_2:
                    max_len_sentence_2 = len(content['sentence2'])
                sentences_two.extend(tokenize_sentences(content['sentence2'],
                                                              lang=self.lang_source))
                sentences_one_parse.extend(tokenize_sentences_unlimited_size(content['sentence1_parse'],
                                                                    lang=self.lang_source))
                sentences_two_parse.extend(tokenize_sentences_unlimited_size(content['sentence2_parse'],
                                                                    lang=self.lang_source))
                sentences_one_binary_parse.extend(tokenize_sentences_unlimited_size(content['sentence1_binary_parse'],
                                                                           lang=self.lang_source))
                sentences_two_binary_parse.extend(tokenize_sentences_unlimited_size(content['sentence2_binary_parse'],
                                                                           lang=self.lang_source))

            print('len sentences_one', len(sentences_one))
            print('len sentences_two', len(sentences_two))
            print('len sentences_one_parse', len(sentences_one_parse))
            print('len sentences_two_parse', len(sentences_two_parse))
            print('len sentences_one_binary_parse', len(sentences_one_binary_parse))
            print('len sentences_two_binary_parse', len(sentences_two_binary_parse))
            print('max_len_sentence_1', max_len_sentence_1)
            print('max_len_sentence_2', max_len_sentence_2)

            # sentence_one_translated = utils.translate(sentences_one, self.snli_file, self.output_dir, self.batch_size)
            # sentence_two_translated = utils.translate(sentences_two, self.snli_file,
            #                                           self.output_dir, self.batch_size)
            # sentences_one_parse_translated = utils.translate(sentences_one_parse,
            #                                                  self.snli_file, self.output_dir, self.batch_size)
            # sentences_two_parse_translated = utils.translate(sentences_two_parse,
            #                                                  self.snli_file, self.output_dir, self.batch_size)
            # sentences_one_binary_parse_translated = utils.translate(sentences_one_binary_parse,
            #                                                         self.snli_file, self.output_dir, self.batch_size)
            # sentences_two_binary_parse_translated = utils.translate(sentences_two_binary_parse,
            #                                                         self.snli_file, self.output_dir, self.batch_size)
            #
            # logging.info('Collected {} sentence to translate'.format(len(sentences_one)))
            #
            # i = 0
            # new_content_lines = []
            # for content in tqdm(content_lines):
            #     content_line = {}
            #     content_line['sentence1'] = sentence_one_translated[i]
            #     content_line['sentence2'] = sentence_two_translated[i]
            #     content_line['sentence1_parse'] = sentences_one_parse_translated[i]
            #     content_line['sentence2_parse'] = sentences_two_parse_translated[i]
            #     content_line['sentence1_binary_parse'] = sentences_one_binary_parse_translated[i]
            #     content_line['sentence2_binary_parse'] = sentences_two_binary_parse_translated[i]
            #     content_line['annotator_labels'] = content['annotator_labels']
            #     content_line['captionID'] = content['captionID']
            #     content_line['gold_label'] = content['gold_label']
            #     content_line['pairID'] = content['pairID']
            #     i = i + 1
            #     print(content_line)
            #     new_content_lines.append(content_line)
            #
            # with open(content_translations_alignments_file, 'wb') as fn:
            #     pickle.dump(new_content_lines, fn)
            #
            # translated_file = os.path.join(self.output_dir,
            #                            os.path.basename(self.snli_file).replace(
            #                                '.json',
            #                                '-{}_small.json'.format(self.lang_target)))
            #
            # with open(translated_file, 'w') as fn:
            #     json.dump(new_content_lines, fn)

        # Load content translated and aligned from file
        else:
            logging.info('Use previously content translations and alignments')
            with open(content_translations_alignments_file, 'rb') as fn:
                self.content_translations_alignments = pickle.load(fn)


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-snli_file', type=str, help='SNLI dataset to translate')
    parser.add_argument('-lang_source', type=str, default='en',
                        help='language of the SNLI dataset to translate (the default value is set to English)')
    parser.add_argument('-lang_target', type=str, help='translation language')
    parser.add_argument('-output_dir', type=str, help='directory where all the generated files are stored')
    parser.add_argument('-answers_from_alignment', action='store_true',
                        help='retrieve translated answers only from the alignment')
    parser.add_argument('-alignment_type', type=str,
                        default='forward', help='use a given translation service')
    parser.add_argument('-batch_size', type=int, default='32', help='batch_size for the translation script '
                                                                    '(change this value in case of CUDA out-of-memory')
    args = parser.parse_args()
    print(args)
    # Create output directory if doesn't exist already
    try:
        os.mkdir(args.output_dir)
    except FileExistsError:
        pass

    translator = SNLITranslator(args.snli_file,
                                 args.lang_source,
                                 args.lang_target,
                                 args.output_dir,
                                 args.alignment_type,
                                 args.answers_from_alignment,
                                 args.batch_size)

    logging.info('Translate SNLI textual content')
    translator.translate_align_content()


    end = time.time()
    logging.info('Total execution time: {} s'.format(round(end - start)))


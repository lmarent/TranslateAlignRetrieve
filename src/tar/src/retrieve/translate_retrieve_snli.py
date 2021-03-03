import json
import time
import subprocess
import csv
from tqdm import tqdm
import os
from collections import defaultdict
import pickle
import argparse
import translate_retrieve_squad_utils as squad_utils
import translate_retrieve_utils as utils
import logging

logging.basicConfig(level=logging.INFO)

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
            for content in tqdm(content_lines):
                sentences_one.extend(squad_utils.tokenize_sentences(content['sentence1'],
                                                              lang=self.lang_source))
                sentences_two.extend(squad_utils.tokenize_sentences(content['sentence2'],
                                                              lang=self.lang_source))
                sentences_one_parse.extend(squad_utils.tokenize_sentences(content['sentence1_parse'],
                                                                    lang=self.lang_source))
                sentences_two_parse.extend(squad_utils.tokenize_sentences(content['sentence2_parse'],
                                                                    lang=self.lang_source))
                sentences_one_binary_parse.extend(squad_utils.tokenize_sentences(content['sentence1_binary_parse'],
                                                                           lang=self.lang_source))
                sentences_two_binary_parse.extend(squad_utils.tokenize_sentences(content['sentence2_binary_parse'],
                                                                           lang=self.lang_source))

            sentence_one_translated = utils.translate(sentences_one, self.snli_file, self.output_dir, self.batch_size)
            sentence_two_translated = utils.translate(sentences_two, self.snli_file,
                                                      self.output_dir, self.batch_size)
            sentences_one_parse_translated = utils.translate(sentences_one_parse,
                                                             self.snli_file, self.output_dir, self.batch_size)
            sentences_two_parse_translated = utils.translate(sentences_two_parse,
                                                             self.snli_file, self.output_dir, self.batch_size)
            sentences_one_binary_parse_translated = utils.translate(sentences_one_binary_parse,
                                                                    self.snli_file, self.output_dir, self.batch_size)
            sentences_two_binary_parse_translated = utils.translate(sentences_two_binary_parse,
                                                                    self.snli_file, self.output_dir, self.batch_size)

            # align sentences
            # sentence_one_translated_align = utils.compute_alignment(sentences_one,  self.lang_source,
            #                                                   sentence_one_translated,
            #                                                   self.lang_target,
            #                                                   self.alignment_type,
            #                                                   self.snli_file,
            #                                                   self.output_dir)
            # sentence_two_translated_align = utils.compute_alignment(sentences_two, self.lang_source,
            #                                                   sentence_two_translated,
            #                                                   self.lang_target,
            #                                                   self.alignment_type,
            #                                                   self.snli_file,
            #                                                   self.output_dir)
            #
            # sentences_one_parse_translated_align = utils.compute_alignment(sentences_one_parse, self.lang_source,
            #                                                   sentences_one_parse_translated,
            #                                                   self.lang_target,
            #                                                   self.alignment_type,
            #                                                   self.snli_file,
            #                                                   self.output_dir)
            # sentences_two_parse_translated_align = utils.compute_alignment(sentences_two_parse, self.lang_source,
            #                                                   sentences_two_parse_translated,
            #                                                   self.lang_target,
            #                                                   self.alignment_type,
            #                                                   self.snli_file,
            #                                                   self.output_dir)
            # sentences_one_binary_parse_translated_align = utils.compute_alignment(sentences_one_binary_parse, self.lang_source,
            #                                                   sentences_one_binary_parse_translated,
            #                                                   self.lang_target,
            #                                                   self.alignment_type,
            #                                                   self.snli_file,
            #                                                   self.output_dir)
            # sentences_two_binary_parse_translated_align = utils.compute_alignment(sentences_two_binary_parse, self.lang_source,
            #                                                   sentences_two_binary_parse_translated,
            #                                                   self.lang_target,
            #                                                   self.alignment_type,
            #                                                   self.snli_file,
            #                                                   self.output_dir)


            logging.info('Collected {} sentence to translate'.format(len(sentences_one)))

            i = 0
            new_content_lines = []
            for content in tqdm(content_lines):
                content_line = {}
                content_line['sentence1'] = sentence_one_translated[i]
                content_line['sentence2'] = sentence_two_translated[i]
                content_line['sentence1_parse'] = sentences_one_parse_translated[i]
                content_line['sentence2_parse'] = sentences_two_parse_translated[i]
                content_line['sentence1_binary_parse'] = sentences_one_binary_parse_translated[i]
                content_line['sentence2_binary_parse'] = sentences_two_binary_parse_translated[i]
                content_line['annotator_labels'] = content['annotator_labels']
                content_line['captionID'] = content['captionID']
                content_line['gold_label'] = content['gold_label']
                content_line['pairID'] = content['pairID']

                print(content_line)
                new_content_lines.append(content_line)

            with open(content_translations_alignments_file, 'wb') as fn:
                pickle.dump(new_content_lines, fn)

            translated_file = os.path.join(self.output_dir,
                                       os.path.basename(self.snli_file).replace(
                                           '.json',
                                           '-{}_small.json'.format(self.lang_target)))

            with open(translated_file, 'w') as fn:
                json.dump(new_content_lines, fn)

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


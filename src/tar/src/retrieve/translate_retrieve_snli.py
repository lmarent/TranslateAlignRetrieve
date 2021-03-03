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
                sentences_one.append(squad_utils.tokenize_sentences(content['sentence1'],
                                                              lang=self.lang_source))
                sentences_two.append(squad_utils.tokenize_sentences(content['sentence2'],
                                                              lang=self.lang_source))
                sentences_one_parse.append(squad_utils.tokenize_sentences(content['sentence1_parse'],
                                                                    lang=self.lang_source))
                sentences_two_parse.append(squad_utils.tokenize_sentences(content['sentence2_parse'],
                                                                    lang=self.lang_source))
                sentences_one_binary_parse.append(squad_utils.tokenize_sentences(content['sentence1_binary_parse'],
                                                                           lang=self.lang_source))
                sentences_two_binary_parse.append(squad_utils.tokenize_sentences(content['sentence2_binary_parse'],
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

            logging.info('Collected {} sentence to translate'.format(len(sentences_one)))

            i = 0
            new_content_lines = []
            for content in tqdm(content_lines):
                content_line = {}
                content_line['sentencia1'] = sentence_one_translated[i]
                content_line['sentencia2'] = sentence_two_translated[i]
                content_line['sentencia_uno_parse'] = sentences_one_parse_translated[i]
                content_line['sentencia_dos_parse'] = sentences_two_parse_translated[i]
                content_line['sentencia_uno_binaria_parse'] = sentences_one_binary_parse_translated[i]
                content_line['sentencia_dos_binaria_parse'] = sentences_two_binary_parse_translated[i]
                content_line['labels_anotador'] = content['annotator_labels']
                content_line['idLeyenda'] = content['captionID']
                content_line['etiqueta_predominante'] = content['gold_label']
                content_line['idPar'] = content['pairID']
                new_content_lines.append(content_line)
            # Translate contexts, questions and answers all together and write to file.
            # Also remove duplicates before to translate with set
            #content_translated = utils.translate(content, self.squad_file, self.output_dir, self.batch_size)

            # Compute alignments
            # context_sentence_questions_answers_alignments = utils.compute_alignment(content,
            #                                                                         self.lang_source,
            #                                                                         content_translated,
            #                                                                         self.lang_target,
            #                                                                         self.alignment_type,
            #                                                                         self.squad_file,
            #                                                                         self.output_dir)
            #
            # # Add translations and alignments
            # for sentence, sentence_translated, alignment in zip(content,
            #                                                     content_translated,
            #                                                     context_sentence_questions_answers_alignments):
            #     self.content_translations_alignments[sentence] = {'translation': sentence_translated,
            #                                                    'alignment': alignment}
            with open(content_translations_alignments_file, 'wb') as fn:
                pickle.dump(new_content_lines, fn)

        # Load content translated and aligned from file
        else:
            logging.info('Use previously content translations and alignments')
            with open(content_translations_alignments_file, 'rb') as fn:
                self.content_translations_alignments = pickle.load(fn)

    def translate_retrieve(self):
        """
            # Parse the SNLI file and replace sentences with their translations
            # using the content_translations_alignments
        """
        with open(self.snli_file) as fn:
            content = json.load(fn)

        for data in tqdm(content['data']):
            title = data['title']
            data['title'] = self.content_translations_alignments[title]['translation']
            for paragraphs in data['paragraphs']:
                context = paragraphs['context']

                context_sentences = [s for s in squad_utils.tokenize_sentences(squad_utils.remove_line_breaks(context),
                                                                         lang=self.lang_source)]

                context_translated = ' '.join(self.content_translations_alignments[s]['translation']
                                              for s in context_sentences)
                context_alignment_tok = utils.compute_context_alignment(
                    [self.content_translations_alignments[s]['alignment']
                     for s in context_sentences])

                # Translate context and replace its value back in the paragraphs
                paragraphs['context'] = context_translated
                for qa in paragraphs['qas']:
                    question = qa['question']
                    question_translated = self.content_translations_alignments[question]['translation']
                    qa['question'] = question_translated

                    # Translate answers and plausible answers for SQUAD v2.0
                    if self.squad_version == 'v2.0':
                        if not qa['is_impossible']:
                            for answer in qa['answers']:
                                answer_translated = self.content_translations_alignments[answer['text']]['translation']
                                answer_translated, answer_translated_start = \
                                    squad_utils.extract_answer_translated(answer,
                                                                    answer_translated,
                                                                    context,
                                                                    context_translated,
                                                                    context_alignment_tok,
                                                                    self.answers_from_alignment)
                                answer['text'] = answer_translated
                                answer['answer_start'] = answer_translated_start

                        else:
                            for plausible_answer in qa['plausible_answers']:
                                plausible_answer_translated = self.content_translations_alignments[plausible_answer['text']]['translation']
                                answer_translated, answer_translated_start = \
                                    squad_utils.extract_answer_translated(plausible_answer,
                                                                    plausible_answer_translated,
                                                                    context,
                                                                    context_translated,
                                                                    context_alignment_tok,
                                                                    self.answers_from_alignment)
                                plausible_answer['text'] = answer_translated
                                plausible_answer['answer_start'] = answer_translated_start

                    # Translate answers for SQUAD v1.1
                    else:
                        for answer in qa['answers']:
                            answer_translated = self.content_translations_alignments[answer['text']]['translation']
                            answer_translated, answer_translated_start = \
                                squad_utils.extract_answer_translated(answer,
                                                                answer_translated,
                                                                context,
                                                                context_translated,
                                                                context_alignment_tok,
                                                                self.answers_from_alignment)
                            answer['text'] = answer_translated
                            answer['answer_start'] = answer_translated_start



        logging.info('Cleaning and refinements...')
        # Parse the file, create a copy of the translated version and clean it from empty answers
        content_translated = content
        content_cleaned = {'version': content['version'], 'data': []}
        total_answers = 0
        total_correct_plausible_answers = 0
        total_correct_answers = 0
        for idx_data, data in tqdm(enumerate(content_translated['data'])):
            content_title = content_translated['data'][idx_data]['title']
            content_cleaned['data'].append({'title': content_title, 'paragraphs': []})
            for par in data['paragraphs']:
                qas_cleaned = []
                for idx_qa, qa in enumerate(par['qas']):
                    question = qa['question']

                    # Extract answers and plausible answers for SQUAD v2.0
                    if self.squad_version  == 'v2.0':
                        if not qa['is_impossible']:
                            correct_answers = []
                            for a in qa['answers']:
                                total_answers += 1
                                if a['text']:
                                    total_correct_answers += 1
                                    correct_answers.append(a)
                            correct_plausible_answers = []
                        else:
                            correct_plausible_answers = []
                            for pa in qa['plausible_answers']:
                                total_answers += 1
                                if pa['text']:
                                    total_correct_plausible_answers += 1
                                    correct_plausible_answers.append(pa)
                            correct_answers = []

                        # add answers and plausible answers to the content cleaned
                        if correct_answers:
                            content_qas_id = qa['id']
                            content_qas_is_impossible = qa['is_impossible']
                            correct_answers_from_context = []
                            for a in qa['answers']:
                                start = a['answer_start']
                                correct_answers_from_context.append(
                                    {'text': par['context'][start:start + len(a['text'])],
                                     'answer_start': start})
                            qa_cleaned = {'question': question,
                                          'answers': correct_answers_from_context,
                                          'id': content_qas_id,
                                          'is_impossible': content_qas_is_impossible}
                            qas_cleaned.append(qa_cleaned)
                        if correct_plausible_answers and not correct_answers:
                            content_qas_id = qa['id']
                            content_qas_is_impossible = qa['is_impossible']
                            correct_answers_from_context = []
                            for a in qa['answers']:
                                start = a['answer_start']
                                correct_answers_from_context.append(
                                    {'text': par['context'][start:start + len(a['text'])],
                                     'answer_start': start})
                            qa_cleaned = {'question': question,
                                          'answers': correct_answers,
                                          'plausible_answers': correct_plausible_answers,
                                          'id': content_qas_id,
                                          'is_impossible': content_qas_is_impossible}
                            qas_cleaned.append(qa_cleaned)

                    # Extract answers for SQUAD v1.0
                    else:
                        correct_answers = []
                        for a in qa['answers']:
                            total_answers += 1
                            if a['text']:
                                total_correct_answers += 1
                                correct_answers.append(a)

                        # add answers and plausible answers to the content cleaned
                        if correct_answers:
                            content_qas_id = qa['id']
                            correct_answers_from_context = []
                            for a in qa['answers']:
                                start = a['answer_start']
                                correct_answers_from_context.append(
                                    {'text': par['context'][start:start + len(a['text'])],
                                     'answer_start': start})
                            qa_cleaned = {'question': question,
                                          'answers': correct_answers_from_context,
                                          'id': content_qas_id}
                            qas_cleaned.append(qa_cleaned)

                # Add the paragraph only if there are non-empty question-answer examples inside
                if qas_cleaned:
                    content_context = par['context']
                    content_cleaned['data'][idx_data]['paragraphs'].append(
                        {'context': content_context, 'qas': qas_cleaned})

        # Write the content back to the translated dataset
        if self.answers_from_alignment:
            translated_file = os.path.join(self.output_dir,
                                           os.path.basename(self.snli_file).replace(
                                               '.json',
                                               '-{}.json'.format(self.lang_target)))
        else:
            translated_file = os.path.join(self.output_dir,
                                           os.path.basename(self.snli_file).replace(
                                               '.json',
                                               '-{}_small.json'.format(self.lang_target)))

        with open(translated_file, 'w') as fn:
            json.dump(content_cleaned, fn)

        # Count correct answers and plausible answers for SQUAD v2.0
        if self.squad_version  == 'v2.0':
            total_correct = total_correct_answers + total_correct_plausible_answers
            accuracy = round((total_correct/ total_answers) * 100, 2)
            logging.info('File: {}:\n'
                  'Percentage of translated examples (correct answers/total answers): {}/{} = {}%\n'
                  'No. of answers: {}\n'
                  'No. of plausible answers: {}'.format(os.path.basename(translated_file),
                                                       total_correct,
                                                       total_answers,
                                                       accuracy,
                                                       total_correct_answers,
                                                       total_correct_plausible_answers))

        # Count correct answers
        else:
            total_correct = total_correct_answers
            accuracy = round((total_correct/ total_answers) * 100, 2)
            logging.info('File: {}:\n'
                  'Percentage of translated examples (correct answers/total answers): {}/{} = {}%\n'
                  'No. of answers: {}'.format(os.path.basename(translated_file),
                                                              total_correct,
                                                              total_answers,
                                                              accuracy,
                                                          total_correct_answers))

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

    logging.info('Translate SNLI textual content and compute alignments...')
    translator.translate_align_content()

    logging.info('Translate and retrieve the SNLI dataset...')
    translator.translate_retrieve()

    end = time.time()
    logging.info('Total execution time: {} s'.format(round(end - start)))


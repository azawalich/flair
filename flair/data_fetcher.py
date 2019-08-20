
import os
from typing import List, Dict, Union
import re
import logging
from enum import Enum
from pathlib import Path
from deprecated import deprecated
import flair
from flair.data import Sentence, Corpus, Token, MultiCorpus
from flair.file_utils import cached_path
log = logging.getLogger('flair')


class NLPTask(Enum):
    CONLL_2000 = 'conll_2000'
    CONLL_03 = 'conll_03'
    CONLL_03_GERMAN = 'conll_03_german'
    CONLL_03_DUTCH = 'conll_03_dutch'
    CONLL_03_SPANISH = 'conll_03_spanish'
    WNUT_17 = 'wnut_17'
    WIKINER_ENGLISH = 'wikiner_english'
    WIKINER_GERMAN = 'wikiner_german'
    WIKINER_FRENCH = 'wikiner_french'
    WIKINER_SPANISH = 'wikiner_spanish'
    WIKINER_ITALIAN = 'wikiner_italian'
    WIKINER_DUTCH = 'wikiner_dutch'
    WIKINER_POLISH = 'wikiner_polish'
    WIKINER_PORTUGUESE = 'wikiner_portuguese'
    WIKINER_RUSSIAN = 'wikiner_russian'
    UD_ENGLISH = 'ud_english'
    UD_GERMAN = 'ud_german'
    UD_DUTCH = 'ud_dutch'
    UD_FRENCH = 'ud_french'
    UD_ITALIAN = 'ud_italian'
    UD_SPANISH = 'ud_spanish'
    UD_PORTUGUESE = 'ud_portuguese'
    UD_ROMANIAN = 'ud_romanian'
    UD_CATALAN = 'ud_catalan'
    UD_POLISH = 'ud_polish'
    UD_CZECH = 'ud_czech'
    UD_SLOVAK = 'ud_slovak'
    UD_SLOVENIAN = 'ud_slovenian'
    UD_CROATIAN = 'ud_croatian'
    UD_SERBIAN = 'ud_serbian'
    UD_BULGARIAN = 'ud_bulgarian'
    UD_RUSSIAN = 'ud_russian'
    UD_SWEDISH = 'ud_swedish'
    UD_DANISH = 'ud_danish'
    UD_NORWEGIAN = 'ud_norwegian'
    UD_FINNISH = 'ud_finnish'
    UD_ARABIC = 'ud_arabic'
    UD_HEBREW = 'ud_hebrew'
    UD_TURKISH = 'ud_turkish'
    UD_PERSIAN = 'ud_persian'
    UD_HINDI = 'ud_hindi'
    UD_INDONESIAN = 'ud_indonesian'
    UD_JAPANESE = 'ud_japanese'
    UD_CHINESE = 'ud_chinese'
    UD_KOREAN = 'ud_korean'
    UD_BASQUE = 'ud_basque'
    UD_GERMAN_HDT = 'ud_german_hdt'
    ONTONER = 'ontoner'
    FASHION = 'fashion'
    GERMEVAL = 'germeval'
    SRL = 'srl'
    WSD = 'wsd'
    CONLL_12 = 'conll_12'
    PENN = 'penn'
    ONTONOTES = 'ontonotes'
    NER_BASQUE = 'eiec'
    IMDB = 'imdb'
    AG_NEWS = 'ag_news'
    TREC_6 = 'trec-6'
    TREC_50 = 'trec-50'
    REGRESSION = 'regression'
    WASSA_ANGER = 'wassa-anger'
    WASSA_FEAR = 'wassa-fear'
    WASSA_JOY = 'wassa-joy'
    WASSA_SADNESS = 'wassa-sadness'


class NLPTaskDataFetcher():

    @staticmethod
    @deprecated(version='0.4.1', reason="Use 'flair.datasets' instead.")
    def load_corpora(tasks, base_path=None):
        return MultiCorpus([NLPTaskDataFetcher.load_corpus(task, base_path) for task in tasks])

    @staticmethod
    @deprecated(version='0.4.1', reason="Use 'flair.datasets' instead.")
    def load_corpus(task, base_path=None):
        '\n        Helper function to fetch a Corpus for a specific NLPTask. For this to work you need to first download\n        and put into the appropriate folder structure the corresponding NLP task data. The tutorials on\n        https://github.com/zalandoresearch/flair give more info on how to do this. Alternatively, you can use this\n        code to create your own data fetchers.\n        :param task: specification of the NLPTask you wish to get\n        :param base_path: path to data folder containing tasks sub folders\n        :return: a Corpus consisting of train, dev and test data\n        '
        if (type(task) is NLPTask):
            NLPTaskDataFetcher.download_dataset(task)
        if (not base_path):
            base_path = (Path(flair.cache_root) / 'datasets')
        if (type(base_path) == str):
            base_path = Path(base_path)
        task = (task.value if (type(task) is NLPTask) else task)
        data_folder = (base_path / task.lower())
        if (task == NLPTask.CONLL_2000.value):
            columns = {
                0: 'text',
                1: 'pos',
                2: 'np',
            }
            return NLPTaskDataFetcher.load_column_corpus(data_folder, columns, tag_to_biloes='np')
        if ((task == NLPTask.CONLL_03.value) or (task == NLPTask.ONTONER.value) or (task == NLPTask.FASHION.value)):
            columns = {
                0: 'text',
                1: 'pos',
                2: 'np',
                3: 'ner',
            }
            return NLPTaskDataFetcher.load_column_corpus(data_folder, columns, tag_to_biloes='ner')
        if (task == NLPTask.CONLL_03_GERMAN.value):
            columns = {
                0: 'text',
                1: 'lemma',
                2: 'pos',
                3: 'np',
                4: 'ner',
            }
            return NLPTaskDataFetcher.load_column_corpus(data_folder, columns, tag_to_biloes='ner')
        if ((task == NLPTask.CONLL_03_DUTCH.value) or task.startswith('wikiner')):
            columns = {
                0: 'text',
                1: 'pos',
                2: 'ner',
            }
            return NLPTaskDataFetcher.load_column_corpus(data_folder, columns, tag_to_biloes='ner')
        if ((task == NLPTask.CONLL_03_SPANISH.value) or (task == NLPTask.WNUT_17.value)):
            columns = {
                0: 'text',
                1: 'ner',
            }
            return NLPTaskDataFetcher.load_column_corpus(data_folder, columns, tag_to_biloes='ner')
        if (task == NLPTask.GERMEVAL.value):
            columns = {
                1: 'text',
                2: 'ner',
            }
            return NLPTaskDataFetcher.load_column_corpus(data_folder, columns, tag_to_biloes='ner')
        if (task == NLPTask.WSD.value):
            columns = {
                0: 'text',
                1: 'lemma',
                2: 'pos',
                3: 'sense',
            }
            return NLPTaskDataFetcher.load_column_corpus(data_folder, columns, train_file='semcor.tsv', test_file='semeval2015.tsv')
        if (task.startswith('ud_') or (task in [NLPTask.ONTONOTES.value, NLPTask.CONLL_12.value, NLPTask.PENN.value])):
            return NLPTaskDataFetcher.load_ud_corpus(data_folder)
        if (task in [NLPTask.IMDB.value, NLPTask.AG_NEWS.value, NLPTask.TREC_6.value, NLPTask.TREC_50.value, NLPTask.REGRESSION.value]):
            use_tokenizer = (
                False if (task in [NLPTask.TREC_6.value, NLPTask.TREC_50.value]) else True)
            return NLPTaskDataFetcher.load_classification_corpus(data_folder, use_tokenizer=use_tokenizer)
        if (task == NLPTask.NER_BASQUE.value):
            columns = {
                0: 'text',
                1: 'ner',
            }
            return NLPTaskDataFetcher.load_column_corpus(data_folder, columns, tag_to_biloes='ner')
        if task.startswith('wassa'):
            return NLPTaskDataFetcher.load_classification_corpus(data_folder, use_tokenizer=True)

    @staticmethod
    @deprecated(version='0.4.1', reason="Use 'flair.datasets' instead.")
    def load_column_corpus(data_folder, column_format, train_file=None, test_file=None, dev_file=None, tag_to_biloes=None):
        '\n        Helper function to get a Corpus from CoNLL column-formatted task data such as CoNLL03 or CoNLL2000.\n\n        :param data_folder: base folder with the task data\n        :param column_format: a map specifying the column format\n        :param train_file: the name of the train file\n        :param test_file: the name of the test file\n        :param dev_file: the name of the dev file, if None, dev data is sampled from train\n        :param tag_to_biloes: whether to convert to BILOES tagging scheme\n        :return: a Corpus with annotated train, dev and test data\n        '
        if (type(data_folder) == str):
            data_folder = Path(data_folder)
        if (train_file is not None):
            train_file = (data_folder / train_file)
        if (test_file is not None):
            test_file = (data_folder / test_file)
        if (dev_file is not None):
            dev_file = (data_folder / dev_file)
        if (train_file is None):
            for file in data_folder.iterdir():
                file_name = file.name
                if file_name.endswith('.gz'):
                    continue
                if (('train' in file_name) and (not ('54019' in file_name))):
                    train_file = file
                if ('dev' in file_name):
                    dev_file = file
                if ('testa' in file_name):
                    dev_file = file
                if ('testb' in file_name):
                    test_file = file
            if (test_file is None):
                for file in data_folder.iterdir():
                    file_name = file.name
                    if file_name.endswith('.gz'):
                        continue
                    if ('test' in file_name):
                        test_file = file
        log.info('Reading data from {}'.format(data_folder))
        log.info('Train: {}'.format(train_file))
        log.info('Dev: {}'.format(dev_file))
        log.info('Test: {}'.format(test_file))
        sentences_train = NLPTaskDataFetcher.read_column_data(
            train_file, column_format)
        if (test_file is not None):
            sentences_test = NLPTaskDataFetcher.read_column_data(
                test_file, column_format)
        else:
            sentences_test = [sentences_train[i]
                              for i in NLPTaskDataFetcher.__sample(len(sentences_train), 0.1)]
            sentences_train = [
                x for x in sentences_train if (x not in sentences_test)]
        if (dev_file is not None):
            sentences_dev = NLPTaskDataFetcher.read_column_data(
                dev_file, column_format)
        else:
            sentences_dev = [sentences_train[i]
                             for i in NLPTaskDataFetcher.__sample(len(sentences_train), 0.1)]
            sentences_train = [
                x for x in sentences_train if (x not in sentences_dev)]
        if (tag_to_biloes is not None):
            for sentence in ((sentences_train + sentences_test) + sentences_dev):
                sentence.convert_tag_scheme(
                    tag_type=tag_to_biloes, target_scheme='iobes')
        return Corpus(sentences_train, sentences_dev, sentences_test, name=data_folder.name)

    @staticmethod
    @deprecated(version='0.4.1', reason="Use 'flair.datasets' instead.")
    def load_ud_corpus(data_folder, train_file=None, test_file=None, dev_file=None):
        '\n        Helper function to get a Corpus from CoNLL-U column-formatted task data such as the UD corpora\n\n        :param data_folder: base folder with the task data\n        :param train_file: the name of the train file\n        :param test_file: the name of the test file\n        :param dev_file: the name of the dev file, if None, dev data is sampled from train\n        :return: a Corpus with annotated train, dev and test data\n        '
        if (train_file is None):
            for file in data_folder.iterdir():
                file_name = file.name
                if ('train' in file_name):
                    train_file = file
                if ('test' in file_name):
                    test_file = file
                if ('dev' in file_name):
                    dev_file = file
                if ('testa' in file_name):
                    dev_file = file
                if ('testb' in file_name):
                    test_file = file
        log.info('Reading data from {}'.format(data_folder))
        log.info('Train: {}'.format(train_file))
        log.info('Dev: {}'.format(dev_file))
        log.info('Test: {}'.format(test_file))
        sentences_train = NLPTaskDataFetcher.read_conll_ud(train_file)
        sentences_test = NLPTaskDataFetcher.read_conll_ud(test_file)
        sentences_dev = NLPTaskDataFetcher.read_conll_ud(dev_file)
        return Corpus(sentences_train, sentences_dev, sentences_test, name=data_folder.name)

    @staticmethod
    @deprecated(version='0.4.1', reason="Use 'flair.datasets' instead.")
    def load_classification_corpus(data_folder, train_file=None, test_file=None, dev_file=None, use_tokenizer=True, max_tokens_per_doc=(- 1)):
        '\n        Helper function to get a Corpus from text classification-formatted task data\n\n        :param data_folder: base folder with the task data\n        :param train_file: the name of the train file\n        :param test_file: the name of the test file\n        :param dev_file: the name of the dev file, if None, dev data is sampled from train\n        :return: a Corpus with annotated train, dev and test data\n        '
        if (type(data_folder) == str):
            data_folder = Path(data_folder)
        if (train_file is not None):
            train_file = (data_folder / train_file)
        if (test_file is not None):
            test_file = (data_folder / test_file)
        if (dev_file is not None):
            dev_file = (data_folder / dev_file)
        if (train_file is None):
            for file in data_folder.iterdir():
                file_name = file.name
                if ('train' in file_name):
                    train_file = file
                if ('test' in file_name):
                    test_file = file
                if ('dev' in file_name):
                    dev_file = file
                if ('testa' in file_name):
                    dev_file = file
                if ('testb' in file_name):
                    test_file = file
        log.info('Reading data from {}'.format(data_folder))
        log.info('Train: {}'.format(train_file))
        log.info('Dev: {}'.format(dev_file))
        log.info('Test: {}'.format(test_file))
        sentences_train = NLPTaskDataFetcher.read_text_classification_file(
            train_file, use_tokenizer=use_tokenizer, max_tokens_per_doc=max_tokens_per_doc)
        sentences_test = NLPTaskDataFetcher.read_text_classification_file(
            test_file, use_tokenizer=use_tokenizer, max_tokens_per_doc=max_tokens_per_doc)
        if (dev_file is not None):
            sentences_dev = NLPTaskDataFetcher.read_text_classification_file(
                dev_file, use_tokenizer=use_tokenizer, max_tokens_per_doc=max_tokens_per_doc)
        else:
            sentences_dev = [sentences_train[i]
                             for i in NLPTaskDataFetcher.__sample(len(sentences_train), 0.1)]
            sentences_train = [
                x for x in sentences_train if (x not in sentences_dev)]
        return Corpus(sentences_train, sentences_dev, sentences_test)

    @staticmethod
    @deprecated(version='0.4.1', reason="Use 'flair.datasets' instead.")
    def read_text_classification_file(path_to_file, max_tokens_per_doc=(- 1), use_tokenizer=True):
        '\n        Reads a data file for text classification. The file should contain one document/text per line.\n        The line should have the following format:\n        __label__<class_name> <text>\n        If you have a multi class task, you can have as many labels as you want at the beginning of the line, e.g.,\n        __label__<class_name_1> __label__<class_name_2> <text>\n        :param path_to_file: the path to the data file\n        :param max_tokens_per_doc: Takes at most this amount of tokens per document. If set to -1 all documents are taken as is.\n        :return: list of sentences\n        '
        label_prefix = '__label__'
        sentences = []
        with open(str(path_to_file), encoding='utf-8') as f:
            for line in f:
                words = line.split()
                labels = []
                l_len = 0
                for i in range(len(words)):
                    if words[i].startswith(label_prefix):
                        l_len += (len(words[i]) + 1)
                        label = words[i].replace(label_prefix, '')
                        labels.append(label)
                    else:
                        break
                text = line[l_len:].strip()
                if (text and labels):
                    sentence = Sentence(text, labels=labels,
                                        use_tokenizer=use_tokenizer)
                    if ((len(sentence) > max_tokens_per_doc) and (max_tokens_per_doc > 0)):
                        sentence.tokens = sentence.tokens[:max_tokens_per_doc]
                    if (len(sentence.tokens) > 0):
                        sentences.append(sentence)
        return sentences

    @staticmethod
    @deprecated(version='0.4.1', reason="Use 'flair.datasets' instead.")
    def read_column_data(path_to_column_file, column_name_map, infer_whitespace_after=True):
        '\n        Reads a file in column format and produces a list of Sentence with tokenlevel annotation as specified in the\n        column_name_map. For instance, by passing "{0: \'text\', 1: \'pos\', 2: \'np\', 3: \'ner\'}" as column_name_map you\n        specify that the first column is the text (lexical value) of the token, the second the PoS tag, the third\n        the chunk and the forth the NER tag.\n        :param path_to_column_file: the path to the column file\n        :param column_name_map: a map of column number to token annotation name\n        :param infer_whitespace_after: if True, tries to infer whitespace_after field for Token\n        :return: list of sentences\n        '
        sentences = []
        try:
            lines = open(str(path_to_column_file),
                         encoding='utf-8').read().strip().split('\n')
        except:
            log.info(
                'UTF-8 can\'t read: {} ... using "latin-1" instead.'.format(path_to_column_file))
            lines = open(str(path_to_column_file),
                         encoding='latin1').read().strip().split('\n')
        text_column = 0
        for column in column_name_map:
            if (column_name_map[column] == 'text'):
                text_column = column
        sentence = Sentence()
        for line in lines:
            if line.startswith('#'):
                continue
            if (line.strip().replace('\ufeff', '') == ''):
                if (len(sentence) > 0):
                    sentence.infer_space_after()
                    sentences.append(sentence)
                sentence = Sentence()
            else:
                fields = re.split('\\s+', line)
                token = Token(fields[text_column])
                for column in column_name_map:
                    if (len(fields) > column):
                        if (column != text_column):
                            token.add_tag(
                                column_name_map[column], fields[column])
                sentence.add_token(token)
        if (len(sentence.tokens) > 0):
            sentence.infer_space_after()
            sentences.append(sentence)
        return sentences

    @staticmethod
    @deprecated(version='0.4.1', reason="Use 'flair.datasets' instead.")
    def read_conll_ud(path_to_conll_file):
        '\n       Reads a file in CoNLL-U format and produces a list of Sentence with full morphosyntactic annotation\n       :param path_to_conll_file: the path to the conll-u file\n       :return: list of sentences\n       '
        sentences = []
        lines = open(path_to_conll_file,
                     encoding='utf-8').read().strip().split('\n')
        sentence = Sentence()
        for line in lines:
            fields = re.split('\t+', line)
            if (line == ''):
                if (len(sentence) > 0):
                    sentences.append(sentence)
                sentence = Sentence()
            elif line.startswith('#'):
                continue
            elif ('.' in fields[0]):
                continue
            elif ('-' in fields[0]):
                continue
            else:
                token = Token(fields[1], head_id=int(fields[6]))
                token.add_tag('lemma', str(fields[2]))
                token.add_tag('upos', str(fields[3]))
                token.add_tag('pos', str(fields[4]))
                token.add_tag('dependency', str(fields[7]))
                for morph in str(fields[5]).split('|'):
                    if (not ('=' in morph)):
                        continue
                    token.add_tag(morph.split(
                        '=')[0].lower(), morph.split('=')[1])
                if ((len(fields) > 10) and (str(fields[10]) == 'Y')):
                    token.add_tag('frame', str(fields[11]))
                sentence.add_token(token)
        if (len(sentence.tokens) > 0):
            sentences.append(sentence)
        return sentences

    @staticmethod
    def __sample(total_number_of_sentences, percentage=0.1):
        import random
        sample_size = round((total_number_of_sentences * percentage))
        sample = random.sample(
            range(1, total_number_of_sentences), sample_size)
        return sample

    @staticmethod
    def download_dataset(task):
        if (task == NLPTask.CONLL_2000):
            conll_2000_path = 'https://www.clips.uantwerpen.be/conll2000/chunking/'
            data_file = (
                ((Path(flair.cache_root) / 'datasets') / task.value) / 'train.txt')
            if (not data_file.is_file()):
                cached_path(''.join(['{}'.format(conll_2000_path), 'train.txt.gz']), (Path(
                    'datasets') / task.value))
                cached_path(''.join(['{}'.format(conll_2000_path), 'test.txt.gz']), (Path(
                    'datasets') / task.value))
                import gzip
                import shutil
                with gzip.open((((Path(flair.cache_root) / 'datasets') / task.value) / 'train.txt.gz'), 'rb') as f_in:
                    with open((((Path(flair.cache_root) / 'datasets') / task.value) / 'train.txt'), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                with gzip.open((((Path(flair.cache_root) / 'datasets') / task.value) / 'test.txt.gz'), 'rb') as f_in:
                    with open((((Path(flair.cache_root) / 'datasets') / task.value) / 'test.txt'), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
        if (task == NLPTask.NER_BASQUE):
            ner_basque_path = 'http://ixa2.si.ehu.eus/eiec/'
            data_path = ((Path(flair.cache_root) / 'datasets') / task.value)
            data_file = (data_path / 'named_ent_eu.train')
            if (not data_file.is_file()):
                cached_path(''.join(['{}'.format(
                    ner_basque_path), '/eiec_v1.0.tgz']), (Path('datasets') / task.value))
                import tarfile
                import shutil
                with tarfile.open((((Path(flair.cache_root) / 'datasets') / task.value) / 'eiec_v1.0.tgz'), 'r:gz') as f_in:
                    corpus_files = ('eiec_v1.0/named_ent_eu.train',
                                    'eiec_v1.0/named_ent_eu.test')
                    for corpus_file in corpus_files:
                        f_in.extract(corpus_file, data_path)
                        shutil.move(
                            ''.join(['{}'.format(data_path), '/', '{}'.format(corpus_file)]), data_path)
        if (task == NLPTask.IMDB):
            imdb_acl_path = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
            data_path = ((Path(flair.cache_root) / 'datasets') / task.value)
            data_file = (data_path / 'train.txt')
            if (not data_file.is_file()):
                cached_path(imdb_acl_path, (Path('datasets') / task.value))
                import tarfile
                with tarfile.open((((Path(flair.cache_root) / 'datasets') / task.value) / 'aclImdb_v1.tar.gz'), 'r:gz') as f_in:
                    datasets = ['train', 'test']
                    labels = ['pos', 'neg']
                    for label in labels:
                        for dataset in datasets:
                            f_in.extractall(data_path, members=[m for m in f_in.getmembers() if (
                                ''.join(['{}'.format(dataset), '/', '{}'.format(label)]) in m.name)])
                            with open(''.join(['{}'.format(data_path), '/', '{}'.format(dataset), '.txt']), 'at') as f_p:
                                current_path = (
                                    ((data_path / 'aclImdb') / dataset) / label)
                                for file_name in current_path.iterdir():
                                    if (file_name.is_file() and file_name.name.endswith('.txt')):
                                        f_p.write(((''.join(['__label__', '{}'.format(
                                            label), ' ']) + file_name.open('rt', encoding='utf-8').read()) + '\n'))
        if task.value.startswith('trec'):
            trec_path = 'http://cogcomp.org/Data/QA/QC/'
            original_filenames = ['train_5500.label', 'TREC_10.label']
            new_filenames = ['train.txt', 'test.txt']
            for original_filename in original_filenames:
                cached_path(''.join(['{}'.format(trec_path), '{}'.format(
                    original_filename)]), ((Path('datasets') / task.value) / 'original'))
            data_path = ((Path(flair.cache_root) / 'datasets') / task.value)
            data_file = (data_path / new_filenames[0])
            if (not data_file.is_file()):
                for (original_filename, new_filename) in zip(original_filenames, new_filenames):
                    with open(((data_path / 'original') / original_filename), 'rt', encoding='latin1') as open_fp:
                        with open((data_path / new_filename), 'wt', encoding='utf-8') as write_fp:
                            for line in open_fp:
                                line = line.rstrip()
                                fields = line.split()
                                old_label = fields[0]
                                question = ' '.join(fields[1:])
                                new_label = '__label__'
                                new_label += (old_label.split(':')
                                              [0] if (task.value == 'trec-6') else old_label)
                                write_fp.write(
                                    ''.join(['{}'.format(new_label), ' ', '{}'.format(question), '\n']))
        if (task == NLPTask.WNUT_17):
            wnut_path = 'https://noisy-text.github.io/2017/files/'
            cached_path(''.join(['{}'.format(wnut_path), 'wnut17train.conll']), (Path(
                'datasets') / task.value))
            cached_path(''.join(['{}'.format(wnut_path), 'emerging.dev.conll']), (Path(
                'datasets') / task.value))
            cached_path(''.join(['{}'.format(wnut_path), 'emerging.test.annotated']), (Path(
                'datasets') / task.value))
        wikiner_path = 'https://raw.githubusercontent.com/dice-group/FOX/master/input/Wikiner/'
        if task.value.startswith('wikiner'):
            lc = ''
            if (task == NLPTask.WIKINER_ENGLISH):
                lc = 'en'
            if (task == NLPTask.WIKINER_GERMAN):
                lc = 'de'
            if (task == NLPTask.WIKINER_DUTCH):
                lc = 'nl'
            if (task == NLPTask.WIKINER_FRENCH):
                lc = 'fr'
            if (task == NLPTask.WIKINER_ITALIAN):
                lc = 'it'
            if (task == NLPTask.WIKINER_SPANISH):
                lc = 'es'
            if (task == NLPTask.WIKINER_PORTUGUESE):
                lc = 'pt'
            if (task == NLPTask.WIKINER_POLISH):
                lc = 'pl'
            if (task == NLPTask.WIKINER_RUSSIAN):
                lc = 'ru'
            data_file = (((Path(flair.cache_root) / 'datasets') / task.value) /
                         ''.join(['aij-wikiner-', '{}'.format(lc), '-wp3.train']))
            if (not data_file.is_file()):
                cached_path(''.join(['{}'.format(wikiner_path), 'aij-wikiner-',
                                     '{}'.format(lc), '-wp3.bz2']), (Path('datasets') / task.value))
                import bz2
                import shutil
                bz_file = bz2.BZ2File((((Path(flair.cache_root) / 'datasets') / task.value) / ''.join(
                    ['aij-wikiner-', '{}'.format(lc), '-wp3.bz2'])), 'rb')
                with bz_file as f, open((((Path(flair.cache_root) / 'datasets') / task.value) / ''.join(['aij-wikiner-', '{}'.format(lc), '-wp3.train'])), 'w') as out:
                    for line in f:
                        line = line.decode('utf-8')
                        words = line.split(' ')
                        for word in words:
                            out.write(('\t'.join(word.split('|')) + '\n'))
        conll_02_path = 'https://www.clips.uantwerpen.be/conll2002/ner/data/'
        if (task == NLPTask.CONLL_03_DUTCH):
            cached_path(''.join(['{}'.format(conll_02_path),
                                 'ned.testa']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(conll_02_path),
                                 'ned.testb']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(conll_02_path),
                                 'ned.train']), (Path('datasets') / task.value))
        if (task == NLPTask.CONLL_03_SPANISH):
            cached_path(''.join(['{}'.format(conll_02_path),
                                 'esp.testa']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(conll_02_path),
                                 'esp.testb']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(conll_02_path),
                                 'esp.train']), (Path('datasets') / task.value))
        ud_path = 'https://raw.githubusercontent.com/UniversalDependencies/'
        if (task == NLPTask.UD_ENGLISH):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_English-EWT/master/en_ewt-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_English-EWT/master/en_ewt-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_English-EWT/master/en_ewt-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_GERMAN):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_German-GSD/master/de_gsd-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_German-GSD/master/de_gsd-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_German-GSD/master/de_gsd-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_DUTCH):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Dutch-Alpino/master/nl_alpino-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Dutch-Alpino/master/nl_alpino-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Dutch-Alpino/master/nl_alpino-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_FRENCH):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_French-GSD/master/fr_gsd-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_French-GSD/master/fr_gsd-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_French-GSD/master/fr_gsd-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_ITALIAN):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Italian-ISDT/master/it_isdt-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Italian-ISDT/master/it_isdt-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Italian-ISDT/master/it_isdt-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_SPANISH):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Spanish-GSD/master/es_gsd-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Spanish-GSD/master/es_gsd-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Spanish-GSD/master/es_gsd-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_PORTUGUESE):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Portuguese-Bosque/blob/master/pt_bosque-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Portuguese-Bosque/blob/master/pt_bosque-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Portuguese-Bosque/blob/master/pt_bosque-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_ROMANIAN):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Romanian-RRT/master/ro_rrt-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Romanian-RRT/master/ro_rrt-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Romanian-RRT/master/ro_rrt-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_CATALAN):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Catalan-AnCora/master/ca_ancora-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Catalan-AnCora/master/ca_ancora-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Catalan-AnCora/master/ca_ancora-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_POLISH):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Polish-LFG/master/pl_lfg-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Polish-LFG/master/pl_lfg-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Polish-LFG/master/pl_lfg-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_CZECH):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Czech-PDT/master/cs_pdt-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Czech-PDT/master/cs_pdt-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Czech-PDT/master/cs_pdt-ud-train-l.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_SLOVAK):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Slovak-SNK/master/sk_snk-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Slovak-SNK/master/sk_snk-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Slovak-SNK/master/sk_snk-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_SWEDISH):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Swedish-Talbanken/master/sv_talbanken-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Swedish-Talbanken/master/sv_talbanken-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Swedish-Talbanken/master/sv_talbanken-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_DANISH):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Danish-DDT/master/da_ddt-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Danish-DDT/master/da_ddt-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Danish-DDT/master/da_ddt-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_NORWEGIAN):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Norwegian-Bokmaal/master/no_bokmaal-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Norwegian-Bokmaal/master/no_bokmaal-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Norwegian-Bokmaal/master/no_bokmaal-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_FINNISH):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Finnish-TDT/master/fi_tdt-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Finnish-TDT/master/fi_tdt-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Finnish-TDT/master/fi_tdt-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_SLOVENIAN):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Slovenian-SSJ/master/sl_ssj-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Slovenian-SSJ/master/sl_ssj-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Slovenian-SSJ/master/sl_ssj-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_CROATIAN):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Croatian-SET/master/hr_set-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Croatian-SET/master/hr_set-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Croatian-SET/master/hr_set-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_SERBIAN):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Serbian-SET/master/sr_set-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Serbian-SET/master/sr_set-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Serbian-SET/master/sr_set-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_BULGARIAN):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Bulgarian-BTB/master/bg_btb-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Bulgarian-BTB/master/bg_btb-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Bulgarian-BTB/master/bg_btb-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_ARABIC):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Arabic-PADT/master/ar_padt-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Arabic-PADT/master/ar_padt-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Arabic-PADT/master/ar_padt-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_HEBREW):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Hebrew-HTB/master/he_htb-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Hebrew-HTB/master/he_htb-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Hebrew-HTB/master/he_htb-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_TURKISH):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Turkish-IMST/master/tr_imst-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Turkish-IMST/master/tr_imst-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Turkish-IMST/master/tr_imst-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_PERSIAN):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Persian-Seraji/master/fa_seraji-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Persian-Seraji/master/fa_seraji-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Persian-Seraji/master/fa_seraji-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_RUSSIAN):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Russian-SynTagRus/master/ru_syntagrus-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Russian-SynTagRus/master/ru_syntagrus-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Russian-SynTagRus/master/ru_syntagrus-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_HINDI):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Hindi-HDTB/master/hi_hdtb-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Hindi-HDTB/master/hi_hdtb-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Hindi-HDTB/master/hi_hdtb-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_INDONESIAN):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Indonesian-GSD/master/id_gsd-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Indonesian-GSD/master/id_gsd-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Indonesian-GSD/master/id_gsd-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_JAPANESE):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Japanese-GSD/master/ja_gsd-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Japanese-GSD/master/ja_gsd-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Japanese-GSD/master/ja_gsd-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_CHINESE):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Chinese-GSD/master/zh_gsd-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Chinese-GSD/master/zh_gsd-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Chinese-GSD/master/zh_gsd-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_KOREAN):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Korean-Kaist/master/ko_kaist-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Korean-Kaist/master/ko_kaist-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Korean-Kaist/master/ko_kaist-ud-train.conllu']), (Path('datasets') / task.value))
        if (task == NLPTask.UD_BASQUE):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Basque-BDT/master/eu_bdt-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Basque-BDT/master/eu_bdt-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_Basque-BDT/master/eu_bdt-ud-train.conllu']), (Path('datasets') / task.value))
        if task.value.startswith('wassa'):
            emotion = task.value[6:]
            for split in ['train', 'dev', 'test']:
                data_file = (((Path(flair.cache_root) / 'datasets') / task.value) /
                             ''.join(['{}'.format(emotion), '-', '{}'.format(split), '.txt']))
                if (not data_file.is_file()):
                    if (split == 'train'):
                        url = ''.join(['http://saifmohammad.com/WebDocs/EmoInt%20Train%20Data/',
                                       '{}'.format(emotion), '-ratings-0to1.train.txt'])
                    if (split == 'dev'):
                        url = ''.join(['http://saifmohammad.com/WebDocs/EmoInt%20Dev%20Data%20With%20Gold/',
                                       '{}'.format(emotion), '-ratings-0to1.dev.gold.txt'])
                    if (split == 'test'):
                        url = ''.join(['http://saifmohammad.com/WebDocs/EmoInt%20Test%20Gold%20Data/',
                                       '{}'.format(emotion), '-ratings-0to1.test.gold.txt'])
                    path = cached_path(url, (Path('datasets') / task.value))
                    with open(path, 'r') as f:
                        with open(data_file, 'w') as out:
                            next(f)
                            for line in f:
                                fields = line.split('\t')
                                out.write(''.join(['__label__', '{}'.format(
                                    fields[3].rstrip()), ' ', '{}'.format(fields[1]), '\n']))
                    os.remove(path)
        if (task == NLPTask.UD_GERMAN_HDT):
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_German-HDT/dev/de_hdt-ud-dev.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(
                ud_path), 'UD_German-HDT/dev/de_hdt-ud-test.conllu']), (Path('datasets') / task.value))
            cached_path(''.join(['{}'.format(ud_path), 'UD_German-HDT/dev/de_hdt-ud-train-a.conllu']),
                        ((Path('datasets') / task.value) / 'original'))
            cached_path(''.join(['{}'.format(ud_path), 'UD_German-HDT/dev/de_hdt-ud-train-b.conllu']),
                        ((Path('datasets') / task.value) / 'original'))
            data_path = ((Path(flair.cache_root) / 'datasets') / task.value)
            train_filenames = ['de_hdt-ud-train-a.conllu',
                               'de_hdt-ud-train-b.conllu']
            new_train_file = (data_path / 'de_hdt-ud-train-all.conllu')
            if (not new_train_file.is_file()):
                with open(new_train_file, 'wt') as f_out:
                    for train_filename in train_filenames:
                        with open(((data_path / 'original') / train_filename), 'rt') as f_in:
                            f_out.write(f_in.read())

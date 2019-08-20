
import os
import re
import logging
from abc import abstractmethod
from collections import Counter
from pathlib import Path
from typing import List, Union, Dict
import gensim
import numpy as np
import torch
from bpemb import BPEmb
from deprecated import deprecated
from torch.nn import ParameterList, Parameter
from pytorch_transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, TransfoXLTokenizer, TransfoXLModel, OpenAIGPTModel, OpenAIGPTTokenizer, GPT2Model, GPT2Tokenizer, XLNetTokenizer, XLMTokenizer, XLNetModel, XLMModel, PreTrainedTokenizer, PreTrainedModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import flair
from flair.data import Corpus
from .nn import LockedDropout, WordDropout
from .data import Dictionary, Token, Sentence
from .file_utils import cached_path, open_inside_zip
from .training_utils import log_line
log = logging.getLogger('flair')


class Embeddings(torch.nn.Module):
    'Abstract base class for all embeddings. Every new type of embedding must implement these methods.'

    @property
    @abstractmethod
    def embedding_length(self):
        'Returns the length of the embedding vector.'
        pass

    @property
    @abstractmethod
    def embedding_type(self):
        pass

    def embed(self, sentences):
        'Add embeddings to all words in a list of sentences. If embeddings are already added, updates only if embeddings\n        are non-static.'
        if (type(sentences) is Sentence):
            sentences = [sentences]
        everything_embedded = True
        if (self.embedding_type == 'word-level'):
            for sentence in sentences:
                for token in sentence.tokens:
                    if (self.name not in token._embeddings.keys()):
                        everything_embedded = False
        else:
            for sentence in sentences:
                if (self.name not in sentence._embeddings.keys()):
                    everything_embedded = False
        if ((not everything_embedded) or (not self.static_embeddings)):
            self._add_embeddings_internal(sentences)
        return sentences

    @abstractmethod
    def _add_embeddings_internal(self, sentences):
        'Private method for adding embeddings to all words in a list of sentences.'
        pass


class TokenEmbeddings(Embeddings):
    'Abstract base class for all token-level embeddings. Ever new type of word embedding must implement these methods.'

    @property
    @abstractmethod
    def embedding_length(self):
        'Returns the length of the embedding vector.'
        pass

    @property
    def embedding_type(self):
        return 'word-level'


class DocumentEmbeddings(Embeddings):
    'Abstract base class for all document-level embeddings. Ever new type of document embedding must implement these methods.'

    @property
    @abstractmethod
    def embedding_length(self):
        'Returns the length of the embedding vector.'
        pass

    @property
    def embedding_type(self):
        return 'sentence-level'


class StackedEmbeddings(TokenEmbeddings):
    'A stack of embeddings, used if you need to combine several different embedding types.'

    def __init__(self, embeddings):
        'The constructor takes a list of embeddings to be combined.'
        super().__init__()
        self.embeddings = embeddings
        for (i, embedding) in enumerate(embeddings):
            self.add_module('list_embedding_{}'.format(i), embedding)
        self.name = 'Stack'
        self.static_embeddings = True
        self.__embedding_type = embeddings[0].embedding_type
        self.__embedding_length = 0
        for embedding in embeddings:
            self.__embedding_length += embedding.embedding_length

    def embed(self, sentences, static_embeddings=True):
        if (type(sentences) is Sentence):
            sentences = [sentences]
        for embedding in self.embeddings:
            embedding.embed(sentences)

    @property
    def embedding_type(self):
        return self.__embedding_type

    @property
    def embedding_length(self):
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences):
        for embedding in self.embeddings:
            embedding._add_embeddings_internal(sentences)
        return sentences

    def __str__(self):
        return ''.join(['StackedEmbeddings [', '{}'.format(','.join([str(e) for e in self.embeddings])), ']'])


class WordEmbeddings(TokenEmbeddings):
    'Standard static word embeddings, such as GloVe or FastText.'

    def __init__(self, embeddings, field=None):
        "\n        Initializes classic word embeddings. Constructor downloads required files if not there.\n        :param embeddings: one of: 'glove', 'extvec', 'crawl' or two-letter language code or custom\n        If you want to use a custom embedding file, just pass the path to the embeddings as embeddings variable.\n        "
        self.embeddings = embeddings
        old_base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/'
        base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.3/'
        embeddings_path_v4 = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/'
        embeddings_path_v4_1 = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4.1/'
        cache_dir = Path('embeddings')
        if ((embeddings.lower() == 'glove') or (embeddings.lower() == 'en-glove')):
            cached_path(''.join(
                ['{}'.format(old_base_path), 'glove.gensim.vectors.npy']), cache_dir=cache_dir)
            embeddings = cached_path(
                ''.join(['{}'.format(old_base_path), 'glove.gensim']), cache_dir=cache_dir)
        elif ((embeddings.lower() == 'turian') or (embeddings.lower() == 'en-turian')):
            cached_path(''.join(
                ['{}'.format(embeddings_path_v4_1), 'turian.vectors.npy']), cache_dir=cache_dir)
            embeddings = cached_path(
                ''.join(['{}'.format(embeddings_path_v4_1), 'turian']), cache_dir=cache_dir)
        elif ((embeddings.lower() == 'extvec') or (embeddings.lower() == 'en-extvec')):
            cached_path(''.join(
                ['{}'.format(old_base_path), 'extvec.gensim.vectors.npy']), cache_dir=cache_dir)
            embeddings = cached_path(
                ''.join(['{}'.format(old_base_path), 'extvec.gensim']), cache_dir=cache_dir)
        elif ((embeddings.lower() == 'crawl') or (embeddings.lower() == 'en-crawl')):
            cached_path(''.join(['{}'.format(
                base_path), 'en-fasttext-crawl-300d-1M.vectors.npy']), cache_dir=cache_dir)
            embeddings = cached_path(''.join(
                ['{}'.format(base_path), 'en-fasttext-crawl-300d-1M']), cache_dir=cache_dir)
        elif ((embeddings.lower() == 'news') or (embeddings.lower() == 'en-news') or (embeddings.lower() == 'en')):
            cached_path(''.join(['{}'.format(
                base_path), 'en-fasttext-news-300d-1M.vectors.npy']), cache_dir=cache_dir)
            embeddings = cached_path(''.join(
                ['{}'.format(base_path), 'en-fasttext-news-300d-1M']), cache_dir=cache_dir)
        elif ((embeddings.lower() == 'twitter') or (embeddings.lower() == 'en-twitter')):
            cached_path(''.join(
                ['{}'.format(old_base_path), 'twitter.gensim.vectors.npy']), cache_dir=cache_dir)
            embeddings = cached_path(
                ''.join(['{}'.format(old_base_path), 'twitter.gensim']), cache_dir=cache_dir)
        elif (len(embeddings.lower()) == 2):
            cached_path(''.join(['{}'.format(embeddings_path_v4), '{}'.format(
                embeddings), '-wiki-fasttext-300d-1M.vectors.npy']), cache_dir=cache_dir)
            embeddings = cached_path(''.join(['{}'.format(embeddings_path_v4), '{}'.format(
                embeddings), '-wiki-fasttext-300d-1M']), cache_dir=cache_dir)
        elif ((len(embeddings.lower()) == 7) and embeddings.endswith('-wiki')):
            cached_path(''.join(['{}'.format(embeddings_path_v4), '{}'.format(
                embeddings[:2]), '-wiki-fasttext-300d-1M.vectors.npy']), cache_dir=cache_dir)
            embeddings = cached_path(''.join(['{}'.format(embeddings_path_v4), '{}'.format(
                embeddings[:2]), '-wiki-fasttext-300d-1M']), cache_dir=cache_dir)
        elif ((len(embeddings.lower()) == 8) and embeddings.endswith('-crawl')):
            cached_path(''.join(['{}'.format(embeddings_path_v4), '{}'.format(
                embeddings[:2]), '-crawl-fasttext-300d-1M.vectors.npy']), cache_dir=cache_dir)
            embeddings = cached_path(''.join(['{}'.format(embeddings_path_v4), '{}'.format(
                embeddings[:2]), '-crawl-fasttext-300d-1M']), cache_dir=cache_dir)
        elif (not Path(embeddings).exists()):
            raise ValueError(''.join(['The given embeddings "', '{}'.format(
                embeddings), '" is not available or is not a valid path.']))
        self.name = str(embeddings)
        self.static_embeddings = True
        if str(embeddings).endswith('.bin'):
            self.precomputed_word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
                str(embeddings), binary=True)
        else:
            self.precomputed_word_embeddings = gensim.models.KeyedVectors.load(
                str(embeddings))
        self.field = field
        self.__embedding_length = self.precomputed_word_embeddings.vector_size
        super().__init__()

    @property
    def embedding_length(self):
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences):
        for (i, sentence) in enumerate(sentences):
            for (token, token_idx) in zip(sentence.tokens, range(len(sentence.tokens))):
                if (('field' not in self.__dict__) or (self.field is None)):
                    word = token.text
                else:
                    word = token.get_tag(self.field).value
                if (word in self.precomputed_word_embeddings):
                    word_embedding = self.precomputed_word_embeddings[word]
                elif (word.lower() in self.precomputed_word_embeddings):
                    word_embedding = self.precomputed_word_embeddings[word.lower(
                    )]
                elif (re.sub('\\d', '#', word.lower()) in self.precomputed_word_embeddings):
                    word_embedding = self.precomputed_word_embeddings[re.sub(
                        '\\d', '#', word.lower())]
                elif (re.sub('\\d', '0', word.lower()) in self.precomputed_word_embeddings):
                    word_embedding = self.precomputed_word_embeddings[re.sub(
                        '\\d', '0', word.lower())]
                else:
                    word_embedding = np.zeros(
                        self.embedding_length, dtype='float')
                word_embedding = torch.FloatTensor(word_embedding)
                token.set_embedding(self.name, word_embedding)
        return sentences

    def __str__(self):
        return self.name

    def extra_repr(self):
        if ('embeddings' not in self.__dict__):
            self.embeddings = self.name
        return ''.join(["'", '{}'.format(self.embeddings), "'"])


class FastTextEmbeddings(TokenEmbeddings):
    'FastText Embeddings with oov functionality'

    def __init__(self, embeddings, use_local=True, field=None):
        "\n        Initializes fasttext word embeddings. Constructor downloads required embedding file and stores in cache\n        if use_local is False.\n\n        :param embeddings: path to your embeddings '.bin' file\n        :param use_local: set this to False if you are using embeddings from a remote source\n        "
        cache_dir = Path('embeddings')
        if use_local:
            if (not Path(embeddings).exists()):
                raise ValueError(''.join(['The given embeddings "', '{}'.format(
                    embeddings), '" is not available or is not a valid path.']))
        else:
            embeddings = cached_path('{}'.format(
                embeddings), cache_dir=cache_dir)
        self.embeddings = embeddings
        self.name = str(embeddings)
        self.static_embeddings = True
        self.precomputed_word_embeddings = gensim.models.FastText.load_fasttext_format(
            str(embeddings))
        self.__embedding_length = self.precomputed_word_embeddings.vector_size
        self.field = field
        super().__init__()

    @property
    def embedding_length(self):
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences):
        for (i, sentence) in enumerate(sentences):
            for (token, token_idx) in zip(sentence.tokens, range(len(sentence.tokens))):
                if (('field' not in self.__dict__) or (self.field is None)):
                    word = token.text
                else:
                    word = token.get_tag(self.field).value
                try:
                    word_embedding = self.precomputed_word_embeddings[word]
                except:
                    word_embedding = np.zeros(
                        self.embedding_length, dtype='float')
                word_embedding = torch.FloatTensor(word_embedding)
                token.set_embedding(self.name, word_embedding)
        return sentences

    def __str__(self):
        return self.name

    def extra_repr(self):
        return ''.join(["'", '{}'.format(self.embeddings), "'"])


class OneHotEmbeddings(TokenEmbeddings):
    'One-hot encoded embeddings.'

    def __init__(self, corpus=Union[(Corpus, List[Sentence])], field='text', embedding_length=300, min_freq=3):
        super().__init__()
        self.name = 'one-hot'
        self.static_embeddings = False
        self.min_freq = min_freq
        tokens = list(map((lambda s: s.tokens), corpus.train))
        tokens = [token for sublist in tokens for token in sublist]
        if (field == 'text'):
            most_common = Counter(
                list(map((lambda t: t.text), tokens))).most_common()
        else:
            most_common = Counter(
                list(map((lambda t: t.get_tag(field)), tokens))).most_common()
        tokens = []
        for (token, freq) in most_common:
            if (freq < min_freq):
                break
            tokens.append(token)
        self.vocab_dictionary = Dictionary()
        for token in tokens:
            self.vocab_dictionary.add_item(token)
        self.__embedding_length = embedding_length
        print(self.vocab_dictionary.idx2item)
        print(''.join(['vocabulary size of ',
                       '{}'.format(len(self.vocab_dictionary))]))
        self.embedding_layer = torch.nn.Embedding(
            len(self.vocab_dictionary), self.__embedding_length)
        torch.nn.init.xavier_uniform_(self.embedding_layer.weight)

    @property
    def embedding_length(self):
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences):
        one_hot_sentences = []
        for (i, sentence) in enumerate(sentences):
            context_idxs = [self.vocab_dictionary.get_idx_for_item(
                t.text) for t in sentence.tokens]
            one_hot_sentences.extend(context_idxs)
        one_hot_sentences = torch.tensor(
            one_hot_sentences, dtype=torch.long).to(flair.device)
        embedded = self.embedding_layer.forward(one_hot_sentences)
        index = 0
        for sentence in sentences:
            for token in sentence:
                embedding = embedded[index]
                token.set_embedding(self.name, embedding)
                index += 1
        return sentences

    def __str__(self):
        return self.name

    def extra_repr(self):
        return 'min_freq={}'.format(self.min_freq)


class BPEmbSerializable(BPEmb):

    def __getstate__(self):
        state = self.__dict__.copy()
        state['spm_model_binary'] = open(self.model_file, mode='rb').read()
        state['spm'] = None
        return state

    def __setstate__(self, state):
        from bpemb.util import sentencepiece_load
        model_file = self.model_tpl.format(lang=state['lang'], vs=state['vs'])
        self.__dict__ = state
        self.cache_dir = (Path(flair.cache_root) / 'embeddings')
        if ('spm_model_binary' in self.__dict__):
            if (not os.path.exists((self.cache_dir / state['lang']))):
                os.makedirs((self.cache_dir / state['lang']))
            self.model_file = (self.cache_dir / model_file)
            with open(self.model_file, 'wb') as out:
                out.write(self.__dict__['spm_model_binary'])
        else:
            self.model_file = self._load_file(model_file)
        state['spm'] = sentencepiece_load(self.model_file)


class MuseCrosslingualEmbeddings(TokenEmbeddings):

    def __init__(self):
        self.name = 'muse-crosslingual'
        self.static_embeddings = True
        self.__embedding_length = 300
        self.language_embeddings = {

        }
        super().__init__()

    def _add_embeddings_internal(self, sentences):
        for (i, sentence) in enumerate(sentences):
            language_code = sentence.get_language_code()
            print(language_code)
            supported = ['en', 'de', 'bg', 'ca', 'hr', 'cs', 'da', 'nl', 'et', 'fi', 'fr',
                         'el', 'he', 'hu', 'id', 'it', 'mk', 'no', 'pl', 'pt', 'ro', 'ru', 'sk']
            if (language_code not in supported):
                language_code = 'en'
            if (language_code not in self.language_embeddings):
                log.info(
                    ''.join(["Loading up MUSE embeddings for '", '{}'.format(language_code), "'!"]))
                webpath = 'https://alan-nlp.s3.eu-central-1.amazonaws.com/resources/embeddings-muse'
                cache_dir = (Path('embeddings') / 'MUSE')
                cached_path(''.join(['{}'.format(webpath), '/muse.', '{}'.format(
                    language_code), '.vec.gensim.vectors.npy']), cache_dir=cache_dir)
                embeddings_file = cached_path(''.join(['{}'.format(
                    webpath), '/muse.', '{}'.format(language_code), '.vec.gensim']), cache_dir=cache_dir)
                self.language_embeddings[language_code] = gensim.models.KeyedVectors.load(
                    str(embeddings_file))
            current_embedding_model = self.language_embeddings[language_code]
            for (token, token_idx) in zip(sentence.tokens, range(len(sentence.tokens))):
                if (('field' not in self.__dict__) or (self.field is None)):
                    word = token.text
                else:
                    word = token.get_tag(self.field).value
                if (word in current_embedding_model):
                    word_embedding = current_embedding_model[word]
                elif (word.lower() in current_embedding_model):
                    word_embedding = current_embedding_model[word.lower()]
                elif (re.sub('\\d', '#', word.lower()) in current_embedding_model):
                    word_embedding = current_embedding_model[re.sub(
                        '\\d', '#', word.lower())]
                elif (re.sub('\\d', '0', word.lower()) in current_embedding_model):
                    word_embedding = current_embedding_model[re.sub(
                        '\\d', '0', word.lower())]
                else:
                    word_embedding = np.zeros(
                        self.embedding_length, dtype='float')
                word_embedding = torch.FloatTensor(word_embedding)
                token.set_embedding(self.name, word_embedding)
        return sentences

    @property
    def embedding_length(self):
        return self.__embedding_length

    def __str__(self):
        return self.name


class BytePairEmbeddings(TokenEmbeddings):

    def __init__(self, language, dim=50, syllables=100000, cache_dir=(Path(flair.cache_root) / 'embeddings')):
        '\n        Initializes BP embeddings. Constructor downloads required files if not there.\n        '
        self.name = ''.join(['bpe-', '{}'.format(language),
                             '-', '{}'.format(syllables), '-', '{}'.format(dim)])
        self.static_embeddings = True
        self.embedder = BPEmbSerializable(
            lang=language, vs=syllables, dim=dim, cache_dir=cache_dir)
        self.__embedding_length = (self.embedder.emb.vector_size * 2)
        super().__init__()

    @property
    def embedding_length(self):
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences):
        for (i, sentence) in enumerate(sentences):
            for (token, token_idx) in zip(sentence.tokens, range(len(sentence.tokens))):
                if (('field' not in self.__dict__) or (self.field is None)):
                    word = token.text
                else:
                    word = token.get_tag(self.field).value
                if (word.strip() == ''):
                    token.set_embedding(self.name, torch.zeros(
                        self.embedding_length, dtype=torch.float))
                else:
                    embeddings = self.embedder.embed(word.lower())
                    embedding = np.concatenate(
                        (embeddings[0], embeddings[(len(embeddings) - 1)]))
                    token.set_embedding(self.name, torch.tensor(
                        embedding, dtype=torch.float))
        return sentences

    def __str__(self):
        return self.name

    def extra_repr(self):
        return 'model={}'.format(self.name)


class ELMoEmbeddings(TokenEmbeddings):
    'Contextual word embeddings using word-level LM, as proposed in Peters et al., 2018.'

    def __init__(self, model='original', options_file=None, weight_file=None):
        super().__init__()
        try:
            import allennlp.commands.elmo
        except:
            log.warning(('-' * 100))
            log.warning('ATTENTION! The library "allennlp" is not installed!')
            log.warning(
                'To use ELMoEmbeddings, please first install with "pip install allennlp"')
            log.warning(('-' * 100))
            pass
        self.name = ('elmo-' + model)
        self.static_embeddings = True
        if ((not options_file) or (not weight_file)):
            options_file = allennlp.commands.elmo.DEFAULT_OPTIONS_FILE
            weight_file = allennlp.commands.elmo.DEFAULT_WEIGHT_FILE
            if (model == 'small'):
                options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
                weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'
            if (model == 'medium'):
                options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json'
                weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5'
            if ((model == 'pt') or (model == 'portuguese')):
                options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pt/elmo_pt_options.json'
                weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pt/elmo_pt_weights.hdf5'
            if (model == 'pubmed'):
                options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_options.json'
                weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5'
        from flair import device
        if re.fullmatch('cuda:[0-9]+', str(device)):
            cuda_device = int(str(device).split(':')[(- 1)])
        elif (str(device) == 'cpu'):
            cuda_device = (- 1)
        else:
            cuda_device = 0
        self.ee = allennlp.commands.elmo.ElmoEmbedder(
            options_file=options_file, weight_file=weight_file, cuda_device=cuda_device)
        dummy_sentence = Sentence()
        dummy_sentence.add_token(Token('hello'))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length = len(
            embedded_dummy[0].get_token(1).get_embedding())

    @property
    def embedding_length(self):
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences):
        sentence_words = []
        for sentence in sentences:
            sentence_words.append([token.text for token in sentence])
        embeddings = self.ee.embed_batch(sentence_words)
        for (i, sentence) in enumerate(sentences):
            sentence_embeddings = embeddings[i]
            for (token, token_idx) in zip(sentence.tokens, range(len(sentence.tokens))):
                word_embedding = torch.cat([torch.FloatTensor(sentence_embeddings[0, token_idx, :]), torch.FloatTensor(
                    sentence_embeddings[1, token_idx, :]), torch.FloatTensor(sentence_embeddings[2, token_idx, :])], 0)
                token.set_embedding(self.name, word_embedding)
        return sentences

    def extra_repr(self):
        return 'model={}'.format(self.name)

    def __str__(self):
        return self.name


class ELMoTransformerEmbeddings(TokenEmbeddings):
    'Contextual word embeddings using word-level Transformer-based LM, as proposed in Peters et al., 2018.'

    def __init__(self, model_file):
        super().__init__()
        try:
            from allennlp.modules.token_embedders.bidirectional_language_model_token_embedder import BidirectionalLanguageModelTokenEmbedder
            from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
        except:
            log.warning(('-' * 100))
            log.warning('ATTENTION! The library "allennlp" is not installed!')
            log.warning(
                'To use ELMoTransformerEmbeddings, please first install a recent version from https://github.com/allenai/allennlp')
            log.warning(('-' * 100))
            pass
        self.name = 'elmo-transformer'
        self.static_embeddings = True
        self.lm_embedder = BidirectionalLanguageModelTokenEmbedder(
            archive_file=model_file, dropout=0.2, bos_eos_tokens=('<S>', '</S>'), remove_bos_eos=True, requires_grad=False)
        self.lm_embedder = self.lm_embedder.to(device=flair.device)
        self.vocab = self.lm_embedder._lm.vocab
        self.indexer = ELMoTokenCharactersIndexer()
        dummy_sentence = Sentence()
        dummy_sentence.add_token(Token('hello'))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length = len(
            embedded_dummy[0].get_token(1).get_embedding())

    @property
    def embedding_length(self):
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences):
        import allennlp.data.tokenizers.token as allen_nlp_token
        indexer = self.indexer
        vocab = self.vocab
        for sentence in sentences:
            character_indices = indexer.tokens_to_indices(
                [allen_nlp_token.Token(token.text) for token in sentence], vocab, 'elmo')['elmo']
            indices_tensor = torch.LongTensor([character_indices])
            indices_tensor = indices_tensor.to(device=flair.device)
            embeddings = self.lm_embedder(indices_tensor)[
                0].detach().cpu().numpy()
            for (token, token_idx) in zip(sentence.tokens, range(len(sentence.tokens))):
                embedding = embeddings[token_idx]
                word_embedding = torch.FloatTensor(embedding)
                token.set_embedding(self.name, word_embedding)
        return sentences

    def extra_repr(self):
        return 'model={}'.format(self.name)

    def __str__(self):
        return self.name


class ScalarMix(torch.nn.Module):
    '\n    Computes a parameterised scalar mixture of N tensors.\n    This method was proposed by Liu et al. (2019) in the paper:\n    "Linguistic Knowledge and Transferability of Contextual Representations" (https://arxiv.org/abs/1903.08855)\n\n    The implementation is copied and slightly modified from the allennlp repository and is licensed under Apache 2.0.\n    It can be found under:\n    https://github.com/allenai/allennlp/blob/master/allennlp/modules/scalar_mix.py.\n    '

    def __init__(self, mixture_size):
        '\n        Inits scalar mix implementation.\n        ``mixture = gamma * sum(s_k * tensor_k)`` where ``s = softmax(w)``, with ``w`` and ``gamma`` scalar parameters.\n        :param mixture_size: size of mixtures (usually the number of layers)\n        '
        super(ScalarMix, self).__init__()
        self.mixture_size = mixture_size
        initial_scalar_parameters = ([0.0] * mixture_size)
        self.scalar_parameters = ParameterList([Parameter(torch.FloatTensor(
            [initial_scalar_parameters[i]]).to(flair.device), requires_grad=False) for i in range(mixture_size)])
        self.gamma = Parameter(torch.FloatTensor(
            [1.0]).to(flair.device), requires_grad=False)

    def forward(self, tensors):
        '\n        Computes a weighted average of the ``tensors``.  The input tensors an be any shape\n        with at least two dimensions, but must all be the same shape.\n        :param tensors: list of input tensors\n        :return: computed weighted average of input tensors\n        '
        if (len(tensors) != self.mixture_size):
            log.error('{} tensors were passed, but the module was initialized to mix {} tensors.'.format(
                len(tensors), self.mixture_size))
        normed_weights = torch.nn.functional.softmax(
            torch.cat([parameter for parameter in self.scalar_parameters]), dim=0)
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)
        pieces = []
        for (weight, tensor) in zip(normed_weights, tensors):
            pieces.append((weight * tensor))
        return (self.gamma * sum(pieces))


def _extract_embeddings(hidden_states, layers, pooling_operation, subword_start_idx, subword_end_idx, use_scalar_mix=False):
    '\n    Extracts subword embeddings from specified layers from hidden states.\n    :param hidden_states: list of hidden states from model\n    :param layers: list of layers\n    :param pooling_operation: pooling operation for subword embeddings (supported: first, last, first_last and mean)\n    :param subword_start_idx: defines start index for subword\n    :param subword_end_idx: defines end index for subword\n    :param use_scalar_mix: determines, if scalar mix should be used\n    :return: list of extracted subword embeddings\n    '
    subtoken_embeddings = []
    for layer in layers:
        current_embeddings = hidden_states[layer][0][subword_start_idx:subword_end_idx]
        first_embedding = current_embeddings[0]
        if (pooling_operation == 'first_last'):
            last_embedding = current_embeddings[(- 1)]
            final_embedding = torch.cat([first_embedding, last_embedding])
        elif (pooling_operation == 'last'):
            final_embedding = current_embeddings[(- 1)]
        elif (pooling_operation == 'mean'):
            all_embeddings = [embedding.unsqueeze(
                0) for embedding in current_embeddings]
            final_embedding = torch.mean(
                torch.cat(all_embeddings, dim=0), dim=0)
        else:
            final_embedding = first_embedding
        subtoken_embeddings.append(final_embedding)
    if use_scalar_mix:
        sm = ScalarMix(mixture_size=len(subtoken_embeddings))
        sm_embeddings = sm(subtoken_embeddings)
        subtoken_embeddings = [sm_embeddings]
    return subtoken_embeddings


def _build_token_subwords_mapping(sentence, tokenizer):
    ' Builds a dictionary that stores the following information:\n    Token index (key) and number of corresponding subwords (value) for a sentence.\n\n    :param sentence: input sentence\n    :param tokenizer: PyTorch-Transformers tokenization object\n    :return: dictionary of token index to corresponding number of subwords\n    '
    token_subwords_mapping = {

    }
    for token in sentence.tokens:
        token_text = token.text
        subwords = tokenizer.tokenize(token_text)
        token_subwords_mapping[token.idx] = len(subwords)
    return token_subwords_mapping


def _build_token_subwords_mapping_gpt2(sentence, tokenizer):
    ' Builds a dictionary that stores the following information:\n    Token index (key) and number of corresponding subwords (value) for a sentence.\n\n    :param sentence: input sentence\n    :param tokenizer: PyTorch-Transformers tokenization object\n    :return: dictionary of token index to corresponding number of subwords\n    '
    token_subwords_mapping = {

    }
    for token in sentence.tokens:
        if (token.idx == 1):
            token_text = token.text
            subwords = tokenizer.tokenize(token_text)
        else:
            token_text = ('X ' + token.text)
            subwords = tokenizer.tokenize(token_text)[1:]
        token_subwords_mapping[token.idx] = len(subwords)
    return token_subwords_mapping


def _get_transformer_sentence_embeddings(sentences, tokenizer, model, name, layers, pooling_operation, use_scalar_mix, bos_token=None, eos_token=None):
    '\n    Builds sentence embeddings for Transformer-based architectures.\n    :param sentences: input sentences\n    :param tokenizer: tokenization object\n    :param model: model object\n    :param name: name of the Transformer-based model\n    :param layers: list of layers\n    :param pooling_operation: defines pooling operation for subword extraction\n    :param use_scalar_mix: defines the usage of scalar mix for specified layer(s)\n    :param bos_token: defines begin of sentence token (used for left padding)\n    :param eos_token: defines end of sentence token (used for right padding)\n    :return: list of sentences (each token of a sentence is now embedded)\n    '
    with torch.no_grad():
        for sentence in sentences:
            token_subwords_mapping = {

            }
            if (name.startswith('gpt2') or name.startswith('roberta')):
                token_subwords_mapping = _build_token_subwords_mapping_gpt2(
                    sentence=sentence, tokenizer=tokenizer)
            else:
                token_subwords_mapping = _build_token_subwords_mapping(
                    sentence=sentence, tokenizer=tokenizer)
            subwords = tokenizer.tokenize(sentence.to_tokenized_string())
            offset = 0
            if bos_token:
                subwords = ([bos_token] + subwords)
                offset = 1
            if eos_token:
                subwords = (subwords + [eos_token])
            indexed_tokens = tokenizer.convert_tokens_to_ids(subwords)
            tokens_tensor = torch.tensor([indexed_tokens])
            tokens_tensor = tokens_tensor.to(flair.device)
            hidden_states = model(tokens_tensor)[(- 1)]
            for token in sentence.tokens:
                len_subwords = token_subwords_mapping[token.idx]
                subtoken_embeddings = _extract_embeddings(hidden_states=hidden_states, layers=layers, pooling_operation=pooling_operation,
                                                          subword_start_idx=offset, subword_end_idx=(offset + len_subwords), use_scalar_mix=use_scalar_mix)
                offset += len_subwords
                final_subtoken_embedding = torch.cat(subtoken_embeddings)
                token.set_embedding(name, final_subtoken_embedding)
    return sentences


class TransformerXLEmbeddings(TokenEmbeddings):

    def __init__(self, pretrained_model_name_or_path='transfo-xl-wt103', layers='1,2,3', use_scalar_mix=False):
        'Transformer-XL embeddings, as proposed in Dai et al., 2019.\n        :param pretrained_model_name_or_path: name or path of Transformer-XL model\n        :param layers: comma-separated list of layers\n        :param use_scalar_mix: defines the usage of scalar mix for specified layer(s)\n        '
        super().__init__()
        self.tokenizer = TransfoXLTokenizer.from_pretrained(
            pretrained_model_name_or_path)
        self.model = TransfoXLModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, output_hidden_states=True)
        self.name = pretrained_model_name_or_path
        self.layers = [int(layer) for layer in layers.split(',')]
        self.use_scalar_mix = use_scalar_mix
        self.static_embeddings = True
        dummy_sentence = Sentence()
        dummy_sentence.add_token(Token('hello'))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length = len(
            embedded_dummy[0].get_token(1).get_embedding())

    @property
    def embedding_length(self):
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences):
        self.model.to(flair.device)
        self.model.eval()
        sentences = _get_transformer_sentence_embeddings(sentences=sentences, tokenizer=self.tokenizer, model=self.model,
                                                         name=self.name, layers=self.layers, pooling_operation='first', use_scalar_mix=self.use_scalar_mix, eos_token='<eos>')
        return sentences

    def extra_repr(self):
        return 'model={}'.format(self.name)

    def __str__(self):
        return self.name


class XLNetEmbeddings(TokenEmbeddings):

    def __init__(self, pretrained_model_name_or_path='xlnet-large-cased', layers='1', pooling_operation='first_last', use_scalar_mix=False):
        'XLNet embeddings, as proposed in Yang et al., 2019.\n        :param pretrained_model_name_or_path: name or path of XLNet model\n        :param layers: comma-separated list of layers\n        :param pooling_operation: defines pooling operation for subwords\n        :param use_scalar_mix: defines the usage of scalar mix for specified layer(s)\n        '
        super().__init__()
        self.tokenizer = XLNetTokenizer.from_pretrained(
            pretrained_model_name_or_path)
        self.model = XLNetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, output_hidden_states=True)
        self.name = pretrained_model_name_or_path
        self.layers = [int(layer) for layer in layers.split(',')]
        self.pooling_operation = pooling_operation
        self.use_scalar_mix = use_scalar_mix
        self.static_embeddings = True
        dummy_sentence = Sentence()
        dummy_sentence.add_token(Token('hello'))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length = len(
            embedded_dummy[0].get_token(1).get_embedding())

    @property
    def embedding_length(self):
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences):
        self.model.to(flair.device)
        self.model.eval()
        sentences = _get_transformer_sentence_embeddings(sentences=sentences, tokenizer=self.tokenizer, model=self.model, name=self.name,
                                                         layers=self.layers, pooling_operation=self.pooling_operation, use_scalar_mix=self.use_scalar_mix, bos_token='<s>', eos_token='</s>')
        return sentences

    def extra_repr(self):
        return 'model={}'.format(self.name)

    def __str__(self):
        return self.name


class XLMEmbeddings(TokenEmbeddings):

    def __init__(self, pretrained_model_name_or_path='xlm-mlm-en-2048', layers='1', pooling_operation='first_last', use_scalar_mix=False):
        '\n        XLM embeddings, as proposed in Guillaume et al., 2019.\n        :param pretrained_model_name_or_path: name or path of XLM model\n        :param layers: comma-separated list of layers\n        :param pooling_operation: defines pooling operation for subwords\n        :param use_scalar_mix: defines the usage of scalar mix for specified layer(s)\n        '
        super().__init__()
        self.tokenizer = XLMTokenizer.from_pretrained(
            pretrained_model_name_or_path)
        self.model = XLMModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, output_hidden_states=True)
        self.name = pretrained_model_name_or_path
        self.layers = [int(layer) for layer in layers.split(',')]
        self.pooling_operation = pooling_operation
        self.use_scalar_mix = use_scalar_mix
        self.static_embeddings = True
        dummy_sentence = Sentence()
        dummy_sentence.add_token(Token('hello'))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length = len(
            embedded_dummy[0].get_token(1).get_embedding())

    @property
    def embedding_length(self):
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences):
        self.model.to(flair.device)
        self.model.eval()
        sentences = _get_transformer_sentence_embeddings(sentences=sentences, tokenizer=self.tokenizer, model=self.model, name=self.name,
                                                         layers=self.layers, pooling_operation=self.pooling_operation, use_scalar_mix=self.use_scalar_mix, bos_token='<s>', eos_token='</s>')
        return sentences

    def extra_repr(self):
        return 'model={}'.format(self.name)

    def __str__(self):
        return self.name


class OpenAIGPTEmbeddings(TokenEmbeddings):

    def __init__(self, pretrained_model_name_or_path='openai-gpt', layers='1', pooling_operation='first_last', use_scalar_mix=False):
        'OpenAI GPT embeddings, as proposed in Radford et al. 2018.\n        :param pretrained_model_name_or_path: name or path of OpenAI GPT model\n        :param layers: comma-separated list of layers\n        :param pooling_operation: defines pooling operation for subwords\n        :param use_scalar_mix: defines the usage of scalar mix for specified layer(s)\n        '
        super().__init__()
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained(
            pretrained_model_name_or_path)
        self.model = OpenAIGPTModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, output_hidden_states=True)
        self.name = pretrained_model_name_or_path
        self.layers = [int(layer) for layer in layers.split(',')]
        self.pooling_operation = pooling_operation
        self.use_scalar_mix = use_scalar_mix
        self.static_embeddings = True
        dummy_sentence = Sentence()
        dummy_sentence.add_token(Token('hello'))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length = len(
            embedded_dummy[0].get_token(1).get_embedding())

    @property
    def embedding_length(self):
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences):
        self.model.to(flair.device)
        self.model.eval()
        sentences = _get_transformer_sentence_embeddings(sentences=sentences, tokenizer=self.tokenizer, model=self.model,
                                                         name=self.name, layers=self.layers, pooling_operation=self.pooling_operation, use_scalar_mix=self.use_scalar_mix)
        return sentences

    def extra_repr(self):
        return 'model={}'.format(self.name)

    def __str__(self):
        return self.name


class OpenAIGPT2Embeddings(TokenEmbeddings):

    def __init__(self, pretrained_model_name_or_path='gpt2-medium', layers='1', pooling_operation='first_last', use_scalar_mix=False):
        'OpenAI GPT-2 embeddings, as proposed in Radford et al. 2019.\n        :param pretrained_model_name_or_path: name or path of OpenAI GPT-2 model\n        :param layers: comma-separated list of layers\n        :param pooling_operation: defines pooling operation for subwords\n        :param use_scalar_mix: defines the usage of scalar mix for specified layer(s)\n        '
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            pretrained_model_name_or_path)
        self.model = GPT2Model.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, output_hidden_states=True)
        self.name = pretrained_model_name_or_path
        self.layers = [int(layer) for layer in layers.split(',')]
        self.pooling_operation = pooling_operation
        self.use_scalar_mix = use_scalar_mix
        self.static_embeddings = True
        dummy_sentence = Sentence()
        dummy_sentence.add_token(Token('hello'))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length = len(
            embedded_dummy[0].get_token(1).get_embedding())

    @property
    def embedding_length(self):
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences):
        self.model.to(flair.device)
        self.model.eval()
        sentences = _get_transformer_sentence_embeddings(sentences=sentences, tokenizer=self.tokenizer, model=self.model, name=self.name, layers=self.layers,
                                                         pooling_operation=self.pooling_operation, use_scalar_mix=self.use_scalar_mix, bos_token='<|endoftext|>', eos_token='<|endoftext|>')
        return sentences


class RoBERTaEmbeddings(TokenEmbeddings):

    def __init__(self, pretrained_model_name_or_path='roberta-base', layers='-1', pooling_operation='first', use_scalar_mix=False):
        'RoBERTa, as proposed by Liu et al. 2019.\n        :param pretrained_model_name_or_path: name or path of RoBERTa model\n        :param layers: comma-separated list of layers\n        :param pooling_operation: defines pooling operation for subwords\n        :param use_scalar_mix: defines the usage of scalar mix for specified layer(s)\n        '
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(
            pretrained_model_name_or_path)
        self.model = RobertaModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, output_hidden_states=True)
        self.name = pretrained_model_name_or_path
        self.layers = [int(layer) for layer in layers.split(',')]
        self.pooling_operation = pooling_operation
        self.use_scalar_mix = use_scalar_mix
        self.static_embeddings = True
        dummy_sentence = Sentence()
        dummy_sentence.add_token(Token('hello'))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length = len(
            embedded_dummy[0].get_token(1).get_embedding())

    @property
    def embedding_length(self):
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences):
        self.model.to(flair.device)
        self.model.eval()
        sentences = _get_transformer_sentence_embeddings(sentences=sentences, tokenizer=self.tokenizer, model=self.model, name=self.name,
                                                         layers=self.layers, pooling_operation=self.pooling_operation, use_scalar_mix=self.use_scalar_mix, bos_token='<s>', eos_token='</s>')
        return sentences


class CharacterEmbeddings(TokenEmbeddings):
    'Character embeddings of words, as proposed in Lample et al., 2016.'

    def __init__(self, path_to_char_dict=None, char_embedding_dim=25, hidden_size_char=25):
        'Uses the default character dictionary if none provided.'
        super().__init__()
        self.name = 'Char'
        self.static_embeddings = False
        if (path_to_char_dict is None):
            self.char_dictionary = Dictionary.load('common-chars')
        else:
            self.char_dictionary = Dictionary.load_from_file(path_to_char_dict)
        self.char_embedding_dim = char_embedding_dim
        self.hidden_size_char = hidden_size_char
        self.char_embedding = torch.nn.Embedding(
            len(self.char_dictionary.item2idx), self.char_embedding_dim)
        self.char_rnn = torch.nn.LSTM(
            self.char_embedding_dim, self.hidden_size_char, num_layers=1, bidirectional=True)
        self.__embedding_length = (self.char_embedding_dim * 2)
        self.to(flair.device)

    @property
    def embedding_length(self):
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences):
        for sentence in sentences:
            tokens_char_indices = []
            for token in sentence.tokens:
                char_indices = [self.char_dictionary.get_idx_for_item(
                    char) for char in token.text]
                tokens_char_indices.append(char_indices)
            tokens_sorted_by_length = sorted(
                tokens_char_indices, key=(lambda p: len(p)), reverse=True)
            d = {

            }
            for (i, ci) in enumerate(tokens_char_indices):
                for (j, cj) in enumerate(tokens_sorted_by_length):
                    if (ci == cj):
                        d[j] = i
                        continue
            chars2_length = [len(c) for c in tokens_sorted_by_length]
            longest_token_in_sentence = max(chars2_length)
            tokens_mask = torch.zeros(
                (len(tokens_sorted_by_length), longest_token_in_sentence), dtype=torch.long, device=flair.device)
            for (i, c) in enumerate(tokens_sorted_by_length):
                tokens_mask[i, :chars2_length[i]] = torch.tensor(
                    c, dtype=torch.long, device=flair.device)
            chars = tokens_mask
            character_embeddings = self.char_embedding(chars).transpose(0, 1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                character_embeddings, chars2_length)
            (lstm_out, self.hidden) = self.char_rnn(packed)
            (outputs, output_lengths) = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
            outputs = outputs.transpose(0, 1)
            chars_embeds_temp = torch.zeros(
                (outputs.size(0), outputs.size(2)), dtype=torch.float, device=flair.device)
            for (i, index) in enumerate(output_lengths):
                chars_embeds_temp[i] = outputs[(i, (index - 1))]
            character_embeddings = chars_embeds_temp.clone()
            for i in range(character_embeddings.size(0)):
                character_embeddings[d[i]] = chars_embeds_temp[i]
            for (token_number, token) in enumerate(sentence.tokens):
                token.set_embedding(
                    self.name, character_embeddings[token_number])

    def __str__(self):
        return self.name


class FlairEmbeddings(TokenEmbeddings):
    'Contextual string embeddings of words, as proposed in Akbik et al., 2018.'

    def __init__(self, model, fine_tune=False, chars_per_chunk=512):
        "\n        initializes contextual string embeddings using a character-level language model.\n        :param model: model string, one of 'news-forward', 'news-backward', 'news-forward-fast', 'news-backward-fast',\n                'mix-forward', 'mix-backward', 'german-forward', 'german-backward', 'polish-backward', 'polish-forward'\n                depending on which character language model is desired.\n        :param fine_tune: if set to True, the gradient will propagate into the language model. This dramatically slows down\n                training and often leads to overfitting, so use with caution.\n        :param  chars_per_chunk: max number of chars per rnn pass to control speed/memory tradeoff. Higher means faster but requires\n                more memory. Lower means slower but less memory.\n        "
        super().__init__()
        cache_dir = Path('embeddings')
        aws_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources'
        self.PRETRAINED_MODEL_ARCHIVE_MAP = {
            'multi-forward': ''.join(['{}'.format(aws_path), '/embeddings-v0.4/lm-multi-forward-v0.1.pt']),
            'multi-backward': ''.join(['{}'.format(aws_path), '/embeddings-v0.4/lm-multi-backward-v0.1.pt']),
            'multi-forward-fast': ''.join(['{}'.format(aws_path), '/embeddings-v0.4/lm-multi-forward-fast-v0.1.pt']),
            'multi-backward-fast': ''.join(['{}'.format(aws_path), '/embeddings-v0.4/lm-multi-backward-fast-v0.1.pt']),
            'en-forward': ''.join(['{}'.format(aws_path), '/embeddings-v0.4.1/big-news-forward--h2048-l1-d0.05-lr30-0.25-20/news-forward-0.4.1.pt']),
            'en-backward': ''.join(['{}'.format(aws_path), '/embeddings-v0.4.1/big-news-backward--h2048-l1-d0.05-lr30-0.25-20/news-backward-0.4.1.pt']),
            'en-forward-fast': ''.join(['{}'.format(aws_path), '/embeddings/lm-news-english-forward-1024-v0.2rc.pt']),
            'en-backward-fast': ''.join(['{}'.format(aws_path), '/embeddings/lm-news-english-backward-1024-v0.2rc.pt']),
            'news-forward': ''.join(['{}'.format(aws_path), '/embeddings-v0.4.1/big-news-forward--h2048-l1-d0.05-lr30-0.25-20/news-forward-0.4.1.pt']),
            'news-backward': ''.join(['{}'.format(aws_path), '/embeddings-v0.4.1/big-news-backward--h2048-l1-d0.05-lr30-0.25-20/news-backward-0.4.1.pt']),
            'news-forward-fast': ''.join(['{}'.format(aws_path), '/embeddings/lm-news-english-forward-1024-v0.2rc.pt']),
            'news-backward-fast': ''.join(['{}'.format(aws_path), '/embeddings/lm-news-english-backward-1024-v0.2rc.pt']),
            'mix-forward': ''.join(['{}'.format(aws_path), '/embeddings/lm-mix-english-forward-v0.2rc.pt']),
            'mix-backward': ''.join(['{}'.format(aws_path), '/embeddings/lm-mix-english-backward-v0.2rc.pt']),
            'ar-forward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-ar-opus-large-forward-v0.1.pt']),
            'ar-backward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-ar-opus-large-backward-v0.1.pt']),
            'bg-forward-fast': ''.join(['{}'.format(aws_path), '/embeddings-v0.3/lm-bg-small-forward-v0.1.pt']),
            'bg-backward-fast': ''.join(['{}'.format(aws_path), '/embeddings-v0.3/lm-bg-small-backward-v0.1.pt']),
            'bg-forward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-bg-opus-large-forward-v0.1.pt']),
            'bg-backward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-bg-opus-large-backward-v0.1.pt']),
            'cs-forward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-cs-opus-large-forward-v0.1.pt']),
            'cs-backward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-cs-opus-large-backward-v0.1.pt']),
            'cs-v0-forward': ''.join(['{}'.format(aws_path), '/embeddings-v0.4/lm-cs-large-forward-v0.1.pt']),
            'cs-v0-backward': ''.join(['{}'.format(aws_path), '/embeddings-v0.4/lm-cs-large-backward-v0.1.pt']),
            'da-forward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-da-opus-large-forward-v0.1.pt']),
            'da-backward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-da-opus-large-backward-v0.1.pt']),
            'de-forward': ''.join(['{}'.format(aws_path), '/embeddings/lm-mix-german-forward-v0.2rc.pt']),
            'de-backward': ''.join(['{}'.format(aws_path), '/embeddings/lm-mix-german-backward-v0.2rc.pt']),
            'de-historic-ha-forward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-historic-hamburger-anzeiger-forward-v0.1.pt']),
            'de-historic-ha-backward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-historic-hamburger-anzeiger-backward-v0.1.pt']),
            'de-historic-wz-forward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-historic-wiener-zeitung-forward-v0.1.pt']),
            'de-historic-wz-backward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-historic-wiener-zeitung-backward-v0.1.pt']),
            'es-forward': ''.join(['{}'.format(aws_path), '/embeddings-v0.4/language_model_es_forward_long/lm-es-forward.pt']),
            'es-backward': ''.join(['{}'.format(aws_path), '/embeddings-v0.4/language_model_es_backward_long/lm-es-backward.pt']),
            'es-forward-fast': ''.join(['{}'.format(aws_path), '/embeddings-v0.4/language_model_es_forward/lm-es-forward-fast.pt']),
            'es-backward-fast': ''.join(['{}'.format(aws_path), '/embeddings-v0.4/language_model_es_backward/lm-es-backward-fast.pt']),
            'eu-forward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-eu-opus-large-forward-v0.1.pt']),
            'eu-backward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-eu-opus-large-backward-v0.1.pt']),
            'eu-v0-forward': ''.join(['{}'.format(aws_path), '/embeddings-v0.4/lm-eu-large-forward-v0.1.pt']),
            'eu-v0-backward': ''.join(['{}'.format(aws_path), '/embeddings-v0.4/lm-eu-large-backward-v0.1.pt']),
            'fa-forward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-fa-opus-large-forward-v0.1.pt']),
            'fa-backward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-fa-opus-large-backward-v0.1.pt']),
            'fi-forward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-fi-opus-large-forward-v0.1.pt']),
            'fi-backward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-fi-opus-large-backward-v0.1.pt']),
            'fr-forward': ''.join(['{}'.format(aws_path), '/embeddings/lm-fr-charlm-forward.pt']),
            'fr-backward': ''.join(['{}'.format(aws_path), '/embeddings/lm-fr-charlm-backward.pt']),
            'he-forward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-he-opus-large-forward-v0.1.pt']),
            'he-backward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-he-opus-large-backward-v0.1.pt']),
            'hi-forward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-hi-opus-large-forward-v0.1.pt']),
            'hi-backward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-hi-opus-large-backward-v0.1.pt']),
            'hr-forward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-hr-opus-large-forward-v0.1.pt']),
            'hr-backward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-hr-opus-large-backward-v0.1.pt']),
            'id-forward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-id-opus-large-forward-v0.1.pt']),
            'id-backward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-id-opus-large-backward-v0.1.pt']),
            'it-forward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-it-opus-large-forward-v0.1.pt']),
            'it-backward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-it-opus-large-backward-v0.1.pt']),
            'ja-forward': ''.join(['{}'.format(aws_path), '/embeddings-v0.4.1/lm__char-forward__ja-wikipedia-3GB/japanese-forward.pt']),
            'ja-backward': ''.join(['{}'.format(aws_path), '/embeddings-v0.4.1/lm__char-backward__ja-wikipedia-3GB/japanese-backward.pt']),
            'nl-forward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-nl-opus-large-forward-v0.1.pt']),
            'nl-backward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-nl-opus-large-backward-v0.1.pt']),
            'nl-v0-forward': ''.join(['{}'.format(aws_path), '/embeddings-v0.4/lm-nl-large-forward-v0.1.pt']),
            'nl-v0-backward': ''.join(['{}'.format(aws_path), '/embeddings-v0.4/lm-nl-large-backward-v0.1.pt']),
            'no-forward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-no-opus-large-forward-v0.1.pt']),
            'no-backward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-no-opus-large-backward-v0.1.pt']),
            'pl-forward': ''.join(['{}'.format(aws_path), '/embeddings/lm-polish-forward-v0.2.pt']),
            'pl-backward': ''.join(['{}'.format(aws_path), '/embeddings/lm-polish-backward-v0.2.pt']),
            'pl-opus-forward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-pl-opus-large-forward-v0.1.pt']),
            'pl-opus-backward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-pl-opus-large-backward-v0.1.pt']),
            'pt-forward': ''.join(['{}'.format(aws_path), '/embeddings-v0.4/lm-pt-forward.pt']),
            'pt-backward': ''.join(['{}'.format(aws_path), '/embeddings-v0.4/lm-pt-backward.pt']),
            'pubmed-forward': ''.join(['{}'.format(aws_path), '/embeddings-v0.4.1/pubmed-2015-fw-lm.pt']),
            'pubmed-backward': ''.join(['{}'.format(aws_path), '/embeddings-v0.4.1/pubmed-2015-bw-lm.pt']),
            'sl-forward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-sl-opus-large-forward-v0.1.pt']),
            'sl-backward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-sl-opus-large-backward-v0.1.pt']),
            'sl-v0-forward': ''.join(['{}'.format(aws_path), '/embeddings-v0.3/lm-sl-large-forward-v0.1.pt']),
            'sl-v0-backward': ''.join(['{}'.format(aws_path), '/embeddings-v0.3/lm-sl-large-backward-v0.1.pt']),
            'sv-forward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-sv-opus-large-forward-v0.1.pt']),
            'sv-backward': ''.join(['{}'.format(aws_path), '/embeddings-stefan-it/lm-sv-opus-large-backward-v0.1.pt']),
            'sv-v0-forward': ''.join(['{}'.format(aws_path), '/embeddings-v0.4/lm-sv-large-forward-v0.1.pt']),
            'sv-v0-backward': ''.join(['{}'.format(aws_path), '/embeddings-v0.4/lm-sv-large-backward-v0.1.pt']),
        }
        if (type(model) == str):
            if (model.lower() in self.PRETRAINED_MODEL_ARCHIVE_MAP):
                base_path = self.PRETRAINED_MODEL_ARCHIVE_MAP[model.lower()]
                model = cached_path(base_path, cache_dir=cache_dir)
            elif (replace_with_language_code(model) in self.PRETRAINED_MODEL_ARCHIVE_MAP):
                base_path = self.PRETRAINED_MODEL_ARCHIVE_MAP[replace_with_language_code(
                    model)]
                model = cached_path(base_path, cache_dir=cache_dir)
            elif (not Path(model).exists()):
                raise ValueError(''.join(['The given model "', '{}'.format(
                    model), '" is not available or is not a valid path.']))
        from flair.models import LanguageModel
        if (type(model) == LanguageModel):
            self.lm = model
            self.name = ''.join(['Task-LSTM-', '{}'.format(self.lm.hidden_size), '-',
                                 '{}'.format(self.lm.nlayers), '-', '{}'.format(self.lm.is_forward_lm)])
        else:
            self.lm = LanguageModel.load_language_model(model)
            self.name = str(model)
        self.fine_tune = fine_tune
        self.static_embeddings = (not fine_tune)
        self.is_forward_lm = self.lm.is_forward_lm
        self.chars_per_chunk = chars_per_chunk
        dummy_sentence = Sentence()
        dummy_sentence.add_token(Token('hello'))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length = len(
            embedded_dummy[0].get_token(1).get_embedding())
        self.eval()

    def train(self, mode=True):
        if ('fine_tune' not in self.__dict__):
            self.fine_tune = False
        if ('chars_per_chunk' not in self.__dict__):
            self.chars_per_chunk = 512
        if (not self.fine_tune):
            pass
        else:
            super(FlairEmbeddings, self).train(mode)

    @property
    def embedding_length(self):
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences):
        gradient_context = (torch.enable_grad()
                            if self.fine_tune else torch.no_grad())
        with gradient_context:
            text_sentences = [sentence.to_tokenized_string()
                              for sentence in sentences]
            longest_character_sequence_in_batch = len(
                max(text_sentences, key=len))
            sentences_padded = []
            append_padded_sentence = sentences_padded.append
            start_marker = '\n'
            end_marker = ' '
            extra_offset = len(start_marker)
            for sentence_text in text_sentences:
                pad_by = (longest_character_sequence_in_batch -
                          len(sentence_text))
                if self.is_forward_lm:
                    padded = '{}{}{}{}'.format(
                        start_marker, sentence_text, end_marker, (pad_by * ' '))
                    append_padded_sentence(padded)
                else:
                    padded = '{}{}{}{}'.format(
                        start_marker, sentence_text[::(- 1)], end_marker, (pad_by * ' '))
                    append_padded_sentence(padded)
            all_hidden_states_in_lm = self.lm.get_representation(
                sentences_padded, self.chars_per_chunk)
            for (i, sentence) in enumerate(sentences):
                sentence_text = sentence.to_tokenized_string()
                offset_forward = extra_offset
                offset_backward = (len(sentence_text) + extra_offset)
                for token in sentence.tokens:
                    offset_forward += len(token.text)
                    if self.is_forward_lm:
                        offset = offset_forward
                    else:
                        offset = offset_backward
                    embedding = all_hidden_states_in_lm[offset, i, :]
                    offset_forward += 1
                    offset_backward -= 1
                    offset_backward -= len(token.text)
                    if (not self.fine_tune):
                        embedding = embedding.detach()
                    token.set_embedding(self.name, embedding.clone())
            all_hidden_states_in_lm = all_hidden_states_in_lm.detach()
            all_hidden_states_in_lm = None
        return sentences

    def __str__(self):
        return self.name


class PooledFlairEmbeddings(TokenEmbeddings):

    def __init__(self, contextual_embeddings, pooling='min', only_capitalized=False, **kwargs):
        super().__init__()
        if (type(contextual_embeddings) is str):
            self.context_embeddings = FlairEmbeddings(
                contextual_embeddings, **kwargs)
        else:
            self.context_embeddings = contextual_embeddings
        self.embedding_length = (self.context_embeddings.embedding_length * 2)
        self.name = (self.context_embeddings.name + '-context')
        self.word_embeddings = {

        }
        self.word_count = {

        }
        self.only_capitalized = only_capitalized
        self.static_embeddings = False
        self.pooling = pooling
        if (pooling == 'mean'):
            self.aggregate_op = torch.add
        elif (pooling == 'fade'):
            self.aggregate_op = torch.add
        elif (pooling == 'max'):
            self.aggregate_op = torch.max
        elif (pooling == 'min'):
            self.aggregate_op = torch.min

    def train(self, mode=True):
        super().train(mode=mode)
        if mode:
            print('train mode resetting embeddings')
            self.word_embeddings = {

            }
            self.word_count = {

            }

    def _add_embeddings_internal(self, sentences):
        self.context_embeddings.embed(sentences)
        for sentence in sentences:
            for token in sentence.tokens:
                local_embedding = token._embeddings[self.context_embeddings.name]
                local_embedding = local_embedding.to(flair.device)
                if (token.text[0].isupper() or (not self.only_capitalized)):
                    if (token.text not in self.word_embeddings):
                        self.word_embeddings[token.text] = local_embedding
                        self.word_count[token.text] = 1
                    else:
                        aggregated_embedding = self.aggregate_op(
                            self.word_embeddings[token.text], local_embedding)
                        if (self.pooling == 'fade'):
                            aggregated_embedding /= 2
                        self.word_embeddings[token.text] = aggregated_embedding
                        self.word_count[token.text] += 1
        for sentence in sentences:
            for token in sentence.tokens:
                if (token.text in self.word_embeddings):
                    base = ((self.word_embeddings[token.text] / self.word_count[token.text]) if (
                        self.pooling == 'mean') else self.word_embeddings[token.text])
                else:
                    base = token._embeddings[self.context_embeddings.name]
                token.set_embedding(self.name, base)
        return sentences

    def embedding_length(self):
        return self.embedding_length


class BertEmbeddings(TokenEmbeddings):

    def __init__(self, bert_model_or_path='bert-base-uncased', layers='-1,-2,-3,-4', pooling_operation='first', use_scalar_mix=False):
        "\n        Bidirectional transformer embeddings of words, as proposed in Devlin et al., 2018.\n        :param bert_model_or_path: name of BERT model ('') or directory path containing custom model, configuration file\n        and vocab file (names of three files should be - config.json, pytorch_model.bin/model.chkpt, vocab.txt)\n        :param layers: string indicating which layers to take for embedding\n        :param pooling_operation: how to get from token piece embeddings to token embedding. Either pool them and take\n        the average ('mean') or use first word piece embedding as token embedding ('first)\n        "
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_or_path)
        self.model = BertModel.from_pretrained(
            pretrained_model_name_or_path=bert_model_or_path, output_hidden_states=True)
        self.layer_indexes = [int(x) for x in layers.split(',')]
        self.pooling_operation = pooling_operation
        self.use_scalar_mix = use_scalar_mix
        self.name = str(bert_model_or_path)
        self.static_embeddings = True

    class BertInputFeatures(object):
        'Private helper class for holding BERT-formatted features'

        def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids, token_subtoken_count):
            self.unique_id = unique_id
            self.tokens = tokens
            self.input_ids = input_ids
            self.input_mask = input_mask
            self.input_type_ids = input_type_ids
            self.token_subtoken_count = token_subtoken_count

    def _convert_sentences_to_features(self, sentences, max_sequence_length):
        max_sequence_length = (max_sequence_length + 2)
        features = []
        for (sentence_index, sentence) in enumerate(sentences):
            bert_tokenization = []
            token_subtoken_count = {

            }
            for token in sentence:
                subtokens = self.tokenizer.tokenize(token.text)
                bert_tokenization.extend(subtokens)
                token_subtoken_count[token.idx] = len(subtokens)
            if (len(bert_tokenization) > (max_sequence_length - 2)):
                bert_tokenization = bert_tokenization[0:(
                    max_sequence_length - 2)]
            tokens = []
            input_type_ids = []
            tokens.append('[CLS]')
            input_type_ids.append(0)
            for token in bert_tokenization:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append('[SEP]')
            input_type_ids.append(0)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = ([1] * len(input_ids))
            while (len(input_ids) < max_sequence_length):
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)
            features.append(BertEmbeddings.BertInputFeatures(unique_id=sentence_index, tokens=tokens, input_ids=input_ids,
                                                             input_mask=input_mask, input_type_ids=input_type_ids, token_subtoken_count=token_subtoken_count))
        return features

    def _add_embeddings_internal(self, sentences):
        'Add embeddings to all words in a list of sentences. If embeddings are already added,\n        updates only if embeddings are non-static.'
        longest_sentence_in_batch = len(max([self.tokenizer.tokenize(
            sentence.to_tokenized_string()) for sentence in sentences], key=len))
        features = self._convert_sentences_to_features(
            sentences, longest_sentence_in_batch)
        all_input_ids = torch.LongTensor(
            [f.input_ids for f in features]).to(flair.device)
        all_input_masks = torch.LongTensor(
            [f.input_mask for f in features]).to(flair.device)
        self.model.to(flair.device)
        self.model.eval()
        (_, _, all_encoder_layers) = self.model(all_input_ids,
                                                token_type_ids=None, attention_mask=all_input_masks)
        with torch.no_grad():
            for (sentence_index, sentence) in enumerate(sentences):
                feature = features[sentence_index]
                subtoken_embeddings = []
                for (token_index, _) in enumerate(feature.tokens):
                    all_layers = []
                    for layer_index in self.layer_indexes:
                        if self.use_scalar_mix:
                            layer_output = all_encoder_layers[int(
                                layer_index)][sentence_index]
                        else:
                            layer_output = all_encoder_layers[int(layer_index)].detach().cpu()[
                                sentence_index]
                        all_layers.append(layer_output[token_index])
                    if self.use_scalar_mix:
                        sm = ScalarMix(mixture_size=len(all_layers))
                        sm_embeddings = sm(all_layers)
                        all_layers = [sm_embeddings]
                    subtoken_embeddings.append(torch.cat(all_layers))
                token_idx = 0
                for token in sentence:
                    token_idx += 1
                    if (self.pooling_operation == 'first'):
                        token.set_embedding(
                            self.name, subtoken_embeddings[token_idx])
                    else:
                        embeddings = subtoken_embeddings[token_idx:(
                            token_idx + feature.token_subtoken_count[token.idx])]
                        embeddings = [embedding.unsqueeze(
                            0) for embedding in embeddings]
                        mean = torch.mean(torch.cat(embeddings, dim=0), dim=0)
                        token.set_embedding(self.name, mean)
                    token_idx += (feature.token_subtoken_count[token.idx] - 1)
        return sentences

    @property
    @abstractmethod
    def embedding_length(self):
        'Returns the length of the embedding vector.'
        return ((len(self.layer_indexes) * self.model.config.hidden_size) if (not self.use_scalar_mix) else self.model.config.hidden_size)


class CharLMEmbeddings(TokenEmbeddings):
    'Contextual string embeddings of words, as proposed in Akbik et al., 2018. '

    @deprecated(version='0.4', reason="Use 'FlairEmbeddings' instead.")
    def __init__(self, model, detach=True, use_cache=False, cache_directory=None):
        "\n        initializes contextual string embeddings using a character-level language model.\n        :param model: model string, one of 'news-forward', 'news-backward', 'news-forward-fast', 'news-backward-fast',\n                'mix-forward', 'mix-backward', 'german-forward', 'german-backward', 'polish-backward', 'polish-forward'\n                depending on which character language model is desired.\n        :param detach: if set to False, the gradient will propagate into the language model. this dramatically slows down\n                training and often leads to worse results, so not recommended.\n        :param use_cache: if set to False, will not write embeddings to file for later retrieval. this saves disk space but will\n                not allow re-use of once computed embeddings that do not fit into memory\n        :param cache_directory: if cache_directory is not set, the cache will be written to ~/.flair/embeddings. otherwise the cache\n                is written to the provided directory.\n        "
        super().__init__()
        cache_dir = Path('embeddings')
        if (model.lower() == 'multi-forward'):
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-multi-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == 'multi-backward'):
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-multi-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == 'news-forward'):
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-news-english-forward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == 'news-backward'):
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-news-english-backward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == 'news-forward-fast'):
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-news-english-forward-1024-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == 'news-backward-fast'):
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-news-english-backward-1024-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == 'mix-forward'):
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-english-forward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == 'mix-backward'):
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-english-backward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == 'german-forward') or (model.lower() == 'de-forward')):
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-german-forward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == 'german-backward') or (model.lower() == 'de-backward')):
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-german-backward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == 'polish-forward') or (model.lower() == 'pl-forward')):
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-polish-forward-v0.2.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == 'polish-backward') or (model.lower() == 'pl-backward')):
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-polish-backward-v0.2.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == 'slovenian-forward') or (model.lower() == 'sl-forward')):
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.3/lm-sl-large-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == 'slovenian-backward') or (model.lower() == 'sl-backward')):
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.3/lm-sl-large-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == 'bulgarian-forward') or (model.lower() == 'bg-forward')):
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.3/lm-bg-small-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == 'bulgarian-backward') or (model.lower() == 'bg-backward')):
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.3/lm-bg-small-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == 'dutch-forward') or (model.lower() == 'nl-forward')):
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-nl-large-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == 'dutch-backward') or (model.lower() == 'nl-backward')):
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-nl-large-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == 'swedish-forward') or (model.lower() == 'sv-forward')):
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-sv-large-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == 'swedish-backward') or (model.lower() == 'sv-backward')):
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-sv-large-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == 'french-forward') or (model.lower() == 'fr-forward')):
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-fr-charlm-forward.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == 'french-backward') or (model.lower() == 'fr-backward')):
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-fr-charlm-backward.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == 'czech-forward') or (model.lower() == 'cs-forward')):
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-cs-large-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == 'czech-backward') or (model.lower() == 'cs-backward')):
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-cs-large-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == 'portuguese-forward') or (model.lower() == 'pt-forward')):
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-pt-forward.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == 'portuguese-backward') or (model.lower() == 'pt-backward')):
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-pt-backward.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif (not Path(model).exists()):
            raise ValueError(''.join(['The given model "', '{}'.format(
                model), '" is not available or is not a valid path.']))
        self.name = str(model)
        self.static_embeddings = detach
        from flair.models import LanguageModel
        self.lm = LanguageModel.load_language_model(model)
        self.detach = detach
        self.is_forward_lm = self.lm.is_forward_lm
        self.cache = None
        if use_cache:
            cache_path = (Path(''.join(['{}'.format(self.name), '-tmp-cache.sqllite'])) if (
                not cache_directory) else (cache_directory / ''.join(['{}'.format(self.name), '-tmp-cache.sqllite'])))
            from sqlitedict import SqliteDict
            self.cache = SqliteDict(str(cache_path), autocommit=True)
        dummy_sentence = Sentence()
        dummy_sentence.add_token(Token('hello'))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length = len(
            embedded_dummy[0].get_token(1).get_embedding())
        self.eval()

    def train(self, mode=True):
        pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state['cache'] = None
        return state

    @property
    def embedding_length(self):
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences):
        if (('cache' in self.__dict__) and (self.cache is not None)):
            all_embeddings_retrieved_from_cache = True
            for sentence in sentences:
                key = sentence.to_tokenized_string()
                embeddings = self.cache.get(key)
                if (not embeddings):
                    all_embeddings_retrieved_from_cache = False
                    break
                else:
                    for (token, embedding) in zip(sentence, embeddings):
                        token.set_embedding(
                            self.name, torch.FloatTensor(embedding))
            if all_embeddings_retrieved_from_cache:
                return sentences
        text_sentences = [sentence.to_tokenized_string()
                          for sentence in sentences]
        longest_character_sequence_in_batch = len(max(text_sentences, key=len))
        sentences_padded = []
        append_padded_sentence = sentences_padded.append
        end_marker = ' '
        extra_offset = 1
        for sentence_text in text_sentences:
            pad_by = (longest_character_sequence_in_batch - len(sentence_text))
            if self.is_forward_lm:
                padded = '\n{}{}{}'.format(
                    sentence_text, end_marker, (pad_by * ' '))
                append_padded_sentence(padded)
            else:
                padded = '\n{}{}{}'.format(
                    sentence_text[::(- 1)], end_marker, (pad_by * ' '))
                append_padded_sentence(padded)
        all_hidden_states_in_lm = self.lm.get_representation(sentences_padded)
        for (i, sentence) in enumerate(sentences):
            sentence_text = sentence.to_tokenized_string()
            offset_forward = extra_offset
            offset_backward = (len(sentence_text) + extra_offset)
            for token in sentence.tokens:
                offset_forward += len(token.text)
                if self.is_forward_lm:
                    offset = offset_forward
                else:
                    offset = offset_backward
                embedding = all_hidden_states_in_lm[offset, i, :]
                offset_forward += 1
                offset_backward -= 1
                offset_backward -= len(token.text)
                token.set_embedding(self.name, embedding)
        if (('cache' in self.__dict__) and (self.cache is not None)):
            for sentence in sentences:
                self.cache[sentence.to_tokenized_string()] = [
                    token._embeddings[self.name].tolist() for token in sentence]
        return sentences

    def __str__(self):
        return self.name


class DocumentMeanEmbeddings(DocumentEmbeddings):

    @deprecated(version='0.3.1', reason="The functionality of this class is moved to 'DocumentPoolEmbeddings'")
    def __init__(self, token_embeddings):
        'The constructor takes a list of embeddings to be combined.'
        super().__init__()
        self.embeddings = StackedEmbeddings(embeddings=token_embeddings)
        self.name = 'document_mean'
        self.__embedding_length = self.embeddings.embedding_length
        self.to(flair.device)

    @property
    def embedding_length(self):
        return self.__embedding_length

    def embed(self, sentences):
        'Add embeddings to every sentence in the given list of sentences. If embeddings are already added, updates\n        only if embeddings are non-static.'
        everything_embedded = True
        if (type(sentences) is Sentence):
            sentences = [sentences]
        for sentence in sentences:
            if (self.name not in sentence._embeddings.keys()):
                everything_embedded = False
        if (not everything_embedded):
            self.embeddings.embed(sentences)
            for sentence in sentences:
                word_embeddings = []
                for token in sentence.tokens:
                    word_embeddings.append(token.get_embedding().unsqueeze(0))
                word_embeddings = torch.cat(
                    word_embeddings, dim=0).to(flair.device)
                mean_embedding = torch.mean(word_embeddings, 0)
                sentence.set_embedding(self.name, mean_embedding)

    def _add_embeddings_internal(self, sentences):
        pass


class DocumentPoolEmbeddings(DocumentEmbeddings):

    def __init__(self, embeddings, fine_tune_mode='linear', pooling='mean'):
        "The constructor takes a list of embeddings to be combined.\n        :param embeddings: a list of token embeddings\n        :param pooling: a string which can any value from ['mean', 'max', 'min']\n        "
        super().__init__()
        self.embeddings = StackedEmbeddings(embeddings=embeddings)
        self.__embedding_length = self.embeddings.embedding_length
        self.fine_tune_mode = fine_tune_mode
        if (self.fine_tune_mode in ['nonlinear', 'linear']):
            self.embedding_flex = torch.nn.Linear(
                self.embedding_length, self.embedding_length, bias=False)
            self.embedding_flex.weight.data.copy_(
                torch.eye(self.embedding_length))
        if (self.fine_tune_mode in ['nonlinear']):
            self.embedding_flex_nonlinear = torch.nn.ReLU(
                self.embedding_length)
            self.embedding_flex_nonlinear_map = torch.nn.Linear(
                self.embedding_length, self.embedding_length)
        self.__embedding_length = self.embeddings.embedding_length
        self.to(flair.device)
        self.pooling = pooling
        if (self.pooling == 'mean'):
            self.pool_op = torch.mean
        elif (pooling == 'max'):
            self.pool_op = torch.max
        elif (pooling == 'min'):
            self.pool_op = torch.min
        else:
            raise ValueError(
                ''.join(['Pooling operation for ', '{}'.format(self.mode), ' is not defined']))
        self.name = ''.join(['document_', '{}'.format(self.pooling)])

    @property
    def embedding_length(self):
        return self.__embedding_length

    def embed(self, sentences):
        'Add embeddings to every sentence in the given list of sentences. If embeddings are already added, updates\n        only if embeddings are non-static.'
        if isinstance(sentences, Sentence):
            sentences = [sentences]
        self.embeddings.embed(sentences)
        for sentence in sentences:
            word_embeddings = []
            for token in sentence.tokens:
                word_embeddings.append(token.get_embedding().unsqueeze(0))
            word_embeddings = torch.cat(
                word_embeddings, dim=0).to(flair.device)
            if (self.fine_tune_mode in ['nonlinear', 'linear']):
                word_embeddings = self.embedding_flex(word_embeddings)
            if (self.fine_tune_mode in ['nonlinear']):
                word_embeddings = self.embedding_flex_nonlinear(
                    word_embeddings)
                word_embeddings = self.embedding_flex_nonlinear_map(
                    word_embeddings)
            if (self.pooling == 'mean'):
                pooled_embedding = self.pool_op(word_embeddings, 0)
            else:
                (pooled_embedding, _) = self.pool_op(word_embeddings, 0)
            sentence.set_embedding(self.name, pooled_embedding)

    def _add_embeddings_internal(self, sentences):
        pass

    def extra_repr(self):
        return ''.join(['fine_tune_mode=', '{}'.format(self.fine_tune_mode), ', pooling=', '{}'.format(self.pooling)])


class DocumentRNNEmbeddings(DocumentEmbeddings):

    def __init__(self, embeddings, hidden_size=128, rnn_layers=1, reproject_words=True, reproject_words_dimension=None, bidirectional=False, dropout=0.5, word_dropout=0.0, locked_dropout=0.0, rnn_type='GRU'):
        "The constructor takes a list of embeddings to be combined.\n        :param embeddings: a list of token embeddings\n        :param hidden_size: the number of hidden states in the rnn\n        :param rnn_layers: the number of layers for the rnn\n        :param reproject_words: boolean value, indicating whether to reproject the token embeddings in a separate linear\n        layer before putting them into the rnn or not\n        :param reproject_words_dimension: output dimension of reprojecting token embeddings. If None the same output\n        dimension as before will be taken.\n        :param bidirectional: boolean value, indicating whether to use a bidirectional rnn or not\n        :param dropout: the dropout value to be used\n        :param word_dropout: the word dropout value to be used, if 0.0 word dropout is not used\n        :param locked_dropout: the locked dropout value to be used, if 0.0 locked dropout is not used\n        :param rnn_type: 'GRU' or 'LSTM'\n        "
        super().__init__()
        self.embeddings = StackedEmbeddings(embeddings=embeddings)
        self.rnn_type = rnn_type
        self.reproject_words = reproject_words
        self.bidirectional = bidirectional
        self.length_of_all_token_embeddings = self.embeddings.embedding_length
        self.static_embeddings = False
        self.__embedding_length = hidden_size
        if self.bidirectional:
            self.__embedding_length *= 4
        self.embeddings_dimension = self.length_of_all_token_embeddings
        if (self.reproject_words and (reproject_words_dimension is not None)):
            self.embeddings_dimension = reproject_words_dimension
        self.word_reprojection_map = torch.nn.Linear(
            self.length_of_all_token_embeddings, self.embeddings_dimension)
        if (rnn_type == 'LSTM'):
            self.rnn = torch.nn.LSTM(self.embeddings_dimension, hidden_size,
                                     num_layers=rnn_layers, bidirectional=self.bidirectional)
        else:
            self.rnn = torch.nn.GRU(self.embeddings_dimension, hidden_size,
                                    num_layers=rnn_layers, bidirectional=self.bidirectional)
        self.name = ('document_' + self.rnn._get_name())
        if (locked_dropout > 0.0):
            self.dropout = LockedDropout(locked_dropout)
        else:
            self.dropout = torch.nn.Dropout(dropout)
        self.use_word_dropout = (word_dropout > 0.0)
        if self.use_word_dropout:
            self.word_dropout = WordDropout(word_dropout)
        torch.nn.init.xavier_uniform_(self.word_reprojection_map.weight)
        self.to(flair.device)
        self.eval()

    @property
    def embedding_length(self):
        return self.__embedding_length

    def embed(self, sentences):
        'Add embeddings to all sentences in the given list of sentences. If embeddings are already added, update\n         only if embeddings are non-static.'
        if (type(sentences) is Sentence):
            sentences = [sentences]
        self.rnn.zero_grad()
        sort_perm = np.argsort([len(s) for s in sentences])[::(- 1)]
        sort_invperm = np.argsort(sort_perm)
        sentences = [sentences[i] for i in sort_perm]
        self.embeddings.embed(sentences)
        longest_token_sequence_in_batch = len(sentences[0])
        lengths = []
        sentence_tensor = torch.zeros([len(sentences), longest_token_sequence_in_batch,
                                       self.embeddings.embedding_length], dtype=torch.float, device=flair.device)
        for (s_id, sentence) in enumerate(sentences):
            lengths.append(len(sentence.tokens))
            sentence_tensor[s_id][:len(sentence)] = torch.cat(
                [token.get_embedding().unsqueeze(0) for token in sentence], 0)
        sentence_tensor = sentence_tensor.transpose_(0, 1)
        if self.use_word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)
        if self.reproject_words:
            sentence_tensor = self.word_reprojection_map(sentence_tensor)
        sentence_tensor = self.dropout(sentence_tensor)
        packed = pack_padded_sequence(sentence_tensor, lengths)
        self.rnn.flatten_parameters()
        (rnn_out, hidden) = self.rnn(packed)
        (outputs, output_lengths) = pad_packed_sequence(rnn_out)
        outputs = self.dropout(outputs)
        for (sentence_no, length) in enumerate(lengths):
            last_rep = outputs[((length - 1), sentence_no)]
            embedding = last_rep
            if self.bidirectional:
                first_rep = outputs[(0, sentence_no)]
                embedding = torch.cat([first_rep, last_rep], 0)
            sentence = sentences[sentence_no]
            sentence.set_embedding(self.name, embedding)
        sentences = [sentences[i] for i in sort_invperm]

    def _add_embeddings_internal(self, sentences):
        pass


@deprecated(version='0.4', reason="The functionality of this class is moved to 'DocumentRNNEmbeddings'")
class DocumentLSTMEmbeddings(DocumentEmbeddings):

    def __init__(self, embeddings, hidden_size=128, rnn_layers=1, reproject_words=True, reproject_words_dimension=None, bidirectional=False, dropout=0.5, word_dropout=0.0, locked_dropout=0.0):
        'The constructor takes a list of embeddings to be combined.\n        :param embeddings: a list of token embeddings\n        :param hidden_size: the number of hidden states in the lstm\n        :param rnn_layers: the number of layers for the lstm\n        :param reproject_words: boolean value, indicating whether to reproject the token embeddings in a separate linear\n        layer before putting them into the lstm or not\n        :param reproject_words_dimension: output dimension of reprojecting token embeddings. If None the same output\n        dimension as before will be taken.\n        :param bidirectional: boolean value, indicating whether to use a bidirectional lstm or not\n        :param dropout: the dropout value to be used\n        :param word_dropout: the word dropout value to be used, if 0.0 word dropout is not used\n        :param locked_dropout: the locked dropout value to be used, if 0.0 locked dropout is not used\n        '
        super().__init__()
        self.embeddings = StackedEmbeddings(embeddings=embeddings)
        self.reproject_words = reproject_words
        self.bidirectional = bidirectional
        self.length_of_all_token_embeddings = self.embeddings.embedding_length
        self.name = 'document_lstm'
        self.static_embeddings = False
        self.__embedding_length = hidden_size
        if self.bidirectional:
            self.__embedding_length *= 4
        self.embeddings_dimension = self.length_of_all_token_embeddings
        if (self.reproject_words and (reproject_words_dimension is not None)):
            self.embeddings_dimension = reproject_words_dimension
        self.word_reprojection_map = torch.nn.Linear(
            self.length_of_all_token_embeddings, self.embeddings_dimension)
        self.rnn = torch.nn.GRU(self.embeddings_dimension, hidden_size,
                                num_layers=rnn_layers, bidirectional=self.bidirectional)
        if (locked_dropout > 0.0):
            self.dropout = LockedDropout(locked_dropout)
        else:
            self.dropout = torch.nn.Dropout(dropout)
        self.use_word_dropout = (word_dropout > 0.0)
        if self.use_word_dropout:
            self.word_dropout = WordDropout(word_dropout)
        torch.nn.init.xavier_uniform_(self.word_reprojection_map.weight)
        self.to(flair.device)

    @property
    def embedding_length(self):
        return self.__embedding_length

    def embed(self, sentences):
        'Add embeddings to all sentences in the given list of sentences. If embeddings are already added, update\n         only if embeddings are non-static.'
        if (type(sentences) is Sentence):
            sentences = [sentences]
        self.rnn.zero_grad()
        sentences.sort(key=(lambda x: len(x)), reverse=True)
        self.embeddings.embed(sentences)
        longest_token_sequence_in_batch = len(sentences[0])
        all_sentence_tensors = []
        lengths = []
        for (i, sentence) in enumerate(sentences):
            lengths.append(len(sentence.tokens))
            word_embeddings = []
            for (token, token_idx) in zip(sentence.tokens, range(len(sentence.tokens))):
                word_embeddings.append(token.get_embedding().unsqueeze(0))
            for add in range((longest_token_sequence_in_batch - len(sentence.tokens))):
                word_embeddings.append(torch.zeros(
                    self.length_of_all_token_embeddings, dtype=torch.float).unsqueeze(0))
            word_embeddings_tensor = torch.cat(
                word_embeddings, 0).to(flair.device)
            sentence_states = word_embeddings_tensor
            all_sentence_tensors.append(sentence_states.unsqueeze(1))
        sentence_tensor = torch.cat(all_sentence_tensors, 1)
        if self.use_word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)
        if self.reproject_words:
            sentence_tensor = self.word_reprojection_map(sentence_tensor)
        sentence_tensor = self.dropout(sentence_tensor)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            sentence_tensor, lengths)
        self.rnn.flatten_parameters()
        (lstm_out, hidden) = self.rnn(packed)
        (outputs, output_lengths) = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
        outputs = self.dropout(outputs)
        for (sentence_no, length) in enumerate(lengths):
            last_rep = outputs[((length - 1), sentence_no)]
            embedding = last_rep
            if self.bidirectional:
                first_rep = outputs[(0, sentence_no)]
                embedding = torch.cat([first_rep, last_rep], 0)
            sentence = sentences[sentence_no]
            sentence.set_embedding(self.name, embedding)

    def _add_embeddings_internal(self, sentences):
        pass


class DocumentLMEmbeddings(DocumentEmbeddings):

    def __init__(self, flair_embeddings):
        super().__init__()
        self.embeddings = flair_embeddings
        self.name = 'document_lm'
        for (i, embedding) in enumerate(flair_embeddings):
            self.add_module('lm_embedding_{}'.format(i), embedding)
            if (not embedding.static_embeddings):
                self.static_embeddings = False
        self._embedding_length = sum(
            (embedding.embedding_length for embedding in flair_embeddings))

    @property
    def embedding_length(self):
        return self._embedding_length

    def _add_embeddings_internal(self, sentences):
        if (type(sentences) is Sentence):
            sentences = [sentences]
        for embedding in self.embeddings:
            embedding.embed(sentences)
            for sentence in sentences:
                sentence = sentence
                if embedding.is_forward_lm:
                    sentence.set_embedding(embedding.name, sentence[(
                        len(sentence) - 1)]._embeddings[embedding.name])
                else:
                    sentence.set_embedding(
                        embedding.name, sentence[0]._embeddings[embedding.name])
        return sentences


class NILCEmbeddings(WordEmbeddings):

    def __init__(self, embeddings, model='skip', size=100):
        "\n        Initializes portuguese classic word embeddings trained by NILC Lab (http://www.nilc.icmc.usp.br/embeddings).\n        Constructor downloads required files if not there.\n        :param embeddings: one of: 'fasttext', 'glove', 'wang2vec' or 'word2vec'\n        :param model: one of: 'skip' or 'cbow'. This is not applicable to glove.\n        :param size: one of: 50, 100, 300, 600 or 1000.\n        "
        base_path = 'http://143.107.183.175:22980/download.php?file=embeddings/'
        cache_dir = (Path('embeddings') / embeddings.lower())
        if (embeddings.lower() == 'glove'):
            cached_path(''.join(['{}'.format(base_path), '{}'.format(
                embeddings), '/', '{}'.format(embeddings), '_s', '{}'.format(size), '.zip']), cache_dir=cache_dir)
            embeddings = cached_path(''.join(['{}'.format(base_path), '{}'.format(
                embeddings), '/', '{}'.format(embeddings), '_s', '{}'.format(size), '.zip']), cache_dir=cache_dir)
        elif (embeddings.lower() in ['fasttext', 'wang2vec', 'word2vec']):
            cached_path(''.join(['{}'.format(base_path), '{}'.format(
                embeddings), '/', '{}'.format(model), '_s', '{}'.format(size), '.zip']), cache_dir=cache_dir)
            embeddings = cached_path(''.join(['{}'.format(base_path), '{}'.format(
                embeddings), '/', '{}'.format(model), '_s', '{}'.format(size), '.zip']), cache_dir=cache_dir)
        elif (not Path(embeddings).exists()):
            raise ValueError(''.join(['The given embeddings "', '{}'.format(
                embeddings), '" is not available or is not a valid path.']))
        self.name = str(embeddings)
        self.static_embeddings = True
        log.info(('Reading embeddings from %s' % embeddings))
        self.precomputed_word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
            open_inside_zip(str(embeddings), cache_dir=cache_dir))
        self.__embedding_length = self.precomputed_word_embeddings.vector_size
        super(TokenEmbeddings, self).__init__()

    @property
    def embedding_length(self):
        return self.__embedding_length

    def __str__(self):
        return self.name


def replace_with_language_code(string):
    string = string.replace('arabic-', 'ar-')
    string = string.replace('basque-', 'eu-')
    string = string.replace('bulgarian-', 'bg-')
    string = string.replace('croatian-', 'hr-')
    string = string.replace('czech-', 'cs-')
    string = string.replace('danish-', 'da-')
    string = string.replace('dutch-', 'nl-')
    string = string.replace('farsi-', 'fa-')
    string = string.replace('persian-', 'fa-')
    string = string.replace('finnish-', 'fi-')
    string = string.replace('french-', 'fr-')
    string = string.replace('german-', 'de-')
    string = string.replace('hebrew-', 'he-')
    string = string.replace('hindi-', 'hi-')
    string = string.replace('indonesian-', 'id-')
    string = string.replace('italian-', 'it-')
    string = string.replace('japanese-', 'ja-')
    string = string.replace('norwegian-', 'no')
    string = string.replace('polish-', 'pl-')
    string = string.replace('portuguese-', 'pt-')
    string = string.replace('slovenian-', 'sl-')
    string = string.replace('spanish-', 'es-')
    string = string.replace('swedish-', 'sv-')
    return string

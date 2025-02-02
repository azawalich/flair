
import os
import shutil
import pytest
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch.optim.adam import Adam
import flair.datasets
from flair.data import Dictionary, Sentence, MultiCorpus
from flair.embeddings import WordEmbeddings, TokenEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import SequenceTagger, TextClassifier, LanguageModel
from flair.samplers import ImbalancedClassificationDatasetSampler
from flair.trainers import ModelTrainer
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from flair.optim import AdamW


@pytest.mark.integration
def test_train_load_use_tagger(results_base_path, tasks_base_path):
    corpus = flair.datasets.ColumnCorpus(data_folder=(tasks_base_path / 'fashion'), column_format={
        0: 'text',
        2: 'ner',
    })
    tag_dictionary = corpus.make_tag_dictionary('ner')
    embeddings = WordEmbeddings('turian')
    tagger = SequenceTagger(hidden_size=64, embeddings=embeddings,
                            tag_dictionary=tag_dictionary, tag_type='ner', use_crf=False)
    trainer = ModelTrainer(tagger, corpus)
    trainer.train(results_base_path, learning_rate=0.1,
                  mini_batch_size=2, max_epochs=2, shuffle=False)
    loaded_model = SequenceTagger.load((results_base_path / 'final-model.pt'))
    sentence = Sentence('I love Berlin')
    sentence_empty = Sentence('       ')
    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_train_load_use_tagger_large(results_base_path, tasks_base_path):
    corpus = flair.datasets.UD_ENGLISH().downsample(0.05)
    tag_dictionary = corpus.make_tag_dictionary('pos')
    embeddings = WordEmbeddings('turian')
    tagger = SequenceTagger(hidden_size=64, embeddings=embeddings,
                            tag_dictionary=tag_dictionary, tag_type='pos', use_crf=False)
    trainer = ModelTrainer(tagger, corpus)
    trainer.train(results_base_path, learning_rate=0.1,
                  mini_batch_size=32, max_epochs=2, shuffle=False)
    loaded_model = SequenceTagger.load((results_base_path / 'final-model.pt'))
    sentence = Sentence('I love Berlin')
    sentence_empty = Sentence('       ')
    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_train_charlm_load_use_tagger(results_base_path, tasks_base_path):
    corpus = flair.datasets.ColumnCorpus(data_folder=(tasks_base_path / 'fashion'), column_format={
        0: 'text',
        2: 'ner',
    })
    tag_dictionary = corpus.make_tag_dictionary('ner')
    embeddings = FlairEmbeddings('news-forward-fast')
    tagger = SequenceTagger(hidden_size=64, embeddings=embeddings,
                            tag_dictionary=tag_dictionary, tag_type='ner', use_crf=False)
    trainer = ModelTrainer(tagger, corpus)
    trainer.train(results_base_path, learning_rate=0.1,
                  mini_batch_size=2, max_epochs=2, shuffle=False)
    loaded_model = SequenceTagger.load((results_base_path / 'final-model.pt'))
    sentence = Sentence('I love Berlin')
    sentence_empty = Sentence('       ')
    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_train_optimizer(results_base_path, tasks_base_path):
    corpus = flair.datasets.ColumnCorpus(data_folder=(tasks_base_path / 'fashion'), column_format={
        0: 'text',
        2: 'ner',
    })
    tag_dictionary = corpus.make_tag_dictionary('ner')
    embeddings = WordEmbeddings('turian')
    tagger = SequenceTagger(hidden_size=64, embeddings=embeddings,
                            tag_dictionary=tag_dictionary, tag_type='ner', use_crf=False)
    optimizer = Adam
    trainer = ModelTrainer(tagger, corpus, optimizer=optimizer)
    trainer.train(results_base_path, learning_rate=0.1,
                  mini_batch_size=2, max_epochs=2, shuffle=False)
    loaded_model = SequenceTagger.load((results_base_path / 'final-model.pt'))
    sentence = Sentence('I love Berlin')
    sentence_empty = Sentence('       ')
    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_train_optimizer_arguments(results_base_path, tasks_base_path):
    corpus = flair.datasets.ColumnCorpus(data_folder=(tasks_base_path / 'fashion'), column_format={
        0: 'text',
        2: 'ner',
    })
    tag_dictionary = corpus.make_tag_dictionary('ner')
    embeddings = WordEmbeddings('turian')
    tagger = SequenceTagger(hidden_size=64, embeddings=embeddings,
                            tag_dictionary=tag_dictionary, tag_type='ner', use_crf=False)
    optimizer = AdamW
    trainer = ModelTrainer(tagger, corpus, optimizer=optimizer)
    trainer.train(results_base_path, learning_rate=0.1, mini_batch_size=2,
                  max_epochs=2, shuffle=False, weight_decay=0.001)
    loaded_model = SequenceTagger.load((results_base_path / 'final-model.pt'))
    sentence = Sentence('I love Berlin')
    sentence_empty = Sentence('       ')
    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_find_learning_rate(results_base_path, tasks_base_path):
    corpus = flair.datasets.ColumnCorpus(data_folder=(tasks_base_path / 'fashion'), column_format={
        0: 'text',
        2: 'ner',
    })
    tag_dictionary = corpus.make_tag_dictionary('ner')
    embeddings = WordEmbeddings('turian')
    tagger = SequenceTagger(hidden_size=64, embeddings=embeddings,
                            tag_dictionary=tag_dictionary, tag_type='ner', use_crf=False)
    optimizer = SGD
    trainer = ModelTrainer(tagger, corpus, optimizer=optimizer)
    trainer.find_learning_rate(results_base_path, iterations=5)
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_load_use_serialized_tagger():
    loaded_model = SequenceTagger.load('ner')
    sentence = Sentence('I love Berlin')
    sentence_empty = Sentence('       ')
    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])
    sentence.clear_embeddings()
    sentence_empty.clear_embeddings()
    loaded_model = SequenceTagger.load('pos')
    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])


@pytest.mark.integration
def test_train_load_use_classifier(results_base_path, tasks_base_path):
    corpus = flair.datasets.ClassificationCorpus((tasks_base_path / 'imdb'))
    label_dict = corpus.make_label_dictionary()
    word_embedding = WordEmbeddings('turian')
    document_embeddings = DocumentRNNEmbeddings(
        [word_embedding], 128, 1, False, 64, False, False)
    model = TextClassifier(document_embeddings, label_dict, False)
    trainer = ModelTrainer(model, corpus)
    trainer.train(results_base_path, max_epochs=2, shuffle=False)
    sentence = Sentence('Berlin is a really nice city.')
    for s in model.predict(sentence):
        for l in s.labels:
            assert (l.value is not None)
            assert (0.0 <= l.score <= 1.0)
            assert (type(l.score) is float)
    loaded_model = TextClassifier.load((results_base_path / 'final-model.pt'))
    sentence = Sentence('I love Berlin')
    sentence_empty = Sentence('       ')
    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_train_classifier_with_sampler(results_base_path, tasks_base_path):
    corpus = flair.datasets.ClassificationCorpus((tasks_base_path / 'imdb'))
    label_dict = corpus.make_label_dictionary()
    word_embedding = WordEmbeddings('turian')
    document_embeddings = DocumentRNNEmbeddings(
        [word_embedding], 32, 1, False, 64, False, False)
    model = TextClassifier(document_embeddings, label_dict, False)
    trainer = ModelTrainer(model, corpus)
    trainer.train(results_base_path, max_epochs=2, shuffle=False,
                  sampler=ImbalancedClassificationDatasetSampler)
    sentence = Sentence('Berlin is a really nice city.')
    for s in model.predict(sentence):
        for l in s.labels:
            assert (l.value is not None)
            assert (0.0 <= l.score <= 1.0)
            assert (type(l.score) is float)
    loaded_model = TextClassifier.load((results_base_path / 'final-model.pt'))
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_train_load_use_classifier_with_prob(results_base_path, tasks_base_path):
    corpus = flair.datasets.ClassificationCorpus((tasks_base_path / 'imdb'))
    label_dict = corpus.make_label_dictionary()
    word_embedding = WordEmbeddings('turian')
    document_embeddings = DocumentRNNEmbeddings(
        [word_embedding], 128, 1, False, 64, False, False)
    model = TextClassifier(document_embeddings, label_dict, False)
    trainer = ModelTrainer(model, corpus)
    trainer.train(results_base_path, max_epochs=2, shuffle=False)
    sentence = Sentence('Berlin is a really nice city.')
    for s in model.predict(sentence, multi_class_prob=True):
        for l in s.labels:
            assert (l.value is not None)
            assert (0.0 <= l.score <= 1.0)
            assert (type(l.score) is float)
    loaded_model = TextClassifier.load((results_base_path / 'final-model.pt'))
    sentence = Sentence('I love Berlin')
    sentence_empty = Sentence('       ')
    loaded_model.predict(sentence, multi_class_prob=True)
    loaded_model.predict([sentence, sentence_empty], multi_class_prob=True)
    loaded_model.predict([sentence_empty], multi_class_prob=True)
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_train_load_use_classifier_multi_label(results_base_path, tasks_base_path):
    corpus = flair.datasets.ClassificationCorpus(
        (tasks_base_path / 'multi_class'))
    label_dict = corpus.make_label_dictionary()
    word_embedding = WordEmbeddings('turian')
    document_embeddings = DocumentRNNEmbeddings(
        embeddings=[word_embedding], hidden_size=32, reproject_words=False, bidirectional=False)
    model = TextClassifier(document_embeddings, label_dict, multi_label=True)
    trainer = ModelTrainer(model, corpus)
    trainer.train(results_base_path, mini_batch_size=1,
                  max_epochs=100, shuffle=False, checkpoint=False)
    sentence = Sentence('apple tv')
    for s in model.predict(sentence):
        for l in s.labels:
            print(l)
            assert (l.value is not None)
            assert (0.0 <= l.score <= 1.0)
            assert (type(l.score) is float)
    sentence = Sentence('apple tv')
    for s in model.predict(sentence):
        assert ('apple' in sentence.get_label_names())
        assert ('tv' in sentence.get_label_names())
        for l in s.labels:
            print(l)
            assert (l.value is not None)
            assert (0.0 <= l.score <= 1.0)
            assert (type(l.score) is float)
    loaded_model = TextClassifier.load((results_base_path / 'final-model.pt'))
    sentence = Sentence('I love Berlin')
    sentence_empty = Sentence('       ')
    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_train_charlm_load_use_classifier(results_base_path, tasks_base_path):
    corpus = flair.datasets.ClassificationCorpus((tasks_base_path / 'imdb'))
    label_dict = corpus.make_label_dictionary()
    embedding = FlairEmbeddings('news-forward-fast')
    document_embeddings = DocumentRNNEmbeddings(
        [embedding], 128, 1, False, 64, False, False)
    model = TextClassifier(document_embeddings, label_dict, False)
    trainer = ModelTrainer(model, corpus)
    trainer.train(results_base_path, max_epochs=2, shuffle=False)
    sentence = Sentence('Berlin is a really nice city.')
    for s in model.predict(sentence):
        for l in s.labels:
            assert (l.value is not None)
            assert (0.0 <= l.score <= 1.0)
            assert (type(l.score) is float)
    loaded_model = TextClassifier.load((results_base_path / 'final-model.pt'))
    sentence = Sentence('I love Berlin')
    sentence_empty = Sentence('       ')
    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_train_language_model(results_base_path, resources_path):
    dictionary = Dictionary.load('chars')
    language_model = LanguageModel(
        dictionary, is_forward_lm=True, hidden_size=128, nlayers=1)
    corpus = TextCorpus((resources_path / 'corpora/lorem_ipsum'),
                        dictionary, language_model.is_forward_lm, character_level=True)
    trainer = LanguageModelTrainer(language_model, corpus, test_mode=True)
    trainer.train(results_base_path, sequence_length=10,
                  mini_batch_size=10, max_epochs=2)
    char_lm_embeddings = FlairEmbeddings(
        str((results_base_path / 'best-lm.pt')))
    sentence = Sentence('I love Berlin')
    char_lm_embeddings.embed(sentence)
    (text, likelihood) = language_model.generate_text(number_of_characters=100)
    assert (text is not None)
    assert (len(text) >= 100)
    shutil.rmtree(results_base_path, ignore_errors=True)


@pytest.mark.integration
def test_train_load_use_tagger_multicorpus(results_base_path, tasks_base_path):
    corpus_1 = flair.datasets.ColumnCorpus(data_folder=(tasks_base_path / 'fashion'), column_format={
        0: 'text',
        2: 'ner',
    })
    corpus_2 = flair.datasets.GERMEVAL(base_path=tasks_base_path)
    corpus = MultiCorpus([corpus_1, corpus_2])
    tag_dictionary = corpus.make_tag_dictionary('ner')
    embeddings = WordEmbeddings('turian')
    tagger = SequenceTagger(hidden_size=64, embeddings=embeddings,
                            tag_dictionary=tag_dictionary, tag_type='ner', use_crf=False)
    trainer = ModelTrainer(tagger, corpus)
    trainer.train(results_base_path, learning_rate=0.1,
                  mini_batch_size=2, max_epochs=2, shuffle=False)
    loaded_model = SequenceTagger.load((results_base_path / 'final-model.pt'))
    sentence = Sentence('I love Berlin')
    sentence_empty = Sentence('       ')
    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_train_resume_text_classification_training(results_base_path, tasks_base_path):
    corpus = flair.datasets.ClassificationCorpus((tasks_base_path / 'imdb'))
    label_dict = corpus.make_label_dictionary()
    embeddings = FlairEmbeddings('news-forward-fast')
    document_embeddings = DocumentRNNEmbeddings([embeddings], 128, 1, False)
    model = TextClassifier(document_embeddings, label_dict, False)
    trainer = ModelTrainer(model, corpus)
    trainer.train(results_base_path, max_epochs=2,
                  shuffle=False, checkpoint=True)
    checkpoint = TextClassifier.load_checkpoint(
        (results_base_path / 'checkpoint.pt'))
    trainer = ModelTrainer.load_from_checkpoint(checkpoint, corpus)
    trainer.train(results_base_path, max_epochs=2,
                  shuffle=False, checkpoint=True)
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_train_resume_sequence_tagging_training(results_base_path, tasks_base_path):
    corpus_1 = flair.datasets.ColumnCorpus(data_folder=(tasks_base_path / 'fashion'), column_format={
        0: 'text',
        2: 'ner',
    })
    corpus_2 = flair.datasets.GERMEVAL(base_path=tasks_base_path)
    corpus = MultiCorpus([corpus_1, corpus_2])
    tag_dictionary = corpus.make_tag_dictionary('ner')
    embeddings = WordEmbeddings('turian')
    model = SequenceTagger(hidden_size=64, embeddings=embeddings,
                           tag_dictionary=tag_dictionary, tag_type='ner', use_crf=False)
    trainer = ModelTrainer(model, corpus)
    trainer.train(results_base_path, max_epochs=2,
                  shuffle=False, checkpoint=True)
    checkpoint = SequenceTagger.load_checkpoint(
        (results_base_path / 'checkpoint.pt'))
    trainer = ModelTrainer.load_from_checkpoint(checkpoint, corpus)
    trainer.train(results_base_path, max_epochs=2,
                  shuffle=False, checkpoint=True)
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_train_resume_language_model_training(resources_path, results_base_path, tasks_base_path):
    dictionary = Dictionary.load('chars')
    language_model = LanguageModel(
        dictionary, is_forward_lm=True, hidden_size=128, nlayers=1)
    corpus = TextCorpus((resources_path / 'corpora/lorem_ipsum'),
                        dictionary, language_model.is_forward_lm, character_level=True)
    trainer = LanguageModelTrainer(language_model, corpus, test_mode=True)
    trainer.train(results_base_path, sequence_length=10,
                  mini_batch_size=10, max_epochs=2, checkpoint=True)
    trainer = LanguageModelTrainer.load_from_checkpoint(
        (results_base_path / 'checkpoint.pt'), corpus)
    trainer.train(results_base_path, sequence_length=10,
                  mini_batch_size=10, max_epochs=2)
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_keep_word_embeddings():
    loaded_model = SequenceTagger.load('ner')
    sentence = Sentence('I love Berlin')
    loaded_model.predict(sentence)
    for token in sentence:
        assert (len(token.embedding.numpy()) == 0)
    loaded_model.predict(sentence, embedding_storage_mode='cpu')
    for token in sentence:
        assert (len(token.embedding.numpy()) > 0)


import pytest
from typing import Tuple
from flair.data import Dictionary, Corpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings
from flair.models.text_regression_model import TextRegressor
from flair.trainers import ModelTrainer


def init(tasks_base_path) -> Tuple[(Corpus, TextRegressor, ModelTrainer)]:
    corpus = NLPTaskDataFetcher.load_corpus(
        NLPTask.REGRESSION, tasks_base_path)
    glove_embedding = WordEmbeddings('glove')
    document_embeddings = DocumentRNNEmbeddings(
        [glove_embedding], 128, 1, False, 64, False, False)
    model = TextRegressor(document_embeddings)
    trainer = ModelTrainer(model, corpus)
    return (corpus, model, trainer)


def test_labels_to_indices(tasks_base_path):
    (corpus, model, trainer) = init(tasks_base_path)
    result = model._labels_to_indices(corpus.train)
    for i in range(len(corpus.train)):
        expected = round(float(corpus.train[i].labels[0].value), 3)
        actual = round(float(result[i].item()), 3)
        assert (expected == actual)


def test_trainer_evaluation(tasks_base_path):
    (corpus, model, trainer) = init(tasks_base_path)
    expected = model.evaluate(corpus.dev)
    assert (expected is not None)

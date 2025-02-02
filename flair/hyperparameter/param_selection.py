
import logging
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Tuple, Union
import numpy as np
from hyperopt import hp, fmin, tpe
import flair.nn
from flair.data import Corpus
from flair.embeddings import DocumentPoolEmbeddings, DocumentRNNEmbeddings
from flair.hyperparameter import Parameter
from flair.hyperparameter.parameter import SEQUENCE_TAGGER_PARAMETERS, TRAINING_PARAMETERS, DOCUMENT_EMBEDDING_PARAMETERS, MODEL_TRAINER_PARAMETERS
from flair.models import SequenceTagger, TextClassifier
from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric, log_line, init_output_file, add_file_handler
log = logging.getLogger('flair')


class OptimizationValue(Enum):
    DEV_LOSS = 'loss'
    DEV_SCORE = 'score'


class SearchSpace(object):

    def __init__(self):
        self.search_space = {

        }

    def add(self, parameter: Parameter, func, **kwargs):
        self.search_space[parameter.value] = func(parameter.value, **kwargs)

    def get_search_space(self):
        return hp.choice('parameters', [self.search_space])


class ParamSelector(object):

    def __init__(self, corpus: Corpus, base_path: Union[(str, Path)], max_epochs: int, evaluation_metric: EvaluationMetric, training_runs: int, optimization_value: OptimizationValue):
        if (type(base_path) is str):
            base_path = Path(base_path)
        self.corpus = corpus
        self.max_epochs = max_epochs
        self.base_path = base_path
        self.evaluation_metric = evaluation_metric
        self.run = 1
        self.training_runs = training_runs
        self.optimization_value = optimization_value
        self.param_selection_file = init_output_file(
            base_path, 'param_selection.txt')

    @abstractmethod
    def _set_up_model(self, params: dict) -> flair.nn.Model:
        pass

    def _objective(self, params: dict):
        log_line(log)
        log.info(''.join(['Evaluation run: ', '{}'.format(self.run)]))
        log.info('Evaluating parameter combination:')
        for (k, v) in params.items():
            if isinstance(v, Tuple):
                v = ','.join([str(x) for x in v])
            log.info(
                ''.join(['\t', '{}'.format(k), ': ', '{}'.format(str(v))]))
        log_line(log)
        for sent in self.corpus.get_all_sentences():
            sent.clear_embeddings()
        scores = []
        vars = []
        for i in range(0, self.training_runs):
            log_line(log)
            log.info(''.join(['Training run: ', '{}'.format((i + 1))]))
            model = self._set_up_model(params)
            training_params = {key: params[key]
                               for key in params if (key in TRAINING_PARAMETERS)}
            model_trainer_parameters = {
                key: params[key] for key in params if (key in MODEL_TRAINER_PARAMETERS)}
            trainer = ModelTrainer(model, self.corpus, **
                                   model_trainer_parameters)
            result = trainer.train(
                self.base_path, max_epochs=self.max_epochs, param_selection_mode=True, **training_params)
            if (self.optimization_value == OptimizationValue.DEV_LOSS):
                curr_scores = result['dev_loss_history'][(- 3):]
            else:
                curr_scores = list(
                    map((lambda s: (1 - s)), result['dev_score_history'][(- 3):]))
            score = (sum(curr_scores) / float(len(curr_scores)))
            var = np.var(curr_scores)
            scores.append(score)
            vars.append(var)
        final_score = (sum(scores) / float(len(scores)))
        final_var = (sum(vars) / float(len(vars)))
        test_score = result['test_score']
        log_line(log)
        log.info('Done evaluating parameter combination:')
        for (k, v) in params.items():
            if isinstance(v, Tuple):
                v = ','.join([str(x) for x in v])
            log.info(''.join(['\t', '{}'.format(k), ': ', '{}'.format(v)]))
        log.info(''.join(
            ['{}'.format(self.optimization_value.value), ': ', '{}'.format(final_score)]))
        log.info(''.join(['variance: ', '{}'.format(final_var)]))
        log.info(''.join(['test_score: ', '{}'.format(test_score), '\n']))
        log_line(log)
        with open(self.param_selection_file, 'a') as f:
            f.write(''.join(['evaluation run ', '{}'.format(self.run), '\n']))
            for (k, v) in params.items():
                if isinstance(v, Tuple):
                    v = ','.join([str(x) for x in v])
                f.write(
                    ''.join(['\t', '{}'.format(k), ': ', '{}'.format(str(v)), '\n']))
            f.write(''.join(['{}'.format(
                self.optimization_value.value), ': ', '{}'.format(final_score), '\n']))
            f.write(''.join(['variance: ', '{}'.format(final_var), '\n']))
            f.write(''.join(['test_score: ', '{}'.format(test_score), '\n']))
            f.write((('-' * 100) + '\n'))
        self.run += 1
        return {
            'status': 'ok',
            'loss': final_score,
            'loss_variance': final_var,
        }

    def optimize(self, space: SearchSpace, max_evals=100):
        search_space = space.search_space
        best = fmin(self._objective, search_space,
                    algo=tpe.suggest, max_evals=max_evals)
        log_line(log)
        log.info('Optimizing parameter configuration done.')
        log.info('Best parameter configuration found:')
        for (k, v) in best.items():
            log.info(''.join(['\t', '{}'.format(k), ': ', '{}'.format(v)]))
        log_line(log)
        with open(self.param_selection_file, 'a') as f:
            f.write('best parameter combination\n')
            for (k, v) in best.items():
                if isinstance(v, Tuple):
                    v = ','.join([str(x) for x in v])
                f.write(
                    ''.join(['\t', '{}'.format(k), ': ', '{}'.format(str(v)), '\n']))


class SequenceTaggerParamSelector(ParamSelector):

    def __init__(self, corpus: Corpus, tag_type: str, base_path: Union[(str, Path)], max_epochs: int = 50, evaluation_metric: EvaluationMetric = EvaluationMetric.MICRO_F1_SCORE, training_runs: int = 1, optimization_value: OptimizationValue = OptimizationValue.DEV_LOSS):
        '\n        :param corpus: the corpus\n        :param tag_type: tag type to use\n        :param base_path: the path to the result folder (results will be written to that folder)\n        :param max_epochs: number of epochs to perform on every evaluation run\n        :param evaluation_metric: evaluation metric used during training\n        :param training_runs: number of training runs per evaluation run\n        :param optimization_value: value to optimize\n        '
        super().__init__(corpus, base_path, max_epochs,
                         evaluation_metric, training_runs, optimization_value)
        self.tag_type = tag_type
        self.tag_dictionary = self.corpus.make_tag_dictionary(self.tag_type)

    def _set_up_model(self, params: dict):
        sequence_tagger_params = {key: params[key] for key in params if (
            key in SEQUENCE_TAGGER_PARAMETERS)}
        tagger = SequenceTagger(tag_dictionary=self.tag_dictionary,
                                tag_type=self.tag_type, **sequence_tagger_params)
        return tagger


class TextClassifierParamSelector(ParamSelector):

    def __init__(self, corpus: Corpus, multi_label: bool, base_path: Union[(str, Path)], document_embedding_type: str, max_epochs: int = 50, evaluation_metric: EvaluationMetric = EvaluationMetric.MICRO_F1_SCORE, training_runs: int = 1, optimization_value: OptimizationValue = OptimizationValue.DEV_LOSS):
        "\n        :param corpus: the corpus\n        :param multi_label: true, if the dataset is multi label, false otherwise\n        :param base_path: the path to the result folder (results will be written to that folder)\n        :param document_embedding_type: either 'lstm', 'mean', 'min', or 'max'\n        :param max_epochs: number of epochs to perform on every evaluation run\n        :param evaluation_metric: evaluation metric used during training\n        :param training_runs: number of training runs per evaluation run\n        :param optimization_value: value to optimize\n        "
        super().__init__(corpus, base_path, max_epochs,
                         evaluation_metric, training_runs, optimization_value)
        self.multi_label = multi_label
        self.document_embedding_type = document_embedding_type
        self.label_dictionary = self.corpus.make_label_dictionary()

    def _set_up_model(self, params: dict):
        embdding_params = {key: params[key] for key in params if (
            key in DOCUMENT_EMBEDDING_PARAMETERS)}
        if (self.document_embedding_type == 'lstm'):
            document_embedding = DocumentRNNEmbeddings(**embdding_params)
        else:
            document_embedding = DocumentPoolEmbeddings(**embdding_params)
        text_classifier = TextClassifier(label_dictionary=self.label_dictionary,
                                         multi_label=self.multi_label, document_embeddings=document_embedding)
        return text_classifier


from pathlib import Path
import flair
import flair.embeddings
import torch
import torch.nn as nn
from typing import List, Union
from flair.datasets import DataLoader
from flair.training_utils import MetricRegression, Result, store_embeddings
from flair.data import Sentence, Label
import logging
log = logging.getLogger('flair')


class TextRegressor(flair.models.TextClassifier):

    def __init__(self, document_embeddings: flair.embeddings.DocumentEmbeddings):
        super(TextRegressor, self).__init__(document_embeddings=document_embeddings,
                                            label_dictionary=flair.data.Dictionary(), multi_label=False)
        log.info('Using REGRESSION - experimental')
        self.loss_function = nn.MSELoss()

    def _labels_to_indices(self, sentences: List[Sentence]):
        indices = [torch.tensor([float(label.value) for label in sentence.labels],
                                dtype=torch.float) for sentence in sentences]
        vec = torch.cat(indices, 0).to(flair.device)
        return vec

    def predict(self, sentences: Union[(Sentence, List[Sentence])], mini_batch_size: int = 32, embedding_storage_mode='none') -> List[Sentence]:
        with torch.no_grad():
            if (type(sentences) is Sentence):
                sentences = [sentences]
            filtered_sentences = self._filter_empty_sentences(sentences)
            store_embeddings(filtered_sentences, 'none')
            batches = [filtered_sentences[x:(
                x + mini_batch_size)] for x in range(0, len(filtered_sentences), mini_batch_size)]
            for batch in batches:
                scores = self.forward(batch)
                for (sentence, score) in zip(batch, scores.tolist()):
                    sentence.labels = [Label(value=str(score[0]))]
                store_embeddings(batch, storage_mode=embedding_storage_mode)
            return sentences

    def _calculate_loss(self, scores: torch.tensor, sentences: List[Sentence]) -> torch.tensor:
        '\n        Calculates the loss.\n        :param scores: the prediction scores from the model\n        :param sentences: list of sentences\n        :return: loss value\n        '
        return self.loss_function(scores.squeeze(1), self._labels_to_indices(sentences))

    def forward_labels_and_loss(self, sentences: Union[(Sentence, List[Sentence])]) -> (List[List[float]], torch.tensor):
        scores = self.forward(sentences)
        loss = self._calculate_loss(scores, sentences)
        return (scores, loss)

    def evaluate(self, data_loader: DataLoader, out_path: Path = None, embeddings_storage_mode: str = 'cpu') -> (Result, float):
        with torch.no_grad():
            eval_loss = 0
            metric = MetricRegression('Evaluation')
            lines = []
            total_count = 0
            for (batch_nr, batch) in enumerate(data_loader):
                if isinstance(batch, Sentence):
                    batch = [batch]
                (scores, loss) = self.forward_labels_and_loss(batch)
                true_values = []
                for sentence in batch:
                    total_count += 1
                    for label in sentence.labels:
                        true_values.append(float(label.value))
                results = []
                for score in scores:
                    if (type(score[0]) is Label):
                        results.append(float(score[0].score))
                    else:
                        results.append(float(score[0]))
                eval_loss += loss
                metric.true.extend(true_values)
                metric.pred.extend(results)
                for (sentence, prediction, true_value) in zip(batch, results, true_values):
                    eval_line = '{}\t{}\t{}\n'.format(
                        sentence.to_original_text(), true_value, prediction)
                    lines.append(eval_line)
                store_embeddings(batch, embeddings_storage_mode)
            eval_loss /= total_count
            if (out_path is not None):
                with open(out_path, 'w', encoding='utf-8') as outfile:
                    outfile.write(''.join(lines))
            log_line = ''.join(['{}'.format(metric.mean_squared_error()), '\t', '{}'.format(
                metric.spearmanr()), '\t', '{}'.format(metric.pearsonr())])
            log_header = 'MSE\tSPEARMAN\tPEARSON'
            detailed_result = ''.join(['AVG: mse: ', '{:.4f}'.format(metric.mean_squared_error()), ' - mae: ', '{:.4f}'.format(
                metric.mean_absolute_error()), ' - pearson: ', '{:.4f}'.format(metric.pearsonr()), ' - spearman: ', '{:.4f}'.format(metric.spearmanr())])
            result = Result(metric.pearsonr(), log_header,
                            log_line, detailed_result)
            return (result, eval_loss)

    def _get_state_dict(self):
        model_state = {
            'state_dict': self.state_dict(),
            'document_embeddings': self.document_embeddings,
        }
        return model_state

    def _init_model_with_state_dict(state):
        model = TextRegressor(document_embeddings=state['document_embeddings'])
        model.load_state_dict(state['state_dict'])
        return model

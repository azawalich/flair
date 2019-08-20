
import logging
from pathlib import Path
from typing import List, Union
import time
import sys
import datetime
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.sgd import SGD
from torch.utils.data.dataset import ConcatDataset
try:
    from apex import amp
except ImportError:
    amp = None
import flair
import flair.nn
from flair.data import MultiCorpus, Corpus
from flair.datasets import DataLoader
from flair.optim import ExpAnnealLR
from flair.training_utils import init_output_file, WeightExtractor, log_line, add_file_handler, Result, store_embeddings
log = logging.getLogger('flair')


class ModelTrainer():

    def __init__(self, model: flair.nn.Model, corpus: Corpus, optimizer: torch.optim.Optimizer = SGD, epoch: int = 0, loss: float = 10000.0, optimizer_state: dict = None, scheduler_state: dict = None, use_tensorboard: bool = False):
        self.model = model
        self.corpus = corpus
        self.optimizer = optimizer
        self.epoch = epoch
        self.loss = loss
        self.scheduler_state = scheduler_state
        self.optimizer_state = optimizer_state
        self.use_tensorboard = use_tensorboard

    def train(self, base_path: Union[(Path, str)], learning_rate: float = 0.1, mini_batch_size: int = 32, eval_mini_batch_size: int = None, max_epochs: int = 100, anneal_factor: float = 0.5, patience: int = 3, min_learning_rate: float = 0.0001, train_with_dev: bool = False, monitor_train: bool = False, monitor_test: bool = False, embeddings_storage_mode: str = 'cpu', checkpoint: bool = False, save_final_model: bool = True, anneal_with_restarts: bool = False, shuffle: bool = True, param_selection_mode: bool = False, num_workers: int = 6, sampler=None, use_amp: bool = False, amp_opt_level: str = 'O1', **kwargs) -> dict:
        "\n        Trains any class that implements the flair.nn.Model interface.\n        :param base_path: Main path to which all output during training is logged and models are saved\n        :param learning_rate: Initial learning rate\n        :param mini_batch_size: Size of mini-batches during training\n        :param eval_mini_batch_size: Size of mini-batches during evaluation\n        :param max_epochs: Maximum number of epochs to train. Terminates training if this number is surpassed.\n        :param anneal_factor: The factor by which the learning rate is annealed\n        :param patience: Patience is the number of epochs with no improvement the Trainer waits\n         until annealing the learning rate\n        :param min_learning_rate: If the learning rate falls below this threshold, training terminates\n        :param train_with_dev: If True, training is performed using both train+dev data\n        :param monitor_train: If True, training data is evaluated at end of each epoch\n        :param monitor_test: If True, test data is evaluated at end of each epoch\n        :param embeddings_storage_mode: One of 'none' (all embeddings are deleted and freshly recomputed),\n        'cpu' (embeddings are stored on CPU) or 'gpu' (embeddings are stored on GPU)\n        :param checkpoint: If True, a full checkpoint is saved at end of each epoch\n        :param save_final_model: If True, final model is saved\n        :param anneal_with_restarts: If True, the last best model is restored when annealing the learning rate\n        :param shuffle: If True, data is shuffled during training\n        :param param_selection_mode: If True, testing is performed against dev data. Use this mode when doing\n        parameter selection.\n        :param num_workers: Number of workers in your data loader.\n        :param sampler: You can pass a data sampler here for special sampling of data.\n        :param kwargs: Other arguments for the Optimizer\n        :return:\n        "
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter()
            except:
                log_line(log)
                log.warning(
                    'ATTENTION! PyTorch >= 1.1.0 and pillow are required for TensorBoard support!')
                log_line(log)
                self.use_tensorboard = False
                pass
        if use_amp:
            if (sys.version_info < (3, 0)):
                raise RuntimeError(
                    'Apex currently only supports Python 3. Aborting.')
            if (amp is None):
                raise RuntimeError(
                    'Failed to import apex. Please install apex from https://www.github.com/nvidia/apex to enable mixed-precision training.')
        if (eval_mini_batch_size is None):
            eval_mini_batch_size = mini_batch_size
        if (type(base_path) is str):
            base_path = Path(base_path)
        log_handler = add_file_handler(log, (base_path / 'training.log'))
        log_line(log)
        log.info(''.join(['Model: "', '{}'.format(self.model), '"']))
        log_line(log)
        log.info(''.join(['Corpus: "', '{}'.format(self.corpus), '"']))
        log_line(log)
        log.info('Parameters:')
        log.info(
            ''.join([' - learning_rate: "', '{}'.format(learning_rate), '"']))
        log.info(''.join([' - mini_batch_size: "',
                          '{}'.format(mini_batch_size), '"']))
        log.info(''.join([' - patience: "', '{}'.format(patience), '"']))
        log.info(
            ''.join([' - anneal_factor: "', '{}'.format(anneal_factor), '"']))
        log.info(''.join([' - max_epochs: "', '{}'.format(max_epochs), '"']))
        log.info(''.join([' - shuffle: "', '{}'.format(shuffle), '"']))
        log.info(
            ''.join([' - train_with_dev: "', '{}'.format(train_with_dev), '"']))
        log_line(log)
        log.info(
            ''.join(['Model training base path: "', '{}'.format(base_path), '"']))
        log_line(log)
        log.info(''.join(['Device: ', '{}'.format(flair.device)]))
        log_line(log)
        log.info(''.join(['Embeddings storage mode: ',
                          '{}'.format(embeddings_storage_mode)]))
        log_train = (True if monitor_train else False)
        log_test = (True if ((not param_selection_mode)
                             and self.corpus.test and monitor_test) else False)
        log_dev = (True if (not train_with_dev) else False)
        loss_txt = init_output_file(base_path, 'loss.tsv')
        weight_extractor = WeightExtractor(base_path)
        optimizer = self.optimizer(
            self.model.parameters(), lr=learning_rate, **kwargs)
        if (self.optimizer_state is not None):
            optimizer.load_state_dict(self.optimizer_state)
        if use_amp:
            (self.model, optimizer) = amp.initialize(
                self.model, optimizer, opt_level=amp_opt_level)
        anneal_mode = ('min' if train_with_dev else 'max')
        scheduler = ReduceLROnPlateau(
            optimizer, factor=anneal_factor, patience=patience, mode=anneal_mode, verbose=True)
        if (self.scheduler_state is not None):
            scheduler.load_state_dict(self.scheduler_state)
        train_data = self.corpus.train
        if train_with_dev:
            train_data = ConcatDataset([self.corpus.train, self.corpus.dev])
        if (sampler is not None):
            sampler = sampler(train_data)
            shuffle = False
        dev_score_history = []
        dev_loss_history = []
        train_loss_history = []
        try:
            previous_learning_rate = learning_rate
            for epoch in range((0 + self.epoch), (max_epochs + self.epoch)):
                log_line(log)
                for group in optimizer.param_groups:
                    learning_rate = group['lr']
                if ((learning_rate != previous_learning_rate) and anneal_with_restarts and (base_path / 'best-model.pt').exists()):
                    log.info('resetting to best model')
                    self.model.load((base_path / 'best-model.pt'))
                previous_learning_rate = learning_rate
                if (learning_rate < min_learning_rate):
                    log_line(log)
                    log.info('learning rate too small - quitting training!')
                    log_line(log)
                    break
                batch_loader = DataLoader(train_data, batch_size=mini_batch_size,
                                          shuffle=shuffle, num_workers=num_workers, sampler=sampler)
                self.model.train()
                train_loss = 0
                seen_batches = 0
                total_number_of_batches = len(batch_loader)
                modulo = max(1, int((total_number_of_batches / 10)))
                batch_time = 0
                for (batch_no, batch) in enumerate(batch_loader):
                    start_time = time.time()
                    loss = self.model.forward_loss(batch)
                    optimizer.zero_grad()
                    if use_amp:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 5.0)
                    optimizer.step()
                    seen_batches += 1
                    train_loss += loss.item()
                    store_embeddings(batch, embeddings_storage_mode)
                    batch_time += (time.time() - start_time)
                    if ((batch_no % modulo) == 0):
                        log.info(''.join(['epoch ', '{}'.format((epoch + 1)), ' - iter ', '{}'.format(batch_no), '/', '{}'.format(total_number_of_batches),
                                          ' - loss ', '{:.8f}'.format((train_loss / seen_batches)), ' - samples/sec: ', '{:.2f}'.format(((mini_batch_size * modulo) / batch_time))]))
                        batch_time = 0
                        iteration = (
                            (epoch * total_number_of_batches) + batch_no)
                        if (not param_selection_mode):
                            weight_extractor.extract_weights(
                                self.model.state_dict(), iteration)
                train_loss /= seen_batches
                self.model.eval()
                log_line(log)
                log.info(''.join(['EPOCH ', '{}'.format((epoch + 1)), ' done: loss ',
                                  '{:.4f}'.format(train_loss), ' - lr ', '{:.4f}'.format(learning_rate)]))
                if self.use_tensorboard:
                    writer.add_scalar('train_loss', train_loss, (epoch + 1))
                current_score = train_loss
                result_line = ''
                if log_train:
                    (train_eval_result, train_loss) = self.model.evaluate(DataLoader(self.corpus.train,
                                                                                     batch_size=eval_mini_batch_size, num_workers=num_workers), embeddings_storage_mode=embeddings_storage_mode)
                    result_line += ''.join(['\t',
                                            '{}'.format(train_eval_result.log_line)])
                    store_embeddings(self.corpus.train,
                                     embeddings_storage_mode)
                if log_dev:
                    (dev_eval_result, dev_loss) = self.model.evaluate(DataLoader(
                        self.corpus.dev, batch_size=eval_mini_batch_size, num_workers=num_workers), embeddings_storage_mode=embeddings_storage_mode)
                    result_line += ''.join(['\t', '{}'.format(dev_loss),
                                            '\t', '{}'.format(dev_eval_result.log_line)])
                    log.info(''.join(['DEV : loss ', '{}'.format(
                        dev_loss), ' - score ', '{}'.format(dev_eval_result.main_score)]))
                    dev_score_history.append(dev_eval_result.main_score)
                    dev_loss_history.append(dev_loss)
                    current_score = dev_eval_result.main_score
                    store_embeddings(self.corpus.dev, embeddings_storage_mode)
                    if self.use_tensorboard:
                        writer.add_scalar('dev_loss', dev_loss, (epoch + 1))
                        writer.add_scalar(
                            'dev_score', dev_eval_result.main_score, (epoch + 1))
                if log_test:
                    (test_eval_result, test_loss) = self.model.evaluate(DataLoader(self.corpus.test, batch_size=eval_mini_batch_size,
                                                                                   num_workers=num_workers), (base_path / 'test.tsv'), embeddings_storage_mode=embeddings_storage_mode)
                    result_line += ''.join(['\t', '{}'.format(test_loss),
                                            '\t', '{}'.format(test_eval_result.log_line)])
                    log.info(''.join(['TEST : loss ', '{}'.format(
                        test_loss), ' - score ', '{}'.format(test_eval_result.main_score)]))
                    store_embeddings(self.corpus.test, embeddings_storage_mode)
                    if self.use_tensorboard:
                        writer.add_scalar('test_loss', test_loss, (epoch + 1))
                        writer.add_scalar(
                            'test_score', test_eval_result.main_score, (epoch + 1))
                scheduler.step(current_score)
                train_loss_history.append(train_loss)
                try:
                    bad_epochs = scheduler.num_bad_epochs
                except:
                    bad_epochs = 0
                for group in optimizer.param_groups:
                    new_learning_rate = group['lr']
                if (new_learning_rate != previous_learning_rate):
                    bad_epochs = (patience + 1)
                log.info(
                    ''.join(['BAD EPOCHS (no improvement): ', '{}'.format(bad_epochs)]))
                with open(loss_txt, 'a') as f:
                    if (epoch == 0):
                        f.write(
                            'EPOCH\tTIMESTAMP\tBAD_EPOCHS\tLEARNING_RATE\tTRAIN_LOSS')
                        if log_train:
                            f.write(
                                ('\tTRAIN_' + '\tTRAIN_'.join(train_eval_result.log_header.split('\t'))))
                        if log_dev:
                            f.write(
                                ('\tDEV_LOSS\tDEV_' + '\tDEV_'.join(dev_eval_result.log_header.split('\t'))))
                        if log_test:
                            f.write(
                                ('\tTEST_LOSS\tTEST_' + '\tTEST_'.join(test_eval_result.log_header.split('\t'))))
                    f.write(''.join(['\n', '{}'.format(epoch), '\t', '{:%H:%M:%S}'.format(datetime.datetime.now(
                    )), '\t', '{}'.format(bad_epochs), '\t', '{:.4f}'.format(learning_rate), '\t', '{}'.format(train_loss)]))
                    f.write(result_line)
                if (checkpoint and (not param_selection_mode)):
                    self.model.save_checkpoint((base_path / 'checkpoint.pt'), optimizer.state_dict(
                    ), scheduler.state_dict(), (epoch + 1), train_loss)
                if ((not train_with_dev) and (not param_selection_mode) and (current_score == scheduler.best)):
                    self.model.save((base_path / 'best-model.pt'))
            if (save_final_model and (not param_selection_mode)):
                self.model.save((base_path / 'final-model.pt'))
        except KeyboardInterrupt:
            log_line(log)
            log.info('Exiting from training early.')
            if self.use_tensorboard:
                writer.close()
            if (not param_selection_mode):
                log.info('Saving model ...')
                self.model.save((base_path / 'final-model.pt'))
                log.info('Done.')
        if self.corpus.test:
            final_score = self.final_test(
                base_path, eval_mini_batch_size, num_workers)
        else:
            final_score = 0
            log.info('Test data not provided setting final score to 0')
        log.removeHandler(log_handler)
        if self.use_tensorboard:
            writer.close()
        return {
            'test_score': final_score,
            'dev_score_history': dev_score_history,
            'train_loss_history': train_loss_history,
            'dev_loss_history': dev_loss_history,
        }

    def final_test(self, base_path: Path, eval_mini_batch_size: int, num_workers: int = 8):
        log_line(log)
        log.info('Testing using best model ...')
        self.model.eval()
        if (base_path / 'best-model.pt').exists():
            self.model = self.model.load((base_path / 'best-model.pt'))
        (test_results, test_loss) = self.model.evaluate(DataLoader(self.corpus.test, batch_size=eval_mini_batch_size,
                                                                   num_workers=num_workers), out_path=(base_path / 'test.tsv'), embeddings_storage_mode='none')
        test_results = test_results
        log.info(test_results.log_line)
        log.info(test_results.detailed_results)
        log_line(log)
        if (type(self.corpus) is MultiCorpus):
            for subcorpus in self.corpus.corpora:
                log_line(log)
                self.model.evaluate(DataLoader(subcorpus.test, batch_size=eval_mini_batch_size, num_workers=num_workers), out_path=(
                    base_path / ''.join(['{}'.format(subcorpus.name), '-test.tsv'])), embeddings_storage_mode='none')
        final_score = test_results.main_score
        return final_score

    @classmethod
    def load_from_checkpoint(cls, checkpoint, corpus: Corpus, optimizer: torch.optim.Optimizer = SGD):
        return ModelTrainer(checkpoint['model'], corpus, optimizer, epoch=checkpoint['epoch'], loss=checkpoint['loss'], optimizer_state=checkpoint['optimizer_state_dict'], scheduler_state=checkpoint['scheduler_state_dict'])

    def find_learning_rate(self, base_path: Union[(Path, str)], file_name: str = 'learning_rate.tsv', start_learning_rate: float = 1e-07, end_learning_rate: float = 10, iterations: int = 100, mini_batch_size: int = 32, stop_early: bool = True, smoothing_factor: float = 0.98, **kwargs) -> Path:
        best_loss = None
        moving_avg_loss = 0
        if (type(base_path) is str):
            base_path = Path(base_path)
        learning_rate_tsv = init_output_file(base_path, file_name)
        with open(learning_rate_tsv, 'a') as f:
            f.write('ITERATION\tTIMESTAMP\tLEARNING_RATE\tTRAIN_LOSS\n')
        optimizer = self.optimizer(
            self.model.parameters(), lr=start_learning_rate, **kwargs)
        train_data = self.corpus.train
        batch_loader = DataLoader(
            train_data, batch_size=mini_batch_size, shuffle=True)
        scheduler = ExpAnnealLR(optimizer, end_learning_rate, iterations)
        model_state = self.model.state_dict()
        model_device = next(self.model.parameters()).device
        self.model.train()
        for (itr, batch) in enumerate(batch_loader):
            loss = self.model.forward_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            optimizer.step()
            scheduler.step(1)
            learning_rate = scheduler.get_lr()[0]
            loss_item = loss.item()
            if (itr == 0):
                best_loss = loss_item
            else:
                if (smoothing_factor > 0):
                    moving_avg_loss = (
                        (smoothing_factor * moving_avg_loss) + ((1 - smoothing_factor) * loss_item))
                    loss_item = (moving_avg_loss /
                                 (1 - (smoothing_factor ** (itr + 1))))
                if (loss_item < best_loss):
                    best_loss = loss
            if (stop_early and ((loss_item > (4 * best_loss)) or torch.isnan(loss))):
                log_line(log)
                log.info('loss diverged - stopping early!')
                break
            if (itr > iterations):
                break
            with open(str(learning_rate_tsv), 'a') as f:
                f.write(''.join(['{}'.format(itr), '\t', '{:%H:%M:%S}'.format(datetime.datetime.now(
                )), '\t', '{}'.format(learning_rate), '\t', '{}'.format(loss_item), '\n']))
        self.model.load_state_dict(model_state)
        self.model.to(model_device)
        log_line(log)
        log.info(''.join(
            ['learning rate finder finished - plot ', '{}'.format(learning_rate_tsv)]))
        log_line(log)
        return Path(learning_rate_tsv)

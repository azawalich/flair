
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
from typing import List
import flair.datasets
from flair.data import Corpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings, CharacterEmbeddings
from flair.training_utils import EvaluationMetric
from flair.visual.training_curves import Plotter
corpus = flair.datasets.UD_ENGLISH()
print(corpus)
tag_type = 'upos'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)
embedding_types = [WordEmbeddings('glove')]
embeddings = StackedEmbeddings(embeddings=embedding_types)
tagger = SequenceTagger(hidden_size=256, embeddings=embeddings,
                        tag_dictionary=tag_dictionary, tag_type=tag_type, use_crf=True)
trainer = ModelTrainer(tagger, corpus)
trainer.train('resources/taggers/example-ner', learning_rate=0.1,
              mini_batch_size=32, max_epochs=20, shuffle=False)
plotter = Plotter()
plotter.plot_training_curves('resources/taggers/example-ner/loss.tsv')
plotter.plot_weights('resources/taggers/example-ner/weights.txt')

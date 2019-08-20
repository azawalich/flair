
from flair.data import Sentence
from flair.models import SequenceTagger
tagger = SequenceTagger.load('ner')
sentence = Sentence('George Washington went to Washington .')
tagger.predict(sentence)
print(('Analysing %s' % sentence))
print('\nThe following NER tags are found: \n')
print(sentence.to_tagged_string())

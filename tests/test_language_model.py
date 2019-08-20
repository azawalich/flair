
import shutil
from flair.data import Dictionary
from flair.trainers.language_model_trainer import TextCorpus


def test_train_resume_language_model_training(resources_path, results_base_path, tasks_base_path):
    dictionary = Dictionary.load('chars')
    corpus = TextCorpus((resources_path / 'corpora/lorem_ipsum'),
                        dictionary, forward=True, character_level=True)
    assert (corpus.test is not None)
    assert (corpus.train is not None)
    assert (corpus.valid is not None)
    assert (len(corpus.train) == 2)


def test_generate_text_with_small_temperatures():
    from flair.embeddings import FlairEmbeddings
    language_model = FlairEmbeddings('news-forward-fast').lm
    (text, likelihood) = language_model.generate_text(
        temperature=0.01, number_of_characters=100)
    assert (text is not None)
    assert (len(text) >= 100)


def test_compute_perplexity():
    from flair.embeddings import FlairEmbeddings
    language_model = FlairEmbeddings('news-forward-fast').lm
    grammatical = 'The company made a profit'
    perplexity_gramamtical_sentence = language_model.calculate_perplexity(
        grammatical)
    ungrammatical = 'Nook negh qapla!'
    perplexity_ungramamtical_sentence = language_model.calculate_perplexity(
        ungrammatical)
    print(''.join(['"', '{}'.format(grammatical), '" - perplexity is ',
                   '{}'.format(perplexity_gramamtical_sentence)]))
    print(''.join(['"', '{}'.format(ungrammatical), '" - perplexity is ',
                   '{}'.format(perplexity_ungramamtical_sentence)]))
    assert (perplexity_gramamtical_sentence <
            perplexity_ungramamtical_sentence)
    language_model = FlairEmbeddings('news-backward-fast').lm
    grammatical = 'The company made a profit'
    perplexity_gramamtical_sentence = language_model.calculate_perplexity(
        grammatical)
    ungrammatical = 'Nook negh qapla!'
    perplexity_ungramamtical_sentence = language_model.calculate_perplexity(
        ungrammatical)
    print(''.join(['"', '{}'.format(grammatical), '" - perplexity is ',
                   '{}'.format(perplexity_gramamtical_sentence)]))
    print(''.join(['"', '{}'.format(ungrammatical), '" - perplexity is ',
                   '{}'.format(perplexity_ungramamtical_sentence)]))
    assert (perplexity_gramamtical_sentence <
            perplexity_ungramamtical_sentence)

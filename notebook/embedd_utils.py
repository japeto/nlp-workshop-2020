"""Data utilitites
This script allows calculate word similarities and save in python dictionaries
"""
from scipy.spatial import distance
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
# own libraries

# embeddings - All Embedding objects cached index by name
embeddings = dict()
#cosines    - Pre-calculated cosine similarities between Embedding objects
cosines = dict()
#euclideans - Pre-calculated euclidean similarities between Embedding objects
euclideans = dict()

def cached_embedding(word):
    """
    Interface to 'embeddings' dictionary
    :param word: string
    :return: Embedding or None
    """
    if word in embeddings:
        return embeddings[word]
    return None

def clear_cache():
    """
    Clears all cached data.
    """
    global embeddings, cosines, euclideans
    embeddings = dict()
    cosines = dict()
    euclideans = dict()

def cosine_similarity(w1_emb, w2_emb):
    """
    Calculates cosine similarity between w1_emb and w2_emb Embedding objects. It will store the calculations
    and if the same embeddings are compared again it will return this saved result.
    :param w1_emb: Embedding
    :param w2_emb: Embedding
    :return: float
    """
    if w2_emb in cosines and w1_emb in cosines[w2_emb]:
        return cosines[w2_emb][w1_emb]
    if w1_emb in cosines and w2_emb in cosines[w1_emb]:
        return cosines[w1_emb][w2_emb]
    if w1_emb not in cosines:
        cosines[w1_emb] = dict()
    cosine_value = -distance.cosine(w1_emb.v, w2_emb.v) / 2 + 1
    cosines[w1_emb][w2_emb] = cosine_value
    return cosine_value

def euclidean_similarity(w1_emb, w2_emb):
    """
    Calculates euclidean similarity between w1_emb and w2_emb Embedding objects. It will store the calculations
    and if the same embeddings are compared again it will return this saved result.
    :param w1_emb: Embedding
    :param w2_emb: Embedding
    :return: float
    """
    if w2_emb in euclideans and w1_emb in euclideans[w2_emb]:
        return euclideans[w2_emb][w1_emb]
    if w1_emb in euclideans and w2_emb in euclideans[w1_emb]:
        return euclideans[w1_emb][w2_emb]
    if w1_emb not in euclideans:
        euclideans[w1_emb] = dict()
    euclidean_value = 1 / (1 + distance.euclidean(w1_emb.v, w2_emb.v))
    euclideans[w1_emb][w2_emb] = euclidean_value
    return euclidean_value

def load_model(word2vec_file=None,limit=10**5, debug=False):
    """
    Loads w2v vectors.
    :param seed_file: string
    :param seed_file: numeric
    :return: Word2Vec model
    """
    model = KeyedVectors.load_word2vec_format(word2vec_file, binary=True, limit=limit )
    print ("Model successfully loaded") if debug else None
    return model
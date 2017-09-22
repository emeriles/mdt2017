"""
Clusterize words from a given file
Usage:
  clusterize.py -k <n> -f <file>
  clusterize.py -h | --help
Options:
  -k <n>            k for k-means clustering algorithm.
  -f <file>         Input file.
  -o <file>         Output file.
  --pos_t <tagger>  POS_tagger ('nltk', or default 'stanford')
"""
import preprocess
from vectorizer import Vectorizer

from sklearn.cluster import KMeans
from docopt import docopt
from scipy.sparse.csr import csr_matrix
import numpy as np

WORDS = []
MATRIX = csr_matrix(0) #    just an empty matrix
KMEANS = None

def get_cluster_sizes(n_clusters):
    cluster_sizes = np.zeros(n_clusters)
    for label in KMEANS.labels_:
        cluster_sizes[label] += 1
    return cluster_sizes

def get_word_from_cluster(c_num):
    words = []
    for word, label in zip(WORDS, KMEANS.labels_):
        if label == c_num:
            words.append(word)
    return words

if __name__ == '__main__':
    # arg parse
    opts = docopt(__doc__)
    # print(opts)
    if not opts['-k'] or not opts['-f']:
        exit()
    if not opts['--pos_t']:
        pos_tagger = 'stanford'
    n_clusters = int(opts['-k'])

    # preprocess corpus
    corpus = preprocess.Corpus_Tokenizer(opts['-f'])

    # vectorizeee
    vectorizer = Vectorizer(corpus, pos_tagger=pos_tagger)
    WORDS, MATRIX = vectorizer.get_vector_matrix()

    # get clusters
    KMEANS = KMeans(n_clusters=n_clusters, init='k-means++',
                n_init=10, max_iter=300,
                tol=0.0001, precompute_distances='auto',
                verbose=0, random_state=None, copy_x=True,
                n_jobs=4, algorithm='auto').fit(MATRIX)




    # PRINTING!!
    #print([x for x in zip(WORDS, KMEANS.labels_)][:10])

    #print(get_cluster_sizes(n_clusters))

    for i in range(n_clusters):
        print('-----------------')
        print('Cluster number ', i)
        words_in_cluster_n = get_word_from_cluster(i)
        print('Cluster size: ', len(words_in_cluster_n))
        print('Words: ')
        print(str.join(' ',words_in_cluster_n))
from nltk.tag import StanfordPOSTagger
#from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.snowball import SpanishStemmer

from sklearn.feature_extraction import DictVectorizer
import random
import numpy
import re

import spacy
import es_core_web_md

from scipy.sparse import vstack
from sklearn.preprocessing import normalize
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import TruncatedSVD



class POS_Tagger():
    def __init__(self, tagger='nltk'):
        self.tagger = tagger

        if self.tagger == 'stanford':
            self.__tag_path_to_model = "extra/spanish-distsim.tagger"
            self.__tag_path_to_jar = "extra/stanford-postagger-3.8.0.jar"

    def get_tagged_sents(self, sents):
        if self.tagger == 'stanford':
            # java_options -mx3000m sets memory use in 3GB
            tagger = StanfordPOSTagger(self.__tag_path_to_model, 
                        self.__tag_path_to_jar, java_options='-mx3000m')
            tagged = tagger.tag_sents(sents)
             # TODO: desdoblar el vector vimp ... así queda 'verb' = True, 'verb & singular'=True
        else:
            tagged = pos_tag(sents, lang='es')
        return tagged

class Vectorizer():
    def __init__(self, corpus_tok):
        self.corpus = corpus_tok
        self.words = None

    def __tag_combinations(self, rawtag):
        """
        returns cominatories of initial words of stanford-like tags
        """
        numb_regxp = re.compile(r'\d')
        letter_tag = re.sub(numb_regxp, '', rawtag)
        return [letter_tag[:i] for i in range(1, len(letter_tag)+1)]

    def get_vector_matrix(self, freq_floor=50, context_words=3):
        
        nlp = es_core_web_md.load()
        STOPWORDS = spacy.es.STOP_WORDS
        def _clean_sent(sent):
            clean_sent = []
            # remove stopwords
            for word in sent:
                word = word.lower()
                if not word in STOPWORDS:
                    if not word.isdigit():
                        clean_sent.append(word)
            return clean_sent

        def _update_feature(word, feature_name, features):
            " dirty update of features "
            counts = 1
            if word in vectors:
                if feature_name in vectors[word]:
                    counts = vectors[word][feature_name] + 1
            features[feature_name] = counts
            return features

        def _update_counts(feature_name, f_counts):
            counts = 1
            if feature_name in f_counts:
                counts = f_counts[feature_name] + 1
            f_counts[feature_name] = counts
            return f_counts

        sents = self.corpus.get_sents()
        stemmer = SpanishStemmer()
        
        # will use the words as keys and dict of features as values
        vectors = {}
        #freq_counts = {}
        for sent in sents:
            # TODO: PARALELLIZE!!
            #for doc in nlp.pipe(texts, batch_size=10000, n_threads=3):
            # take off stopwords && to get context_words!
            cleaned_sent = _clean_sent(sent)
            doc = nlp(' '.join(sent))
            for word_idx in range(len(doc)):
                # get the word and the pos tag
                spacy_word = doc[word_idx]
                word = spacy_word.text.lower()

                pos_tag = spacy_word.pos_

                if len(word) <= 2:
                    continue
                if word in STOPWORDS:
                    continue
                if word.isdigit():
                    continue

                # if not seen word
                if not word in vectors:
                    features = {}
                else:
                    features = vectors[word]

                # counts of frequency to normalze later
                #freq_counts = _update_counts(pos_tag, freq_counts)

                # context related (POS and words stemmed)
                features = _update_feature(word, pos_tag, features)
                if word_idx > 0:
                    prev_tag = doc[word_idx-1].pos_
                    feature_name = prev_tag + '_pos_prev'
                    features = _update_feature(word, feature_name, features)
                if word_idx < len(sent)-1:
                    post_tag = doc[word_idx+1].pos_
                    feature_name = post_tag + '_pos_post'
                    features = _update_feature(word, feature_name, features)


                # dependency features. the objective of the dep is stemmed!
                dep_type = spacy_word.dep_
                if dep_type!='ROOT':
                    dep_obj = stemmer.stem(spacy_word.head.text.lower())
                    feature_name = 'DEP:' + dep_type + '-' + dep_obj
                    features = _update_feature(word, feature_name, features)

                # get n words from context as features (stemmed...!)
                for i in range(context_words):
                    ctxt_word = (random.choice(cleaned_sent))
                    feature_word = stemmer.stem(ctxt_word)
                    feature_name = ctxt_word + '_ctxt_word'
                    features = _update_feature(word, feature_name, features)
                # agregar feature de synset (wordnet) :0
                features['word'] = word
                
                # frequency counting
                features = _update_feature(word, 'freq', features)
                

                vectors[word] = features

        # sacar palabras con < 'freq'
        words_to_pop=set()
        for word, f_dict in vectors.items():
            if f_dict['freq'] <= freq_floor:
                words_to_pop.add(word)
        for word in words_to_pop:
            vectors.pop(word)

        for word, f_dict in vectors.items():
            #print(word, f_dict)
            f_dict['freq'] = 0
            vectors[word] = f_dict # delete an irrelevant dimension!
        # normalizar los contextos de POS
        #for word, f_dict in vectors.items():
        #    f_dict[]

        # agregar palabra de contexto. .. LEMATIZADA !

        # NORMALIZAR TODOS LOS CONTEXTOS! -> diccionario de frequencias de ... TODOS los features que ocurrieron
        self.words = list(vectors.keys())          # thankfully in the same order as vectors.values

        vectorizer = DictVectorizer(dtype=numpy.int32)
        vec_matrix = vectorizer.fit_transform(list(vectors.values()))
        vectors_shape = vec_matrix.get_shape()
        print(vectors_shape)

        """
        freqs_vector = vectorizer.transform(freq_counts)

        vec_matrix = vstack([freqs_vector, vec_matrix])
        print(s.get_shape)
        print(s)
        print(vectorizer.inverse_transform(s))
        """

        # normalization
        vec_matrix = normalize(vec_matrix, copy=False)

        ####### reduccion de dim no sup
        # reducir dimensionalidad con variance treshold
        #selector = VarianceThreshold(threshold = 0.0)
        #vec_matrix = selector.fit_transform(vec_matrix)

        # SVD (PCA)
        Trunc_svd = TruncatedSVD(n_components=1500)
        vec_matrix = Trunc_svd.fit_transform(vec_matrix)

        # reducir dimensionalidad con percentile de varianza
        #selected = SelectPercentile(chi2, percentile = 10)
        #word_vecs_new=selected.fit_transform(new_word_vecs,target_vec)


        print(vectorizer.inverse_transform(vec_matrix)) # -> to see features!

        return self.words, vec_matrix
from nltk.tag import StanfordPOSTagger
from nltk import pos_tag

from sklearn.feature_extraction import DictVectorizer
import numpy

import re

class POS_Tagger():
    def __init__(self, tagger='nltk'):
        self.tagger = tagger

        if self.tagger == 'stanford':
            self.__tag_path_to_model = "extra/spanish-distsim.tagger"
            self.__tag_path_to_jar = "extra/stanford-postagger-3.8.0.jar"

    def get_tagged_sents(self, sents):
        if self.tagger == 'stanford':
            tagger = StanfordPOSTagger(self.__tag_path_to_model, self.__tag_path_to_jar)
            tagged = tagger.tag_sents(sents)
             # TODO: desdoblar el vector vimp ... así queda 'verb' = True, 'verb & singular'=True
        else:
            tagged = pos_tag(sents, lang='es')
        return tagged

class Vectorizer():
    def __init__(self, corpus, vec_type='sintactic', pos_tagger='nltk'):
        self.corpus = corpus
        self.vec_type = vec_type
        self.pos_tagger = POS_Tagger(pos_tagger)
        self.words = None

    def __tag_combinations(self, rawtag):
        """
        returns cominatories of initial words of stanford-like tags
        """
        numb_regxp = re.compile(r'\d')
        letter_tag = re.sub(numb_regxp, '', rawtag)
        return [letter_tag[:i] for i in range(1, len(letter_tag)+1)]

    def get_vector_matrix(self):
        def _update_pos_t_feature():
            f_val = 1
            if word in vectors:
                if feature_name in vectors[word]:
                    f_val = vectors[word][feature_name] + 1 # if seen before!
            features[feature_name] = f_val

        sents = self.corpus.get_sents()
        tagged_sents = self.pos_tagger.get_tagged_sents(sents)
        
        # will use the words as keys and dict of features as values
        vectors = {}
        for sent in tagged_sents:
            for word_idx in range(len(sent)):
                features = {}
                word = sent[word_idx][0]
                pos_tag = sent[word_idx][1]

                # POS-related features
                features[pos_tag] = 1
                for sub_tag in self.__tag_combinations(pos_tag):
                    features[sub_tag] = 1
                if word_idx > 0:
                    prev_tag = sent[word_idx-1][1]
                    feature_name = prev_tag + '_prev'
                    _update_pos_t_feature()
                if word_idx < len(sent)-1:
                    post_tag = sent[word_idx+1][1]
                    feature_name = post_tag + '_post'
                    _update_pos_t_feature()

                # frequency counting
                if (word in vectors):
                    counts = vectors[word]['freq'] + 1
                else:
                    counts = 1
                features['freq'] = counts

                vectors[word] = features

        self.words = list(vectors.keys())          # thankfully in the same order as vectors.values

        vectorizer = DictVectorizer(dtype=numpy.int32)
        vec_matrix = vectorizer.fit_transform(list(vectors.values()))
        #print (vectorizer.inverse_transform(vec_matrix)[:10]) # -> to see features!
        return self.words, vec_matrix
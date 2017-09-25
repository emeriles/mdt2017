from nltk.tag import StanfordPOSTagger
from nltk import pos_tag
from nltk.corpus import stopwords

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
    def __init__(self, corpus, vec_type='morphosyntactic', pos_tagger='nltk'):
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

    def get_vector_matrix(self, freq_floor=10):
        def _update_pos_t_feature():
            f_val = 1
            if word in vectors:
                if feature_name in vectors[word]:
                    f_val = vectors[word][feature_name] + 1 # if seen before!
            features[feature_name] = f_val

        STOPWORDS = stopwords.words('spanish')
        def _clean_sent(sent):
            clean_sent = []
            # remove stopwords
            for word, tag in sent:
                word = word.lower()
                if not word in STOPWORDS:
                    if not word.isdigit():
                        clean_sent.append((word, tag))
            return clean_sent

        sents = self.corpus.get_sents()
        tagged_sents = self.pos_tagger.get_tagged_sents(sents)
        
        # will use the words as keys and dict of features as values
        vectors = {}
        for sent in tagged_sents:
            # take off stopwords
            cleaned_sent = _clean_sent(sent)
            for word_idx in range(len(cleaned_sent)):
                features = {}
                word = cleaned_sent[word_idx][0]
                pos_tag = cleaned_sent[word_idx][1]

                # dirty noise catcher
                if len(word) <= 2:
                    continue

                # POS-related features
                features[pos_tag] = 1
                for sub_tag in self.__tag_combinations(pos_tag):
                    features[sub_tag] = 1
                if word_idx > 0:
                    prev_tag = cleaned_sent[word_idx-1][1][0]
                    feature_name = prev_tag + '_prev'       # USAR PREFIJO DE PREV_TAG !!?
                    _update_pos_t_feature()     # TODO: tf-iwf !!!!!!!     D:>
                if word_idx < len(cleaned_sent)-1:
                    post_tag = cleaned_sent[word_idx+1][1][0]
                    feature_name = post_tag + '_post'
                    _update_pos_t_feature()
                #agregar feature de synset (wordnet) :0

                # frequency counting
                if (word in vectors):
                    counts = vectors[word]['freq'] + 1
                else:
                    counts = 1
                features['freq'] = counts

                vectors[word] = features

        # sacar palabras con < 'freq'
        words_to_pop=set()
        for word, f_dict in vectors.items():
            if f_dict['freq'] <= freq_floor:
                words_to_pop.add(word)
        for word in words_to_pop:
            vectors.pop(word)

        # agregar palabra

        # NORMALIZAR TODOS LOS CONTEXTOS! -> diccionario de frequencias de ... TODOS los features que ocurrieron
        self.words = list(vectors.keys())          # thankfully in the same order as vectors.values

        vectorizer = DictVectorizer(dtype=numpy.int32)
        vec_matrix = vectorizer.fit_transform(list(vectors.values()))
        print(vec_matrix.get_shape())
        #print (vectorizer.inverse_transform(vec_matrix)[:10]) # -> to see features!
        return self.words, vec_matrix
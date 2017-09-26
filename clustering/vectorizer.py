from nltk.tag import StanfordPOSTagger
from nltk import pos_tag
from nltk.corpus import stopwords

from sklearn.feature_extraction import DictVectorizer
import random
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
            # java_options -mx3000m sets memory use in 3GB
            tagger = StanfordPOSTagger(self.__tag_path_to_model, 
                        self.__tag_path_to_jar, java_options='-mx3000m', verbose=True)
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

    def get_vector_matrix(self, freq_floor=50, context_words=3):
        
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

        def _update_feature(word, feature_name, features):
            " dirty update of features "
            counts = 1
            if word in vectors:
                if feature_name in vectors[word]:
                    counts = vectors[word][feature_name] + 1
            features[feature_name] = counts
            return features

        sents = self.corpus.get_sents()
        tagged_sents = self.pos_tagger.get_tagged_sents(sents)
        
        # will use the words as keys and dict of features as values
        vectors = {}
        freq_counts = {}
        for sent in tagged_sents:
            # take off stopwords && to get context_words!
            cleaned_sent = _clean_sent(sent)
            for word_idx in range(len(sent)):
                # get the word and the pos tag
                word = sent[word_idx][0]
                word = word.lower()
                pos_tag = sent[word_idx][1]

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
                #if pos_tag[0] in freq_counts:
                #    freq_counts[pos_tag[0]] += 1
                #else:
                #    freq_counts[pos_tag[0]] = 1

                # context related (POS and words stemmed)
                for sub_tag in self.__tag_combinations(pos_tag):
                    features = _update_feature(word, sub_tag, features)
                if word_idx > 0:
                    prev_tag = sent[word_idx-1][1][0]
                    feature_name = prev_tag + '_pos_prev'       # USAR PREFIJO DE PREV_TAG !!?
                    features = _update_feature(word, feature_name, features)     # TODO: tf-iwf !!!!!!!     D:>
                if word_idx < len(sent)-1:
                    post_tag = sent[word_idx+1][1][0]
                    feature_name = post_tag + '_pos_post'
                    features = _update_feature(word, feature_name, features)

                # get n words from context as features (stemmed...)
                for i in range(context_words):
                    ctxt_word = (random.choice(cleaned_sent))[0]
                    feature_name = ctxt_word + '_ctxt_word'
                    features = _update_feature(word, feature_name, features)
                #agregar feature de synset (wordnet) :0

                
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

        #print (vectorizer.inverse_transform(vec_matrix)[:10]) # -> to see features!
        return self.words, vec_matrix
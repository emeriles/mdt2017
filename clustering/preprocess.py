from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import RegexpTokenizer
from nltk.data import LazyLoader
import codecs
import re
from os.path import basename, dirname


pattern = r'''(?ix)      # set flag to allow verbose regexps
     (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
   | \w+(?:-\w+)*        # words with optional internal hyphens
'''
regex = re.compile(pattern)		# lavoz tokenizer
tokenizer = RegexpTokenizer(pattern)	# other tokenizer

class Corpus_Tokenizer():
    #re.sub(r"&#\d*", "", strr)

    def __init__(self, filepath):
        self.filepath = filepath
        self.__is_la_voz = ('lavoz' in basename(filepath))
        if True:
            with codecs.open(self.filepath, 'r', 'utf-8') as f:
                corpus = f.read()
            corpus_sent = LazyLoader('tokenizers/punkt/spanish.pickle').tokenize(corpus)
            self.corpus_clean = [re.sub('&#\d+', '', sents) for sents in corpus_sent]
        else:
            root = dirname(filepath)
            basen = basename(filepath)
            self.corpus_clean = PlaintextCorpusReader('.' if root=='' else root, basen)

    def get_sents(self):
        if True:
            sents = [regex.findall(sents) for sents in self.corpus_clean]
        else:
            sents = self.corpus_clean.sents()
        return sents
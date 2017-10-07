from nltk.data import LazyLoader
import codecs
import re
from os.path import basename, dirname


pattern = r'''(?ix)      # set flag to allow verbose regexps
     (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
   | \w+(?:-\w+)*        # words with optional internal hyphens
'''
regex = re.compile(pattern)		# lavoz tokenizer

class Corpus_Tokenizer():
    #re.sub(r"&#\d*", "", strr)

    def __init__(self, filepath):
        self.filepath = filepath
        with codecs.open(self.filepath, 'r', 'utf-8') as f:
            corpus = f.read()
        corpus = re.sub('\\n', '. ', corpus)
        corpus_sent = LazyLoader('tokenizers/punkt/spanish.pickle').tokenize(corpus)
        self.corpus_clean = [re.sub('&#\d+', '', sents) for sents in corpus_sent]

    def get_sents(self):
        sents = [regex.findall(sents) for sents in self.corpus_clean]
        return sents
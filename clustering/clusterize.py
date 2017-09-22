import preprocess


from vectorizer import Vectorizer

from sklearn.cluster import KMeans
#import docopt

if __name__ == '__main__':
    corpus = preprocess.Corpus_Tokenizer('../../../lavoz_head.txt')
    vectorizer = Vectorizer(corpus, pos_tagger='stanford')
    words, matrix = vectorizer.get_vector_matrix()

    #print([x for x in zip(words, matrix)][:10])
    
    




"""
kmeans = KMeans(n_clusters=20, init='k-means++',
                n_init=10, max_iter=300,
                tol=0.0001, precompute_distances='auto',
                verbose=0, random_state=None, copy_x=True,
                n_jobs=1, algorithm='auto').fit(dataset_matrix)
kmeans.labels_ # que hace?
#kmeans.predict([[0, 0], [4, 4]])  ... predice a que cluster
len(kmeans.cluster_centers_[0]) # 322 ... por ahora 

counts = 0
for dic,c_lab in zip(vectorizer.inverse_transform(dataset_matrix), kmeans.labels_):
    if c_lab == 15:
        counts += 1
        print (dic, " ", c_lab)
print(counts)
"""
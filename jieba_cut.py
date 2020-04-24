import os
from os import listdir
import jieba
import sys
import numpy as np
import re
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from ckiptagger import WS

def read_file(file):
    with open(file, 'r') as f:
        return f.read().replace('\n', '')

def classification(number):
    type_list = []
    for i in range(1, number+1):
        files = listdir(f'/home/aa0918358122/git/crawler/articles_{sys.argv[2]}/{i}/')
        type_list.append(files)
    return type_list

if __name__ == "__main__":

    corpus = []
    x = [] # tfidf for each article
    y = [] # this article comes from which newspaper

    # jieba
    jieba.load_userdict('./jieba_dict/dict.txt.big')
    stopword_set = set()
    with open('./jieba_dict/stopwords.txt', 'r', encoding = 'utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))
        stopword_set.add('\n')
    # ckiptagger
    ws = WS('./model')

    for i, file in enumerate(classification(7)):
        # print(len(file))
        for j in range(len(file)):
            # print(file[j])
            article = read_file(f'/home/aa0918358122/git/crawler/articles_{sys.argv[2]}/{i+1}/{file[j]}')
            y.append(int(file[j][0]))

            # use jieba or ckiptagger
            if sys.argv[1] == 'jieba':
                seg_list = jieba.cut(article, cut_all = False)
            elif sys.argv[1] == 'ckiptagger':
                seg_list = ws([article])[0]

            res = []
            for word in seg_list:
                if word not in stopword_set:
                    res.append(word)
            text = ' '.join(res)
            corpus.append(text)
    # print(corpus)

    # find tfidf by sklearn
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    word = vectorizer.get_feature_names()
    weight = tfidf.toarray()

    feature_array = np.array(word)
    tfidf_sorting = np.argsort(weight).flatten()[::-1]
    n = 90000
    least_n = feature_array[tfidf_sorting][n:]
    print(least_n(edgeitems=5000))

#    for i in range(len(weight)):
#        dimension = [] # input demention of tfidf for each article
#        for j in range(len(word)):
#            dimension.append(word[j])
#        x.append(weight[i])
    # print(len(dimension))

    #np.save(f'{sys.argv[1]}_{sys.argv[2]}_x.npy', np.asarray(x))
    #np.save(f'{sys.argv[1]}_{sys.argv[2]}_y.npy', np.asarray(y))


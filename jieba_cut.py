import os
from os import listdir
import jieba
import sys
import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from ckiptagger import WS

def read_file(file):
    with open(file, 'r') as f:
        # f.encode('utf-8')
        return f.read().replace('\n', '')

if __name__ == "__main__":

    files = listdir('./data/')
    corpus = []
    y = []

    ws = WS('./model')
    jieba.load_userdict('./jieba_dict/dict.txt.big')
    stopword_set = set()
    with open('./jieba_dict/stopwords.txt', 'r', encoding = 'utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))
        stopword_set.add('\n')

    for file in files:
        y.append(int(file[0]))
        article = read_file('./data/' + file)

        seg_list = jieba.cut(article, cut_all = False)
        # seg_list = ws([article])[0]

        res = []
        for word in seg_list:
            if word not in stopword_set:
                res.append(word)

        text = ' '.join(res)
        corpus.append(text)

    print(corpus)

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    word = vectorizer.get_feature_names()
    weight = tfidf.toarray()

    x = []
    for i in range(len(weight)):
        e = []
        for j in range(len(word)):
            e.append(word[j])
        x.append(weight[i])

    np.save('x.npy', np.asarray(x))
    np.save('y.npy', np.asarray(y))


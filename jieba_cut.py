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
from datetime import datetime

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
    a = []

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
        count = []
        for j in range(len(file)):
            # print(file[j])

            # choose specific date before or after some big event
            date = re.findall(r'-.*-', file[j])
            date = datetime.strptime(f'{date[0][1:9]}', '%Y%m%d')
            start_date = datetime.strptime('20200215', '%Y%m%d')
            end_date = datetime.strptime('20200625', '%Y%m%d')
            duration1 = date - start_date
            duration2 = date - end_date
            if duration1.total_seconds() >= 0 and duration2.total_seconds() <= 0:

                # choose all date
                article = read_file(f'/home/aa0918358122/git/crawler/articles_{sys.argv[2]}/{i+1}/{file[j]}')
                count.append(int(file[j][0]))
                y.append(int(file[j][0]))
                # use jieba
                if sys.argv[1] == 'jieba':
                    seg_list = jieba.cut(article, cut_all = False)
                # use ckiptagger
                elif sys.argv[1] == 'ckiptagger':
                    seg_list = ws([article])[0]
                res = []
                for word in seg_list:
                    if word not in stopword_set:
                        res.append(word)
                text = ' '.join(res)
                corpus.append(text)
                # choose same numbers of articles from each newspaper office
                if len(count) >= 100:
                    break
        # print(len(count))
    # print(corpus)


    # find tfidf by sklearn
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    word = vectorizer.get_feature_names() #type(word) = list
    weight = tfidf.toarray() # type(weight) = nparray


    # choose top n tfidf
    total_weight = np.sum(weight, axis=0) # sum tfidf of each article by column
    feature_array = np.array(word)
    tfidf_sorting_through_index = np.argsort(-total_weight)
    tfidf_sorting_through_value = total_weight[np.argsort(-total_weight)]
    n = 200
    top_n = feature_array[tfidf_sorting_through_index][:n] #type(top_n) = nparray
    # least_n = feature_array[tfidf_sorting_through_index][n:]
    # print(top_n)
    for i in range(len(weight)):
        a.append(weight[i][tfidf_sorting_through_index][:n])


    # use all tfidf
    for i in range(len(weight)):
        dimension = [] # input demention of tfidf for each article
        for j in range(len(word)):
            dimension.append(word[j])
        x.append(weight[i])
    # print(len(dimension))


    np.save(f'{sys.argv[1]}_{sys.argv[2]}_x.npy', np.asarray(x))
    np.save(f'{sys.argv[1]}_{sys.argv[2]}_y.npy', np.asarray(y))

    np.save(f'{sys.argv[1]}_{sys.argv[2]}_a.npy', np.asarray(a))
    np.save(f'{sys.argv[1]}_{sys.argv[2]}_topn.npy', top_n)


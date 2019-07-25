from os import listdir
import copy
import math

#read file
def read_file(filename):
    with open(filename, 'r') as article:        #'r' for read , 'w' for write
        return article.read().rstrip('\n')      #file.read()   #str.rstrip()

#delete the redundant words in articles

if "__main__" == __name__:
    all_article = listdir('./data/')
    articles = []
    set_article = set()

    for article in all_article:
        path = './data/' + article
        delete_article = read_file(path)
        deletion = ['，', '。', '、', '「', '」', ' ','；', '（', '）']
        for char in deletion:
            delete_article = delete_article.replace(char, '')        #str.replace()
        articles.append(delete_article)          #list.append()
        set_article.update(delete_article)       #set.update()

    dictionary = dict()                          #dictionary()
    for i, article in enumerate(articles):       #enumerate()
        dictionary[i] = dict()
        for char in set_article:
            dictionary[i][char] = 0

#count word frequency

        for char in article:
            dictionary[i][char] += 1

#count total word

        dictionary[i]['total'] = len(article)

        #count = 0
        #for key in dictionary[i]:
            #count += dictionary[i][key]

#count tf

    tf_dict = copy.deepcopy(dictionary)
    for i in range(len(articles)):       #dictionary deep copy
        for key in tf_dict[i]:
            tf_dict[i][key] = tf_dict[i][key]/tf_dict[i]['total']

#count idf

    idf_dict = dict()
    for char in set_article:
        count = 0
        for i in range(len(articles)):
            if dictionary[i][char] != 0:
                count += 1
        idf = math.log((10/count), 10)
        idf_dict[char] = idf
        idf_dict['total'] = 0

#calculate tf-idf

    tfidf_dict = copy.deepcopy(dictionary)
    for i in range(len(articles)):
        for key in tf_dict[i]:
            tfidf_dict[i][key] = tf_dict[i][key]*idf_dict[key]

    # a = [[v[1],v[0]] for v in tfidf_dict[0].items()]
    # a.sort(reverse=True)
    # print([ v for v in sorted(tfidf_dict[0].values())] )

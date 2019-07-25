from os import listdir
import copy
import math

# read file
def read_file(file):
    with open(file, 'r') as f:        # 'r' for read , 'w' for write
        return f.read().rstrip('\n')  # file.read()   #str.rstrip()

if "__main__" == __name__:
    files = listdir('./data/')
    articles = []
    char_set = set()

    for file in files:
        article = read_file('./data/' + file)

        # delete the redundant words in articles
        deletion = ['，', '。', '、', '「', '」', ' ','；', '（', '）']
        for char in deletion:
            article = article.replace(char, '') # str.replace()

        articles.append(article)    # list.append()
        char_set.update(article)    # set.update()

    count_dict = dict()                          # count_dict()
    for i, article in enumerate(articles):       # enumerate()
        count_dict[i] = dict()
        # initialize dict value
        for char in char_set:
            count_dict[i][char] = 0

        # count word frequency
        for char in article:
            count_dict[i][char] += 1

    idf_dict = dict()
    tfidf_dict = copy.deepcopy(count_dict)

    #count idf
    for char in char_set:
        count = 0
        for i in range(len(articles)):
            if count_dict[i][char]:
                count += 1
        idf_dict[char] = math.log((10/count), 10)

    #count tf, tf-idf
    for i in range(len(articles)):
        for char in char_set:
            tfidf_dict[i][char] = count_dict[i][char] / len(articles[i]) * idf_dict[char]

    print(tfidf_dict)

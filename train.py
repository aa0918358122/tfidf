import numpy as np
import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score

if sys.argv[1] == 'jieba':
    x = np.load('jieba_politics_x.npy')
    y = np.load('jieba_politics_y.npy')
elif sys.argv[1] == 'ckiptagger':
    x = np.load('ckiptagger_politics_x.npy')
    y = np.load('ckiptagger_politics_y.npy')

# x = np.load(f'{sys.argv[1]}_x.npy')
# y = np.load(f'{sys.argv[1]}_y.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0, shuffle = True, stratify = y)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# GridSearch
def grid_search():
    grid_1 = {'n_estimators': [100*n for n in range(1, 10)],
              'max_depth': [m for m in range(1, 20)]}
    score = []
    clf = RandomForestClassifier()
    grid_search = GridSearchCV(clf, grid_1, cv = 3)
    grid_search.fit(x_train, y_train)
    score.append(grid_search.best_score_)
    score.append(grid_search.best_params_)
    print(score)


# hyperopt
def hyperopt():
    global best
    best = 0
    def hyperopt_model(params):
        clf = RandomForestClassifier(**params)
        return cross_val_score(clf, x_train, y_train, cv = 3).mean()

    def fn(params):
        global best
        acc = hyperopt_model(params)
        if acc > best:
            best = acc
        return{'loss' : -acc, 'status' : STATUS_OK}

    param_space = {'n_estimators' : hp.choice('n_estimators', range(100, 1000)),
                   'max_depth' : hp.choice('max_depth', range(1, 20))}
    trials = Trials()
    best = fmin(fn, param_space, algo = tpe.suggest, max_evals = 100, trials = trials)
    print(best)


def test():
    for i in range(10):
        clf = RandomForestClassifier(n_estimators = 765, random_state = i, max_depth = 37)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        print(confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4, 5, 6, 7]))
        print(accuracy_score(y_test, y_pred))


if '__main__' == __name__:
    # grid_search()
    # hyperopt()
    test()


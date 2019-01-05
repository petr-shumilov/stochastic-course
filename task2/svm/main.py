import numpy as np
import matplotlib.pyplot as plot

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import os



def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    return np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def draw_models(data, models, titles, files_url):
    
    objects = data[:, :2]
    X0, X1 = objects[:, 0], objects[:, 1]
    xx, yy = make_meshgrid(X0, X1)
        
    for clf, title, file_url in zip(models, titles, files_url):
        _, sub = plot.subplots(1, 1)
        plot.subplots_adjust(wspace=0.4, hspace=0.4)

        plot_contours(sub, clf, xx, yy, cmap=plot.cm.coolwarm, alpha=0.8)
        sub.scatter(X0, X1, c=data[:, 2], cmap=plot.cm.coolwarm, s=20, edgecolors='k')
        sub.set_xlim(xx.min(), xx.max())
        sub.set_ylim(yy.min(), yy.max())
        sub.set_xticks(())
        sub.set_yticks(())
        sub.set_title(title)
        plot.savefig(file_url)
        


def get_best_model(config, data):

    objects, classes = data[:, :2], data[:, 2]
    models = []
    titles = []
    filenames = []

    for _config in config:
        if _config['type'] == 'svc':
            model = GridSearchCV(SVC(), _config['params'], cv=5, scoring=_config['metric']).fit(objects, classes)
        elif _config['type'] == 'knn':   
            model = GridSearchCV(KNeighborsClassifier(), _config['params'], cv=5, scoring=_config['metric']).fit(objects, classes)
        
        filenames.append('%s_algorithm_with_the_%s_metric' % (_config['type'].upper(), _config['metric']))    
        titles.append('%s algorithm with the %s metric = %f' % (_config['type'].upper(), _config['metric'], model.best_score_))    
        models.append(model.best_estimator_)
        
    return models, titles, filenames
    


def main():

    # path to file 
    INPUT_FILE_PATH = '/homework/stochastic-course/task2/input.txt'
    # INPUT_FILE_PATH = 'input.txt'
    
    RESULT_DIR = '/homework/stochastic-course/task2/results'
    # RESULT_DIR = 'results'


    # config 
    config = [{
        'type': 'svc',
        'params': {
            'kernel': ('linear', 'poly', 'rbf'),
            'C': np.arange(1., 10., 0.1),
            'gamma': ['auto', 'scale']
        },
        'metric': 'balanced_accuracy'
    }, {
        'type': 'knn',
        'params': {
            'n_neighbors': range(1, 10),
            'algorithm': ['auto', 'brute', 'kd_tree', 'ball_tree']
        },
        'metric': 'balanced_accuracy'
    }, {
        'type': 'svc',
        'params': {
            'kernel': ('linear', 'poly', 'rbf'),
            'C': np.arange(1., 10., 0.1),
            'gamma': ['auto', 'scale']
        },
        'metric': 'average_precision'
    }, {
        'type': 'knn',
        'params': {
            'n_neighbors': range(1, 10),
            'algorithm': ['auto', 'brute', 'kd_tree', 'ball_tree']
        },
        'metric': 'average_precision'
    }]

    data = np.loadtxt(INPUT_FILE_PATH, delimiter=',')

    models, titles, filenames = get_best_model(config, data)

    os.makedirs(RESULT_DIR, exist_ok=True)
    files_url = [os.path.join(RESULT_DIR, '{}.png'.format(filename)) for filename in filenames]
    
    draw_models(data, models, titles, files_url)

    print('DONE!')

if __name__ == '__main__':
    main()


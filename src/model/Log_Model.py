# -*- coding: utf-8 -*-
"""
This code creates regression models to predict the scores of the games
"""

import numpy as np
import pandas as pd
from sklearn import linear_model, grid_search, ensemble, pipeline, svm
from sklearn.cross_validation import KFold


class Log_Model(object):
    """
    Produces a regression model object
    """

    def __init__(self):
        self.best_model = None
        self.tx = None
        self.ty = None
        self.steps = None
        self.params = None
        self.model_type = None
        self.acc = None
        self.input_cols = None

#Sets training dataset for the model to be built on
    def set_training(self, x, y):
        self.tx = x
        self.ty = y
        self.input_cols = set(self.tx.columns.values)
        #self.cap_and_floor()

#Sets pipeline
    def set_pipeline(self, steps, params):
        self.steps = steps
        self.params = params

#Creates the model
    def calc_model(self):
        lr = self.logistic_reg(self.tx, self.ty)
        svc_linear = self.SVC_linear(self.tx, self.ty)
        svc_other = self.SVC_other(self.tx, self.ty)

        if (lr.best_score_>svc_linear.best_score_) and (lr.best_score_>svc_other.best_score_):
            self.best_model = lr
            self.model_type = 'lr'
        elif svc_linear.best_score_ > svc_other.best_score_:
            self.best_model = svc_linear
            self.model_type = 'svc_linear'
        else:
            self.best_model = svc_other
            self.model_type = 'svc_other'

        self.acc = self.best_model.best_score_

    #Returns predictions for given x-values
    def get_pred(self, test_x):
        if self.best_model == None:
            return None
        else:
            return self.best_model.predict(test_x)

    #Returns probabiltiies of win for given x-values
    def get_prob(self, test_x):
        if self.best_model == None:
            return None
        else:
            probs = self.best_model.predict_proba(test_x)
            return [a[0] for a in probs]

    #Returns MSE of best model
    def get_mse(self):
        return self.mse

    #Returns model type with best MSE
    def get_model_type(self):
        return self.model_type

    #Caps and floors x and y to +/- 2 sd from mean
    def cap_and_floor(self):
        for col in self.tx.columns:
            avg = np.mean(self.tx[col])
            std_dev = np.std(self.tx[col])
            floor = avg - 2*std_dev
            cap = avg + 2*std_dev
            #self.tx[col] = [cap if x>cap else x for x in self.tx[col]]
            #self.tx[col] = [floor if x<floor else x for x in self.tx[col]]
            self.tx.loc[:,col] = [cap if x>cap else x for x in self.tx[col]]
            self.tx.loc[:,col] = [floor if x<floor else x for x in self.tx[col]]
        avg = np.mean(self.ty)
        std_dev = np.std(self.ty)
        floor = avg - 2 * std_dev
        cap = avg + 2 * std_dev
        self.ty = [cap if y>cap else y for y in self.ty]
        self.ty = [floor if y<floor else y for y in self.ty]

    def get_x(self):
        return self.tx

    def get_y(self):
        return self.ty

    def get_acc(self):
        return self.acc

    def get_model_type(self):
        return self.model_type

    def logistic_reg(self, x, y):
        lr = linear_model.LogisticRegression()
        parameters = {'penalty':['l1'], 'C':[.01, .1, 1, 100]}

        if len(y) < 5:
            kf = None
        else:
            kf = KFold(n=len(y), n_folds=5, shuffle=True)

        if self.params is None:
            clf = grid_search.GridSearchCV(lr, parameters, cv=kf)
        else:
            steps = self.steps + [('lr', lr)]
            p_line = pipeline.Pipeline(steps)
            parameters = self.params.copy()
            parameters['lr__C'] = [.1, 1, 10, 100]
            parameters['lr__penalty'] = ['l1']
            clf = grid_search.GridSearchCV(p_line, param_grid=parameters, cv=kf)
        clf.fit(x, y)
        return clf


    def SVC_linear(self, x, y):
        svc = svm.LinearSVC()
        parameters = {'C':[.01, .1, 1, 20, 100]}

        if len(y) < 5:
            kf = None
        else:
            kf = KFold(n=len(y), n_folds=5, shuffle=True)

        if self.params is None:
            clf = grid_search.GridSearchCV(svc, parameters, cv=kf)
        else:
            steps = self.steps + [('svc', svc)]
            p_line = pipeline.Pipeline(steps)
            parameters = self.params.copy()
            parameters['svc__C'] = [.1, 1, 10, 100]
            clf = grid_search.GridSearchCV(p_line, param_grid=parameters, cv=kf)
        clf.fit(x, y)
        return clf


    def SVC_other(self, x, y):
        svc = svm.SVC()
        parameters = {'C':[.01, .1, 1, 100], 'kernel':['rbf', 'sigmoid']}

        if len(y) < 5:
            kf = None
        else:
            kf = KFold(n=len(y), n_folds=5, shuffle=True)

        if self.params is None:
            clf = grid_search.GridSearchCV(svc, parameters, cv=kf)
        else:
            steps = self.steps + [('svc', svc)]
            p_line = pipeline.Pipeline(steps)
            parameters = self.params.copy()
            parameters['svc__C'] = [.1, 1, 10, 100]
            parameters['svc__kernel'] = ['rbf', 'sigmoid']
            clf = grid_search.GridSearchCV(p_line, param_grid=parameters, cv=kf)
        clf.fit(x, y)
        return clf

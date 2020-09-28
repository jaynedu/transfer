# -*- coding: utf-8 -*-
# @Date    : 2020/9/20 16:40
# @Author  : Du Jing
# @FileName: experiments
# ---- Description ----

import os
import sys
import numpy as np
import pandas as pd
from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import utils
from data_loader import DataLoader
from mmd import mmd


class Run:
    source_dir = r'E:\z\Database-CSV\DAIC\wavfeaturestest'
    target_dir = r'E:\z\Database-CSV\LZ\lanzhoutest'
    loader = DataLoader(source_dir, target_dir)
    Xs, Ys, Xt, Yt = loader.load_raw_data()
    # Xs, Ys, Xt, Yt = Xs[:500], Ys[:500], Xt[:100], Yt[:100]
    raw_config = utils.ConfigDict(
        kernel='sigmoid',
        gamma=68.34902294058122,
        coef0=6.1456262680953895,
        C=30.60194111421361
    )
    tca_config = utils.ConfigDict(
        kernel='primal',
        lamb=0.16341741375495933,
        dim=2,
    )
    jda_config = utils.ConfigDict(
        kernel='linear',
        gamma=1,
        lamb=0.45,
        dim=2,
        T=9
    )
    marker = ['o', 's', '^', '*']
    color = ['b', 'b', 'r', 'r']
    desc = [
        'source health control',
        'source depression',
        'target health control',
        'target depression'
    ]

    @classmethod
    def svm_using_raw_feature(cls):
        params = utils.ConfigDict(
            C=hp.uniform('C', 0.001, 100),
            svm_kernel=hp.choice('svm_kernel', [
                {
                    'name': 'rbf',
                    'gamma': hp.uniform('rbf_gamma_uniform', 0.001, 10),
                },
                {
                    'name': 'linear',
                },
                {
                    'name': 'sigmoid',
                    'gamma': hp.uniform('sigmoid_gamma_uniform', 0.001, 100),
                    'coef0': hp.uniform('sigmoid_coef0', 0, 10)
                },
                # {
                #     'name': 'poly',
                #     'gamma': hp.uniform('poly_gamma_uniform', 0.001, 100),
                #     'coef0': hp.uniform('poly_coef0', 0, 10),
                #     'degree': hp.uniformint('poly_degree', 2, 3),
                # }
            ]),
        )

        def choose_best_params(config):
            print('\n---------------------------------------------------------------------------------------')
            print('Params: ')
            print(config)
            svm_kernel_config = config['svm_kernel']
            svm_kernel = svm_kernel_config.pop('name')
            C = config['C']

            clf = make_pipeline(StandardScaler(),
                                SVC(kernel=svm_kernel, tol=0.001, random_state=666, shrinking=True, C=C,
                                    **svm_kernel_config))
            clf.fit(cls.Xs, cls.Ys)
            Yt_pred = clf.predict(cls.Xt)
            distance = mmd(cls.Xs, cls.Xt)
            matrix = confusion_matrix(cls.Yt, Yt_pred)
            report = classification_report(cls.Yt, Yt_pred)
            print('Result: ')
            print(matrix)
            print(report)
            print('---------------------------------------------------------------------------------------\n')
            report_print = classification_report(cls.Yt, Yt_pred, output_dict=True)
            uar = report_print['macro avg']['recall']
            return {
                'loss': distance - uar,
                'status': STATUS_OK
            }

        trials = Trials()
        best = fmin(fn=choose_best_params, space=params, algo=tpe.suggest, max_evals=100, trials=trials)

        print(best)
        print('best: ', trials.best_trial)

    @classmethod
    def svm_using_tca_feature(cls):
        params = utils.ConfigDict(
            C=hp.uniform('C', 0.001, 100),
            svm_kernel=hp.choice('svm_kernel', [
                {
                    'name': 'rbf',
                    'gamma': hp.uniform('rbf_gamma_uniform', 0.001, 10)
                },
                {
                    'name': 'linear',
                },
                {
                    'name': 'sigmoid',
                    'gamma': hp.uniform('sigmoid_gamma_uniform', 0.001, 100),
                    'coef0': hp.uniform('sigmoid_coef0', 0, 10)
                },
                # {
                #     'name': 'poly',
                #     'gamma': hp.uniform('poly_gamma_uniform', 0.001, 100),
                #     'coef0': hp.uniform('poly_coef0', 0, 10),
                #     'degree': hp.uniformint('poly_degree', 2, 3),
                # }
            ]),
            tca_kernel=hp.choice('tca_kernel', [
                {
                    'name': 'primal',
                    'gamma': 1
                },
                {
                    'name': 'linear',
                    'gamma': 1
                },
                {
                    'name': 'rbf',
                    'gamma': hp.uniform('gamma', 0.001, 10)
                }
            ]),
            lamb=hp.uniform('lamb', 0.001, 1)
        )

        def choose_best_params(config):
            print('\n---------------------------------------------------------------------------------------')
            print('Params: ')
            print(config)
            tca_kernel_config = config['tca_kernel']
            tca_kernel = tca_kernel_config['name']
            tca_gamma = tca_kernel_config['gamma']
            lamb = config['lamb']
            Xs_new, Ys_new, Xt_new, Yt_new = cls.loader.load_tca_data(cls.Xs, cls.Ys, cls.Xt, cls.Yt, tca_kernel, dim=2, gamma=tca_gamma, lamb=lamb)

            svm_kernel_config = config['svm_kernel']
            svm_kernel = svm_kernel_config.pop('name')
            C = config['C']

            # uar
            clf = make_pipeline(StandardScaler(),
                                SVC(kernel=svm_kernel, tol=0.001, random_state=666, shrinking=True, C=C, **svm_kernel_config))
            clf.fit(Xs_new, Ys_new)
            Yt_pred = clf.predict(Xt_new)
            distance = mmd(Xs_new, Xt_new)
            matrix = confusion_matrix(Yt_new, Yt_pred)
            report = classification_report(Yt_new, Yt_pred)
            print('Result: ')
            print(matrix)
            print(report)
            print('---------------------------------------------------------------------------------------\n')
            report_print = classification_report(Yt_new, Yt_pred, output_dict=True)
            uar = report_print['macro avg']['recall']
            return {
                'loss': distance - uar,
                'status': STATUS_OK
            }

        trials = Trials()
        best = fmin(fn=choose_best_params, space=params, algo=tpe.suggest, max_evals=100, trials=trials)

        print(best)

        print('best: ', trials.best_trial)

    @classmethod
    def svm_using_bda_feature(cls):
        params = utils.ConfigDict(
            C=hp.uniform('C', 0.001, 100),
            svm_kernel=hp.choice('svm_kernel', [
                {
                    'name' : 'rbf',
                    'gamma': hp.uniform('rbf_gamma_uniform', 0.001, 10)
                },
                {
                    'name': 'linear',
                },
                {
                    'name' : 'sigmoid',
                    'gamma': hp.uniform('sigmoid_gamma_uniform', 0.001, 100),
                    'coef0': hp.uniform('sigmoid_coef0', 0, 10)
                },
                # {
                #     'name'  : 'poly',
                #     'gamma' : hp.uniform('poly_gamma_uniform', 0.001, 100),
                #     'coef0' : hp.uniform('poly_coef0', 0, 10),
                #     'degree': hp.uniformint('poly_degree', 2, 3),
                # }
            ]),
            bda_kernel=hp.choice('bda_kernel', [
                {
                    'name': 'primal',
                    'gamma': 1,
                },
                {
                    'name': 'linear',
                    'gamma': 1,
                },
                {
                    'name': 'rbf',
                    'gamma': hp.uniform('bda_gamma', 0.001, 10)
                }
            ]),
            lamb=hp.uniform('bda_lamb', 0.001, 1),
            mu=hp.uniform('bda_mu', 0.001, 1),
            T=hp.uniformint('bda_iterations', 1, 20),
            mode=hp.choice('bda_mode', ['BDA', 'WBDA']),
        )

        def choose_best_params(config):
            print('\n---------------------------------------------------------------------------------------')
            print('Params: ')
            print(config)
            bda_kernel_config = config['bda_kernel']
            bda_kernel = bda_kernel_config.pop('name')
            Xs_new, Ys_new, Xt_new, Yt_new = cls.loader.load_bda_data(cls.Xs, cls.Ys, cls.Xt, cls.Yt,
                                                                      kernel=bda_kernel,
                                                                      **bda_kernel_config,
                                                                      lamb=config['lamb'],
                                                                      mu=config['mu'],
                                                                      T=config['T'],
                                                                      mode=config['mode'],
                                                                      estimate_mu=False)
            svm_kernel_config = config['svm_kernel']
            svm_kernel = svm_kernel_config.pop('name')
            C = config['C']

            # uar
            clf = make_pipeline(StandardScaler(),
                                SVC(kernel=svm_kernel, tol=0.001, random_state=666, shrinking=True, C=C,
                                    **svm_kernel_config))
            clf.fit(Xs_new, Ys_new)
            Yt_pred = clf.predict(Xt_new)
            distance = mmd(Xs_new, Xt_new)
            matrix = confusion_matrix(Yt_new, Yt_pred)
            report = classification_report(Yt_new, Yt_pred)
            print('Result: ')
            print(matrix)
            print(report)
            print('---------------------------------------------------------------------------------------\n')
            report_print = classification_report(Yt_new, Yt_pred, output_dict=True)
            uar = report_print['macro avg']['recall']
            return {
                'loss'  : distance - uar,
                'status': STATUS_OK
            }

        trials = Trials()
        best = fmin(fn=choose_best_params, space=params, algo=tpe.suggest, max_evals=100, trials=trials)

        print(best)
        print('best: ', trials.best_trial)

    @classmethod
    def svm_using_jda_feature(cls):
        params = utils.ConfigDict(
            C=hp.uniform('C', 0.001, 100),
            svm_kernel=hp.choice('svm_kernel', [
                {
                    'name' : 'rbf',
                    'gamma': hp.uniform('rbf_gamma_uniform', 0.001, 10)
                },
                {
                    'name': 'linear',
                },
                {
                    'name' : 'sigmoid',
                    'gamma': hp.uniform('sigmoid_gamma_uniform', 0.001, 100),
                    'coef0': hp.uniform('sigmoid_coef0', 0, 10)
                },
                # {
                #     'name'  : 'poly',
                #     'gamma' : hp.uniform('poly_gamma_uniform', 0.001, 100),
                #     'coef0' : hp.uniform('poly_coef0', 0, 10),
                #     'degree': hp.uniformint('poly_degree', 2, 3),
                # }
            ]),
            jda_kernel=hp.choice('jda_kernel', [
                {
                    'name' : 'primal',
                    'gamma': 1,
                },
                {
                    'name' : 'linear',
                    'gamma': 1,
                },
                {
                    'name' : 'rbf',
                    'gamma': hp.uniform('jda_gamma', 0.001, 10)
                }
            ]),
            lamb=hp.uniform('jda_lamb', 0.001, 1),
            T=hp.uniformint('jda_iterations', 1, 20),
        )

        def choose_best_params(config):
            print('\n---------------------------------------------------------------------------------------')
            print('Params: ')
            print(config)
            jda_kernel_config = config['jda_kernel']
            jda_kernel = jda_kernel_config.pop('name')
            Xs_new, Ys_new, Xt_new, Yt_new = cls.loader.load_jda_data(cls.Xs, cls.Ys, cls.Xt, cls.Yt,
                                                                      kernel=jda_kernel,
                                                                      **jda_kernel_config,
                                                                      lamb=config['lamb'],
                                                                      T=config['T'])
            svm_kernel_config = config['svm_kernel']
            svm_kernel = svm_kernel_config.pop('name')
            C = config['C']

            # uar
            clf = make_pipeline(StandardScaler(),
                                SVC(kernel=svm_kernel, tol=0.001, random_state=666, shrinking=True, C=C,
                                    **svm_kernel_config))
            clf.fit(Xs_new, Ys_new)
            Yt_pred = clf.predict(Xt_new)
            distance = mmd(Xs_new, Xt_new)
            matrix = confusion_matrix(Yt_new, Yt_pred)
            report = classification_report(Yt_new, Yt_pred)
            print('Result:')
            print(matrix)
            print(report)
            print('---------------------------------------------------------------------------------------\n')
            report_print = classification_report(Yt_new, Yt_pred, output_dict=True)
            uar = report_print['macro avg']['recall']
            return {
                'loss'  : distance - uar,
                'status': STATUS_OK
            }

        trials = Trials()
        best = fmin(fn=choose_best_params, space=params, algo=tpe.suggest, max_evals=100, trials=trials)

        print(best)
        print('best: ', trials.best_trial)



if __name__ == '__main__':
    sys.stdout = utils.Logger(sys.stdout)  # 将输出记录到log
    exps = Run()

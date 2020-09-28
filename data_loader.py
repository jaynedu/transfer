# -*- coding: utf-8 -*-
# @Date    : 2020/9/20 16:48
# @Author  : Du Jing
# @FileName: data_loader
# ---- Description ----

import os
import tqdm
import random
import pandas as pd
import numpy as np
import sklearn

import utils
import feature_reduction as fr


class DataLoader:
    '''
    load_func 在init之后可以直接运行
    '''
    def __init__(self, source_dir, target_dir):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.features = utils.data.get_feature_name(target_dir)

    def get_X_y(self, dir, shuffle=True):
        '''
        X: n x nfeature; nfeature: [], len = 93
        y: n x 1
        '''
        result = utils.ConfigDict(X=[], y=[])
        filenames = os.listdir(dir)
        if shuffle:
            random.shuffle(filenames)
        for filename in tqdm.tqdm(filenames, desc='loading data from %s...' % dir):
            mean_value = []
            df = pd.read_csv(os.path.join(dir, filename))
            for feature in self.features:
                mean_value.append(np.mean(df[feature].values))
            result.X.append(mean_value)
            category = os.path.splitext(filename)[0].split('_')[-1]
            if category == 'depr':
                result.y.append(0)  # depr
            else:
                result.y.append(1)  # nor
        return result

    def load_raw_data(self):
        '''
        从source dir和target dir 读取数据，所有特征组合成列表
        X: n x nfeature (mean value) nfeature是一个长度为93的列表，每个值对应每类特征的平均值，共n条数据
        Y: n x 1
        '''
        assert self.source_dir is not None
        assert self.target_dir is not None

        source = self.get_X_y(self.source_dir)
        target = self.get_X_y(self.target_dir)
        Xs, Ys = source.X, source.y
        Xt, Yt = target.X, target.y
        return np.array(Xs), np.array(Ys), np.array(Xt), np.array(Yt)

    def load_tca_data(self, Xs, Ys, Xt, Yt, kernel='primal', dim=2, lamb=1, gamma=1):
        '''
        从source dir和target dir读取dim个tca特征
        X: n x dim
        Y: n x 1
        '''
        tca = fr.TCA(kernel_type=kernel, dim=dim, lamb=lamb, gamma=gamma)
        Xs_new, Xt_new = tca.fit(Xs, Xt)
        return Xs_new, Ys, Xt_new, Yt

    def load_bda_data(self, Xs, Ys, Xt, Yt, kernel='primal', dim=2, lamb=1, mu=0.5, gamma=1, T=20, mode='BDA', estimate_mu=True):
        '''
        从source_dir和target_dir读取dim个bda特征
        '''
        bda = fr.BDA(kernel, dim, lamb, mu, gamma, T, mode, estimate_mu)
        Xs_new, Xt_new = bda.fit(Xs, Ys, Xt)
        return Xs_new, Ys, Xt_new, Yt

    def load_jda_data(self, Xs, Ys, Xt, Yt, kernel='primal', dim=2, lamb=1, gamma=1, T=10):
        '''
        从source_dir和target_dir读取dim个jda特征
        :param Xs:
        '''
        jda = fr.JDA(kernel, dim, lamb, gamma, T)
        Xs_new, Xt_mew = jda.fit(Xs, Ys, Xt)
        return Xs_new, Ys, Xt_mew, Yt

    def load_feature_scatter(self,
                             Xs, Ys, Xt, Yt,
                             select_features=None,
                             use_tca=False,
                             use_jda=False,
                             use_bda=False,
                             **kwargs):
        '''
        当use_tca为False时：必须指定2～3个feature用于画图
        :returns: 每个变量对应长度为2或3的二维列表，列表中的每个元素分别是不同特征的前n个样本值组成的列表
                  该函数返回的4个二维列表可直接用于Draw类画图
        '''
        assert not (use_tca & use_jda)
        if use_tca:
            Xs, Ys, Xt, Yt = self.load_tca_data(Xs, Ys, Xt, Yt, **kwargs)
            Ns = Xs[np.where(Ys > 0)]
            Ds = Xs[np.where(Ys == 0)]
            Nt = Xt[np.where(Yt > 0)]
            Dt = Xt[np.where(Yt == 0)]
            Ns = np.array((Ns[:, 0], Ns[:, 1]))
            Ds = np.array((Ds[:, 0], Ds[:, 1]))
            Nt = np.array((Nt[:, 0], Nt[:, 1]))
            Dt = np.array((Dt[:, 0], Dt[:, 1]))
        elif use_jda:
            Xs, Ys, Xt, Yt = self.load_jda_data(Xs, Ys, Xt, Yt, **kwargs)
            Ns = Xs[np.where(Ys > 0)]
            Ds = Xs[np.where(Ys == 0)]
            Nt = Xt[np.where(Yt > 0)]
            Dt = Xt[np.where(Yt == 0)]
            Ns = np.array((Ns[:, 0], Ns[:, 1]))
            Ds = np.array((Ds[:, 0], Ds[:, 1]))
            Nt = np.array((Nt[:, 0], Nt[:, 1]))
            Dt = np.array((Dt[:, 0], Dt[:, 1]))
        elif use_bda:
            Xs, Ys, Xt, Yt = self.load_bda_data(Xs, Ys, Xt, Yt, **kwargs)
            Ns = Xs[np.where(Ys > 0)]
            Ds = Xs[np.where(Ys == 0)]
            Nt = Xt[np.where(Yt > 0)]
            Dt = Xt[np.where(Yt == 0)]
            Ns = np.array((Ns[:, 0], Ns[:, 1]))
            Ds = np.array((Ds[:, 0], Ds[:, 1]))
            Nt = np.array((Nt[:, 0], Nt[:, 1]))
            Dt = np.array((Dt[:, 0], Dt[:, 1]))
        else:
            assert isinstance(select_features, list) and len(select_features) > 1
            i, j = self.features.index(select_features[0]), self.features.index(select_features[1])
            Ns = Xs[np.where(Ys > 0)]
            Ds = Xs[np.where(Ys == 0)]
            Nt = Xt[np.where(Yt > 0)]
            Dt = Xt[np.where(Yt == 0)]
            Ns = np.array((Ns[:, i], Ns[:, j]))
            Ds = np.array((Ds[:, i], Ds[:, j]))
            Nt = np.array((Nt[:, i], Nt[:, j]))
            Dt = np.array((Dt[:, i], Dt[:, j]))

        return Ns, Ds, Nt, Dt

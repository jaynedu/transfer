# -*- coding: utf-8 -*-
# @Date    : 2020/9/18 20:49
# @Author  : Du Jing
# @FileName: mmd
# ---- Description ----

import numpy as np
from sklearn import metrics


class MMD:

    @staticmethod
    def _linear(source, target):
        """MMD using linear kernel (i.e., k(x,y) = <x,y>)
        Note that this is not the original linear MMD, only the reformulated and faster version.
        The original version is:
            def mmd_linear(X, Y):
                XX = np.dot(X, X.T)
                YY = np.dot(Y, Y.T)
                XY = np.dot(X, Y.T)
                return XX.mean() + YY.mean() - 2 * XY.mean()
        Arguments:
            X {[n_sample1, dim]} -- [X matrix]
            Y {[n_sample2, dim]} -- [Y matrix]
        Returns:
            [scalar] -- [MMD value]
        """
        delta = source.mean(0) - target.mean(0)
        return delta.dot(delta.T)

    @staticmethod
    def _rbf(source, target, gamma=1.0):
        """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
        Arguments:
            X {[n_sample1, dim]} -- [X matrix]
            Y {[n_sample2, dim]} -- [Y matrix]
        Keyword Arguments:
            gamma {float} -- [kernel parameter] (default: {1.0})
        Returns:
            [scalar] -- [MMD value]
        """
        XX = metrics.pairwise.rbf_kernel(source, source, gamma).mean()
        YY = metrics.pairwise.rbf_kernel(target, target, gamma).mean()
        XY = metrics.pairwise.rbf_kernel(source, target, gamma).mean()
        YX = metrics.pairwise.rbf_kernel(target, source, gamma).mean()
        loss = XX + YY - XY - YX
        return loss

    @staticmethod
    def _poly(source, target, degree=2, gamma=1, coef0=0):
        """MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
        Arguments:
            X {[n_sample1, dim]} -- [X matrix]
            Y {[n_sample2, dim]} -- [Y matrix]
        Keyword Arguments:
            degree {int} -- [degree] (default: {2})
            gamma {int} -- [gamma] (default: {1})
            coef0 {int} -- [constant item] (default: {0})
        Returns:
            [scalar] -- [MMD value]
        """
        XX = metrics.pairwise.polynomial_kernel(source, source, degree, gamma, coef0)
        YY = metrics.pairwise.polynomial_kernel(target, target, degree, gamma, coef0)
        XY = metrics.pairwise.polynomial_kernel(source, target, degree, gamma, coef0)
        return XX.mean() + YY.mean() - 2 * XY.mean()

    def __call__(self, source, target, kernel_type='rbf', **kwargs):
        if kernel_type == 'linear':
            d = self._linear(source, target)
            print('MMD - linear: %s' % d)
            return d
        if kernel_type == 'rbf':
            d = self._rbf(source, target, **kwargs)
            print('MMD - rbf: %s' % d)
            return d
        if kernel_type == 'poly':
            d = self._poly(source, target, **kwargs)
            print('MMD - poly: %s' % d)
            return d

mmd = MMD()

# -*- coding: utf-8 -*-
# # Source Free Knowledge Transfer for Privacy-Preserving Unsupervised Motor Imagery Classification
# Author: Wen Zhang and Dongrui Wu
# Date: Oct., 2021
# E-mail: wenz@hust.edu.cn
# Refer: https://github.com/chamwen/MEKT

import numpy as np
from pyriemann.utils.covariance import covariances
from pyriemann.utils.mean import mean_covariance
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import fractional_matrix_power

class centroid_align(BaseEstimator, TransformerMixin):
    def __init__(self, center_type='euclid', cov_type='cov'):
        self.center_type = center_type
        self.cov_type = cov_type  # 'cov', 'scm', 'oas'

    def fit(self, X):
        tmp_cov = covariances(X, estimator=self.cov_type)
        center_cov = self._compute_center(tmp_cov, type=self.center_type)
        if center_cov is None:
            print('mean covriance matrix is none...')
            return None
        self.ref_matrix = fractional_matrix_power(center_cov, -0.5)

        return self

    def transform(self, X):
        num_trial, num_chn, num_point = X.shape[0], X.shape[1], X.shape[2]
        tmp_cov = covariances(X, estimator=self.cov_type)

        cov_new = np.zeros([num_trial, num_chn, num_chn])
        X_new = np.zeros(X.shape)
        for j in range(num_trial):
            trial_cov = np.squeeze(tmp_cov[j, :, :])
            trial_data = np.squeeze(X[j, :, :])
            cov_new[j, :, :] = np.dot(np.dot(self.ref_matrix, trial_cov), self.ref_matrix)
            X_new[j, :, :] = np.dot(self.ref_matrix, trial_data)
        return cov_new, X_new

    def fit_transform(self, X, **kwargs):
        tmp_cov = covariances(X, estimator=self.cov_type)
        center_cov = self._compute_center(tmp_cov, type=self.center_type)
        if center_cov is None:
            print('mean covriance matrix is none...')
            return None
        ref_matrix = fractional_matrix_power(center_cov, -0.5)

        num_trial, num_chn, num_point = X.shape[0], X.shape[1], X.shape[2]
        cov_new = np.zeros([num_trial, num_chn, num_chn])
        X_new = np.zeros(X.shape)
        for j in range(num_trial):
            trial_cov = np.squeeze(tmp_cov[j, :, :])
            trial_data = np.squeeze(X[j, :, :])
            cov_new[j, :, :] = np.dot(np.dot(ref_matrix, trial_cov), ref_matrix)
            X_new[j, :, :] = np.dot(ref_matrix, trial_data)
        return cov_new, X_new

    def _compute_center(self, tmp_cov, type='riemann'):
        center_cov = None
        if type == 'riemann':
            center_cov = mean_covariance(tmp_cov, metric='riemann')
        elif type == 'logeuclid':
            center_cov = mean_covariance(tmp_cov, metric='logeuclid')
        elif type == 'euclid':
            center_cov = np.mean(tmp_cov, axis=0)
        else:
            print("unsupport center...")
        return center_cov


if __name__ == '__main__':
    X = np.random.rand(144, 22, 750)  # 144*22*750
    print(X.shape)

    # way 1
    ca = centroid_align(center_type='euclid', cov_type='lwf')
    ca.fit(X)
    cov_new, X_new = ca.transform(X)
    print(cov_new.shape, X_new.shape)

    # way 2
    ca = centroid_align(center_type='euclid', cov_type='lwf')
    cov_new, X_new = ca.fit_transform(X)
    print(cov_new.shape, X_new.shape)
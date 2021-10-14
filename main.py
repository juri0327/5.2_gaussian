# -*- coding: utf-8 -*-

import numpy as np

class Gaussian:
    def __init__(self, dim):
        '''コンストラクタ（みたいなもの）
        オブジェクトを作るときに初めに実行される。
        内部状態の初期化に使う
        '''
        self.dim = dim
        '''
        self = オブジェクトを指す。 self.dim は、オブジェクトの dim という変数を指す。
        上の命令は、 self.dim に dim の値を代入することを表す
        '''
        self.mean = np.random.randn(dim) # オブジェクトの mean という変数をランダムに初期化
        self.cov = np.identity(dim)

    def log_pdf(self, X):
        ''' 確率密度関数の対数を返す

        Parameters
        ----------
        X : numpy.array, shape (sample_size, dim)

        Returns
        -------
        log_pdf : array, shape (sample_size,)
        '''
        centered_X = X - self.mean
        cov_inv_centered_X = np.linalg.solve(self.cov, centered_X.T).T
        log_pdf = - 0.5 * self.dim * np.log(2 * np.pi) \
            - 0.5 * np.linalg.slogdet(self.cov)[1] \
            - 0.5 * np.sum(centered_X * cov_inv_centered_X, axis=1)
        return log_pdf

    def fit(self, X):
        ''' X を使って最尤推定をする

        Parameters
        ----------
        X : numpy.array, shape (sample_size, dim)
        '''
        pass

    def sample(self, sample_size):
        ''' 現状のパラメタを使って `sample_size` のサイズのサンプルを生成する

        Parameters
        ----------------
        sample_size : int

        Returns
        -----------
        X : numpy.array, shape (sample_size, dim)
            各行は平均 `self.mean`, 分散 `self.cov` の正規分布に従う
        '''
        pass

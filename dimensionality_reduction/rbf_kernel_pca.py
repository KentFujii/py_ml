from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np


def rbf_kernel_pca(X, gamma, n_components):
    # それぞれ行すべての組み合わせの距離を求める
    sq_dists = pdist(X, 'sqeuclidean')
    # 上で求めたそれぞれの組み合わせの距離を行列で表現する
    mat_sq_dists = squareform(sq_dists)
    # rbf対称カーネル行列を計算する
    K = exp(-gamma * mat_sq_dists)
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    # カーネル行列Kの中心化を行う
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    eigvals, eigvecs = eigh(K)
    # n_componentsの数だけ固有値を抽出する
    alphas = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))
    # n_componentsの数だけ固有ベクトルを抽出する
    lambdas = [eigvals[-i] for i in range(1, n_components + 1)]
    return alphas, lambdas
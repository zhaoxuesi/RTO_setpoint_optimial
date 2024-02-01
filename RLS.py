import numpy as np


# This function is  recursive_least_squares for the liner problem: y = beta*x
# note!!! The shape of x and y should be n*1 and 1*1.
# beta.shape = (1,n), cov_matrix.shape=(n*n)
def recursive_least_squares(x_point, y_point, beta, cov_matrix):
    # 解包新数据点
    x = x_point
    y = y_point

    # 计算预测值
    y_pred = beta @ x

    # 计算残差
    residual = y - y_pred
    # 更新协方差矩阵
    cov_matrix = (cov_matrix + np.outer((cov_matrix @ x), x.T) @ cov_matrix) / (1 + (x.T @ cov_matrix @ x))

    # 计算增量权重
    weight = (cov_matrix @ x) / (1 + x.T @ cov_matrix @ x)

    # 更新参数
    beta = beta + weight * residual

    return beta, cov_matrix

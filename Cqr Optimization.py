
def admm_cqr(y, X, tau_list, rho=1.0, max_iter=1000, tol=1e-6):
    """ADMM 求解复合分位数回归"""
    n, p = X.shape
    K = len(tau_list)

    # 初始化变量
    beta = np.zeros(p)
    b = np.zeros(K)
    z = np.zeros(n * K)
    u = np.zeros(n * K)
    loss_history = []

    # 定义 reshape 函数，便于处理分位点数据
    def reshape_vars(v):
        return v.reshape((K, n))

    for iteration in range(max_iter):
        # 更新 beta
        z_u = reshape_vars(z - u)
        for k in range(K):
            residuals = y - z_u[k, :]
            beta = np.linalg.solve(X.T @ X + rho * np.eye(p), X.T @ residuals)

        # 更新 z
        z_old = z.copy()
        for k in range(K):
            residuals = y - X @ beta - b[k]
            z_k = residuals + u.reshape((K, n))[k, :]
            z_k = np.maximum(z_k - tau_list[k] / rho, 0) + np.minimum(z_k + (1 - tau_list[k]) / rho, 0)
            z[k * n:(k + 1) * n] = z_k

        # 更新 u
        u += z - z_old

        # 计算当前的损失
        loss = 0
        for k, tau in enumerate(tau_list):
            residuals = y - b[k] - X @ beta
            loss += np.sum(tau * np.maximum(0, residuals) + (1 - tau) * np.maximum(0, -residuals))
        loss /= n
        loss_history.append(loss)

        # 检查收敛性
        if np.linalg.norm(z - z_old) < tol:
            break

    return beta, b, loss_history

def beta_difference_2_norm(beta1, beta2):
    # 计算两个β向量的差
    difference = np.array(beta1) - np.array(beta2)
    # 计算差向量的二范数
    norm = np.linalg.norm(difference, 2)
    return norm


def calculate_metrics(y_true, y_pred):
    """
    使用NumPy在CPU上计算回归指标：ME, MAE, MSE, RMSE

    参数：
    y_true : array-like（支持列表、NumPy数组等）
    y_pred : array-like（支持列表、NumPy数组等）

    返回：
    dict，包含各指标的名称和数值（浮点型）
    """
    # 将输入转换为NumPy数组
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 检查输入维度
    if y_true.shape != y_pred.shape:
        raise ValueError(f"形状不一致！y_true: {y_true.shape}, y_pred: {y_pred.shape}")

    # 计算误差
    error = y_true - y_pred

    # 计算指标
    metrics = {
        "ME": np.mean(error).item(),
        "MAE": np.mean(np.abs(error)).item(),
        "MSE": np.mean(error ** 2).item(),
        "RMSE": np.sqrt(np.mean(error ** 2)).item(),
    }

    return metrics



# 示例数据
if __name__ == "__main__":
    import numpy as np

    n, p , m= 300, 10, 100
    N = n*m
    # 设置均值向量和协方差矩阵
    mean = np.zeros(p)  # p维均值向量，均值为0
    def generate_covariance_matrix(size):
        # 初始化协方差矩阵
        covariance_matrix = np.zeros((size, size))

        # 填充协方差矩阵
        for j in range(size):
            for k in range(size):
                covariance_matrix[j, k] = 2*(0.5 ** abs(j - k))

        return covariance_matrix
    cov = generate_covariance_matrix(p)
    mean1 = 0
    std_dev = np.sqrt(16)  # 方差为16，标准差为4
    # 分位点
    tau_list = np.linspace(0.05, 0.95, 19)
    lambda_ = 0.1  # SCAD惩罚参数


    # 使用 ADMM 方法
    num_iterations = 100  # 重复次数
    true_beta = np.ones(p)


    bias_all_list = []
    beta_admm_list = []  # 新增：存储每次迭代的beta_admm
    b_admm_list = []  # 新增：存储每次迭代的b_admm

    for _ in range(num_iterations):
        # 生成 n 个 p 维的多元正态随机数
        X = np.random.multivariate_normal(mean, cov, size=N)
        true_beta = np.ones(p)
        errors = np.random.chisquare(2, N)
        y = X @ true_beta + errors

        # ADMM估计结果
        beta_admm, b_admm, loss_history_admm = admm_cqr(y, X, tau_list)

        # 存储当前迭代的结果
        biasall = beta_difference_2_norm(beta_admm, true_beta)
        bias_all_list.append(biasall)
        beta_admm_list.append(beta_admm)  # 新增
        b_admm_list.append(b_admm)  # 新增

        print(f"Iteration {_}: Bias = {biasall:.4f}")

    # 计算所有指标的平均值
    average_bias = np.mean(bias_all_list)
    average_beta = np.mean(np.array(beta_admm_list), axis=0)  # 按列平均
    average_b = np.mean(np.array(b_admm_list), axis=0)  # 按列平均

    print("\nFinal Results:")
    print(f"Average Bias: {average_bias:.4f}")
    print(f"Average beta_admm: {average_beta}")
    print(f"Average b_admm: {average_b}")
    print(calculate_metrics(average_beta, true_beta))

def beta_difference_2_norm(beta1, beta2):
    # 计算两个β向量的差
    difference = np.array(beta1) - np.array(beta2)
    # 计算差向量的二范数
    norm = np.linalg.norm(difference, 2)
    return norm

class CQRSCAD:
    def quantile_loss_gradient(y, X, beta, b, tau_list):
        """计算梯度"""
        n, p = X.shape
        K = len(tau_list)
        grad_beta = np.zeros(p)
        grad_b = np.zeros(K)

        for k, tau in enumerate(tau_list):
            residuals = y - b[k] - X @ beta
            indicator = (residuals > 0).astype(float) - tau
            grad_beta -= X.T @ indicator
            grad_b[k] -= np.sum(indicator)

        return grad_beta / n, grad_b / n


    def scad_penalty(beta, lambda_, a=3.7):
        """SCAD惩罚函数"""
        penalty = np.zeros_like(beta)
        for j in range(len(beta)):
            if abs(beta[j]) <= lambda_:
                penalty[j] = lambda_ * abs(beta[j])
            elif abs(beta[j]) <= a * lambda_:
                penalty[j] = -((beta[j] ** 2 - 2 * a * lambda_ * abs(beta[j]) + lambda_ ** 2) / (2 * (a - 1)))
            else:
                penalty[j] = (a + 1) * lambda_ ** 2 / 2
        return penalty

    def gradient_descent_scad(y, X, tau_list, lambda_, lr=0.01, max_iter=500, tol=1e-6):
        """梯度下降法求解带SCAD惩罚的复合分位数回归"""
        n, p = X.shape
        K = len(tau_list)
        beta = np.zeros(p)
        b = np.zeros(K)
        loss_history = []

        for iteration in range(max_iter):
            grad_beta, grad_b = CQRSCAD.quantile_loss_gradient(y, X, beta, b, tau_list)
            grad_beta += CQRSCAD.scad_penalty(beta, lambda_)  # 加上SCAD惩罚梯度

            # 更新参数
            beta -= lr * grad_beta
            b -= lr * grad_b

            # 计算当前的损失
            loss = 0
            for k, tau in enumerate(tau_list):
                residuals = y - b[k] - X @ beta
                loss += np.sum(tau * np.maximum(0, residuals) + (1 - tau) * np.maximum(0, -residuals))
            loss += np.sum(CQRSCAD.scad_penalty(beta, lambda_))  # 加入SCAD惩罚项
            loss /= n
            loss_history.append(loss)

            # 检查收敛性
            if len(loss_history) > 1 and abs(loss_history[-1] - loss_history[-2]) < tol:
                break

        return beta, b, loss_history

if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    n, p , m= 80, 100, 5
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
    # 生成 n 个 p 维的多元正态随机数
    X = np.random.multivariate_normal(mean, cov, size=N)
    # 创建一个全为零的向量
    true_beta = np.zeros(p)
    # 设置特定的值
    true_beta[0] = 3
    true_beta[1] = 1.5
    true_beta[4] = 2
    errors = np.random.standard_t(3, size=N)
    y = X @ true_beta + errors

    # 分位点
    tau_list = np.linspace(0.05, 0.95, 19)
    lambda_ = 0.1  # SCAD惩罚参数

    # # 使用梯度下降法
    # beta_gd, b_gd, loss_history_gd = gradient_descent(y, X, tau_list)
    # print("Gradient Descent - Estimated beta:", beta_gd)
    # print("Gradient Descent - Estimated b (intercepts):", b_gd)


    # 使用带SCAD惩罚的梯度下降法
    beta_scad, b_scad, loss_history_scad = CQRSCAD.gradient_descent_scad(y, X, tau_list, lambda_)
    print("SCAD Gradient Descent - Estimated beta:", beta_scad)
    print("SCAD Gradient Descent - Estimated b (intercepts):", b_scad)
    bias = beta_difference_2_norm(true_beta, beta_scad)
    print(bias)



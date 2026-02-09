#改成cupy的形式
import cupy as cp
from cupyx.scipy.fft import fft, ifft
from cupyx.scipy.special import ndtr  # GPU加速的正态分布累积分布函数
from tqdm import tqdm  # 引入进度条库
from joblib import Parallel, delayed
from scipy.integrate import quad
import numpy as np
def beta_difference_2_norm(beta1, beta2):
    difference = cp.array(beta1) - cp.array(beta2)
    norm = cp.linalg.norm(difference, 2)
    return norm
def K_tilde(u, kernel_type='gaussian'):
    if kernel_type == 'gaussian':
        return cp.exp(-0.5 * u ** 2) / cp.sqrt(2 * cp.pi)
    elif kernel_type == 'epanechnikov':
        return 0.75 * (1 - u ** 2) * (cp.abs(u) <= 1)
    else:
        raise ValueError("不支持的核函数类型")
def K_h(u, h):
    return K_tilde(u / h) / h
def K_bar_scalar(u):
    return ndtr(u)
def K_bar(u):
    return cp.vectorize(K_bar_scalar)(u)
#----------------------------------
def compute_gradients(X_batch, y_batch, beta_h, b_tau, tau, h):
    n, K = X_batch.shape[0], len(tau)
    U = -(y_batch[:, cp.newaxis] - X_batch @ beta_h[:, cp.newaxis] - b_tau[cp.newaxis, :]) / h
    W = K_bar(U)
    grad_beta_h = X_batch.T @ (W - tau) @ cp.ones(K) / (n *K)
    grad_b_tau = cp.sum(W - tau, axis=0) / (n *K)
    return grad_beta_h, grad_b_tau
def gradient(beta_h, b_tau, X, y, tau, h):
    grad_beta_h, grad_b_tau = compute_gradients(X, y, beta_h, b_tau, tau, h)
    return grad_beta_h, grad_b_tau
def rho_tau(u, tau):
    """
    定义检查函数 rho_tau(u)。
    """
    u = cp.asarray(u)  # 确保 u 是 CuPy 数组
    return u * (tau - cp.where(u < 0, 1.0, 0.0))  # 使用 cp.where 处理标量和数组

def cumulative_loss_fft(y_diff, b, tau, kernel_func):
    """
    使用 FFT 快速计算累积损失函数 \ell_{h, \tau}(u)。
    """
    n = y_diff.shape[0]

    # 1. 构造离散点网格
    v_points = cp.linspace(-3 * b, 3 * b, n)  # 限定积分范围 [-3b, 3b]
    dv = v_points[1] - v_points[0]  # 网格间距

    # 2. 计算高斯核值
    kernel_values = kernel_func(v_points, b)

    # 3. 对 y_diff 和 tau 计算 rho_tau(v)
    rho_values = rho_tau(v_points[:, None], tau[None, :])  # 扩展维度，支持多分位点

    # 4. 使用 FFT 快速卷积
    fft_kernel = fft(kernel_values)  # 高斯核的 FFT
    fft_rho = fft(rho_values, axis=0)  # 检查函数的 FFT
    conv_result = ifft(fft_kernel[:, None] * fft_rho, axis=0).real  # 卷积结果

    # 5. 离散积分求和
    integral_values = cp.sum(conv_result, axis=0) * dv  # 对每个分位点求和

    return integral_values

def Q_h_C(beta, b_tau, X0, y0, tau, b):
    """
    优化后的目标函数 Q_h_C，使用 FFT 加速高斯核卷积。
    """
    n, M = X0.shape[0], len(tau)

    # 1. 计算每个样本的 y_diff 矩阵
    y_diff_matrix = y0[:, cp.newaxis] - b_tau[cp.newaxis, :] - X0 @ beta[:, cp.newaxis]  # (n, M)

    # 2. 批量计算累积损失值
    ell_h_tau_matrix = cumulative_loss_fft(y_diff_matrix, b, tau, K_h)  # (n, M)

    # 3. 归一化并返回损失值
    Q_h_C_value = cp.sum(ell_h_tau_matrix) / (n * M)
    return Q_h_C_value





def scad_derivative(x, lambda_penalty, a=3.7):
    """
    SCAD 惩罚的导数函数
    x: 输入值或数组
    lambda_penalty: 惩罚参数 λ
    a: SCAD 参数（默认 3.7）
    返回: SCAD 惩罚的导数
    """
    x = cp.asarray(x)  # 确保输入可以是标量或数组
    grad = cp.zeros_like(x)  # 初始化梯度

    # 条件 1: |x| <= λ
    condition1 = cp.abs(x) <= lambda_penalty
    grad[condition1] = lambda_penalty * cp.sign(x[condition1])

    # 条件 2: λ < |x| <= aλ
    condition2 = (cp.abs(x) > lambda_penalty) & (cp.abs(x) <= a * lambda_penalty)
    grad[condition2] = (x[condition2] - lambda_penalty * cp.sign(x[condition2])) / (a - 1)

    # 条件 3: |x| > aλ
    # 在此条件下，grad 已经初始化为 0，因此不需要额外处理
    return grad

def mcp_derivative(x, lambda_penalty, gamma=3.0):
    """
    MCP 惩罚的导数函数
    x: 输入值或数组
    lambda_penalty: 惩罚参数 λ
    gamma: MCP 参数（默认 3.0）
    返回: MCP 惩罚的导数
    """
    x = cp.asarray(x)  # 确保输入可以是标量或数组
    grad = cp.zeros_like(x)  # 初始化梯度

    # 条件 1: |x| <= γλ
    condition1 = cp.abs(x) <= gamma * lambda_penalty
    grad[condition1] = lambda_penalty * cp.sign(x[condition1]) * (1 - cp.abs(x[condition1]) / (gamma * lambda_penalty))

    # 条件 2: |x| > γλ (默认梯度已是 0)
    return grad

def shrink(x, threshold):
    """
    Lasso 惩罚的 Shrink 操作
    x: 输入向量
    threshold: 阈值
    """
    return cp.sign(x) * cp.maximum(cp.abs(x) - threshold, 0)

def shrink_scalar(x, threshold):
    """
    Lasso 惩罚的 Shrink 操作，适用于单个数值。
    Args:
        x: 输入标量
        threshold: 阈值（标量）
    Returns:
        Shrink 结果（标量）
    """
    return cp.sign(x) * max(cp.abs(x) - threshold, 0)

def shrink_vector(x, threshold):
    """
    对向量逐元素应用 Shrink 操作。
    Args:
        x: 输入向量
        threshold: 阈值
    Returns:
        Shrink 结果向量
    """
    return cp.array([shrink_scalar(xi, ti) for xi, ti in zip(x, threshold)])




def LAMM(X, y, tau, b, lambda_penalty, penalty_type='scad',
         max_iter=1000, tol=1e-6, c0=0.46, gamma=1.1, delta=1e-5, a=3.7,
         gamma_mcp=3.0):
    """
    实现 LAMM 算法，支持 LASSO、SCAD 和 MCP 惩罚
    data: 数据集 (x, y)
    tau: 分位数向量 tau
    lambda_penalty: 惩罚参数 λ
    penalty_type: 惩罚类型，'lasso', 'scad', 或 'mcp'
    max_iter: 最大迭代次数
    tol: 收敛阈值
    c0: 初始梯度下降步长参数
    gamma: 步长增大因子
    delta: 收敛终止阈值
    a: SCAD 惩罚的超参数
    gamma_mcp: MCP 惩罚的超参数

    返回值: 最优参数 beta, b_tau
    """
    # 初始化参数
    K = len(tau)  # 分位点的数量
    p = X.shape[1]  # beta 的维度由 x 的特征数决定
    # beta = cp.random.normal(0, 0.01, p)  # 初始化为随机值
    # b_tau = cp.random.uniform(-0.01, 0.01, K)  # 初始化为随机值
    beta = cp.zeros(p)
    b_tau = cp.zeros(K)
    ck = c0  # 初始步长参数

    for w in range(max_iter):
        # Step 1: 计算梯度
        d_beta, db_tau = gradient(beta, b_tau, X, y, tau, b)
        # print(d_beta[0])
        # Step 2: 更新参数
        b_tau_new = b_tau - (1 / ck) * db_tau

        # 根据惩罚类型选择 Shrink 操作
        if penalty_type == 'lasso':
            beta_new = shrink_vector(beta - (1 / ck) * d_beta, (1 / ck) * lambda_penalty* cp.ones(p))
        elif penalty_type == 'l1':
            beta_new = shrink_vector(beta - (1 / ck) * d_beta, (1 / ck) * cp.ones(p))
        elif penalty_type == 'scad':
            absolute_beta = cp.abs(beta)
            lambdaSCAD = scad_derivative(absolute_beta, lambda_penalty, a=a)
            # print(lambdaSCAD[0])
            beta_new = shrink_vector(beta - (1 / ck) * d_beta, (1 / ck) * lambdaSCAD)
        elif penalty_type == 'mcp':
            absolute_beta = cp.abs(beta)
            lambdaMCP = mcp_derivative(absolute_beta, lambda_penalty, gamma=gamma_mcp)
            beta_new = shrink_vector(beta - (1 / ck) * d_beta, (1 / ck) * lambdaMCP)
        else:
            raise ValueError("Unsupported penalty type. Choose from 'lasso', 'scad', or 'mcp'.")

        # Step 3: 判断目标函数值下降
        Q_h_new = Q_h_C(beta_new, b_tau_new, X, y, tau, b)
        #print(beta_new[0])
        F_new = compute_F(beta, b_tau, b_tau_new, beta_new, tau, b, X, y, ck)
        # print(Q_h_new, F_new,beta[0])
        # print(w, "--------------------------")

        if Q_h_new > F_new:
            ck *= gamma  # 增大步长参数
            # Step 4: 更新参数
            b_tau, beta = b_tau_new, beta_new
            continue
        elif Q_h_new <= F_new and cp.linalg.norm(beta_new - beta) < delta:
            break
        # # Step 4: 更新参数
        # b_tau, beta = b_tau_new, beta_new
        #
        # # Step 5: 检查收敛
        # if cp.linalg.norm(beta_new - beta) < delta:
        #     print(f"在 {w} 次迭代后收敛。")
        #     break

    return beta, b_tau


def compute_F(beta, b_tau, b_tau_new, beta_new, tau, b, X, y, phi_step_size):
    phi = cp.concatenate((beta_new, b_tau_new))
    phi_prev = cp.concatenate((beta, b_tau))

    # 计算 Q_h(phi_prev)
    Q_h_val = Q_h_C(beta_new, b_tau_new, X, y, tau, b)

    # 计算梯度项
    beta_prev, b_tau_prev = gradient(beta, b_tau, X, y, tau, b)
    grad_phi_prev = cp.concatenate((beta_prev, b_tau_prev))
    grad_term = cp.dot(grad_phi_prev, phi - phi_prev)

    # 计算二次正则化项
    quadratic_term = (phi_step_size / 2) * cp.linalg.norm(phi - phi_prev) ** 2

    # 组合结果
    F_val = Q_h_val + grad_term + quadratic_term
    return F_val


def cross_validation_lambdas(X, y, tau, b, lambdas, beta_true, k_folds=5, penalty_type='scad'):
    """
    针对 lambda_penalty 参数进行交叉验证，选择最佳的 lambda。
    评估标准为 beta 参数与真实值 beta_true 的二范数偏差。

    Args:
        X: 输入特征矩阵
        y: 标签向量
        tau: 分位数向量
        b: 平滑参数
        lambdas: 待测试的 lambda_penalty 参数列表
        beta_true: 真实的 beta 参数
        k_folds: 交叉验证的折数
        penalty_type: 惩罚类型（支持 'lasso', 'scad', 'mcp'）

    Returns:
        best_lambda: 最优的 lambda_penalty
        best_beta: 对应最优 lambda 的 beta 参数
        best_b_tau: 对应最优 lambda 的 b_tau 参数
    """
    n_samples = X.shape[0]
    fold_size = n_samples // k_folds
    indices = cp.random.permutation(n_samples)  # 打乱样本索引

    best_lambda = None
    best_score = float('inf')
    best_beta, best_b_tau = None, None

    for lambda_penalty in lambdas:
        validation_scores = []

        # 交叉验证
        for fold in range(k_folds):
            # 分割训练集和验证集
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < k_folds - 1 else n_samples
            val_indices = indices[val_start:val_end]
            train_indices = cp.concatenate((indices[:val_start], indices[val_end:]))

            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]

            # 使用 LAMM 算法训练模型
            beta, b_tau = LAMM(X_train, y_train, tau, b, lambda_penalty, penalty_type=penalty_type)

            # 计算 beta 的二范数偏差
            beta_diff = beta_difference_2_norm(beta, beta_true)
            validation_scores.append(beta_diff)

        # 计算当前 lambda 的平均验证误差（beta 的二范数偏差）
        mean_val_score = cp.mean(cp.array(validation_scores))
        print(f"Lambda: {lambda_penalty}, Beta Difference (2-norm): {mean_val_score}")

        # 更新最佳 lambda
        if mean_val_score < best_score:
            best_score = mean_val_score
            best_lambda = lambda_penalty
            best_beta, best_b_tau = beta, b_tau

    return best_lambda, best_beta, best_b_tau



cp.random.seed(None)  # 每次运行时随机生成种子

# Inputs
n, p, m = 200, 500, 1
N = n * m
nu = 1e-4

# 设置均值向量和协方差矩阵
mean = cp.zeros(p)  # p维均值向量，均值为0

def generate_covariance_matrix(size):
    # 初始化协方差矩阵
    covariance_matrix = cp.zeros((size, size))

    # 填充协方差矩阵
    for j in range(size):
        for k in range(size):
            covariance_matrix[j, k] = 2 * (0.5 ** abs(j - k))

    return covariance_matrix

cov = generate_covariance_matrix(p)

# 生成 n 个 p 维的多元正态随机数
X = cp.random.multivariate_normal(mean.get(), cov.get(), size=N)

# 创建一个全为零的向量
beta_true = cp.zeros(p)

# 设置特定的值
beta_true[0] = 3
beta_true[1] = 1.5
beta_true[4] = 0.5
# 设置参数
loc = 0  # 位置参数
scale = 1  # 扩展参数

# 生成 Cauchy 分布随机样本
# errors = cp.random.standard_cauchy(size=N) * scale + loc
errors = cp.random.standard_t(3, size=N)
# errors = cp.random.normal(0, 3, N)
y = X @ beta_true + errors

# 分位数水平和参数
tau = cp.linspace(0.05, 0.95, 19)  # tau 从 0 到 1 均匀划分为 20 等份

# 计算 h
b = (cp.log(p)) / n
b = b ** (1 / 4) * 0.5  # 开二分之一次方
h = (cp.log(p)) / N
h = h ** (1 / 4) * 0.5  # 开二分之一次方




beta0, b_tau0 = LAMM(X, y, tau,  b, lambda_penalty=0.11,c0=0.46)
# print('beta1:', beta1)
# print('b_tau1:', b_tau1)
bias1D = beta_difference_2_norm(beta0, beta_true)
print('bias1D:', bias1D)
# print(h,b)



# #交叉验证c0的选取
# result = []
# for i in np.arange(0.01, 1, 0.05):
#     beta1, b_tau1 = LAMM(X, y, tau, b, lambda_penalty=0.11,c0=i)
#     # print('beta1:', beta1)
#     # print('b_tau1:', b_tau1)
#     bias1D = beta_difference_2_norm(beta1, beta_true)
#     result.append(bias1D)
#     result.append(i)
#     print(beta1[0])
#     print('bias1D:', bias1D,"c0大小",i)
# print(result)


# # 设置待测试的 lambda_penalty 参数列表
# lambda_candidates = cp.linspace(0.1, 2.0, 100)

# # 调用交叉验证函数
# best_lambda, best_beta, best_b_tau = cross_validation_lambdas(
#     X, y, tau, b, lambda_candidates, beta_true, k_folds=5, penalty_type='scad'
# )
# biasbest = beta_difference_2_norm(beta_true, best_beta)
# print(f"最佳 Lambda: {best_lambda}")
# print(f"对应的 biasbest 参数: {biasbest}")


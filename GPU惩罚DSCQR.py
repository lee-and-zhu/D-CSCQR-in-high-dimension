import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from tqdm import tqdm  # 引入进度条库
import statsmodels.api as sm
from joblib import Parallel, delayed
import cupy as cp
from cupyx.scipy.special import ndtr  # GPU加速的正态分布累积分布函数
from cupyx.scipy.fft import fft, ifft


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

    def gradient_descent_scad(y, X, tau_list, lambda_, lr=0.01, max_iter=500, tol=1e-4):
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


# ----------------------------------

def compute_gradients(X_batch, y_batch, beta_h, b_tau, tau, h):
    n, K = X_batch.shape[0], len(tau)
    U = -(y_batch[:, cp.newaxis] - X_batch @ beta_h[:, cp.newaxis] - b_tau[cp.newaxis, :]) / h
    W = K_bar(U)
    grad_beta_h = X_batch.T @ (W - tau) @ cp.ones(K) / (n * K)
    grad_b_tau = cp.sum(W - tau, axis=0) / (n * K)
    return grad_beta_h, grad_b_tau


def gradient(beta_h, b_tau, X, y, tau, h):
    grad_beta_h, grad_b_tau = compute_gradients(X, y, beta_h, b_tau, tau, h)
    return grad_beta_h, grad_b_tau


# 定义移位损失的梯度
def gradient_Q(beta_h, b_tau, X, y, tau, h, b, beta0, b_tau0, X0, y0):
    # grad_beta_h, grad_b_tau = gradient(beta0, b_tau0, X, y, tau, h)#全局梯度
    # 使用并行计算梯度
    # results = Parallel(n_jobs=-1)(
    #     delayed(gradient)(beta, b_tau, X, y, tau, h) for beta in [beta0]
    # )
    # grad_beta_h, grad_b_tau = results[0]
    grad_beta_h, grad_b_tau = gradient(beta0, b_tau0, X, y, tau, h)
    grad_beta_b, grad_b_taub = gradient(beta_h, b_tau, X0, y0, tau, b)
    grad_beta_b0, grad_b_taub0 = gradient(beta0, b_tau0, X0, y0, tau, b)  # 局部梯度
    grad_shift_beta = grad_beta_b - grad_beta_b0 + grad_beta_h
    grad_shift_b_tau = grad_b_taub - grad_b_taub0 + grad_b_tau
    return grad_shift_beta, grad_shift_b_tau


def BB_step(delta, eta):
    lambda_1 = np.dot(delta, delta) / np.dot(delta, eta)
    lambda_2 = np.dot(delta, eta) / np.dot(eta, eta)
    if lambda_1 > 0:
        return min(lambda_1, lambda_2, 40)
    else:
        return 1.0


# 使用梯度下降和 BB 步长求解 CSQR 估计
def solve_CSQR(X, y, tau, h, nu=1e-4, max_iter=1000):
    n, p = X.shape
    M = len(tau)

    # 初始化
    beta_h = np.zeros(p)
    b_tau = np.zeros(M)

    beta_h_prev = beta_h.copy()
    b_tau_prev = b_tau.copy()

    # 第一步梯度下降
    grad_beta_h, grad_b_tau = gradient(beta_h, b_tau, X, y, tau, h)
    beta_h -= grad_beta_h
    b_tau -= grad_b_tau

    # 迭代过程，加入进度条
    for t in tqdm(range(1, max_iter + 1), desc="CSQR 迭代进度"):
        # 计算增量 delta 和梯度差异 eta
        delta = np.concatenate([(beta_h - beta_h_prev), (b_tau - b_tau_prev)])
        grad_beta_h, grad_b_tau = gradient(beta_h, b_tau, X, y, tau, h)
        grad_combined = np.concatenate([grad_beta_h, grad_b_tau])

        # 计算 BB 步长
        step_size = BB_step(delta, grad_combined)

        # 更新 beta_h 和 b_tau
        beta_h_prev = beta_h.copy()
        b_tau_prev = b_tau.copy()
        beta_h -= step_size * grad_beta_h
        b_tau -= step_size * grad_b_tau
        # print(step_size, beta_h, b_tau)
        # 判断收敛条件
        if np.linalg.norm(beta_h - beta_h_prev) < nu:
            print(f"在 {t} 次迭代后收敛。")
            break
    else:
        print("达到最大迭代次数，未完全收敛。")
    beta_true = np.ones(p)
    print(beta_difference_2_norm(beta_h, beta_true))
    return beta_h, b_tau


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


# Qhat,局部样本SCQR
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


# 可能是没有标准化
# 修正目标函数 Q̃(β, bτ)
def Q_h(beta, b_tau, beta0, b_tau0, X, y, tau, h, b, X0, y0):
    """
    修正目标函数 Q̃(β, bτ)
    """
    # 原始 Q_h^C(β, bτ),是局部的
    Q_original = Q_h_C(beta, b_tau, X0, y0, tau, b)
    # print("原本的Q", Q_original)

    # 初始梯度 ∇Q_b^C(β, bτ) 局部和 ∇Q_h(β, bτ)全局
    grad_beta_init, grad_b_tau_init = gradient(beta0, b_tau0, X0, y0, tau, b)
    grad_beta_current, grad_b_tau_current = gradient(beta0, b_tau0, X, y, tau, h)

    # 修正项
    correction_term = cp.dot(grad_beta_init - grad_beta_current, beta) + \
                      cp.dot(grad_b_tau_init - grad_b_tau_current, b_tau)
    # print('内积部分', correction_term)
    # print("移位损失函数", Q_original - correction_term)
    return Q_original - correction_term


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


def LAMM(beta0, b_tau0, X, y, tau, h, b, X0, y0, lambda_penalty, penalty_type='scad',
         max_iter=1000, tol=1e-6, c0=0.05, gamma=1.25, delta=1e-5, a=3.7,
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
    # beta = cp.zeros(p)
    # b_tau = cp.zeros(K)
    beta = cp.random.normal(0, 0.01, p)  # 初始化为随机值
    b_tau = cp.random.uniform(-0.01, 0.01, K)  # 初始化为随机值
    # beta0, b_tau0 = solve_CSQR(X0, y0, tau, b, nu=1e-4, max_iter=10000)
    # beta0, b_tau0,loss_history_scad = CQRSCAD.gradient_descent_scad(y0, X0, tau, 0.05)
    # print('beta0', beta_difference_2_norm(beta0, beta_true))
    # print('beta0', beta0)
    ck = c0  # 初始步长参数
    for w in range(max_iter):
        # Step 1: 计算梯度
        d_beta, db_tau = gradient_Q(beta, b_tau, X, y, tau, h, b, beta0, b_tau0, X0, y0)
        # print("d_beta, db_tau",d_beta, db_tau)
        # Step 2: 更新参数
        b_tau_new = b_tau - (1 / ck) * db_tau
        # print(d_beta[0])
        # 根据惩罚类型选择 Shrink 操作,可能有问题
        if penalty_type == 'lasso':
            beta_new = shrink_vector(beta - (1 / ck) * d_beta, (1 / ck) * lambda_penalty * np.ones(p))
        elif penalty_type == 'l1':
            beta_new = shrink_vector(beta - (1 / ck) * d_beta, (1 / ck) * np.ones(p))
        elif penalty_type == 'scad':
            absolute_beta = cp.abs(beta)
            lambdaSCAD = scad_derivative(absolute_beta, lambda_penalty, a=a)
            print(lambdaSCAD[0],"lambdaSCAD")
            beta_new = shrink_vector(beta - (1 / ck) * d_beta, (1 / ck) * lambdaSCAD)
        elif penalty_type == 'mcp':
            absolute_beta = cp.abs(beta)
            lambdaMCP = mcp_derivative(absolute_beta, lambda_penalty, gamma=gamma_mcp)
            beta_new = shrink_vector(beta - (1 / ck) * d_beta, (1 / ck) * lambdaMCP)
        else:
            raise ValueError("Unsupported penalty type. Choose from 'lasso', 'scad', or 'mcp'.")
        print(beta_new[0],"更新的系数")
        # Step 3: 判断目标函数值下降
        Q_h_new = Q_h(beta_new, b_tau_new, beta0, b_tau0, X, y, tau, h, b, X0, y0)
        # Q_h_old = Q_h(beta, b_tau,beta0, b_tau0, X, y, tau, h,b, X0, y0, kernel_func=K_tilde)

        # F的定义应该有问题，其他应该差不多了
        # F_new = (Q_h_old + np.linalg.norm(b_tau_new - b_tau) ** 2 / (2 * ck)
        #          + np.linalg.norm(beta_new - beta) ** 2 / (
        #             2 * ck))
        F_new = compute_F(beta, b_tau, b_tau_new, beta_new, beta0, b_tau0, tau, h, b, X0, y0, ck)
        # print(Q_h_new, F_new)
        # print(w,"--------------------------")
        if Q_h_new > F_new:
            ck *= gamma  # 增大步长参数
            b_tau, beta = b_tau_new, beta_new
            continue
        else:
            break
        # # Step 4: 更新参数
        # b_tau, beta = b_tau_new, beta_new
        #
        # # Step 5: 检查收敛
        # if np.linalg.norm(beta_new - beta) < delta:
        #     print(f"在 {w} 次迭代后收敛。")
        #     print(np.linalg.norm(beta_new - beta))
        #     break

    return beta, b_tau


def compute_F(beta, b_tau, b_tau_new, beta_new, beta0, b_tau0, tau, h, b, X0, y0, phi_step_size):
    """
    计算目标函数值 F(φ; φ_k, φ^(k-1))
    :param phi: 当前变量向量 φ
    :param phi_prev: 上一次迭代的变量向量 φ^(k-1)
    :param phi_step_size: 当前步长 φ_k
    :param Q_h_func: 计算 Q_h 的函数
    :param gradient_func: 计算梯度的函数 ∇Q_h
    :return: F(φ; φ_k, φ^(k-1)) 的值
    """
    phi = cp.concatenate((beta_new, b_tau_new))
    phi_prev = cp.concatenate((beta, b_tau))

    # 计算 Q_h(phi_prev)
    Q_h_val = Q_h(beta, b_tau, beta0, b_tau0, X0, y0, tau, h, b, X0, y0)
    # print("q_h_val = ", Q_h_val)

    # 计算梯度项
    beta_prev, b_tau_prev = gradient_Q(beta, b_tau, X0, y0, tau, h, b, beta0, b_tau0, X0, y0)
    grad_phi_prev = cp.concatenate((beta_prev, b_tau_prev))
    grad_term = cp.dot(grad_phi_prev, phi - phi_prev)
    # print("grad_term = ", grad_term)

    # 计算二次正则化项
    quadratic_term = (phi_step_size / 2) * cp.linalg.norm(phi - phi_prev) ** 2
    # print("quadratic_term = ", quadratic_term)

    # 组合结果
    F_val = Q_h_val + grad_term + quadratic_term
    # print("计算的F", F_val)
    return F_val


def compute_fp_fn(beta_pred, beta_true, epsilon=1e-4):
    """
    计算 False Positive (FP) 和 False Negative (FN) 指标。

    Args:
        beta_pred: 模型预测的系数 (CuPy 数组)
        beta_true: 真实的系数 (CuPy 数组)
        epsilon: 判断系数是否为非零的阈值

    Returns:
        FP: False Positive 数量
        FN: False Negative 数量
    """
    # 判断非零系数的条件
    pred_nonzero = cp.abs(beta_pred) > epsilon  # 模型预测为非零的条件
    true_nonzero = cp.abs(beta_true) > epsilon  # 真实非零的条件

    # 计算 FP 和 FN
    FP = cp.sum(pred_nonzero & ~true_nonzero)  # 预测为非零，但真实为零
    FN = cp.sum(~pred_nonzero & true_nonzero)  # 预测为零，但真实为非零

    return FP, FN


# 生成合成数据
cp.random.seed(None)  # 每次运行时随机生成种子

# Inputs
n, p, m = 200, 500, 80
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
# errors = cp.random.standard_cauchy(size=N) * scale + loc
# errors = cp.random.standard_t(3, size=N)
errors = cp.random.normal(0, 3, N)
y = X @ beta_true + errors

# 分位数水平和参数
tau = cp.linspace(0.05, 0.95, 19)  # tau 从 0 到 1 均匀划分为 20 等份

# 计算 h
b = (cp.log(p)) / n
b = b ** (1 / 4) * 0.5  # 开二分之一次方
h = (cp.log(p)) / N
h = h ** (1 / 4) * 0.5  # 开二分之一次方

# 定义分布式，将总体随机分配到 m 个机器
# 随机打乱数据
shuffled_indices = cp.random.permutation(N)
shuffled_X = X[shuffled_indices]
shuffled_errors = errors[shuffled_indices]
shuffled_dataY = shuffled_X @ beta_true + shuffled_errors

# 将数据分成 m 份
split_dataX = cp.array_split(shuffled_X, m)
split_dataErrors = cp.array_split(shuffled_errors, m)

# 为每一份定义一个向量并存储它们
distubitedX = [part for part in split_dataX]
distubitedErrors = [part for part in split_dataErrors]

# 初始化结果列表
distubitedY = []
for i in range(m):
    distubitedY0 = distubitedX[i] @ beta_true + distubitedErrors[i]
    distubitedY.append(distubitedY0)


def select_lambda(X, scale=1.0):
    """
    根据理论公式 λ ∝ sqrt(ln(p)) / n 自动选择正则化参数 λ。

    参数:
    - X: 输入特征矩阵 (n_samples, n_features)
    - scale: 调节系数，用于调整 λ 的比例（默认为 1.0）

    返回:
    - lambda_value: 计算得到的 λ 值
    """
    n, p = X.shape  # 样本数 n 和特征数 p
    lambda_value = scale * (np.sqrt(np.log(p)) / n)
    return lambda_value


lambda_auto = select_lambda(distubitedX[0], scale=1.0)
beta1, b_tau1 = LAMM(beta0, b_tau0, X, y, tau, h, b, distubitedX[0], distubitedY[0], lambda_penalty=0.1, c0=0.26)
print('beta1:', beta1[0])
# print('b_tau1:', b_tau1)
bias1D = beta_difference_2_norm(beta1, beta_true)
FP, FN = compute_fp_fn(beta1, beta_true)
print(print(f"FP: {FP}, FN: {FN}"))
print('bias1D:', bias1D)
# print(h,b,lambda_auto)

# #交叉验证c0的选取
# result = []
# for i in np.arange(0.01, 2, 0.05):
#     beta1, b_tau1 = LAMM(beta0, b_tau0,X, y, tau, h, b, distubitedX[0], distubitedY[0], lambda_penalty=i,c0=0.26)
#     # print('beta1:', beta1)
#     # print('b_tau1:', b_tau1)
#     bias1D = beta_difference_2_norm(beta1, beta_true)
#     result.append(bias1D)
#     result.append(i)
#     print(beta1[0])
#     print('bias1D:', bias1D,"c0大小",i)
# print(result)
import cupy as cp
from scipy.stats import norm
from scipy.integrate import quad
from tqdm import tqdm  # 引入进度条库
import statsmodels.api as sm
from joblib import Parallel, delayed
#改成cupy的形式
import cupy as cp
from cupyx.scipy.special import ndtr  # GPU加速的正态分布累积分布函数
from tqdm import tqdm  # 引入进度条库
from joblib import Parallel, delayed


class CQRSCAD:
    def quantile_loss_gradient(y, X, beta, b, tau_list):
        """计算梯度"""
        n, p = X.shape
        K = len(tau_list)
        grad_beta = cp.zeros(p)
        grad_b = cp.zeros(K)

        for k, tau in enumerate(tau_list):
            residuals = y - b[k] - X @ beta
            indicator = (residuals > 0).astype(float) - tau
            grad_beta -= X.T @ indicator
            grad_b[k] -= cp.sum(indicator)

        return grad_beta / n, grad_b / n

    def scad_penalty(beta, lambda_, a=3.7):
        """SCAD惩罚函数"""
        penalty = cp.zeros_like(beta)
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
        beta = cp.zeros(p)
        b = cp.zeros(K)
        loss_history = []

        for iteration in range(max_iter):
            grad_beta, grad_b = CQRSCAD.quantile_loss_gradient(y, X, beta, b, tau_list)
            grad_beta += CQRSCAD.scad_penalty(beta, lambda_)  # 加上SCAD惩罚梯度

            # 更新参数
            beta -= lr * grad_beta
            b -= lr * grad_b

            # 计算当前的损失
            residuals = y[:, cp.newaxis] - b - X @ beta  # 向量化计算残差
            # 确保 tau_list 的形状与 residuals 兼容
            tau_matrix = cp.array(tau_list)[cp.newaxis, :]  # 将 tau_list 转换为 (1, K) 的形状
            loss = cp.sum(cp.maximum(0, residuals) * tau_matrix + cp.maximum(0, -residuals) * (1 - tau_matrix))
            loss += cp.sum(CQRSCAD.scad_penalty(beta, lambda_))  # 加入SCAD惩罚项
            loss /= n
            loss_history.append(loss)

            # 检查收敛性
            if len(loss_history) > 1 and abs(loss_history[-1] - loss_history[-2]) < tol:
                break

        return beta, b, loss_history
def beta_difference_2_norm(beta1, beta2):
    # 计算两个β向量的差
    difference = cp.array(beta1) - cp.array(beta2)
    # 计算差向量的二范数
    norm = cp.linalg.norm(difference, 2)
    return norm

# 核平滑函数，支持不同核函数
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
def compute_gradients(X_batch, y_batch, beta_h, b_tau, tau, h):
    n, K = X_batch.shape[0], len(tau)
    U = -(y_batch[:, cp.newaxis] - X_batch @ beta_h[:, cp.newaxis] - b_tau[cp.newaxis, :]) / h
    W = K_bar(U)
    grad_beta_h = X_batch.T @ (W - tau) / (n * K)  # 向量化计算
    grad_b_tau = cp.sum(W - tau, axis=0) / (n * K)  # 向量化计算
    return grad_beta_h, grad_b_tau
def gradient(beta_h, b_tau, X, y, tau, h):
    grad_beta_h, grad_b_tau = compute_gradients(X, y, beta_h, b_tau, tau, h)
    return grad_beta_h, grad_b_tau

# 目标函数的梯度,快速算法 (平滑分位数回归)
def gradient1(beta_h, b_tau, X, y, tau, h):
    """
    计算梯度的向量化版本。
    """
    n, K = X.shape[0], len(tau)

    # 计算误差矩阵 U (n x K)
    U = -(y[:, cp.newaxis] - X @ beta_h[:, cp.newaxis] - b_tau[cp.newaxis, :]) / h

    # 计算权重矩阵 W (n x K) 使用累积核函数 K_bar
    W = K_bar(U)

    # 计算梯度 grad_beta_h 和 grad_b_tau
    # grad_beta_h: sum over (W - tau) * X
    grad_beta_h = X.T @ (W - tau) @ cp.ones(K)

    # grad_b_tau: sum over (W - tau)
    grad_b_tau = cp.sum(W - tau, axis=0)

    # Normalize gradients
    grad_beta_h /= n * K
    grad_b_tau /= n * K

    return grad_beta_h, grad_b_tau

# 初代梯度，强行循环计算
def gradient2(beta_h, b_tau, X, y, tau, h):
    n, K = X.shape[0], len(tau)
    grad_beta_h = cp.zeros_like(beta_h)
    grad_b_tau = cp.zeros_like(b_tau)

    for i in range(n):
        for k in range(K):  # 循环可能有问题
            u = -(y[i] - X[i] @ beta_h - b_tau[k]) / h
            weight = K_bar(u)  # 使用 K_bar_h(u, h) 而不是 K_tilde
            grad_beta_h += (weight - tau[k]) * X[i]
            grad_b_tau[k] += (weight - tau[k])

    grad_beta_h /= n * K
    grad_b_tau /= n * K  # 加了K后就能跑了
    return grad_beta_h, grad_b_tau

# 定义移位损失的梯度
def gradient_Q(beta_h, b_tau, X, y, tau, h, b, beta0, b_tau0, X0, y0):
    grad_beta_h, grad_b_tau = gradient(beta0, b_tau0, X, y, tau, h)
    grad_beta_b, grad_b_taub = gradient(beta_h, b_tau, X0, y0, tau, b)
    grad_beta_b0, grad_b_taub0 = gradient(beta0, b_tau0, X0, y0, tau, b)  # 局部梯度
    grad_shift_beta = grad_beta_b - grad_beta_b0 + grad_beta_h
    grad_shift_b_tau = grad_b_taub - grad_b_taub0 + grad_b_tau
    return grad_shift_beta, grad_shift_b_tau

def BB_step(delta, eta):
    lambda_1 = cp.dot(delta, delta) / cp.dot(delta, eta)
    lambda_2 = cp.dot(delta, eta) / cp.dot(eta, eta)
    return min(lambda_1, lambda_2, 40) if lambda_1 > 0 else 1.0

# 使用梯度下降和 BB 步长求解 CSQR 估计
def solve_CSQR(X, y, tau, h, nu=1e-4, max_iter=1000):
    n, p = X.shape
    M = len(tau)

    # 初始化
    beta_h = cp.zeros(p)
    b_tau = cp.zeros(M)

    beta_h_prev = beta_h.copy()
    b_tau_prev = b_tau.copy()

    # 第一步梯度下降
    grad_beta_h, grad_b_tau = gradient(beta_h, b_tau, X, y, tau, h)
    beta_h -= grad_beta_h
    b_tau -= grad_b_tau

    # 迭代过程，加入进度条
    for t in tqdm(range(1, max_iter + 1), desc="CSQR 迭代进度"):
        # 计算增量 delta 和梯度差异 eta
        delta = cp.concatenate([(beta_h - beta_h_prev), (b_tau - b_tau_prev)])
        grad_beta_h, grad_b_tau = gradient(beta_h, b_tau, X, y, tau, h)
        grad_combined = cp.concatenate([grad_beta_h, grad_b_tau])

        # 计算 BB 步长
        step_size = BB_step(delta, grad_combined)

        # 更新 beta_h 和 b_tau
        beta_h_prev = beta_h.copy()
        b_tau_prev = b_tau.copy()
        beta_h -= step_size * grad_beta_h
        b_tau -= step_size * grad_b_tau

        # 判断收敛条件
        if cp.linalg.norm(beta_h - beta_h_prev) < nu:
            print(f"在 {t} 次迭代后收敛。")
            break
    else:
        print("达到最大迭代次数，未完全收敛。")

    return beta_h, b_tau

def rho_tau(u, tau):
    """
    定义检查函数 rho_tau(u)。
    """
    u = cp.asarray(u)  # 确保 u 是 CuPy 数组
    tau_array = cp.asarray(tau)  # 将 tau 转换为 CuPy 数组
    return u * (tau_array - cp.where(u < 0, 1.0, 0.0))  # 使用 cp.where 处理标量和数组

def cumulative_loss(y_diff, h, tau, kernel_func):
    """
    累积损失函数
    """
    y_diff = cp.asarray(y_diff)  # 确保 y_diff 是 CuPy 数组
    def integrand(u):
        return rho_tau(u, tau) * kernel_func((u - y_diff) / h)  # 被积函数
    integral, _ = quad(integrand, -cp.inf, cp.inf)  # 数值积分
    return integral

#Qhat,局部样本SCQR
def Q_h_C(beta, b_tau, X0, y0, tau, b, kernel_func=K_h):
    n, M = X0.shape[0], len(tau)
    y_diff = y0[:, cp.newaxis] - b_tau - X0 @ beta  # 向量化计算差值
    Q_h_C_value = cp.sum(cumulative_loss(y_diff, b, tau[k], kernel_func) for k in range(M))  # 向量化累积损失
    Q_h_C_value /= n  # 样本数归一化
    return Q_h_C_value

# 修正目标函数 Q̃(β, bτ)
def Q_h(beta, b_tau, beta0, b_tau0, X, y, tau, h, b, X0, y0, kernel_func=K_tilde):
    """
    修正目标函数 Q̃(β, bτ)
    """
    # 原始 Q_h^C(β, bτ),是局部的
    Q_original = Q_h_C(beta, b_tau, X0, y0, tau, b, kernel_func)
    print("原本的Q",Q_original)
    # 初始梯度 ∇Q_b^C(β, bτ) 局部和 ∇Q_h(β, bτ)全局
    grad_beta_init, grad_b_tau_init = gradient(beta0, b_tau0, X0, y0, tau, b)
    grad_beta_current, grad_b_tau_current = gradient(beta0, b_tau0, X, y, tau, h)

    # 修正项
    correction_term = cp.dot(grad_beta_init - grad_beta_current, beta) - \
                      cp.dot(grad_b_tau_init - grad_b_tau_current, b_tau)
    print('内积部分',correction_term)
    print("移位损失函数",Q_original - correction_term)
    return Q_original - correction_term


def shrink(x, threshold):
    """
    Lasso 惩罚的 Shrink 操作
    x: 输入向量
    threshold: 阈值
    """
    return cp.sign(x) * cp.maximum(cp.abs(x) - threshold, 0)


def scad_shrink(x, lambda_val, a=3.7):
    """
    SCAD 惩罚的 Shrink 操作
    x: 输入向量
    lambda_val: 惩罚参数
    a: SCAD 的超参数，通常取 3.7
    """
    abs_x = cp.abs(x)
    shrunk = cp.zeros_like(x)

    # 第一阶段：|x| <= lambda
    mask1 = abs_x <= lambda_val
    shrunk[mask1] = cp.sign(x[mask1]) * cp.maximum(abs_x[mask1] - lambda_val, 0)

    # 第二阶段：lambda < |x| <= a * lambda
    mask2 = (abs_x > lambda_val) & (abs_x <= a * lambda_val)
    shrunk[mask2] = ((a - 1) * x[mask2] - cp.sign(x[mask2]) * lambda_val) / (a - 1)

    # 第三阶段：|x| > a * lambda
    mask3 = abs_x > a * lambda_val
    shrunk[mask3] = x[mask3]

    return shrunk

def mcp_shrink(x, lambda_val, gamma=3.0):
    """
    MCP 惩罚的 Shrink 操作
    x: 输入向量
    lambda_val: 惩罚参数
    gamma: MCP 的超参数
    """
    abs_x = cp.abs(x)
    shrunk = cp.zeros_like(x)

    # 第一阶段：|x| <= gamma * lambda
    mask1 = abs_x <= gamma * lambda_val
    shrunk[mask1] = cp.sign(x[mask1]) * cp.maximum(abs_x[mask1] - lambda_val, 0) / (1 - 1 / gamma)

    # 第二阶段：|x| > gamma * lambda
    mask2 = abs_x > gamma * lambda_val
    shrunk[mask2] = x[mask2]

    return shrunk
def LAMM(X, y, tau, h, b, X0, y0, lambda_penalty, penalty_type='scad',
         max_iter=1000, tol=1e-6, c0=0.01, gamma=1.1, delta=1e-5, a=3.7,
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
    beta = cp.zeros(p)  # 初始化 beta
    b_tau = cp.zeros(K)  # 初始化分位点参数
    beta0, b_tau0,loss_history_scad = CQRSCAD.gradient_descent_scad(y0, X0, tau, 0.1)

    ck = c0  # 初始步长参数
    for w in range(max_iter):
        # Step 1: 计算梯度
        d_beta, db_tau = gradient_Q(beta, b_tau, X, y, tau, h, b, beta0, b_tau0, X0, y0)

        # Step 2: 更新参数
        b_tau_new = b_tau - (1 / ck) * db_tau

        # 根据惩罚类型选择 Shrink 操作
        if penalty_type == 'lasso':
            beta_new = shrink(beta - (1 / ck) * d_beta, (1 / ck) * lambda_penalty)
        elif penalty_type == 'scad':
            beta_new = scad_shrink(beta - (1 / ck) * d_beta, (1 / ck) * lambda_penalty, a=a)
        elif penalty_type == 'mcp':
            beta_new = mcp_shrink(beta - (1 / ck) * d_beta, (1 / ck) * lambda_penalty, gamma=gamma_mcp)
        else:
            raise ValueError("Unsupported penalty type. Choose from 'lasso', 'scad', or 'mcp'.")

        # Step 3: 判断目标函数值下降
        Q_h_new = Q_h(beta_new, b_tau_new, beta0, b_tau0, X, y, tau, h, b, X0, y0, kernel_func=K_tilde)

        # F的定义应该有问题，其他应该差不多了
        F_new = compute_F(beta, b_tau, b_tau_new, beta_new, beta0, b_tau0, tau, h, b, X0, y0, ck)
        print(Q_h_new, F_new)
        print(w,"--------------------------")
        if F_new > Q_h_new:
            ck *= gamma  # 增大步长参数
            continue

        # Step 4: 更新参数
        b_tau, beta = b_tau_new, beta_new

        # Step 5: 检查收敛
        if cp.linalg.norm(beta_new - beta) < delta:
            print(f"在 {w} 次迭代后收敛。")
            print(cp.linalg.norm(beta_new - beta))
            break

    return beta, b_tau


def compute_F(beta, b_tau, b_tau_new, beta_new, beta0, b_tau0, tau, h, b, X0, y0, phi_step_size):

    phi = cp.concatenate((beta_new, b_tau_new))
    phi_prev = cp.concatenate((beta, b_tau))
    # 计算 Q_h(phi_prev)
    Q_h_val = Q_h(beta, b_tau, beta0, b_tau0, X0, y0, tau, h, b, X0, y0, kernel_func=K_tilde)
    print("q_h_val = ", Q_h_val)

    # 计算梯度项
    beta_prev, b_tau_prev = gradient_Q(beta, b_tau, X0, y0, tau, h, b, beta0, b_tau0, X0, y0)
    grad_phi_prev = cp.concatenate((beta_prev, b_tau_prev))
    grad_term = cp.dot(grad_phi_prev, phi - phi_prev)
    print("grad_term = ", grad_term)

    # 计算二次正则化项
    quadratic_term = (phi_step_size / 2) * cp.linalg.norm(phi - phi_prev) ** 2
    print("quadratic_term = ", quadratic_term)

    # 组合结果
    F_val = Q_h_val + grad_term + quadratic_term
    print("计算的F", F_val)
    return F_val

# 生成合成数据
n, p, m = 300, 500, 50
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
            covariance_matrix[j, k] = 2 * (0.5 ** cp.abs(j - k))

    return covariance_matrix
cov = generate_covariance_matrix(p)
# 生成 n 个 p 维的多元正态随机数
X = cp.random.multivariate_normal(mean, cov, size=N)
# 创建一个全为零的向量
beta_true = cp.zeros(p)
# 设置特定的值
beta_true[0] = 3
beta_true[1] = 1.5
beta_true[4] = 2
errors = cp.random.standard_t(3, size=N)
y = X @ beta_true + errors
# 分位数水平和参数
tau = cp.linspace(0.05, 0.95, 19)  # tau 从 0 到 1 均匀划分为 20 等份
# 计算 h
b = (cp.log(p)) / n
b = b ** (1 / 2) * 0.75  # 开二分之一次方
h = (cp.log(p)) / N
h = h ** (1 / 2) * 0.75  # 开二分之一次方

# 分布式，将总体随机分配到m个机器
# 随机打乱数据
shuffled_indices = cp.random.permutation(N)
shuffled_X = X[shuffled_indices]
shuffled_errors = errors[shuffled_indices]
shuffled_dataY = shuffled_X @ beta_true + shuffled_errors
# 将数据分成m份
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
    """
    n, p = X.shape  # 样本数 n 和特征数 p
    lambda_value = scale * (cp.sqrt(cp.log(p)) / n)
    return lambda_value


lambda_auto = select_lambda(distubitedX[0], scale=1.0)
beta1, b_tau1 = LAMM(X, y, tau, h, b, distubitedX[0], distubitedY[0], lambda_auto)
bias1D = beta_difference_2_norm(beta1, beta_true)
print('bias1D:', bias1D)
print(h, b)

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.optimize import minimize

class AdaptSVM:
    def __init__(self, C=10.0):
        self.y_sv = None
        self.X_sv = None
        self.alpha = None
        self.f = None
        self.beta = None
        self.f_prime = None
        self.x_aux_sv = None
        self.y_aux_sv = None
        self.C = C
        self.alpha_prime = None
        self.beta_prime = None

    def Adapt_SVM_aux(self, x_aux, y_aux): # 辅助数据SVM
        n = len(y_aux)
        f = -1 * np.ones(n)
        Aeq = y_aux  # 约束条件
        beq = 0
        lb = np.zeros(n)
        ub = self.C * np.ones(n)
        a0 = np.zeros(n)

        H = linear_kernel(x_aux, y_aux)

        def objective(a):
            return 0.5 * np.dot(a, np.dot(H, a)) + np.dot(f, a)

        constraints = [{'type': 'eq', 'fun': lambda a: np.dot(Aeq, a) - beq}]
        bounds = list(zip(lb, ub))
        result = minimize(objective, a0, method='SLSQP', bounds=bounds, constraints=constraints)
        a = result.x

        epsilon = 1e-8
        sv_label = np.where(np.abs(a) > epsilon)[0]  # 支持向量

        self.alpha_prime = a[sv_label] # 拉格朗日乘子
        self.y_aux_sv = y_aux[sv_label]
        self.x_aux_sv = x_aux[:, sv_label]
        self.beta_prime = np.sum((a * y_aux) * x_aux,axis=1)
        self.f_prime = np.dot(self.x_aux_sv.T, self.beta_prime)

    def Adapt_SVM_main(self, x_main, y_main):
        n = len(y_main)
        f = -1 * np.ones(n)
        Aeq = y_main  # 约束条件
        beq = 0
        lb = np.zeros(n)
        ub = self.C * np.ones(n)
        a0 = np.zeros(n)

        H = linear_kernel(x_main, y_main)
        f_prime_main = self.f_prime[:n]

        def objective(a):
            return 0.5 * np.dot(a, np.dot(H, a)) + np.dot(f, (1 - np.dot(y_main, f_prime_main)) * a) # 拉格朗日对偶目标函数
        constraints = [{'type': 'eq', 'fun': lambda a: np.dot(Aeq, a) - beq}]
        bounds = list(zip(lb, ub))
        result = minimize(objective, a0, method='SLSQP', bounds=bounds, constraints=constraints)
        a = result.x

        epsilon = 1e-8
        sv_label = np.where(np.abs(a) > epsilon)[0]

        self.alpha = a[sv_label]
        self.X_sv = x_main[:, sv_label]
        self.y_sv = y_main[sv_label]
        self.beta = np.sum((a * y_main) * x_main, axis=1) + self.beta_prime
        self.f = f_prime_main + np.sum(self.alpha * self.y_sv * np.dot(x_main.T, self.X_sv), axis=1)

    def test_main_data(self, x_test):
        score_test = np.dot(self.beta, x_test)
        y_pred = np.sign(score_test)

        return y_pred

    def linear_svm_test(self, Xt, Yt):
        w = np.sum((self.alpha_prime * self.y_aux_sv) * self.x_aux_sv, axis=1)
        score = np.dot(w, Xt)
        Y = np.sign(score)

        result = {
            'score': score,
            'Y': Y,
            'accuracy': np.sum(Y == Yt) / len(Yt)
        }

        return result

def linear_kernel(X, Y):
    return np.outer(Y, Y) * np.dot(X.T, X)

def main():
    C = 10
    n = 50

    data = scipy.io.loadmat('datanew.mat')

    # 辅助数据（先验）
    xp_aux = data['X_P'][:n, :].T
    yp_aux = np.ones(n)
    xn_aux = data['X_N'][:n, :].T
    yn_aux = -np.ones(n)

    x_aux = np.hstack((xp_aux, xn_aux))
    y_aux = np.hstack((yp_aux, yn_aux))

    # 主数据labelled
    xp_main = data['X_P'][n: n + 10, :].T
    yp_main = np.ones(10)
    xn_main = data['X_N'][n: n + 10, :].T
    yn_main = -np.ones(10)

    x_main = np.hstack((xp_main, xn_main))
    y_main = np.hstack((yp_main, yn_main))

    #主数据unlabelled
    offset = np.array([1, 0.5])
    xp_test = data['X_P'][n + 10:, :].T
    xp_test = xp_test + offset.reshape((2, 1))
    yp_test = np.ones(40)
    xn_test = data['X_N'][n + 10:, :].T
    xn_test = xn_test + offset.reshape((2, 1))
    yn_test = -np.ones(40)

    x_test = np.hstack((xp_test, xn_test))
    y_test = np.hstack((yp_test, yn_test))

    # 辅助数据SVM
    svm = AdaptSVM(C=C)
    svm.Adapt_SVM_aux(x_aux, y_aux)

    # # 画图
    # plt.figure()
    # plt.plot(xp_aux[0, :], xp_aux[1, :], 'bx', xn_aux[0, :], xn_aux[1, :], 'g.')
    # plt.axis([-11, 8, -11, 8])
    # plt.gca().set_prop_cycle(None)
    #
    # plt.scatter(svm.x_aux_sv[:, 0], svm.x_aux_sv[:, 1], c='r', marker='o')
    #
    # # 找决策边界：Xt在整个屏幕上生成足够多的点，Yd是每个点的预测状况，在预测交界处为决策边界。(并不是测试的意思
    # xp_aux, xn_aux = np.meshgrid(np.arange(-11, 8, 0.05), np.arange(-11, 8, 0.05))
    # Xt = np.vstack((xp_aux.ravel(), xn_aux.ravel()))
    # result_aux = svm.linear_svm_test(Xt, np.zeros(len(Xt.T)))
    #
    # Yd = result_aux['Y'].reshape(xp_aux.shape)
    # plt.contour(xp_aux, xn_aux, Yd, colors='m')
    #
    # plt.show()

    # 主数据中有标签的部分训练模型
    svm.Adapt_SVM_main(x_main, y_main)

    # 主数据中无标签的部分测试
    y_pred = svm.test_main_data(x_test)
    accuracy = np.sum(y_pred == y_test)/len(y_test)
    compare = svm.linear_svm_test(x_test, y_test)

    print("Accuracy linear SVM:", compare['accuracy'])
    print("Accuracy Adapt-SVM:", accuracy)

main()
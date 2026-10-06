import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as stats
from scipy.special import beta as beta_f
from scipy import linalg
from torch.autograd import Function
import matplotlib.pyplot as plt


## Beta Mixture model from https://github.com/PaulAlbert31/LabelNoiseCorrection/blob/master/utils.py
## Some functions from https://github.com/thuml/Separate_to_Adapt/blob/master/utilities.py

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()
    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1)
        return b

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class WeightedForwardLayerF(Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * ctx.beta
        return output, None

def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta

class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):  # p(k)*p(l|k) == p(y)*p(x|y)
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps
        self.score_history = []
        self.weight_0 = []
        self.weight_1 = []
        for i in range(self.max_iters):
            # E-step
            r = self.responsibilities(x)
            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

            neg_log_likelihood = np.sum([self.score_samples(i) for i in x])
            self.score_history.append(neg_log_likelihood)
            self.weight_0.append(self.weight[0])
            self.weight_1.append(self.weight[1])
        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l

    def look_lookup(self, x, loss_max, loss_min, testing=False):
        if testing:
            x_i = x
        else:
            x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def plot(self, title, save_dir, save_signal=False):
        x = np.linspace(0, 1, 100)
        plt.plot(x, self.weighted_likelihood(x, 0), label='known')
        plt.plot(x, self.weighted_likelihood(x, 1), label='unknown')
        plt.plot(x, self.probability(x), lw=2, label='mixture')
        plt.legend()
        if save_signal:
            plt.title(title)
            plt.savefig(save_dir, dpi=300)
        plt.close()

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)

    def calculate_criteria(self):
        self.K = ( self.weight[0] * beta_f(self.alphas[1], self.betas[1])) / ( self.weight[1] * beta_f(self.alphas[0], self.betas[0]))
        self.criteria = ((np.log(self.K)) - (self.betas[1] - self.betas[0])) / ( (self.alphas[1]-self.alphas[0]) - (self.betas[1]-self.betas[0]) )
        print(self.K, self.alphas[1]-self.alphas[0], beta_f(2,3))
        return self.criteria


def CrossEntropyLoss(label, predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-12):
    N, C = label.size()
    N_, C_ = predict_prob.size()

    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
        instance_normalize = N
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        instance_normalize = torch.sum(instance_level_weight) + epsilon
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    ce = -label * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * ce * class_level_weight) / float(instance_normalize)


# variational_beta_mixture.py

import numpy as np
from scipy.special import gammaln, digamma
from scipy.stats import beta
import copy

def log_beta_pdf(x, alpha, beta_):
    """
    log Beta(x | alpha, beta) = log[Gamma(alpha+beta)/Gamma(alpha)/Gamma(beta)]
                               + (alpha-1)*log(x) + (beta-1)*log(1-x)
    """
    return (  gammaln(alpha+beta_) - gammaln(alpha) - gammaln(beta_)
            + (alpha - 1)*np.log(x) + (beta_ - 1)*np.log(1 - x) )

def gamma_log_prior(alpha, a0, b0):
    # 以 Gamma(a0, b0) 为先验, 其对数形式 ~ (a0 - 1)*log(alpha) - b0*alpha
    if alpha <= 0:
        return -1e10
    return ( (a0 - 1)*np.log(alpha) - b0*alpha )

class VariationalBetaMixture1D:
    def __init__(self,
                 max_iters=50,
                 # 初始值
                 alphas_init=[1.,2.],
                 betas_init =[2.,1.],
                 weights_init=[0.5,0.5],
                 # VB先验超参
                 gamma=[1.,1.],   # Dirichlet(\gamma_1, \gamma_2)
                 a0=1.0, b0=0.5,  # Gamma(a0,b0)先验
                 step_size=1e-3
                 ):
        # 初始化
        self.max_iters = max_iters
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas  = np.array(betas_init,  dtype=np.float64)
        self.weights = np.array(weights_init, dtype=np.float64)  # 对应 \pi_1, \pi_2
        # 先验超参
        self.gamma = np.array(gamma, dtype=np.float64)
        self.a0, self.b0 = a0, b0
        self.step_size = step_size
        self.eps_nan = 1e-12

    def e_step_responsibility(self, x):
        r"""
        q(z_i = k) \propto pi_k * Beta(x_i | alpha_k, beta_k)
        """
        log_resp = []
        for k in range(2):
            log_pk = np.log(self.weights[k] + self.eps_nan) \
                     + log_beta_pdf(x, self.alphas[k], self.betas[k])
            log_resp.append(log_pk)
        log_resp = np.vstack(log_resp)  # shape (2, len(x))

        max_log = np.max(log_resp, axis=0, keepdims=True)
        log_resp -= max_log
        resp = np.exp(log_resp)
        resp_sum = np.sum(resp, axis=0, keepdims=True)
        resp /= resp_sum
        return resp  # shape (2, len(x))

    def m_step_params(self, x, resp):
        """
        对 \alpha_k,\beta_k 做近似VB更新:
          maximize( 期望对数似然 + 对数先验 ), 用梯度上升(MAP).
        """
        for k in range(2):
            alpha_k = self.alphas[k]
            beta_k = self.betas[k]
            r_k = resp[k]

            sum_rk_logx     = np.sum(r_k * np.log(x))
            sum_rk_log1mx   = np.sum(r_k * np.log(1.0 - x))
            psi_ab   = digamma(alpha_k + beta_k)
            psi_a    = digamma(alpha_k)
            psi_b    = digamma(beta_k)

            d_alpha = np.sum(r_k * (psi_ab - psi_a + np.log(x))) \
                      + ((self.a0 - 1)/alpha_k - self.b0)
            d_beta  = np.sum(r_k * (psi_ab - psi_b + np.log(1.0 - x))) \
                      + ((self.a0 - 1)/beta_k - self.b0)

            # alpha_k_new = alpha_k + self.step_size * d_alpha
            # beta_k_new  = beta_k  + self.step_size * d_beta
            alpha_k_new = alpha_k + self.step_size * (d_alpha - 0.01 * alpha_k)  # 加入正则项
            beta_k_new = beta_k + self.step_size * (d_beta - 0.01 * beta_k)      # 加入正则项

            alpha_k_new = max(alpha_k_new, 1e-6)
            beta_k_new  = max(beta_k_new,  1e-6)

            self.alphas[k] = alpha_k_new
            self.betas[k]  = beta_k_new

    def m_step_weights(self, resp):
        r"""
        Dirichlet先验对 \pi 做更新:
        posterior = Dirichlet(gamma_1 + sum(r_1), gamma_2 + sum(r_2))
        """
        sum_r = np.sum(resp, axis=1)
        gamma_post = self.gamma + sum_r
        gamma_sum = np.sum(gamma_post)
        self.weights = gamma_post / gamma_sum
        self.weights = np.maximum(self.weights, 1e-6)  # 最小权重限制
        self.weights /= self.weights.sum()

    def fit(self, x):
        x = np.copy(x)
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps]     = eps

        self.score_history = []

        for it in range(self.max_iters):
            # E-step
            resp = self.e_step_responsibility(x)
            # M-step
            self.m_step_params(x, resp)
            self.m_step_weights(resp)

            # 简单记录 -log似然(仅供观察)
            log_prob = []
            for i in range(len(x)):
                mixture = 0.
                for k in range(2):
                    mixture += self.weights[k]*beta.pdf(x[i], self.alphas[k], self.betas[k])
                log_prob.append(-np.log(mixture+1e-12))
            nll = np.sum(log_prob)
            self.score_history.append(nll)

            if it>1 and abs(self.score_history[-1] - self.score_history[-2])<1e-6:
                break
        return self

    def create_lookup(self, step=1):
        """
        在区间[0,1]中按某个步长采样, 预计算Beta PDF / Posterior等.
        """
        self.lookup_x = np.linspace(0.0, 1.0, 101)
        self.lookup_pdf_0 = beta.pdf(self.lookup_x, self.alphas[0], self.betas[0])
        self.lookup_pdf_1 = beta.pdf(self.lookup_x, self.alphas[1], self.betas[1])
        # ...

    def posterior(self, x, k):
        numerator = self.weights[k]*beta.pdf(x, self.alphas[k], self.betas[k])
        denominator = 0.
        for j in range(2):
            denominator += self.weights[j]*beta.pdf(x, self.alphas[j], self.betas[j])
        return numerator/(denominator+1e-12)

    def predict(self, x):
        post_1 = self.posterior(x, k=1)
        return (post_1 > 0.5).astype(np.int32)

    def __str__(self):
        return f"VariationalBetaMixture1D(weights={self.weights}, alphas={self.alphas}, betas={self.betas})"

if __name__=="__main__":
    data = np.random.rand(200)
    vbmm = VariationalBetaMixture1D()
    vbmm.fit(data)

    # 调用 create_lookup(1)
    vbmm.create_lookup(step=1)

    # 使用 vbmm 做后续推断或可视化
    print("Lookup PDF 0:", vbmm.lookup_pdf_0[:5])
    print("Lookup PDF 1:", vbmm.lookup_pdf_1[:5])

"""
Solutions for Question 1 of hwk3.
@author: Shawn Tan and Jae Hyun Lim
"""
import math
import numpy as np
import torch

torch.manual_seed(42)

def log_likelihood_bernoulli(mu, target):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Bernoulli random variables p(x=1).
    :param target: (FloatTensor) - shape: (batch_size x input_size) - Target samples (binary values).
    :return: (FloatTensor) - shape: (batch_size,) - log-likelihood of target samples on the Bernoulli random variables.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    target = target.view(batch_size, -1)
    
    # Calcul de la log-vraisemblance
    # log p(x|mu) = sum_i[x_i * log(mu_i) + (1 - x_i) * log(1 - mu_i)]
    ll_bernoulli = target * torch.log(mu + 1e-8) + (1 - target) * torch.log(1 - mu + 1e-8)
    
    # somme sur la dimension des pixels (input_size) pour obtenir une valeur par echantillon
    ll_bernoulli = ll_bernoulli.sum(dim=1)
    
    return ll_bernoulli


def log_likelihood_normal(mu, logvar, z):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Normal distributions.
    :param logvar: (FloatTensor) - shape: (batch_size x input_size) - The log variance of Normal distributions.
    :param z: (FloatTensor) - shape: (batch_size x input_size) - Target samples.
    :return: (FloatTensor) - shape: (batch_size,) - log probability of the sames on the given Normal distributions.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    logvar = logvar.view(batch_size, -1)
    z = z.view(batch_size, -1)

    # Calcul de la log-vraisemblance
    # log p(z | mu, sigma) = sum_i [ -1/2 * log(2pi) - 1/2 * logvar_i - (z_i - mu_i)^2 / (2 * exp(logvar_i)) ]
    log_2pi = torch.log(torch.tensor(2 * torch.pi))
    ll_normal = -0.5 * (log_2pi + logvar + ((z - mu) ** 2) / torch.exp(logvar))
    
    # Somme sur la dimension des composantes (input_size) pour obtenir une valeur par échantillon
    ll_normal = ll_normal.sum(dim=1)
    return ll_normal


def log_mean_exp(y):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param y: (FloatTensor) - shape: (batch_size x sample_size) - Values to be evaluated for log_mean_exp. For example log proababilies
    :return: (FloatTensor) - shape: (batch_size,) - Output for log_mean_exp.
    """
    # init
    batch_size = y.size(0)
    sample_size = y.size(1)

    # Calcul de a_i = max_k y_i^(k)
    a_i = torch.max(y, dim=1)[0]  # Shape: (batch_size,)

    # Calcul de exp(y_i^(k) - a_i)
    exp_terms = torch.exp(y - a_i.unsqueeze(1))  # Shape: (batch_size, sample_size)

    # Moyenne sur sample_size (1/K * sum_k exp(y_i^(k) - a_i))
    mean_exp = torch.mean(exp_terms, dim=1)  # Shape: (batch_size,)

    # Logarithme et ajout de a_i
    lme = torch.log(mean_exp) + a_i  # Shape: (batch_size,)


    return lme 


def kl_gaussian_gaussian_analytic(mu_q, logvar_q, mu_p, logvar_p):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    mu_q = mu_q.view(batch_size, -1)
    logvar_q = logvar_q.view(batch_size, -1)
    mu_p = mu_p.view(batch_size, -1)
    logvar_p = logvar_p.view(batch_size, -1)

    # Calcul de la divergence KL
    # D_KL = 1/2 * sum_i [ (logvar_p - logvar_q) - 1 + exp(logvar_q - logvar_p) + (mu_q - mu_p)^2 / exp(logvar_p) ]
    kl_gg = 0.5 * (
        logvar_p - logvar_q - 1 +
        torch.exp(logvar_q - logvar_p) +
        ((mu_q - mu_p) ** 2) / torch.exp(logvar_p)
    )

    # Somme sur la dimension input_size pour obtenir une valeur par échantillon
    kl_gg = kl_gg.sum(dim=1)


    return kl_gg


def kl_gaussian_gaussian_mc(mu_q, logvar_q, mu_p, logvar_p, num_samples=1):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :param num_samples: (int) - shape: () - The number of sample for Monte Carlo estimate for KL-divergence
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    input_size = np.prod(mu_q.size()[1:])
    mu_q = mu_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_q = logvar_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    mu_p = mu_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_p = logvar_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)

    # Échantillonnage Monte Carlo
    # z = mu_q + sigma_q * epsilon, où epsilon ~ N(0, I)
    sigma_q = torch.exp(0.5 * logvar_q)  # Shape: (batch_size, num_samples, input_size)
    epsilon = torch.randn(batch_size, num_samples, input_size, device=mu_q.device)
    z = mu_q + sigma_q * epsilon  # Shape: (batch_size, num_samples, input_size)

    # Calcul de log q(z|x)
    log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=mu_q.device))
    log_q = -0.5 * (log_2pi + logvar_q + ((z - mu_q) ** 2) / torch.exp(logvar_q))
    log_q = log_q.sum(dim=2)  # Somme sur input_size: (batch_size, num_samples)

    # Calcul de log p(z)
    log_p = -0.5 * (log_2pi + logvar_p + ((z - mu_p) ** 2) / torch.exp(logvar_p))
    log_p = log_p.sum(dim=2)  # Somme sur input_size: (batch_size, num_samples)

    # Estimation Monte Carlo de la KL
    kl_mc = (log_q - log_p).mean(dim=1)  # Moyenne sur num_samples: (batch_size,)


    return kl_mc

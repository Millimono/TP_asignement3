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

    #TODO: compute log_likelihood_bernoulli

    # On va  s'assurer que mu est dans [ε, 1−ε] pour éviter log(0)
    eps = 1e-6
    mu = torch.clamp(mu, eps, 1. - eps)

    # log-likelihood
    ll = target * torch.log(mu) + (1 - target) * torch.log(1 - mu)

    # Somme sur les dimensions d'entrée
    ll_bernoulli = ll.sum(dim=1)

   # raise NotImplementedError

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

    #TODO: compute log normal
    
    # Calcul de la variance à partir du log-variance
    var = torch.exp(logvar)

    # log proba 
    ll_elementwise = -0.5 * (math.log(2 * math.pi) + logvar + (z - mu) ** 2 / var)

    # On somme sur les dimensions d'entrée
    ll_normal = ll_elementwise.sum(dim=1)

    return ll_normal

    #raise NotImplementedError
    
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

    #TODO: compute log_mean_exp

    # Calcul de a_i
    a_i = torch.max(y, dim=1, keepdim=True)[0]  # max par ligne (sur l'ensemble des k)

    # pour chaque k et somme sur les k
    exp_term = torch.exp(y - a_i)
    mean_exp = torch.mean(exp_term, dim=1)  # moyenne sur la dimension des échantillons (k)

    # Ajouter a_i à la somme et appliquer le log
    lme = torch.log(mean_exp + 1e-6) + a_i.squeeze()  # ajouter epsilon pour éviter log(0)


    #raise NotImplementedError

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

    #TODO: compute kld

    #exponentielles
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)

    # Calcul des termes un par un
    log_var_ratio = logvar_q - logvar_p               # log(σ²_q / σ²_p)
    var_ratio = var_p / var_q                         # σ²_p / σ²_q
    mean_diff_term = ((mu_q - mu_p)**2) / var_q        # (μ_q - μ_p)² / σ²_q

    kl_elementwise = log_var_ratio - 1 + var_ratio + mean_diff_term

    kl_gg = 0.5 * kl_elementwise.sum(dim=1)   # On somme sur toutes les dimensions


    #raise NotImplementedError

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
    

    #TODO: compute kld
    #raise NotImplementedError
     #---------------------
    # Standard deviations # σ_q = exp(logvar_q / 2)
    std_q = torch.exp(0.5 * logvar_q)

    # Sample from q(z|x) using reparameterization trick
    eps = torch.randn_like(std_q)
    z = mu_q + std_q * eps

    # log q(z)
    log_qz = -0.5 * (logvar_q + np.log(2 * np.pi) + ((z - mu_q) ** 2) / torch.exp(logvar_q))
    log_qz = log_qz.sum(dim=-1)  # sum over dimensions

    # log p(z)
    log_pz = -0.5 * (logvar_p + np.log(2 * np.pi) + ((z - mu_p) ** 2) / torch.exp(logvar_p))
    log_pz = log_pz.sum(dim=-1)

    # KL estimation
    # KL(q||p) = log q(z) - log p(z)
    kl_mc = (log_qz - log_pz).mean(dim=1) 
    
    return kl_mc

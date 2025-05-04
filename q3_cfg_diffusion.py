# %%
import torch
import torch.utils.data
import torchvision
from torch import nn
from typing import Tuple, Optional
import torch.nn.functional as F
from tqdm import tqdm
from easydict import EasyDict
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
import os 

from cfg_utils.args import * 


class CFGDiffusion():
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.n_steps = n_steps
        
        # self.lambda_min = -20
        # self.lambda_max = 20
        # Définir lambda_min et lambda_max comme tenseurs
        self.lambda_min = torch.tensor(-20.0, device=device, dtype=torch.float)
        self.lambda_max = torch.tensor(20.0, device=device, dtype=torch.float)


    ### UTILS
    def get_exp_ratio(self, l: torch.Tensor, l_prim: torch.Tensor):
        return torch.exp(l-l_prim)
    
    def get_lambda(self, t: torch.Tensor): 
        # Calcule lambda_t basé sur le calendrier de bruit cosinus
        # t est dans [0, n_steps-1], mapper vers u dans [0,1]
        u = t / self.n_steps
        # Selon le papier : lambda = -2 * log(tan(a*u + b)), où
        device = t.device
        
        b = torch.atan(torch.exp(-self.lambda_max / 2))
        a = torch.atan(torch.exp(-self.lambda_min / 2)) - b
        lambda_t = -2 * torch.log(torch.tan(a * u + b))
        # Redimensionner en (batch_size, 1, 1, 1)
        return lambda_t.view(-1, 1, 1, 1)
        
    def alpha_lambda(self, lambda_t: torch.Tensor): 
        # Selon Eq. (1) : alpha_lambda^2 = 1 / (1 + exp(-lambda))
        var = 1 / (1 + torch.exp(-lambda_t))
        return var.sqrt()
    
    def sigma_lambda(self, lambda_t: torch.Tensor): 
        # Selon Eq. (1) : sigma_lambda^2 = 1 - alpha_lambda^2
        alpha_sq = 1 / (1 + torch.exp(-lambda_t))
        var = 1 - alpha_sq
        return var.sqrt()
    
    ## Forward sampling
    def q_sample(self, x: torch.Tensor, lambda_t: torch.Tensor, noise: torch.Tensor):
        # Selon Eq. (1) : z_lambda = alpha_lambda * x + sigma_lambda * epsilon
        alpha_t = self.alpha_lambda(lambda_t)
        sigma_t = self.sigma_lambda(lambda_t)
        z_lambda_t = alpha_t * x + sigma_t * noise

        return z_lambda_t
               
    def sigma_q(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        # Selon Eq. (2) : sigma_lambda'^2 = (1 - exp(lambda - lambda')) * sigma_lambda^2
        sigma_lambda = self.sigma_lambda(lambda_t)
        exp_ratio = self.get_exp_ratio(lambda_t, lambda_t_prim)
        var_q = (1 - exp_ratio) * sigma_lambda**2

    
        return var_q.sqrt()
    
    def sigma_q_x(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        # Selon Eq. (3) : sigma_lambda'^2 = (1 - exp(lambda - lambda')) * sigma_lambda'^2
        sigma_lambda_prim = self.sigma_lambda(lambda_t_prim)
        exp_ratio = self.get_exp_ratio(lambda_t, lambda_t_prim)
        sigma_sq = (1 - exp_ratio) * sigma_lambda_prim**2

        return sigma_sq.sqrt()

    ### REVERSE SAMPLING
    def mu_p_theta(self, z_lambda_t: torch.Tensor, x: torch.Tensor, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        # Selon Eq. (3) : mu_lambda'|lambda = exp(lambda - lambda') * (alpha_lambda' / alpha_lambda) * z_lambda + (1 - exp(lambda - lambda')) * alpha_lambda' * x
        alpha_lambda = self.alpha_lambda(lambda_t)
        alpha_lambda_prim = self.alpha_lambda(lambda_t_prim)
        exp_ratio = self.get_exp_ratio(lambda_t, lambda_t_prim)
        mu = exp_ratio * (alpha_lambda_prim / alpha_lambda) * z_lambda_t + (1 - exp_ratio) * alpha_lambda_prim * x
        return mu

    def var_p_theta(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor, v: float=0.3):
        # Selon Eq. (4) : var = (sigma_lambda'|lambda^2)^(1-v) * (sigma_lambda|lambda'^2)^v
        sigma_q = self.sigma_q(lambda_t, lambda_t_prim)
        sigma_q_x = self.sigma_q_x(lambda_t, lambda_t_prim)
        var = (sigma_q_x**2)**(1 - v) * (sigma_q**2)**v
        return var
    
    def p_sample(self, z_lambda_t: torch.Tensor, lambda_t : torch.Tensor, lambda_t_prim: torch.Tensor,  x_t: torch.Tensor, set_seed=False):
        # TODO: Write a function that sample z_{lambda_t_prim} from p_theta(•|z_lambda_t) according to (4) 
        # Note that x_t correspond to x_theta(z_lambda_t)
        # Échantillonne z_lambda_t_prim depuis p_theta(z_lambda_t_prim | z_lambda_t) en utilisant Eq. (4)
        if set_seed:
            torch.manual_seed(42)
        mu = self.mu_p_theta(z_lambda_t, x_t, lambda_t, lambda_t_prim)
        var = self.var_p_theta(lambda_t, lambda_t_prim)
        noise = torch.randn_like(z_lambda_t)
        sample = mu + var.sqrt() * noise
    
        return sample 

    ### LOSS
    def loss(self, x0: torch.Tensor, labels: torch.Tensor, noise: Optional[torch.Tensor] = None, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        batch_size = x0.shape[0]
        dim = tuple(range(1, x0.ndim))
        t = torch.randint(
            0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long
        )
        
       # Échantillonne z_lambda_t en utilisant q_sample
        lambda_t = self.get_lambda(t)
        
        if noise is None:
            noise = torch.randn_like(x0)
        
        z_lambda_t = self.q_sample(x0, lambda_t, noise)

        # Calcule le bruit prédit
        epsilon_pred = self.eps_model(z_lambda_t, labels) if labels is not None else self.eps_model(z_lambda_t, None)
        
        # Calcule la perte MSE
        loss = (epsilon_pred - noise) ** 2
        loss = loss.sum(dim=dim).mean()
        return loss



    
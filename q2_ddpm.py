import torch 
from torch import nn 
from typing import Optional, Tuple


class DenoiseDiffusion():
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta



    ### UTILS
    def gather(self, c: torch.Tensor, t: torch.Tensor):
        c_ = c.gather(-1, t)
        return c_.reshape(-1, 1, 1, 1)

    ### FORWARD SAMPLING
    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Moyenne : sqrt(alpha_bar_t) * x0
        mean = torch.sqrt(self.gather(self.alpha_bar, t)) * x0
        # Variance : (1 - alpha_bar_t) * I
        var = self.gather(1.0 - self.alpha_bar, t)

        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None:
            eps = torch.randn_like(x0)
        #  return x_t sampled from q(•|x_0) according to (1)
        # Échantillonnage : sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
        mean, var = self.q_xt_x0(x0, t)
        sample = mean + torch.sqrt(var) * eps
        return sample


    ### REVERSE SAMPLING
    def p_xt_prev_xt(self, xt: torch.Tensor, t: torch.Tensor):
        #  return mean and variance of p_theta(x_{t-1} | x_t) according to (2)
        # Prédiction du bruit par le modèle
        eps_theta = self.eps_model(xt, t)
        # Coefficients
        alpha_t = self.gather(self.alpha, t)
        beta_t = self.gather(self.beta, t)
        alpha_bar_t = self.gather(self.alpha_bar, t)
        # Moyenne : (1/sqrt(alpha_t)) * (xt - (beta_t/sqrt(1-alpha_bar_t)) * eps_theta)
        mu_theta = (xt - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_theta) / torch.sqrt(alpha_t)
        # Variance : beta_t
        var = beta_t
        return mu_theta, var

    #  sample x_{t-1} from p_theta(•|x_t) according to (3)
    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        # Calculer la moyenne et la variance
        mu_theta, var = self.p_xt_prev_xt(xt, t)
        # Échantillonnage : mu_theta + sqrt(var) * z
        z = torch.randn_like(xt) if not set_seed else torch.randn_like(xt)
        sample = mu_theta + torch.sqrt(var) * z

        return sample

    ### LOSS
    # : compute loss according to (4)
    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        batch_size = x0.shape[0]
        dim = list(range(1, x0.ndim))
        t = torch.randint(
            0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long
        )
        if noise is None:
            noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        eps_theta = self.eps_model(xt, t)
        assert noise.shape == eps_theta.shape, f"Noise shape {noise.shape} doesn't match eps_theta shape {eps_theta.shape}"
        error = (noise - eps_theta) ** 2
        loss = torch.mean(error) * torch.prod(torch.tensor(x0.shape[1:]).float())       
        return loss

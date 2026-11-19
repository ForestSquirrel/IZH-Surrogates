"""
model.py — gated SwiGLU embedder + FrEIA conditional normalizing flow.

Layout (see cinn.md §4):

    cond (2F = 44 dims)                theta = (a,b,c,d) standardized (4 dims)
          |                                        |
     Gated embedder                         Coupling-block stack
     (residual SwiGLU)  --- H (embed_hidden) ---> (conditioned on H)
          |                                        |
                                                   v
                                         z ~ N(0, I), 4-dim latent

The bijection is between theta (4D) and z (4D) only; H merely conditions each
coupling subnet, so there is no dimension-padding bookkeeping. Each block is an
AllInOneBlock-style unit: ActNorm -> learnable soft (orthogonal) permutation ->
affine or rational-quadratic-spline coupling. Spline blocks are interleaved
every `spline_every`-th block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import FrEIA.framework as Ff
import FrEIA.modules as Fm

from spline import RQSplineCoupling


# ============================================================
# Gated embedder (condition network)
# ============================================================

class SwiGLU(nn.Module):
    """SwiGLU feed-forward: split a projection into (gate, value); SiLU(gate)*value."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.in_proj = nn.Linear(dim, 2 * hidden)
        self.out_proj = nn.Linear(hidden, dim)

    def forward(self, x):
        gate, value = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(F.silu(gate) * value)


class ResidualSwiGLU(nn.Module):
    """LayerNorm -> SwiGLU -> residual add (keeps gradients sane across 3-4 layers)."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = SwiGLU(dim, hidden)

    def forward(self, x):
        return x + self.ff(self.norm(x))


class Embedder(nn.Module):
    """
    44 -> embed_hidden gated embedder. The (value, mask) input layout lets each
    SwiGLU gate route around undefined-feature slots using the mask channel as
    the natural gating signal. Output H conditions every coupling subnet.
    """

    def __init__(self, in_dim, hidden, layers):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList(ResidualSwiGLU(hidden, hidden) for _ in range(layers))
        self.out_norm = nn.LayerNorm(hidden)

    def forward(self, x):
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.out_norm(h)


# ============================================================
# Learnable soft permutation (orthogonal mixing, logdet = 0)
# ============================================================

class SoftPermute(Fm.InvertibleModule):
    """
    Learnable orthogonal channel mixing via the matrix exponential of a
    skew-symmetric parameter: Q = expm(W - W^T) is orthogonal with det = +1, so
    its log-Jacobian determinant is exactly 0. At only 4 dims a learned mixing
    matters more than a fixed random permutation (cinn.md §4.2).
    """

    def __init__(self, dims_in, dims_c=None):
        super().__init__(dims_in, dims_c)
        self.n = dims_in[0][0]
        self.w = nn.Parameter(0.01 * torch.randn(self.n, self.n))

    def _orthogonal(self):
        skew = self.w - self.w.t()
        return torch.matrix_exp(skew)

    def forward(self, x, c=None, rev=False, jac=True):
        x = x[0]
        Q = self._orthogonal()
        # forward: y = x Q^T ;  inverse: x = y Q  (Q orthogonal => Q^{-1} = Q^T)
        y = x @ Q if rev else x @ Q.t()
        logdet = x.new_zeros(x.shape[0])
        return (y,), logdet

    def output_dims(self, input_dims):
        return input_dims


# ============================================================
# Coupling-subnet constructor
# ============================================================

def _make_subnet(hidden):
    """
    FrEIA subnet_constructor(c_in, c_out) -> MLP. The final layer is zero-init'd
    so every coupling block starts as (near-)identity, which stabilizes the first
    few hundred steps of flow training.
    """
    def subnet(c_in, c_out):
        net = nn.Sequential(
            nn.Linear(c_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, c_out),
        )
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
        return net
    return subnet


# ============================================================
# Flow builder
# ============================================================

def build_inn(cfg, param_dim, cond_dim):
    """
    Build the conditioned SequenceINN over `param_dim` channels, each coupling
    block conditioned on the `cond_dim`-wide embedder output H.
    """
    subnet = _make_subnet(cfg.coupling_hidden)

    def is_spline(k):
        # spline_every: every k-th block is a spline block; interleaved uniformly.
        # 0 or >= n_blocks -> all-affine baseline; 1 -> all spline.
        se = cfg.spline_every
        if se <= 0 or se >= cfg.n_coupling_blocks:
            return False
        return (k % se) == (se - 1)

    inn = Ff.SequenceINN(param_dim)
    for k in range(cfg.n_coupling_blocks):
        inn.append(Fm.ActNorm)
        if cfg.permute_soft:
            inn.append(SoftPermute)
        else:
            inn.append(Fm.PermuteRandom)

        if is_spline(k):
            inn.append(RQSplineCoupling, cond=0, cond_shape=(cond_dim,),
                       subnet_constructor=subnet, bins=cfg.spline_bins)
        else:
            inn.append(Fm.GLOWCouplingBlock, cond=0, cond_shape=(cond_dim,),
                       subnet_constructor=subnet, clamp=2.0)
    return inn


# ============================================================
# Full cINN
# ============================================================

class CINN(nn.Module):
    """Embedder + conditioned flow. Forward maps (theta, cond) -> (z, log|det J|)."""

    def __init__(self, cfg, cond_dim, param_dim=4):
        super().__init__()
        self.param_dim = param_dim
        self.cond_dim = cond_dim
        self.embedder = Embedder(cond_dim, cfg.embed_hidden, cfg.embed_layers)
        self.inn = build_inn(cfg, param_dim, cfg.embed_hidden)

    def embed(self, cond):
        return self.embedder(cond)

    def forward(self, theta, cond):
        """theta (B,4) standardized, cond (B,2F) -> z (B,4), log_jac_det (B,)."""
        H = self.embedder(cond)
        z, log_jac_det = self.inn(theta, c=[H])
        return z, log_jac_det

    def nll(self, theta, cond):
        """Per-sample negative log-likelihood: 0.5||z||^2 - log|det J| (+ const)."""
        z, log_jac_det = self.forward(theta, cond)
        return 0.5 * z.pow(2).sum(dim=1) - log_jac_det

    @torch.no_grad()
    def sample(self, cond_row, n_samples, temperature=1.0):
        """
        Draw `n_samples` posterior theta (standardized) for ONE conditioning row.
        cond_row: (1, 2F) or (2F,). Returns (n_samples, 4).

        `temperature` scales the latent draw to z ~ N(0, T^2 I). T = 1 is the
        posterior the model was trained to represent and is the only setting that
        is statistically correct -- anything else is a deliberately wrong, wider
        proposal.

        T > 1 exists for OUT-OF-DISTRIBUTION conditioning, where the sharpness
        that a low NLL buys becomes a liability. On a target whose features fall
        outside the atlas (a foreign neuron model), the condition is not something
        the flow has ever seen, so it concentrates confidently on the wrong place.
        Downstream consumers like run_surrogate.py do not want the density anyway:
        they use these draws as a PROPOSAL for a rejection search scored against
        the real Izhikevich simulator, and a proposal is only useful if it covers.
        Measured on Hindmarsh-Rose against cinn_model_1e8_v2 (4096 draws, scored
        with full feature overlap), the target's p=2 regime is unreachable at
        T=1 and comes back as T rises:

            T=1  0 draws at p=2      T=2   5 draws, best d2 91.5
            T=3  27 draws, best d2 11.6

        It is not free: widening also proposes more non-firing theta, so the
        fraction of draws that score against the full feature set falls (2216 ->
        1193 of 4096 between T=1 and T=3). Raise it only for OOD work, and never
        for a calibration or density claim.
        """
        if cond_row.dim() == 1:
            cond_row = cond_row.unsqueeze(0)
        H = self.embedder(cond_row)                       # (1, embed_hidden)
        H = H.expand(n_samples, -1).contiguous()          # (n, embed_hidden)
        z = torch.randn(n_samples, self.param_dim, device=H.device, dtype=H.dtype)
        if temperature != 1.0:
            z = z * temperature
        theta_std, _ = self.inn(z, c=[H], rev=True)
        return theta_std

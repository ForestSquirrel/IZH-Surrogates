"""
spline.py — conditioned rational-quadratic spline (RQS) coupling as a FrEIA
InvertibleModule.

FrEIA 0.2 ships affine (GLOW) coupling but no spline coupling, and cinn.md calls
for interleaved rational-quadratic spline capacity (a spline block wherever a
sharp/graded parameter boundary happens to sit). This is a self-contained monotone
RQS transform (Durkan et al., "Neural Spline Flows", 2019) with linear tails
outside [-B, B], wrapped as a one-sided coupling: the identity half (plus the
conditioning H) drives the spline parameters for the transformed half, and the
SoftPermute mixing between blocks ensures every channel eventually gets warped.
"""

import numpy as np
import torch
import torch.nn.functional as F

import FrEIA.modules as Fm

_MIN_BIN_WIDTH = 1e-3
_MIN_BIN_HEIGHT = 1e-3
_MIN_DERIVATIVE = 1e-3


def _searchsorted(bin_locations, inputs):
    """Index of the bin containing each input, in [0, num_bins-1].

    The raw count is num_bins+1 for an input sitting exactly on the last knot
    (x == +tail_bound is reachable in float32 and the tail mask below is
    inclusive), which would gather one past the last bin — a device-side assert
    on CUDA. Clamping puts that input in the last bin, where the RQS maps it to
    the knot value itself, so the transform stays continuous with the tail.
    """
    num_bins = bin_locations.shape[-1] - 1
    idx = torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1
    return idx.clamp(0, num_bins - 1)


def _rqs(inputs, unnorm_w, unnorm_h, unnorm_d, inverse,
         left, right, bottom, top):
    """Core monotone RQS on the finite interval, all inputs guaranteed inside."""
    num_bins = unnorm_w.shape[-1]

    widths = F.softmax(unnorm_w, dim=-1)
    widths = _MIN_BIN_WIDTH + (1 - _MIN_BIN_WIDTH * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, (1, 0), value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = _MIN_DERIVATIVE + F.softplus(unnorm_d)

    heights = F.softmax(unnorm_h, dim=-1)
    heights = _MIN_BIN_HEIGHT + (1 - _MIN_BIN_HEIGHT * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, (1, 0), value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = _searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = _searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    input_heights = heights.gather(-1, bin_idx)[..., 0]
    delta = input_heights / input_bin_widths

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    if inverse:
        a = ((inputs - input_cumheights) * (input_derivatives
             + input_derivatives_plus_one - 2 * delta)
             + input_heights * (delta - input_derivatives))
        b = (input_heights * input_derivatives
             - (inputs - input_cumheights) * (input_derivatives
             + input_derivatives_plus_one - 2 * delta))
        c = -delta * (inputs - input_cumheights)
        discriminant = b.pow(2) - 4 * a * c
        discriminant = torch.clamp(discriminant, min=0.0)
        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = delta + ((input_derivatives + input_derivatives_plus_one
                                - 2 * delta) * theta_one_minus_theta)
        derivative_numerator = delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)
        numerator = input_heights * (delta * theta.pow(2)
                                     + input_derivatives * theta_one_minus_theta)
        denominator = delta + ((input_derivatives + input_derivatives_plus_one
                                - 2 * delta) * theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, logabsdet


def unconstrained_rqs(inputs, unnorm_w, unnorm_h, unnorm_d, inverse, tail_bound):
    """RQS on [-B, B] with linear (identity) tails outside; params are the raw
    subnet outputs: widths (K), heights (K), inner derivatives (K-1).

    Branch-free by construction. The obvious implementation guards the spline
    call with ``if inside.any():`` and gathers with boolean masks, but that costs
    a device->host sync and a data-dependent output shape on *every* block --- a
    hard graph break under torch.compile, and 10 of them per forward once every
    block is a spline block. Running `_rqs` on the clamped input unconditionally
    and selecting with `torch.where` is measurably faster (2.5x end-to-end at
    all-spline) and numerically identical: the discarded lane is exactly the
    identity tail either way.

    The clamp is on the *input*, not the output. `torch.where` alone would still
    backprop NaNs from the unselected branch; `_rqs` restricted to [-B, B] is
    finite everywhere, so there is nothing to poison the gradient.
    """
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)

    # pad inner derivatives with the tail-matching boundary derivative (=1)
    unnorm_d = F.pad(unnorm_d, (1, 1))
    constant = float(np.log(np.exp(1 - _MIN_DERIVATIVE) - 1))
    unnorm_d[..., 0] = constant
    unnorm_d[..., -1] = constant

    out_in, ld_in = _rqs(
        inputs.clamp(-tail_bound, tail_bound), unnorm_w, unnorm_h, unnorm_d,
        inverse=inverse,
        left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound)

    # outside: identity with unit derivative (logabsdet 0)
    zero = torch.zeros((), device=inputs.device, dtype=inputs.dtype)
    return torch.where(inside, out_in, inputs), torch.where(inside, ld_in, zero)


class RQSplineCoupling(Fm.InvertibleModule):
    """
    One-sided rational-quadratic spline coupling conditioned on H.

    The first half of the channels passes through unchanged and, concatenated
    with the condition, feeds the subnet that produces the spline parameters for
    the second half. Pair this with SoftPermute between blocks so both channels
    get warped over the stack.
    """

    def __init__(self, dims_in, dims_c=None, subnet_constructor=None,
                 bins=8, tail_bound=5.0):
        super().__init__(dims_in, dims_c)
        self.channels = dims_in[0][0]
        self.split_a = self.channels // 2
        self.split_b = self.channels - self.split_a
        self.bins = bins
        self.tail_bound = tail_bound
        cond_ch = dims_c[0][0] if dims_c else 0
        # per transformed channel: K widths + K heights + (K-1) inner derivatives
        self.params_per_dim = 3 * bins - 1
        self.subnet = subnet_constructor(
            self.split_a + cond_ch, self.split_b * self.params_per_dim)

    def forward(self, x, c=None, rev=False, jac=True):
        x = x[0]
        xa, xb = x[:, :self.split_a], x[:, self.split_a:]
        cond = c[0] if c else None
        net_in = torch.cat([xa, cond], dim=1) if cond is not None else xa
        params = self.subnet(net_in)
        params = params.view(x.shape[0], self.split_b, self.params_per_dim)
        uw = params[..., :self.bins]
        uh = params[..., self.bins:2 * self.bins]
        ud = params[..., 2 * self.bins:]

        yb, logabsdet = unconstrained_rqs(
            xb, uw, uh, ud, inverse=rev, tail_bound=self.tail_bound)
        y = torch.cat([xa, yb], dim=1)
        logdet = logabsdet.sum(dim=1)
        return (y,), logdet

    def output_dims(self, input_dims):
        return input_dims

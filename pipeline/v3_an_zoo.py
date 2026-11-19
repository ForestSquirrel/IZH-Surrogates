"""
v3_an_zoo.py -- the out-of-family target set, as V3Spec objects.

Six targets. Per-model `v_th`, rheobase bracket, step and bootstrap window are
by design, not tuning slack: an out-of-family neuron has its own units, and the
harness reports feature definedness so a starved extraction is visible rather
than silently scored.

ACKNOWLEDGED LIMITATION -- the HR BURSTING BAND is out of domain, quantified.

§51's bifurcation maps settle this. At r = 0.015 the period changes 62 times
across an I range of 1.5, with single-grid-cell windows; in a box 0.004 x 0.3 the
mean drive-distance between period changes is 0.0018, i.e. ~0.04% of the
amplitude. The schema's protocols span gamma = 1.02 to 4.0, so ONE HR cell hands
them three different regimes at once (at r = 0.020: period 2 at gamma=1.2, 3 at
1.02, 4 at 2.0). The 18-dim vector is a collage of several dynamical systems, not
a description of one.

Nothing in the pipeline fixes that: a single shared amplitude would delete six
columns, §50's auto-widening already reached 16x, and finer DT does not touch it
because the aperiodicity is the model's own. HR targets are kept and scored, and
their scores are honest measurements of an unmatchable target rather than
evidence of pipeline failure.

"HR tonic" is additionally absent for a separate reason.

At gamma = 2*i_rh the Hindmarsh-Rose tonic parameterization fires 7 spikes and
none after the discard window: an onset burst, then silence. There is no settled
cycle, so T_cycle is undefined and the target has no anchor. The cause is
inherited, not a v3 regression -- rheobase means ">= N_MIN_SPIKES anywhere in
the window" in v2 and v3 alike, so a pure onset transient qualifies, and v2
survived it only because `t_ref` (mean of last <=5 ISIs) can be read off a
handful of transient spikes where v3's settled-window T_cycle cannot.

Redefining rheobase as SUSTAINED firing would fix the class, and was rejected
here for a specific reason: it changes i_rh for the Izhikevich atlas too, which
invalidates the 29-hour build. The regime is excluded and quantified instead.
HR BURSTING is included and is the regime Hindmarsh-Rose is actually used for.
"""
import numpy as np

from v3_an_features import V3Spec


def reset_targets():
    """[(name, rhs, params)] for models that RESET, run through v3's Izhikevich
    path rather than the reset-free one.

    AdEx qualifies exactly: 2-D state, spike declared at V_TH = 30, and the reset
    is V -> params[2], w -> w + params[3] -- the same convention v3_protocols
    already applies for (c, d). The IC v3 builds, y[1] = params[1]*V_REST, is 0
    for AdEx's params[1] = 0, which is AdEx's correct w(0). So no new kernel is
    needed; only the right entry point.

    Four Naud-style regimes at fixed a / tau_w, as in v2's target set.
    """
    from models_offamily import adex_rhs, ADEX_TARGETS
    # rheo_hi = 2000 pA: AdEx is in pA with C = 200 pF, so its rheobase runs
    # 150-400 -- v3's Izhikevich-tuned 200 clipped the two bursting regimes.
    return [(lbl, adex_rhs, np.asarray(pr, float), 2000.0)
            for lbl, pr in ADEX_TARGETS]


def targets():
    """[(name, V3Spec)]. Reset-free models only -- a RESET model diverges here
    and belongs in reset_targets()."""
    from models_offamily import resetfree_specs
    out = []
    for name, ns in resetfree_specs():
        if name == "HR tonic":
            continue                      # see the module docstring
        out.append((name, V3Spec(
            rhs=ns.rhs, y0=np.asarray(ns.y0, float),
            params=np.asarray(ns.params, float), v_th=ns.v_th,
            rheo_lo=ns.rheo_lo, rheo_hi=ns.rheo_hi, h=ns.h,
            boot_t_trans=ns.t_discard, boot_t_win=ns.t_win,
            boot_t_disc=ns.t_discard, name=name)))
    # ---- Wilson, at three spike-detection thresholds -----------------------
    # v_th is not a property of the model; it is where we choose to call a spike,
    # and Wilson's V is in units of 100 mV so these are 0, -25 and -50 mV. §49
    # showed 0 and -0.5 give the SAME 21 spikes, so Wilson's p=2 doublets are real
    # dynamics rather than a detection artifact -- the ladder is kept so that
    # remains checkable rather than asserted. Wilson lives in
    # sweeps.resetfree_sweeps(), not models_offamily; rheo_hi = 1.5 because it
    # enters depolarization block above ~1.5 in these units (bracket scan, §8.1).
    from sweeps import wilson_rhs, hr_rhs
    for vth in (0.0, -0.25, -0.5):
        tag = "Wilson" if vth == 0.0 else f"Wilson vth{vth:g}"
        out.append((tag, V3Spec(
            rhs=wilson_rhs, y0=np.array([-0.7, 0.088, 0.562, 1.687]),
            params=np.array([26.0, 1.0, 13.0, 0.0]), v_th=vth,
            rheo_lo=0.0, rheo_hi=1.5, h=0.005,
            boot_t_trans=100.0, boot_t_win=800.0, boot_t_disc=100.0, name=tag)))

    # ---- Hindmarsh-Rose, three burst regimes -------------------------------
    # params = (r, s, x_r). `r` sets the slow-variable speed and therefore the
    # burst structure. Chosen for REGIME diversity rather than to span the range:
    # a scan gives p = 1 at r = 0.008, p = 0 (aperiodic) at 0.015, p = 2 at 0.020
    # and p = 1 again at 0.030, so 0.008/0.015/0.020 covers three distinct
    # regimes where 0.008/0.015/0.030 would cover two.
    #
    # r <= 0.006 is out of domain: at gamma = 2*i_rh it fires an onset burst and
    # stops, so there is no settled cycle to anchor. Confirmed rather than
    # assumed -- the auto-widening (§50) took the window to 16x and it still
    # returned no anchor. Same limitation as HR tonic, see the module docstring.
    x_r = -1.6
    for r in (0.008, 0.015, 0.020):
        tag = f"Hindmarsh-Rose r={r:g}"
        out.append((tag, V3Spec(
            rhs=hr_rhs, y0=np.array([x_r, 1.0 - 5.0*x_r*x_r, 0.0]),
            params=np.array([r, 7.0, x_r, 0.0]), v_th=1.0,
            rheo_lo=0.0, rheo_hi=5.0, h=0.01,
            boot_t_trans=200.0, boot_t_win=2000.0, boot_t_disc=200.0, name=tag)))
    return out

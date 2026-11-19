"""
encode.py -- stimulus events for the input neurons (times on the 1 ms grid of the paper).

MNIST: pixel / 255 is the per-ms probability that the input neuron receives a pulse
(CODE: ceil(p - U[0,1)) each timestep), for T_SAMPLE ms.
N-MNIST: every event within the first 250 ms is a pulse at its 1 ms bin (A.1).
"""
import numpy as np
from numba import njit
import spec as S


def mnist_events(img, rng):
    """img uint8 (784,) -> (times ms, channels)"""
    p = img.astype(np.float32) / 255.0
    T = int(S.T_SAMPLE)
    hit = rng.random((T, len(p))) < p
    t, ch = np.nonzero(hit)
    return t.astype(np.float64), ch.astype(np.int32)


def nmnist_events(t_ms, ch):
    tb = np.floor(t_ms).astype(np.float64)
    key = np.unique(tb.astype(np.int64) * 4096 + ch)        # one pulse per (ms, channel)
    return (key // 4096).astype(np.float64), (key % 4096).astype(np.int32)


@njit(cache=True)
def _lif_input(t_ms, ch, nin, T, inv_tau, theta, tref, amp):
    """Ivanov's LIF input neurons (Euler, 1 ms) driven by 1 ms pulses of amp at the stimulus
    events; as in the swapped network, a pulse acts on the step after its event.
    -> (spike times ms, channels)"""
    drive = np.zeros((T, nin), np.float32)
    for k in range(len(t_ms)):
        s = int(t_ms[k]) + 1
        if s < T:
            drive[s, ch[k]] += amp
    out_t = np.empty(T * nin, np.float64)
    out_c = np.empty(T * nin, np.int32)
    n = 0
    for c in range(nin):
        v = np.float32(0.0)
        r = 0
        for s in range(T):
            v = v - v * inv_tau + drive[s, c]
            if r > 0:
                r -= 1
            elif v >= theta:
                v = np.float32(0.0)
                r = tref
                out_t[n] = s
                out_c[n] = c
                n += 1
    return out_t[:n], out_c[:n]


def lif_input_spikes(events, nin):
    """stimulus events -> spike trains of the paper's LIF input layer (LIF-input control)"""
    t, ch = events
    return _lif_input(np.asarray(t, np.float64), np.asarray(ch, np.int32), nin, int(S.T_SAMPLE),
                      np.float32(S.DT_LIF / S.TAU_V), np.float32(S.THETA), int(round(S.T_REF / S.DT_LIF)),
                      np.float32(S.I_PULSE))

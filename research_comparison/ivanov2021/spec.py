"""
spec.py -- Ivanov & Michmizos 2021 (NeurIPS, arXiv 2111.01760), baseline LSM (static weights,
no STDP, no astrocyte). PAPER = text/appendix; CODE = github.com/combra-lab/NALSM, used only
where the paper is silent; CHOSEN = ours.
"""
# liquid (§2.1, App. A.3)
N_LIQ = 1000
SHAPE = (10, 10, 10)            # CHOSEN (code takes the dims as arguments; 1000 = 10^3)
F_INH = 0.2
C = ((0.2, 0.1), (0.05, 0.3))   # [pre][post] 0 E / 1 I: EE .2, EI .1, IE .05, II .3 (A.3)
LAM = 3.0
IN_DENSITY = 0.15               # input -> liquid, random sign (§2.1)
# LIF (Table S1)
THETA, TAU_V, TAU_U, BIAS, T_REF = 20.0, 64.0, 1.0, 0.0, 2.0
DT_LIF = 1.0                    # ms; 250 timesteps = 250 ms (A.1)
# input neurons (CODE: INPUT_CURRENT 100, one-step pulse when Bernoulli(pixel) fires)
I_PULSE = 100.0
T_PULSE = 1.0                   # ms
T_SAMPLE = 250.0                # ms
# readout (§2.2)
BATCH, LR, L2, MAX_EPOCHS = 250, 0.1, 5e-10, 5000
SPLIT = (50000, 10000, 10000)   # train / val (last 10k of train) / test (A.1)
# liquid-weight sweep (§2.3)
W_SWEEP = {"mnist": (0.4, 1.2), "nmnist": (0.8, 1.35)}
N_NETS = 10
REF = {"mnist": dict(top=95.44), "nmnist": dict(top=95.35)}    # baseline LSM, Table 1 / §3

# time-scaled input neurons ('@fastin', input_speed.py): speed s per zoo model, pulse
# A = max(2 thr_s, 2 i_rh) so single pulses fire and fused pulses stay in the DC firing band
FASTIN_SPEED = {"hh": 4, "ml_c1": 32, "adex_adapting": 16}

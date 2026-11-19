"""
spec.py -- Schulte to Brinke, Duarte & Morrison 2022 (Front. Integr. Neurosci. 16:923468),
data-based circuit (replication of Haeusler & Maass 2007), spike-pattern tasks.

Sources: Fig 1 (populations, PSP amplitudes A [mV], probabilities), Tables 1-5, §2.2.
Where the paper and the authors' code (Zenodo 7037649) differ we follow the PAPER unless
the paper gives no value; CODE marks values taken from the code because the paper is
silent; CHOSEN marks our own choices. Units: ms, mV, nS, pA.
"""
N = 560
POPS = {"E23": 134, "I23": 34, "E4": 89, "I4": 23, "E5": 224, "I5": 56}   # Fig 1
PTYPE = {k: (0 if k[0] == "E" else 1) for k in POPS}

# Fig 1: [src][dst] = (A mV, p)
CONN = {
    "E23": {"E23": (1.7, .26), "I23": (1.9, .21), "I4": (1.6, .08), "E5": (1.4, .55)},
    "I23": {"E23": (0.65, .16), "I23": (1.35, .25), "E5": (5.2, .20)},
    "E4":  {"E23": (4.0, .28), "I23": (0.15, .10), "E4": (1.1, .17), "I4": (3.7, .19)},
    "I4":  {"E23": (1.75, .50), "I23": (1.5, .20), "E4": (0.85, .10), "I4": (1.55, .50)},
    "E5":  {"E23": (0.3, .03), "E5": (1.7, .09), "I5": (0.9, .10)},
    "I5":  {"E5": (1.2, .12), "I5": (1.2, .60)},
}
IN1 = {"E23": 0.2, "E4": 0.8, "I4": 0.5, "E5": 0.1}      # Fig 1, stream 1
IN2 = {"E23": 0.2}                                        # Fig 1, stream 2
A_IN = 1.924779612734342                                  # mV (Table 4, code value)

# weights (Table 4, eq. 27-28): w = S * A * gL / |E - V_mean|
G_L, V_MEAN, E_EX, E_IN = 15.5862, -65.0, 0.0, -75.0
S_RW = 66825.0 / N                                        # code value (paper 60000/N)
S1, S2 = 14.85, 36.498                                    # code values (paper 14, 33)
W_CV = 0.7                                                # b_w

# Table 3: [src type][dst type] -> (U, tau_rec ms, tau_fac ms); delays ms
STP = (((0.5, 1100.0, 50.0), (0.05, 125.0, 1200.0)),
       ((0.25, 700.0, 20.0), (0.32, 144.0, 60.0)))
DELAY = ((1.5, 0.8), (0.8, 0.8))
UDF_CV, D_CV = 0.5, 0.1                                   # b_U = b_rec = b_fac, b_d
DELAY_IN = 1.0      # CODE: input synapses are static NEST synapses with the default delay

TAU_SYN = (3.0, 6.0)                                      # Table 1, exc / inh
NOISE = dict(tau=(2.7, 10.5), sig=(3.0, 6.6))             # Table 2 (means are in the neuron)
V0 = (-70.0, -60.0)                                       # Table 1, uniform

# tasks (§2.2, Table 5)
N_CH = 40           # spike trains per stream (pattern tasks)
RATE = 20.0         # Hz, CODE (paper: not stated in the replication)
N_SEG, T_SEG = 15, 30.0
JITTER = 1.0        # ms SD, once per trial, same for every target neuron
N_TRAIN, N_TEST = 1500, 300
DISCARD_SEG = 1     # CODE: one 30 ms segment discarded (needed by the delayed task)
TAU_FILTER = 15.0   # readout spike filter
READOUTS = ("E23", "E5")    # Table 6 rows L23 / L5 (text says "layer 4"; code and Table 6: L5)
THRESH = 0.5
N_RUNS = 20

DT = 0.1            # CHOSEN (paper: NEST 0.2 ms); HH integrated with 4 RK4 substeps of 0.025

# Table 6, data-based column (kappa; 20 runs, HH with noise)
REF = {"tcl1 L23": 0.87, "tcl2 L23": 0.947, "tcl1 L5": 0.693, "tcl2 L5": 0.968,
       "tcl1(t-1) L23": 0.389, "tcl2(t-1) L23": 0.352, "tcl1(t-1) L5": 0.418,
       "tcl2(t-1) L5": 0.49, "XOR L23": 0.304, "XOR L5": 0.383}

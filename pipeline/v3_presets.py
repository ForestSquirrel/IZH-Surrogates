"""
v3_presets.py -- Izhikevich's twenty canonical regimes.

Lifted verbatim from `src/run_cpu_debug.py`. That module imports `models`,
`features` and `config` at the top level, so importing it would pull four files
of the v1/v2 tree in for the sake of a dict literal.

Each entry is (a, b, c, d) plus the drive `I` Izhikevich used in the original
figure. The pipeline uses only (a, b, c, d) -- every protocol amplitude here is
set in rheobase units (gamma), so the published `I` is carried for reference and
is NOT the current any protocol injects.

Two entries are out of scope by design and are expected to fail:
`J subthreshold oscillations` and `Q DAP` are defined by sub-threshold waveform
shape, which no component of the schema observes. `S`/`T` are inhibition-induced
and inhibition is out of scope in v3; `T` has no anchor at all.
"""
PRESETS = {
    "A tonic spiking":                dict(a=0.02,  b=0.2,  c=-65, d=6,    I=14),
    "B phasic spiking":               dict(a=0.02,  b=0.25, c=-65, d=6,    I=0.5),
    "C tonic bursting":               dict(a=0.02,  b=0.2,  c=-50, d=2,    I=15),
    "D phasic bursting":              dict(a=0.02,  b=0.25, c=-55, d=0.05, I=0.6),
    "E mixed mode":                   dict(a=0.02,  b=0.2,  c=-55, d=4,    I=10),
    "F spike freq adapt":             dict(a=0.01,  b=0.2,  c=-65, d=8,    I=30),
    "G Class 1":                      dict(a=0.02,  b=-0.1, c=-55, d=6,    I=0),
    "H Class 2":                      dict(a=0.2,   b=0.26, c=-65, d=0,    I=0),
    "I spike latency":                dict(a=0.02,  b=0.2,  c=-65, d=6,    I=7),
    "J subthreshold oscillations":    dict(a=0.05,  b=0.26, c=-60, d=0,    I=0),
    "K resonator":                    dict(a=0.1,   b=0.26, c=-60, d=-1,   I=0),
    "L integrator":                   dict(a=0.02,  b=-0.1, c=-55, d=6,    I=0),
    "M rebound spike":                dict(a=0.03,  b=0.25, c=-60, d=4,    I=0),
    "N rebound burst":                dict(a=0.03,  b=0.25, c=-52, d=0,    I=0),
    "O threshold variability":        dict(a=0.03,  b=0.25, c=-60, d=4,    I=0),
    "P bistability":                  dict(a=1,     b=1.5,  c=-60, d=0,    I=-65),
    "Q DAP":                          dict(a=1,     b=0.2,  c=-60, d=-21,  I=0),
    "R accomodation":                 dict(a=0.02,  b=1,    c=-55, d=4,    I=0),
    "S inhibition-induced spiking":   dict(a=-0.02, b=-1,   c=-60, d=8,    I=80),
    "T inhibition-induced bursting":  dict(a=-0.026,b=-1,   c=-45, d=0,    I=80),
}

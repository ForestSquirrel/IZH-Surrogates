"""
texdirs.py -- where the generated manuscript material goes: tex_comparison/<section>/{figs,tables}.

    gain/            synaptic gain of a surrogate network (general section)
    ivanov_static/   Ivanov & Michmizos 2021 baseline LSM on MNIST (static weights)
    schulte/         Schulte 2022 / Haeusler-Maass lamina-specific HH microcircuit
"""
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tex_comparison")
SECTIONS = ("gain", "ivanov_static", "schulte")


def figs(section):
    assert section in SECTIONS, section
    return os.path.join(ROOT, section, "figs")


def tables(section):
    assert section in SECTIONS, section
    return os.path.join(ROOT, section, "tables")

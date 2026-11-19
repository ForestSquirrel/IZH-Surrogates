"""Neuron models for the network engines: name -> module with step(y, p, I) -> spike."""
from . import lif, izh, hh_dp, lif_euler, zoo

MODELS = {"lif": lif, "izh": izh, "hh_dp": hh_dp, "lif_euler": lif_euler,
          "zoo_hh": zoo.HH, "zoo_ml": zoo.ML, "zoo_adex": zoo.ADEX}
# column of P holding the RK4 substep count (compute records)
NSUB_COL = {"izh": 5, "hh_dp": 4, "zoo_hh": 1, "zoo_ml": 1, "zoo_adex": 1}

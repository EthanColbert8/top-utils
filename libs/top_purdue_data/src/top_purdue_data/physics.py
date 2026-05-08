"""
May eventually include this in a separate physics utilities library, but that
seems overkill for now.
"""

import logging
import numpy as np
import awkward as ak
# import vector

particle_masses = {
    5: 4.8, # b quark
    6: 172.5, # top quark
    11: 0.000511, # electron
    13: 0.10566, # muon
    15: 1.77686, # tau
    24: 80.369, # W boson
}

def masses_from_pdgids(pdgids: ak.Array) -> ak.Array:
    mass_array = ak.zeros_like(pdgids, dtype=np.float32)
    for pdgid, mass in particle_masses.items():
        mask = np.abs(pdgids) == pdgid
        mass_array = ak.where(mask, mass, mass_array)
    
    if ak.any(mass_array == 0):
        logging.warning("Found pdgids with no assigned mass. Setting mass to 0 for these particles.")
    
    return mass_array

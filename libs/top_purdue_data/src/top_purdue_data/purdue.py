import os
import re
import logging
import numpy as np
import awkward as ak
import vector
import uproot
from typing import List, Optional

try:
    from XRootD import client
    from XRootD.client.flags import DirListFlags
except ImportError:
    logging.debug("XRootD library not found. EOS file listing will not work, and will error. Please install XRootD if needed.")

# from .. import physics
from . import selections

ttbar_run2_pattern = re.compile(r"(?:ee|emu|mumu)_ttbarsignal(?:plus|via)tau_fromDilepton.*\.root")
ttbar_run3_pattern = re.compile(r"(?:ee|emu|mumu)_ttbar_fromDilepton.*\.root")

run2_eras = set(["2016preVFP", "2016postVFP", "2017", "2018"])
run3_eras = set(["2022preEE", "2022postEE", "2023preBPix", "2023postBPix"])

run2_branches = [
    # Jets, leptons, MET
    'jet_pts', 'jet_etas', 'jet_phis', 'jet_masses', 'jet_b_tags',
    'lepton_pts', 'lepton_etas', 'lepton_phis', 'lepton_pdgids', #'lepton_masses',
    'lepton_index', 'antilepton_index', # Needed to get selected leptons
    'met_pt', 'met_phi',

    # Reco-level objects: bs, neutrinos, tops, ttbar system
    'b_pt', 'b_eta', 'b_phi', 'bbar_pt', 'bbar_eta', 'bbar_phi', 'b_mass', 'bbar_mass',
    'nu_pt', 'nu_eta', 'nu_phi', 'nubar_pt', 'nubar_eta', 'nubar_phi', #'nu_mass', 'nubar_mass',
    'top_pt', 'top_eta', 'top_phi', 'top_mass', 'tbar_pt', 'tbar_eta', 'tbar_phi', 'tbar_mass',
    'ttbar_pt', 'ttbar_eta', 'ttbar_phi', 'ttbar_mass',

    # # Reco-level spin observables
    # 'b1k', 'b2k', 'b1r', 'b2r', 'b1n', 'b2n',
    # 'c_kk', 'c_rr', 'c_nn', 'c_rk', 'c_kn', 'c_nr', 'c_kr', 'c_rn', 'c_nk',
    # 'll_cHel',

    # Gen-level objects: bs, leptons, neutrinos, tops, ttbar system
    'gen_b_pt', 'gen_b_eta', 'gen_b_phi', 'gen_bbar_pt', 'gen_bbar_eta', 'gen_bbar_phi', 'gen_b_mass', 'gen_bbar_mass',
    'gen_l_pt', 'gen_l_eta', 'gen_l_phi', 'gen_lbar_pt', 'gen_lbar_eta', 'gen_lbar_phi', 'gen_l_pdgid', 'gen_lbar_pdgid', #'gen_l_mass', 'gen_lbar_mass',
    'gen_nu_pt', 'gen_nu_eta', 'gen_nu_phi', 'gen_nubar_pt', 'gen_nubar_eta', 'gen_nubar_phi', #'gen_nu_mass', 'gen_nubar_mass',
    'gen_top_pt', 'gen_top_eta', 'gen_top_phi', 'gen_top_mass', 'gen_tbar_pt', 'gen_tbar_eta', 'gen_tbar_phi', 'gen_tbar_mass',
    'gen_ttbar_pt', 'gen_ttbar_eta', 'gen_ttbar_phi', 'gen_ttbar_mass',

    # # Gen-level spin observables
    # 'gen_b1k', 'gen_b2k', 'gen_b1r', 'gen_b2r', 'gen_b1n', 'gen_b2n',
    # 'gen_c_kk', 'gen_c_rr', 'gen_c_nn', 'gen_c_rk', 'gen_c_kn', 'gen_c_nr', 'gen_c_kr', 'gen_c_rn', 'gen_c_nk',
    # 'gen_ll_cHel',

    # Event weight
    'eventWeight',
]

run3_branches = [
    # Jets, leptons, MET
    'jets_pt', 'jets_eta', 'jets_phi', 'jets_mass', 'jets_BtagRobustParTAK4B', #'jets_BtagDeepFlavB',
    'l_pt', 'l_eta', 'l_phi', 'lbar_pt', 'lbar_eta', 'lbar_phi', 'l_mass', 'lbar_mass',
    'met_pt', 'met_phi',

    # # Reco-level objects: bs, neutrinos, tops, ttbar system
    # 'b_pt', 'b_eta', 'b_phi', 'bbar_pt', 'bbar_eta', 'bbar_phi', 'b_mass', 'bbar_mass',
    # 'nu_pt', 'nu_eta', 'nu_phi', 'nubar_pt', 'nubar_eta', 'nubar_phi', #'nu_mass', 'nubar_mass',
    # 'top_pt', 'top_eta', 'top_phi', 'top_mass', 'tbar_pt', 'tbar_eta', 'tbar_phi', 'tbar_mass',
    # 'ttbar_pt', 'ttbar_eta', 'ttbar_phi', 'ttbar_mass',

    # # Reco-level spin observables
    # 'b1k', 'b2k', 'b1r', 'b2r', 'b1n', 'b2n',
    # 'c_kk', 'c_rr', 'c_nn', 'c_rk', 'c_kn', 'c_nr', 'c_kr', 'c_rn', 'c_nk',
    # 'll_cHel',

    # Gen-level objects: bs, leptons, neutrinos, tops, ttbar system
    'gen_b_pt', 'gen_b_eta', 'gen_b_phi', 'gen_bbar_pt', 'gen_bbar_eta', 'gen_bbar_phi', 'gen_b_mass', 'gen_bbar_mass',
    'gen_l_pt', 'gen_l_eta', 'gen_l_phi', 'gen_lbar_pt', 'gen_lbar_eta', 'gen_lbar_phi', 'gen_l_mass', 'gen_lbar_mass', #'gen_l_pdgid', 'gen_lbar_pdgid',
    'gen_nu_pt', 'gen_nu_eta', 'gen_nu_phi', 'gen_nubar_pt', 'gen_nubar_eta', 'gen_nubar_phi', #'gen_nu_mass', 'gen_nubar_mass',
    'gen_top_pt', 'gen_top_eta', 'gen_top_phi', 'gen_top_mass', 'gen_tbar_pt', 'gen_tbar_eta', 'gen_tbar_phi', 'gen_tbar_mass',
    'gen_ttbar_pt', 'gen_ttbar_eta', 'gen_ttbar_phi', 'gen_ttbar_mass',

    # # Gen-level spin observables
    # 'gen_b1k', 'gen_b2k', 'gen_b1r', 'gen_b2r', 'gen_b1n', 'gen_b2n',
    # 'gen_c_kk', 'gen_c_rr', 'gen_c_nn', 'gen_c_rk', 'gen_c_kn', 'gen_c_nr', 'gen_c_kr', 'gen_c_rn', 'gen_c_nk',
    # 'gen_ll_cHel',

    # Event weight
    'eventWeight',
]

#####################################################################################################
######## Physics utility for particle masses - should be factored out into a physics library ########
#####################################################################################################
particle_masses = {
    11: 0.000511, # electron
    13: 0.10566, # muon
    15: 1.77686, # tau
    5: 4.8, # b quark
}

def masses_from_pdgids(pdgids: ak.Array) -> ak.Array:
    mass_array = ak.zeros_like(pdgids, dtype=np.float32)
    for pdgid, mass in particle_masses.items():
        mask = (pdgids == pdgid) | (pdgids == -pdgid)
        mass_array = ak.where(mask, mass, mass_array)
    
    if ak.any(mass_array == 0):
        print("WARNING: Found pdgids with no assigned mass. Setting mass to 0 for these particles.")
    
    return mass_array
###################################### End of physics utility. ######################################

def get_common_run_for_eras(eras: List[str]) -> int:
    if all(era in run2_eras for era in eras):
        return 2
    elif all(era in run3_eras for era in eras):
        return 3
    else:
        raise ValueError(f"All eras must be from the same run (either Run 2 or Run 3). Provided eras: {eras}")

def generate_file_list(
    base_dir: str,
    era: str,
    suffix: Optional[str] = "ttBar_treeVariables_step7",
    use_eos: Optional[bool] = False
) -> List[str]:
    if (era in run2_eras):
        pattern = ttbar_run2_pattern
    elif (era in run3_eras):
        pattern = ttbar_run3_pattern
    else:
        raise ValueError(f"Unknown era: {era}")
    
    if use_eos:
        purdue_eos = client.FileSystem("root://eos.cms.rcac.purdue.edu/")
    
    file_list = []
    for channel in ["ee", "emu", "mumu"]:
        channel_dir = os.path.join(base_dir, channel)
        if not (os.path.isdir(channel_dir)):
            print(f"Warning: Channel directory does not exist: {channel_dir}")
            continue
        
        if use_eos:
            status, listing = purdue_eos.dirlist(channel_dir, DirListFlags.STAT)
            # print(status) # TODO: Check status and report errors better
            all_files = [entry.name for entry in listing]
        else:
            all_files = os.listdir(channel_dir)
        
        correct_files = list(filter(pattern.search, all_files))
        correct_files = [f"{channel_dir}/{f}:{suffix}" for f in correct_files]

        file_list.extend(correct_files)
    
    return file_list

def generate_file_list_from_minitrees(
    eras: List[str],
    version: str,
    suffix: Optional[str] = "ttBar_treeVariables_step7"
):
    file_list = []
    for era in eras:
        # TODO: Accept an optional base directory and use_eos option ... also probably use eos as default
        base_dir = f"/depot/cms/users/jduarteq/{era}/spinCorrInput_{era}_{version}/Nominal"
        if (not os.path.isdir(base_dir)):
            print(f"Warning: directory does not exist: {base_dir}")
            continue
        
        file_list.extend(generate_file_list(base_dir, era, suffix))
    
    return file_list

def load_dataset_info_from_files(file_list: List[str], eras_included: List[str]) -> ak.Array:
    is_run2 = get_common_run_for_eras(eras_included) == 2
    if is_run2:
        return uproot.concatenate(file_list, run2_branches, library="ak")
    else:
        return uproot.concatenate(file_list, run3_branches, library="ak")

def create_vectors_run2(data_info: ak.Array) -> dict:
    """
    Creates the vectors for a reconstruction dataset based on Run 2 MC.
    """
    # leptons_masses = physics.masses_from_pdgids(data_info["lepton_pdgids"])
    leptons_masses = masses_from_pdgids(data_info["lepton_pdgids"])
    all_leptons = vector.Array(ak.zip({
        "pt": data_info["lepton_pts"],
        "eta": data_info["lepton_etas"],
        "phi": data_info["lepton_phis"],
        "mass": leptons_masses,
    }))

    selected_leptons, successful_events = selections.lepton_selection_emulation_run2(all_leptons, data_info["lepton_pdgids"], data_info["lepton_index"], data_info["antilepton_index"])

    selected_data = data_info[successful_events] # Hopefully this works for the entire  set of data info, else we have to build a dict

    jets_vectors = vector.Array(ak.zip({
        "pt": selected_data["jet_pts"],
        "eta": selected_data["jet_etas"],
        "phi": selected_data["jet_phis"],
        "mass": selected_data["jet_masses"],
    }))
    met_vectors = vector.array({
        "pt": selected_data["met_pt"].to_numpy()[:, np.newaxis],
        "phi": selected_data["met_phi"].to_numpy()[:, np.newaxis],
    })

    # gen_l_masses = physics.masses_from_pdgids(selected_data["gen_l_pdgid"]).to_numpy()
    # gen_lbar_masses = physics.masses_from_pdgids(selected_data["gen_lbar_pdgid"]).to_numpy()
    gen_l_masses = masses_from_pdgids(selected_data["gen_l_pdgid"]).to_numpy()
    gen_lbar_masses = masses_from_pdgids(selected_data["gen_lbar_pdgid"]).to_numpy()

    vector_info = {
        "reco": {
            "jets": jets_vectors,
            "leptons": selected_leptons,
            "met": met_vectors,
        },
        "gen": {
            "l": vector.array({"pt": selected_data["gen_l_pt"].to_numpy(), "eta": selected_data["gen_l_eta"].to_numpy(), "phi": selected_data["gen_l_phi"].to_numpy(), "mass": gen_l_masses}),
            "lbar": vector.array({"pt": selected_data["gen_lbar_pt"].to_numpy(), "eta": selected_data["gen_lbar_eta"].to_numpy(), "phi": selected_data["gen_lbar_phi"].to_numpy(), "mass": gen_lbar_masses}),
            "b": vector.array({"pt": selected_data["gen_b_pt"].to_numpy(), "eta": selected_data["gen_b_eta"].to_numpy(), "phi": selected_data["gen_b_phi"].to_numpy(), "mass": selected_data["gen_b_mass"].to_numpy()}),
            "bbar": vector.array({"pt": selected_data["gen_bbar_pt"].to_numpy(), "eta": selected_data["gen_bbar_eta"].to_numpy(), "phi": selected_data["gen_bbar_phi"].to_numpy(), "mass": selected_data["gen_bbar_mass"]}),
            "nu": vector.array({"pt": selected_data["gen_nu_pt"].to_numpy(), "eta": selected_data["gen_nu_eta"].to_numpy(), "phi": selected_data["gen_nu_phi"].to_numpy(), "mass": np.zeros_like(selected_data["gen_nu_pt"].to_numpy())}),
            "nubar": vector.array({"pt": selected_data["gen_nubar_pt"].to_numpy(), "eta": selected_data["gen_nubar_eta"].to_numpy(), "phi": selected_data["gen_nubar_phi"].to_numpy(), "mass": np.zeros_like(selected_data["gen_nubar_pt"].to_numpy())}),
            "top": vector.array({"pt": selected_data["gen_top_pt"].to_numpy(), "eta": selected_data["gen_top_eta"].to_numpy(), "phi": selected_data["gen_top_phi"].to_numpy(), "mass": selected_data["gen_top_mass"].to_numpy()}),
            "tbar": vector.array({"pt": selected_data["gen_tbar_pt"].to_numpy(), "eta": selected_data["gen_tbar_eta"].to_numpy(), "phi": selected_data["gen_tbar_phi"].to_numpy(), "mass": selected_data["gen_tbar_mass"].to_numpy()}),
            "ttbar": vector.array({"pt": selected_data["gen_ttbar_pt"].to_numpy(), "eta": selected_data["gen_ttbar_eta"].to_numpy(), "phi": selected_data["gen_ttbar_phi"].to_numpy(), "mass": selected_data["gen_ttbar_mass"].to_numpy()}),
        },
        "mlb_reweighting": {
            "b": vector.array({"pt": selected_data["b_pt"].to_numpy(), "eta": selected_data["b_eta"].to_numpy(), "phi": selected_data["b_phi"].to_numpy(), "mass": selected_data["b_mass"].to_numpy()}),
            "bbar": vector.array({"pt": selected_data["bbar_pt"].to_numpy(), "eta": selected_data["bbar_eta"].to_numpy(), "phi": selected_data["bbar_phi"].to_numpy(), "mass": selected_data["bbar_mass"].to_numpy()}),
            "nu": vector.array({"pt": selected_data["nu_pt"].to_numpy(), "eta": selected_data["nu_eta"].to_numpy(), "phi": selected_data["nu_phi"].to_numpy(), "mass": np.zeros_like(selected_data["nu_pt"].to_numpy())}),
            "nubar": vector.array({"pt": selected_data["nubar_pt"].to_numpy(), "eta": selected_data["nubar_eta"].to_numpy(), "phi": selected_data["nubar_phi"].to_numpy(), "mass": np.zeros_like(selected_data["nubar_pt"].to_numpy())}),
            "top": vector.array({"pt": selected_data["top_pt"].to_numpy(), "eta": selected_data["top_eta"].to_numpy(), "phi": selected_data["top_phi"].to_numpy(), "mass": selected_data["top_mass"].to_numpy()}),
            "tbar": vector.array({"pt": selected_data["tbar_pt"].to_numpy(), "eta": selected_data["tbar_eta"].to_numpy(), "phi": selected_data["tbar_phi"].to_numpy(), "mass": selected_data["tbar_mass"].to_numpy()}),
            "ttbar": vector.array({"pt": selected_data["ttbar_pt"].to_numpy(), "eta": selected_data["ttbar_eta"].to_numpy(), "phi": selected_data["ttbar_phi"].to_numpy(), "mass": selected_data["ttbar_mass"].to_numpy()}),
        },
        "weights": {
            "eventWeight": selected_data["eventWeight"].to_numpy(),
        },
        # "selections": {
        #     "lepton_emulation": successful_events,
        #     "mlb_reweighting": selected_data["top_pt"].to_numpy() >= 0, # Branches are filled with -999 for events that fail mlb-weighting
        # }
    }

    vector_info["reco"]["lepton_charge"] = np.concatenate([np.ones([len(selected_leptons),1]), np.full([len(selected_leptons),1], -1.0)], axis=1)
    vector_info["reco"]["jet_btag"] = selected_data["jet_b_tags"]

    return vector_info

def create_vectors_run3(data_info: ak.Array) -> dict:
    """
    Creates the vectors for a reconstruction dataset based on Run 3 MC.
    """
    leptons = vector.array({
        "pt": np.column_stack((data_info["l_pt"].to_numpy(), data_info["lbar_pt"].to_numpy())),
        "eta": np.column_stack((data_info["l_eta"].to_numpy(), data_info["lbar_eta"].to_numpy())),
        "phi": np.column_stack((data_info["l_phi"].to_numpy(), data_info["lbar_phi"].to_numpy())),
        "mass": np.column_stack((data_info["l_mass"].to_numpy(), data_info["lbar_mass"].to_numpy())),
    })
    jets = vector.Array(ak.zip({
        "pt": data_info["jets_pt"],
        "eta": data_info["jets_eta"],
        "phi": data_info["jets_phi"],
        "mass": data_info["jets_mass"],
    }))
    met = vector.array({
        "pt": data_info["met_pt"].to_numpy()[:, np.newaxis],
        "phi": data_info["met_phi"].to_numpy()[:, np.newaxis],
    })

    vector_info = {
        "reco": {
            "jets": jets,
            "leptons": leptons,
            "met": met,
        },
        "gen": {
            "l": vector.array({"pt": data_info["gen_l_pt"].to_numpy(), "eta": data_info["gen_l_eta"].to_numpy(), "phi": data_info["gen_l_phi"].to_numpy(), "mass": data_info["gen_l_mass"].to_numpy()}),
            "lbar": vector.array({"pt": data_info["gen_lbar_pt"].to_numpy(), "eta": data_info["gen_lbar_eta"].to_numpy(), "phi": data_info["gen_lbar_phi"].to_numpy(), "mass": data_info["gen_lbar_mass"].to_numpy()}),
            "b": vector.array({"pt": data_info["gen_b_pt"].to_numpy(), "eta": data_info["gen_b_eta"].to_numpy(), "phi": data_info["gen_b_phi"].to_numpy(), "mass": data_info["gen_b_mass"].to_numpy()}),
            "bbar": vector.array({"pt": data_info["gen_bbar_pt"].to_numpy(), "eta": data_info["gen_bbar_eta"].to_numpy(), "phi": data_info["gen_bbar_phi"].to_numpy(), "mass": data_info["gen_bbar_mass"].to_numpy()}),
            "nu": vector.array({"pt": data_info["gen_nu_pt"].to_numpy(), "eta": data_info["gen_nu_eta"].to_numpy(), "phi": data_info["gen_nu_phi"].to_numpy(), "mass": np.zeros_like(data_info["gen_nu_pt"].to_numpy())}),
            "nubar": vector.array({"pt": data_info["gen_nubar_pt"].to_numpy(), "eta": data_info["gen_nubar_eta"].to_numpy(), "phi": data_info["gen_nubar_phi"].to_numpy(), "mass": np.zeros_like(data_info["gen_nubar_pt"].to_numpy())}),
            "top": vector.array({"pt": data_info["gen_top_pt"].to_numpy().squeeze(), "eta": data_info["gen_top_eta"].to_numpy().squeeze(), "phi": data_info["gen_top_phi"].to_numpy().squeeze(), "mass": data_info["gen_top_mass"].to_numpy().squeeze()}),
            "tbar": vector.array({"pt": data_info["gen_tbar_pt"].to_numpy(), "eta": data_info["gen_tbar_eta"].to_numpy(), "phi": data_info["gen_tbar_phi"].to_numpy(), "mass": data_info["gen_tbar_mass"].to_numpy()}),
            "ttbar": vector.array({"pt": data_info["gen_ttbar_pt"].to_numpy().squeeze(), "eta": data_info["gen_ttbar_eta"].to_numpy().squeeze(), "phi": data_info["gen_ttbar_phi"].to_numpy().squeeze(), "mass": data_info["gen_ttbar_mass"].to_numpy().squeeze()}),
        },
        "weights": {
            "eventWeight": data_info["eventWeight"].to_numpy(),
        }
    }

    vector_info["reco"]["lepton_charge"] = np.concatenate([np.ones([len(data_info["l_pt"]),1]), np.full([len(data_info["lbar_pt"]),1], -1.0)], axis=1)
    vector_info["reco"]["jet_btag"] = data_info["jets_BtagRobustParTAK4B"]

    return vector_info

def create_vectors(data_info, eras_included):
    is_run2 = get_common_run_for_eras(eras_included) == 2
    if is_run2:
        return create_vectors_run2(data_info)
    else:
        return create_vectors_run3(data_info)

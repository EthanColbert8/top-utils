import numpy as np
import awkward as ak
import vector

def lepton_selection_emulation_run2(leptons, pdgids, l_index, lbar_index):
    """
    Attempts to emulate the lepton selection from run 2 given available information.
    """
    all_leptons = leptons

    not_already_selected = l_index < 0
    has_extra_leptons = (ak.num(leptons, axis=1) > 2).to_numpy().squeeze()
    needs_extra_selection = not_already_selected & has_extra_leptons
    needs_charge_selection = not_already_selected & ~has_extra_leptons
    
    ######## If only 2 leptons in array, easy selection based on charge. ########
    idx_of_lepton = ak.fill_none(ak.argmax(pdgids, axis=1), -999).to_numpy()
    idx_of_lbar = ak.fill_none(ak.argmin(pdgids, axis=1), -999).to_numpy()

    selected_l_idx = np.where(needs_charge_selection, idx_of_lepton, l_index).to_numpy()
    selected_lbar_idx = np.where(needs_charge_selection, idx_of_lbar, lbar_index).to_numpy()

    ######## Tough part - all the extra cuts to try to deduce the actual selection. ########
    indices_needing_selection = np.nonzero(needs_extra_selection)[0]
    leptons_for_selection = leptons[indices_needing_selection]
    pdgids_for_selection = pdgids[indices_needing_selection]

    pair_indices = ak.argcombinations(pdgids_for_selection, 2, replacement=False, axis=1, fields=['l1', 'l2'])

    individual_pt_eta_cut = (leptons_for_selection.pt > 20) & (np.abs(leptons_for_selection.eta) < 2.4)
    pt_eta_cut = individual_pt_eta_cut[pair_indices['l1']] & individual_pt_eta_cut[pair_indices['l2']]

    charge_cut = (pdgids_for_selection[pair_indices['l1']] * pdgids_for_selection[pair_indices['l2']]) < 0

    overall_cut = pt_eta_cut & charge_cut #& pdgid_cut
    valid_pair_indices = pair_indices[overall_cut]

    num_valid_pairs = ak.num(valid_pair_indices, axis=1).to_numpy()
    valid_selection = num_valid_pairs == 1
    selected_pair_indices = valid_pair_indices[:, 0]

    selected_l1 = np.where(valid_selection, selected_pair_indices['l1'], -1) # like to find a way to make it -999, but need valid index for now
    selected_l2 = np.where(valid_selection, selected_pair_indices['l2'], -1)
    lepton_mask = (pdgids_for_selection[np.arange(len(pdgids_for_selection)), selected_l1] > 0).to_numpy()

    selected_l_index = np.where(lepton_mask, selected_l1, selected_l2)
    selected_lbar_index = np.where(lepton_mask, selected_l2, selected_l1)

    for i, large_idx in enumerate(indices_needing_selection):
        selected_l_idx[large_idx] = selected_l_index[i]
        selected_lbar_idx[large_idx] = selected_lbar_index[i]
    
    # Get the actual selected leptons from their indices, and a mask over successful selection
    successful_events = selected_l_idx >= 0
    all_leptons = all_leptons[successful_events]
    selected_l_idx = selected_l_idx[successful_events]
    selected_lbar_idx = selected_lbar_idx[successful_events]

    selected_l = vector.MomentumNumpy4D(all_leptons[np.arange(len(all_leptons)), selected_l_idx])
    selected_lbar = vector.MomentumNumpy4D(all_leptons[np.arange(len(all_leptons)), selected_lbar_idx])

    selected_leptons = vector.array({
        "pt": np.column_stack([selected_l.pt, selected_lbar.pt]),
        "eta": np.column_stack([selected_l.eta, selected_lbar.eta]),
        "phi": np.column_stack([selected_l.phi, selected_lbar.phi]),
        "M": np.column_stack([selected_l.M, selected_lbar.M]),
    })

    return selected_leptons, successful_events

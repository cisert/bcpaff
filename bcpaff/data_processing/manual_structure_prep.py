"""
© 2023, ETH Zurich
"""

import copy
import json
import os
import pickle
import tempfile
from typing import Dict, List, Set, Tuple, Union

import numpy as np
from openbabel.pybel import readfile
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdchem import BondType
from scipy.spatial.distance import cdist

CH_BOND_LENGTH = 1.09  # https://en.wikipedia.org/wiki/Carbon%E2%80%93hydrogen_bond (06.09.22)


from numpy import cross, dot, eye
from scipy.linalg import expm, norm


def M(axis: np.array, theta: float) -> np.array:
    """Generate rotation matrix. https://stackoverflow.com/questions/6802577/rotation-of-3d-vector (Accessed 26.10.22)"""
    return expm(cross(eye(3), axis / norm(axis) * theta))


def split_resname_and_resnum(s: str) -> Tuple[str, int]:
    """Split residue name and residue number from mol2-files.

    Parameters
    ----------
    s : str
        combined resname and resnum

    Returns
    -------
    tuple(str, str)
        individual resname, resnum
    """
    resnum, resname = [], []
    still_at_resnum = True
    for x in s[::-1]:
        if still_at_resnum and x.isnumeric():
            resnum.append(x)
        else:
            resname.append(x)
            still_at_resnum = False
    resnum = int("".join(resnum[::-1])) if len(resnum) else 9999
    resname = "".join(resname[::-1])
    return resname, resnum


def addInfoFromMol2(mol: Chem.Mol, mol_path: str, is_ligand: bool = False, dataset: str = "pdbbind"):
    """Read residue infos and formal charges from mol2-file and add to RDKit molecule.
    Tripos mol2 files aren't perfectly supported by RDKit, so we need to add some info
    ourselves.

    Parameters
    ----------
    mol : RDKit molecule
        protein mol
    mol_path : str
        path to mol2 file for protein
    """
    with open(mol_path, "r") as f:
        lines = [l.rstrip("\n") for l in f.readlines()]
    ATOM_START_LINE = "@<TRIPOS>ATOM"
    ATOM_END_LINE = "@<TRIPOS>UNITY_ATOM_ATTR" if dataset == "d2dr" else "@<TRIPOS>BOND"
    atom_lines = lines[lines.index(ATOM_START_LINE) + 1 : lines.index(ATOM_END_LINE)]
    assert mol.GetNumAtoms() == len(atom_lines)  # sanity check
    elems = [l.split()[5].split(".")[0] for l in atom_lines]
    elems_compare = ["*" if x == "Du" else x for x in elems]  # different dummy atom representations
    assert [a.GetSymbol() for a in mol.GetAtoms()] == elems_compare  #  sanity check
    resnames, resnums = zip(*[split_resname_and_resnum(l.split()[7]) for l in atom_lines])
    c_alphas = [l.split()[1] == "CA" for l in atom_lines]
    c_betas = [l.split()[1] == "CB" for l in atom_lines]
    formal_charges = [float(l.split()[-1]) for l in atom_lines]
    if is_ligand:
        mol.SetIntProp("formal_charge", round(sum(formal_charges)))
    for a, resname, resnum, c_alpha, c_beta, formal_charge in zip(
        mol.GetAtoms(), resnames, resnums, c_alphas, c_betas, formal_charges
    ):
        info = Chem.AtomPDBResidueInfo()
        if is_ligand:
            info.SetResidueName("LIG")
            info.SetIsHeteroAtom(True)
            # no formal charges for ligand because only Gasteiger charges in input files
        else:
            info.SetResidueName(resname)
            a.SetFormalCharge(int(formal_charge))
        info.SetResidueNumber(int(resnum))
        a.SetMonomerInfo(info)
        a.SetBoolProp("c_alpha", c_alpha)
        a.SetBoolProp("c_beta", c_beta)


def cut_Calpha_Cbeta_bonds(protein: Chem.Mol) -> List:
    """Remove bonds between alpha and beta carbons

    Parameters
    ----------
    protein : RDKit molecule
        protein mol

    Returns
    -------
    list
        list of properties which should be assigned to individual atoms denoting in which direction bonds where cut
    """
    bonds_to_remove = []
    prop_assignment = []
    positions = protein.GetConformer().GetPositions()
    for a in protein.GetAtoms():
        if a.GetSymbol() != "H" and a.GetBoolProp("c_alpha"):
            for n in a.GetNeighbors():
                if n.GetSymbol() != "H" and n.GetBoolProp("c_beta"):  # Hs not annotated as c_beta
                    aidx, nidx = a.GetIdx(), n.GetIdx()
                    bonds_to_remove.append((aidx, nidx))
                    cut_direction = positions[nidx] - positions[aidx]
                    prop_assignment.append((aidx, f"cut_direction_{aidx}_{nidx}", cut_direction))
                    prop_assignment.append(
                        (nidx, f"cut_direction_{nidx}_{aidx}", -cut_direction)
                    )  # opposite direction
    for b in bonds_to_remove:
        protein.RemoveBond(*b)
    return prop_assignment


def cut_backbone(protein: Chem.Mol) -> List:
    """Remove bonds in the backbone by cutting between alpha carbon and adjacent carboxy

    Parameters
    ----------
    protein : RDKit molecule
        protein mol

    Returns
    -------
    list
        list of properties which should be assigned to individual atoms denoting in which direction bonds where cut
    """
    bonds_to_remove = []
    prop_assignment = []
    positions = protein.GetConformer().GetPositions()
    for a in protein.GetAtoms():
        if a.GetSymbol() != "H" and a.GetBoolProp("c_alpha"):
            for n in a.GetNeighbors():
                if n.GetSymbol() == "C" and not n.GetBoolProp("c_beta"):  # potential carboxy C
                    next_neighbors = sorted([x.GetSymbol() for x in n.GetNeighbors()])
                    if next_neighbors == ["C", "N", "O"]:
                        aidx, nidx = a.GetIdx(), n.GetIdx()
                        bonds_to_remove.append((aidx, nidx))
                        cut_direction = positions[nidx] - positions[aidx]
                        prop_assignment.append((aidx, f"cut_direction_{aidx}_{nidx}", cut_direction))
                        prop_assignment.append(
                            (nidx, f"cut_direction_{nidx}_{aidx}", -cut_direction)
                        )  # opposite direction
    for b in bonds_to_remove:
        protein.RemoveBond(*b)
    return prop_assignment


def get_idxs_within_cutoff(protein: Chem.Mol, ligand: Chem.Mol, cutoff: int = 6) -> Tuple[Set[int], List]:
    """Get atom indices of fragments where at least one atom is within cutoff of the ligand

    Parameters
    ----------
    protein : RDKit molecule
        protein mol
    ligand : RDKit molecule
        ligand mol
    cutoff : int, optional
        cutoff distance in Angstrom, by default 6

    Returns
    -------
    set, list
        indices within cutoff and combined properties (cut directions which should be assigned)
    """
    protein = copy.deepcopy(protein)  # don't cut the real protein
    prop_assignment = cut_Calpha_Cbeta_bonds(protein)
    prop_assignment2 = cut_backbone(protein)
    prop_assignment.extend(prop_assignment2)
    frags = Chem.GetMolFrags(protein)  # individual fragments after cutting

    protein_pos = protein.GetConformer().GetPositions()
    ligand_pos = ligand.GetConformer().GetPositions()
    distances = cdist(protein_pos, ligand_pos)
    idxs_within_cutoff = []
    for frag in frags:
        shortest_distance_to_ligand = distances[frag, :].min()
        if shortest_distance_to_ligand < cutoff:
            idxs_within_cutoff.extend(list(frag))
    return set(idxs_within_cutoff), prop_assignment


def fix_carboxylic_acids(mol: Chem.Mol, dataset="pdbbind"):
    """Assign charges & aromaticitiy to deprotonated carboxylic acids --> avoid radicals"""
    patt = (
        Chem.MolFromSmarts("[#6D3](=[#8D1])[#8D1]") if dataset == "d2dr" else Chem.MolFromSmarts("[cD3](:o)(:o)")
    )  # special expression because of format in D2DR
    mol.UpdatePropertyCache()
    matches = mol.GetSubstructMatches(patt)

    for match in matches:
        for aidx in match:
            if mol.GetAtomWithIdx(aidx).GetSymbol() == "C":
                c = mol.GetAtomWithIdx(aidx)
        oxygen_ids = [idx for idx in match if idx != c.GetIdx()]
        o1 = mol.GetAtomWithIdx(oxygen_ids[0])
        o2 = mol.GetAtomWithIdx(oxygen_ids[1])
        b1 = mol.GetBondBetweenAtoms(c.GetIdx(), o1.GetIdx())
        b2 = mol.GetBondBetweenAtoms(c.GetIdx(), o2.GetIdx())
        if dataset != "d2dr":  # formatted as single/double in D2DR
            assert b1.GetBondType() == b2.GetBondType() == BondType.AROMATIC
        b1.SetBondType(BondType.SINGLE)
        o1.SetFormalCharge(-1)
        b2.SetBondType(BondType.DOUBLE)
        o2.SetFormalCharge(0)
        c.SetIsAromatic(False)
        o1.SetIsAromatic(False)
        o2.SetIsAromatic(False)


def fix_guadinine_groups(mol: Chem.Mol):
    """Assign charges to guadenine groups --> avoid radicals"""

    patt = Chem.MolFromSmarts("[#6]([#7X3])([#7X3])([#7X3])")
    mol.UpdatePropertyCache()
    matches = mol.GetSubstructMatches(patt)

    for match in matches:
        for aidx in match:
            if mol.GetAtomWithIdx(aidx).GetSymbol() == "C":
                c = mol.GetAtomWithIdx(aidx)
        nitrogen_ids = [idx for idx in match if idx != c.GetIdx()]
        n1 = mol.GetAtomWithIdx(nitrogen_ids[0])
        n2 = mol.GetAtomWithIdx(nitrogen_ids[1])
        n3 = mol.GetAtomWithIdx(nitrogen_ids[2])
        b1 = mol.GetBondBetweenAtoms(c.GetIdx(), n1.GetIdx())
        b2 = mol.GetBondBetweenAtoms(c.GetIdx(), n2.GetIdx())
        b3 = mol.GetBondBetweenAtoms(c.GetIdx(), n3.GetIdx())
        assert b1.GetBondType() == b2.GetBondType() == b3.GetBondType() == BondType.AROMATIC
        n1.SetFormalCharge(1)
        b1.SetBondType(BondType.DOUBLE)
        n2.SetFormalCharge(0)
        b2.SetBondType(BondType.SINGLE)
        n3.SetFormalCharge(0)
        b3.SetBondType(BondType.SINGLE)
        c.SetFormalCharge(0)
        [a.SetIsAromatic(False) for a in [c, n1, n2, n3]]


def fix_nitrogen_ion_groups(mol: Chem.Mol):
    """Assign charges to nitrogen ion groups --> avoid radicals"""
    patt = Chem.MolFromSmarts("[#6X3]([#7X3])([#7X3])([#6])")
    mol.UpdatePropertyCache()
    matches = mol.GetSubstructMatches(patt)

    for match in matches:
        for aidx in match:
            if mol.GetAtomWithIdx(aidx).GetSymbol() == "C":
                c_candidate = mol.GetAtomWithIdx(aidx)
                if sorted([a.GetSymbol() for a in c_candidate.GetNeighbors()]) == ["C", "N", "N"]:
                    c = c_candidate
                    if c.IsInRing():  # hacky fix because this SMARTS pattern also matches amine-N-heterocycles
                        return
                    break  # otherwise continue to get the other carbon
        nitrogen_ids = [idx for idx in match if idx != c.GetIdx()]
        n1 = mol.GetAtomWithIdx(nitrogen_ids[0])
        n2 = mol.GetAtomWithIdx(nitrogen_ids[1])
        b1 = mol.GetBondBetweenAtoms(c.GetIdx(), n1.GetIdx())
        b2 = mol.GetBondBetweenAtoms(c.GetIdx(), n2.GetIdx())
        assert b1.GetBondType() == b2.GetBondType() == BondType.AROMATIC  # THIS STILL MATCHES GUADININE
        n1.SetFormalCharge(1)
        b1.SetBondType(BondType.DOUBLE)
        n2.SetFormalCharge(0)
        b2.SetBondType(BondType.SINGLE)
        c.SetFormalCharge(0)
        [a.SetIsAromatic(False) for a in [c, n1, n2]]


def fix_phosphoric_acids(mol: Chem.Mol):
    """Assign charges to deprotonated phosphoric acids --> avoid radicals"""
    patt = Chem.MolFromSmarts("[#15]([#8X1])([#8X1])([#8X1])")
    matches = mol.GetSubstructMatches(patt)

    for match in matches:
        for aidx in match:
            if mol.GetAtomWithIdx(aidx).GetSymbol() == "P":
                p = mol.GetAtomWithIdx(aidx)
        oxygen_ids = [idx for idx in match if idx != p.GetIdx()]
        o1 = mol.GetAtomWithIdx(oxygen_ids[0])
        o2 = mol.GetAtomWithIdx(oxygen_ids[1])
        o3 = mol.GetAtomWithIdx(oxygen_ids[2])
        b1 = mol.GetBondBetweenAtoms(p.GetIdx(), o1.GetIdx())
        b2 = mol.GetBondBetweenAtoms(p.GetIdx(), o2.GetIdx())
        b3 = mol.GetBondBetweenAtoms(p.GetIdx(), o3.GetIdx())
        assert b1.GetBondType() == b2.GetBondType() == b3.GetBondType() == BondType.AROMATIC
        b1.SetBondType(BondType.SINGLE)
        o1.SetFormalCharge(-1)
        b2.SetBondType(BondType.SINGLE)
        o2.SetFormalCharge(-1)
        b3.SetBondType(BondType.DOUBLE)
        o3.SetFormalCharge(0)
        p.SetFormalCharge(0)
        [a.SetIsAromatic(False) for a in [p, o1, o2, o3]]


def fix_phosphoric_esters(mol: Chem.Mol):
    """Assign charges to deprotonated phosphoric esters --> avoid radicals"""
    patt = Chem.MolFromSmarts("[#15]([*])([#8X2])([#8X1])([#8X1])")
    matches = mol.GetSubstructMatches(patt)

    for match in matches:
        for aidx in match:
            if mol.GetAtomWithIdx(aidx).GetSymbol() == "P":
                p = mol.GetAtomWithIdx(aidx)
        oxygen_ids = [idx for idx in match if idx != p.GetIdx()]
        os_single_neighbor = []
        os_two_neighbors = []
        for oid in oxygen_ids:
            a = mol.GetAtomWithIdx(oid)
            if len(a.GetNeighbors()) == 1:
                os_single_neighbor.append(a)
            elif len(a.GetNeighbors()) > 1:
                os_two_neighbors.append(a)
            else:
                raise ValueError(f"Unexpected number of neighbors for atom {oid}")
        assert len(os_single_neighbor) == len(os_two_neighbors) == 2

        o1_single_neighbor = os_single_neighbor[0]
        o2_single_neighbor = os_single_neighbor[1]
        o1_two_neighbors = os_two_neighbors[0]
        o2_two_neighbors = os_two_neighbors[1]

        b1_single_neighbor = mol.GetBondBetweenAtoms(p.GetIdx(), o1_single_neighbor.GetIdx())
        b2_single_neighbor = mol.GetBondBetweenAtoms(p.GetIdx(), o2_single_neighbor.GetIdx())
        b1_two_neighbors = mol.GetBondBetweenAtoms(p.GetIdx(), o1_two_neighbors.GetIdx())
        b2_two_neighbors = mol.GetBondBetweenAtoms(p.GetIdx(), o2_two_neighbors.GetIdx())

        b1_single_neighbor.SetBondType(BondType.DOUBLE)
        o1_single_neighbor.SetFormalCharge(0)

        b2_single_neighbor.SetBondType(BondType.SINGLE)
        o2_single_neighbor.SetFormalCharge(-1)

        b1_two_neighbors.SetBondType(BondType.SINGLE)
        o1_two_neighbors.SetFormalCharge(0)
        b2_two_neighbors.SetBondType(BondType.SINGLE)
        o2_two_neighbors.SetFormalCharge(0)
        p.SetFormalCharge(0)
        [
            a.SetIsAromatic(False)
            for a in [p, o1_single_neighbor, o2_single_neighbor, o1_two_neighbors, o2_two_neighbors]
        ]


def remove_protein_overlap_with_ligand(ligand: Chem.Mol, protein: Chem.Mol):
    """Some files (e.g., 9abp) have the ligand also in the protein file, so it overlaps --> need to remove the one in the protein."""
    frags = Chem.GetMolFrags(protein)
    protein_pos = protein.GetConformer().GetPositions()
    ligand_pos = ligand.GetConformer().GetPositions()
    potential_frags = [frag for frag in frags if len(frag) == ligand.GetNumAtoms()]
    atoms_to_remove = []
    for frag in potential_frags:
        frag_pos = protein_pos[frag, :]
        min_dist = cdist(frag_pos, ligand_pos).min(axis=1)
        num_very_close = (
            min_dist < CH_BOND_LENGTH / 3
        ).sum()  # using 1/3 of a CH bond length as a distance that we really shouldn't be below
        if num_very_close > 1:
            atoms_to_remove.extend(frag)
    atoms_to_remove.sort(reverse=True)
    for idx in atoms_to_remove:
        protein.RemoveAtom(int(idx))


def fix_radical_aldehydes(protein: Chem.Mol):
    patt = Chem.MolFromSmarts("[CX2](=O)[#6]")  # aldehyde with missing proton
    protein.UpdatePropertyCache()
    matches = protein.GetSubstructMatches(patt)
    for match in matches:
        for aidx in match:
            atom = protein.GetAtomWithIdx(aidx)
            if atom.GetSymbol() == "O":
                assert len(atom.GetNeighbors()) == 1
                carbonyl_c = atom.GetNeighbors()[0]  # only neighbor of oxygen
                break
        neighbors = carbonyl_c.GetNeighbors()
        carbonyl_c_pos = protein.GetConformer().GetPositions()[carbonyl_c.GetIdx()]
        assert len(neighbors) == 2  # oxygen and adjacent carbon
        neighbor1_pos = protein.GetConformer().GetPositions()[neighbors[0].GetIdx()]
        neighbor2_pos = protein.GetConformer().GetPositions()[neighbors[1].GetIdx()]
        vec1 = carbonyl_c_pos - neighbor1_pos
        vec2 = carbonyl_c_pos - neighbor2_pos
        proton_vec = (vec1 + vec2) / np.linalg.norm(vec1 + vec2) * CH_BOND_LENGTH
        new_location = carbonyl_c_pos + proton_vec
        new_atom_id = protein.AddAtom(Chem.Atom("H"))  # pad with hydrogen
        protein.GetConformer().SetAtomPosition(new_atom_id, new_location)
        protein.AddBond(carbonyl_c.GetIdx(), new_atom_id, Chem.rdchem.BondType.SINGLE)


def fix_carbons_with_missing_hydrogens(protein: Chem.Mol):
    """Adds hydrogens to carbons which don't have enough --> some specific terminal CH2 (instead of CH3) groups were observed

    Parameters
    ----------
    protein : RDKit molecule
        protein to be fixed
    """
    patt = Chem.MolFromSmarts("[#6X3]([#1])([#1])([#6])")  # carbon with missing hydrogen
    # this SMARTS isn't elegant, but directly using valence doesn't work if we didn't previously
    # sanitize the molecule, so we need to work around this
    protein.UpdatePropertyCache()
    matches = protein.GetSubstructMatches(patt)
    for match in matches:
        for aidx in match:
            if protein.GetAtomWithIdx(aidx).GetSymbol() == "C":
                c_candidate = protein.GetAtomWithIdx(aidx)
                if sorted([a.GetSymbol() for a in c_candidate.GetNeighbors()]) == ["C", "H", "H"]:
                    c = c_candidate
                    c_rot = [a for a in c_candidate.GetNeighbors() if a.GetSymbol() == "C"][0]
                    break  # otherwise continue to get the other carbon
        hydrogen_ids = [idx for idx in match if idx != c.GetIdx()]
        h1 = protein.GetAtomWithIdx(hydrogen_ids[0])
        h2 = protein.GetAtomWithIdx(hydrogen_ids[1])

        c_pos = protein.GetConformer().GetPositions()[c.GetIdx()]
        c_rot_pos = protein.GetConformer().GetPositions()[c_rot.GetIdx()]
        h1_pos = protein.GetConformer().GetPositions()[h1.GetIdx()]
        h2_pos = protein.GetConformer().GetPositions()[h2.GetIdx()]

        # rotate around C-C bond
        start_vec = c_pos - (h1_pos + h2_pos) / 2
        rotation_axis = c_pos - c_rot_pos
        proton_vec = dot(M(rotation_axis, np.pi), start_vec)
        proton_vec = proton_vec / np.linalg.norm(proton_vec) * CH_BOND_LENGTH
        new_location = c_pos - proton_vec

        new_atom_id = protein.AddAtom(Chem.Atom("H"))  # pad with hydrogen
        protein.GetConformer().SetAtomPosition(new_atom_id, new_location)
        protein.AddBond(c.GetIdx(), new_atom_id, Chem.rdchem.BondType.SINGLE)


def fix_nitrogens_with_missing_hydrogens(protein: Chem.Mol):
    """Adds hydrogens to nitrogens which don't have enough --> some specific terminal NH (instead of NH2) groups were observed

    Parameters
    ----------
    protein : RDKit molecule
        protein to be fixed
    """
    patt = Chem.MolFromSmarts("[#7X2]([#1])([#6])")  # carbon with missing hydrogen
    # this SMARTS isn't elegant, but directly using valence doesn't work if we didn't previously
    # sanitize the molecule, so we need to work around this
    protein.UpdatePropertyCache()
    matches = protein.GetSubstructMatches(patt)
    for match in matches:
        for aidx in match:
            if protein.GetAtomWithIdx(aidx).GetSymbol() == "N":
                n = protein.GetAtomWithIdx(aidx)
            if protein.GetAtomWithIdx(aidx).GetSymbol() == "H":
                h = protein.GetAtomWithIdx(aidx)
            if protein.GetAtomWithIdx(aidx).GetSymbol() == "C":
                c = protein.GetAtomWithIdx(aidx)

        n_pos = protein.GetConformer().GetPositions()[n.GetIdx()]
        h_pos = protein.GetConformer().GetPositions()[h.GetIdx()]
        c_pos = protein.GetConformer().GetPositions()[c.GetIdx()]

        vec1 = n_pos - h_pos
        vec2 = n_pos - c_pos
        proton_vec = (vec1 + vec2) / np.linalg.norm(vec1 + vec2) * CH_BOND_LENGTH
        new_location = n_pos + proton_vec
        new_atom_id = protein.AddAtom(Chem.Atom("H"))  # pad with hydrogen
        protein.GetConformer().SetAtomPosition(new_atom_id, new_location)
        protein.AddBond(n.GetIdx(), new_atom_id, Chem.rdchem.BondType.SINGLE)
        inheritPDBResidueInfo(n, protein.GetAtomWithIdx(new_atom_id))


def cut_remote_fragments(protein: Chem.Mol, idxs_within_cutoff: List[int]):
    """Remove fragments where no element is within cutoff distance of the ligand

    Parameters
    ----------
    protein : RDKit molecule
        protein mol
    idxs_within_cutoff : set
        indices of atoms that should be kept (at least one atom in the fragment close to ligand)
    """
    bonds_to_remove = []
    for a in protein.GetAtoms():
        if a.GetSymbol() != "H" and a.GetBoolProp("c_alpha"):
            for n in a.GetNeighbors():
                aidx, nidx = a.GetIdx(), n.GetIdx()
                if n.GetSymbol() != "H" and n.GetBoolProp("c_beta"):  #
                    if not (aidx in idxs_within_cutoff and nidx in idxs_within_cutoff):
                        # remove bond between C_alpha and C_beta unless both residues are within cutoff
                        assert a.HasProp(f"cut_direction_{aidx}_{nidx}")
                        bonds_to_remove.append((aidx, nidx))
                        a.SetBoolProp(f"needs_padding_{aidx}_{nidx}", True)
                        n.SetBoolProp(f"needs_padding_{nidx}_{aidx}", True)
                elif n.GetSymbol() == "C" and not n.GetBoolProp("c_beta"):  # potential carboxy C (cut backbone)
                    next_neighbors = sorted([x.GetSymbol() for x in n.GetNeighbors()])
                    if next_neighbors == ["C", "N", "O"]:
                        if not (aidx in idxs_within_cutoff and nidx in idxs_within_cutoff):
                            assert a.HasProp(f"cut_direction_{aidx}_{nidx}")
                            assert n.HasProp(f"cut_direction_{nidx}_{aidx}")
                            bonds_to_remove.append((aidx, nidx))
                            a.SetBoolProp(f"needs_padding_{aidx}_{nidx}", True)
                            n.SetBoolProp(f"needs_padding_{nidx}_{aidx}", True)
    for b in bonds_to_remove:
        protein.RemoveBond(*b)

    atoms_to_remove = list(set((np.arange(protein.GetNumAtoms()))) - idxs_within_cutoff)
    atoms_to_remove.sort(reverse=True)
    for idx in atoms_to_remove:
        protein.RemoveAtom(int(idx))


def transfer_props(protein: Chem.Mol, prop_assignment: List[int]):
    """Transfer atom properties to protein. Need to do this like this because we've used a copy of the protein for preliminary cuts before

    Parameters
    ----------
    protein : RDKit molecule
        protein mol
    prop_assignment : list
        properties to assign
    """
    for idx, key, cut_direction in prop_assignment:
        protein.GetAtomWithIdx(idx).SetProp(key, f"{cut_direction[0]} {cut_direction[1]} {cut_direction[2]}")


def find_pad_atoms(protein: Chem.Mol) -> List[Chem.Atom]:
    """Determine which atoms need to be padded with hydrogens because we cut a bond to them.

    Parameters
    ----------
    protein : RDKit molecule
        protein mol

    Returns
    -------
    list
        list of atoms that need to be padded
    """
    pad_atoms = []
    for a in protein.GetAtoms():
        if len([key for key in a.GetPropsAsDict().keys() if key.startswith("needs_padding")]):
            pad_atoms.append(a)
    return pad_atoms


def pad_with_protons(protein: Chem.Mol):
    """Pad protein with proton (H) where we cut bonds

    Parameters
    ----------
    protein : RDKit molecule
        protein mol
    """
    pad_atoms = find_pad_atoms(protein)

    for a in pad_atoms:
        needs_padding = [key for key in a.GetPropsAsDict().keys() if key.startswith("needs_padding")]
        # don't use all cut directions! an atom can have multiple cut directions but only one is
        # actually cut, that's why it's labelled as needs_padding
        for nps in needs_padding:
            aidx, nidx = int(nps.split("_")[2]), int(nps.split("_")[3])
            cut_direction = [float(x) for x in a.GetProp(f"cut_direction_{aidx}_{nidx}").split()]
            start_position = protein.GetConformer().GetPositions()[a.GetIdx()]
            displacement = cut_direction / np.linalg.norm(cut_direction) * CH_BOND_LENGTH
            new_location = start_position + displacement
            new_atom_id = protein.AddAtom(Chem.Atom("H"))  # pad with hydrogen
            new_atom = protein.GetAtomWithIdx(new_atom_id)
            protein.GetConformer().SetAtomPosition(new_atom_id, new_location)
            protein.AddBond(a.GetIdx(), new_atom_id, Chem.rdchem.BondType.SINGLE)
            inheritPDBResidueInfo(a, new_atom)


def inheritPDBResidueInfo(a: Chem.Atom, new_atom: Chem.Atom):
    """add PDBResidueInfo to new atom

    Parameters
    ----------
    a : Chem.Atom
        atom from which to inherit PDBResidueInfo
    new_atom : Chem.Atom
        atom which inherits PDBResidueInfo
    """
    existing_info = a.GetPDBResidueInfo()
    new_info = Chem.AtomPDBResidueInfo()
    new_info.SetResidueName(existing_info.GetResidueName())
    new_info.SetResidueNumber(int(existing_info.GetResidueNumber()))
    new_atom.SetMonomerInfo(new_info)


def write_to_file(
    output_basepath: str, structure_id: str, protein: Chem.Mol, ligand: Chem.Mol, fragments_have_radicals: bool
):
    """Save results to files. Sometimes RDKit can't write successfully, then we'll work around this...

    Parameters
    ----------
    output_basepath : str
        folder path of output basepath
    pdb_id : str
        PDB-ID
    protein : RDKit molecule
        protein mol, after preparation
    ligand : RDKit molecule
        ligand mol
    ligand_path : str
        path to input file for ligand
    """
    basepath = os.path.join(output_basepath, structure_id)
    os.makedirs(basepath, exist_ok=True)

    # write protein
    try:
        w = Chem.SDWriter(os.path.join(basepath, f"{structure_id}_pocket_with_hydrogens.sdf"))
        w.SetKekulize(False)
        w.write(protein)
        w.flush()
        w.close()
    except:
        pass
    # XYZ not as nice for visualization, but works
    Chem.rdmolfiles.MolToXYZFile(protein, os.path.join(basepath, f"{structure_id}_pocket_with_hydrogens.xyz"))

    # write ligand
    try:
        w = Chem.SDWriter(os.path.join(basepath, f"{structure_id}_ligand_with_hydrogens.sdf"))
        w.SetKekulize(False)
        w.write(ligand)
        w.flush()
        w.close()
    except:
        print("Couldn't write ligand")

    pl_complex = Chem.CombineMols(ligand, protein)
    # write complex

    try:
        w = Chem.SDWriter(os.path.join(basepath, "pl_complex.sdf"))
        w.SetKekulize(False)
        w.write(pl_complex)
        w.flush()
        w.close()
    except:
        pass
    Chem.rdmolfiles.MolToXYZFile(pl_complex, os.path.join(basepath, "pl_complex.xyz"))

    # write xTB input data
    charge = Chem.GetFormalCharge(protein) + Chem.GetFormalCharge(ligand)
    num_unpaired_electrons = Descriptors.NumRadicalElectrons(pl_complex)  # assuming no radicals in ligands
    frags = Chem.GetMolFrags(pl_complex)

    with open(os.path.join(basepath, "input.inp"), "w") as f:
        f.write(f"$chrg {charge}\n")
        f.write(f"$spin {num_unpaired_electrons}\n")
        f.write("$split\n")
        for i, frag in enumerate(frags):
            f.write(f"   fragment: {i+1},{','.join([str(fr) for fr in frag])}\n")
    with open(os.path.join(basepath, "chrg_uhf.json"), "w") as f:
        json.dump({"charge": charge, "num_unpaired_electrons": num_unpaired_electrons}, f)

    try:
        with open(os.path.join(basepath, "psi4_input.pkl"), "wb") as f:
            pickle.dump(get_psi4_input(pl_complex), f, protocol=pickle.HIGHEST_PROTOCOL)
    except:
        pass

    # write warning about radicals:
    if fragments_have_radicals:
        with open(os.path.join(basepath, "radicals.txt"), "w") as f:
            f.write("This complex contains radicals.")


def get_psi4_input(pl_complex: Chem.Mol) -> Dict:
    """Generate all the data Psi4 needs to construct a molecule

    Parameters
    ----------
    pl_complex : Chem.Mol
        complex of protein and ligand

    Returns
    -------
    Dict
        Psi4 input data
    """
    frags = Chem.GetMolFrags(pl_complex, asMols=True, sanitizeFrags=False)

    elements_by_fragment = [[a.GetAtomicNum() for a in mol.GetAtoms()] for mol in frags]
    elements_flattened = [elem for frag_elems in elements_by_fragment for elem in frag_elems]
    fragment_separators = [sum([len(elems) for elems in elements_by_fragment[:i]]) for i in range(1, len(frags))]
    fragment_charges = [int(Chem.GetFormalCharge(f)) for f in frags]
    fragment_multiplicities = [int(Descriptors.NumRadicalElectrons(f) * 0.5 * 2 + 1) for f in frags]
    molecular_charge = int(Chem.GetFormalCharge(pl_complex))
    molecular_multiplicity = int(Descriptors.NumRadicalElectrons(pl_complex) * 0.5 * 2 + 1)
    geom = np.concatenate([f.GetConformer().GetPositions() for f in frags])

    res = {
        "elez": elements_flattened,
        "fragment_separators": fragment_separators,
        "fragment_charges": fragment_charges,
        "fragment_multiplicities": fragment_multiplicities,
        "molecular_charge": molecular_charge,
        "molecular_multiplicity": molecular_multiplicity,
        "geom": geom,
    }
    return res


def remove_dummy_atoms(mol: Chem.Mol):
    """Remove unphysical atom types

    Parameters
    ----------
    mol : Chem.Mol
        molecule to clean
    """
    atoms_to_remove = []
    for a in mol.GetAtoms():
        if a.GetSymbol() in set(["*", "Du"]):
            atoms_to_remove.append(a.GetIdx())
    atoms_to_remove.sort(reverse=True)
    for idx in atoms_to_remove:
        mol.RemoveAtom(int(idx))


def has_radical(mol: Chem.Mol):
    """Check that the molecule doesn't have radicals. (not an exhaustive check,
    only briefly to catch odd number of electrons which can mess with the xTB calculation)

    Parameters
    ----------
    mol : RDKit molecule
        protein or ligand

    Raises
    ------
    ValueError
        If odd number of electrons
    """
    if mol.GetNumAtoms() == 1:
        return True  # most likely radicals in metals
    num_protons = sum([a.GetAtomicNum() for a in mol.GetAtoms()])
    assert sum([a.GetFormalCharge() for a in mol.GetAtoms()]) == Chem.GetFormalCharge(mol)
    total_charge = Chem.GetFormalCharge(mol)
    num_electrons = num_protons - total_charge
    if num_electrons % 2 != 0:
        return True
    else:
        return False


def check_radical_fragments(mol: Chem.Mol):
    """Check that the molecule doesn't have radicals. (not an exhaustive check,
    only briefly to catch odd number of electrons which can mess with the xTB calculation)

    Parameters
    ----------
    mol : RDKit molecule
        protein or ligand

    Returns
    ------
    bool
        Whether the molecule has radicals (True = has radicals)

    """
    assert sum([a.GetFormalCharge() for a in mol.GetAtoms()]) == Chem.GetFormalCharge(mol)  # sanity check
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    for frag in frags:
        if has_radical(frag):
            return True
    return False


def read_protein_and_ligand(folder: str, structure_id: str, dataset: str = "pdbbind") -> Tuple[Chem.Mol, Chem.Mol]:
    """Read protein and ligand from respective files (depending on dataset) as RDKit molecules

    Parameters
    ----------
    folder : str
        folder with protein and ligand files
    structure_id : str
        structure_identifier
    dataset : str, optional
        name of dataset, by default "pdbbind", alternative "pde10a"

    Returns
    -------
    Tuple[Chem.Mol, Chem.Mol]
        read-in molecules

    Raises
    ------
    ValueError
        if unknown dataset
    """
    if dataset == "pdbbind":
        pdb_id = structure_id
        structure_folder = os.path.join(folder, pdb_id)

        # read molecules
        protein_path = os.path.join(structure_folder, f"{pdb_id}_protein.mol2")
        ligand_path = os.path.join(structure_folder, f"{pdb_id}_ligand.mol2")
        protein = Chem.rdmolfiles.MolFromMol2File(
            protein_path, removeHs=False, sanitize=False, cleanupSubstructures=False
        )
        assert protein is not None
        protein = Chem.RWMol(protein)

        # add monomer info
        addInfoFromMol2(protein, protein_path, is_ligand=False, dataset=dataset)
        ligand = Chem.rdmolfiles.MolFromMol2File(
            ligand_path, removeHs=False, sanitize=False, cleanupSubstructures=False
        )
        addInfoFromMol2(ligand, ligand_path, is_ligand=True, dataset=dataset)
    elif dataset == "pde10a":
        structure_folder = os.path.join(folder, structure_id)

        # read molecules
        protein_path = os.path.join(structure_folder, f"{structure_id}_protein.mol2")
        ligand_path = os.path.join(structure_folder, f"{structure_id}_ligand.mol2")
        protein = Chem.rdmolfiles.MolFromMol2File(
            protein_path, removeHs=False, sanitize=False, cleanupSubstructures=False
        )
        assert protein is not None
        protein = Chem.RWMol(protein)

        # add monomer info
        addInfoFromMol2(protein, protein_path, is_ligand=False, dataset=dataset)
        ligand = Chem.rdmolfiles.MolFromMol2File(
            ligand_path, removeHs=False, sanitize=False, cleanupSubstructures=False
        )
        addInfoFromMol2(ligand, ligand_path, is_ligand=True, dataset=dataset)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    remove_dummy_atoms(ligand)
    remove_dummy_atoms(protein)
    assert ligand is not None
    assert protein is not None
    return ligand, protein


def add_hydrogens(path: str) -> Chem.Mol:
    """Helper function to read an SDF file and add hydrogens. NOT currently used because all
    used datasets (PDBbind Volkov & PDE10A) come pre-protonated.

    Parameters
    ----------
    path : str
        path to SDF file

    Returns
    -------
    Chem.Mol
        mol with hydrogens
    """
    mol_obabel = next(readfile("sdf", path))
    mol_obabel.OBMol.AddHydrogens(False, True, 7.4)  # polaronly, correctForPH, pH (physiological)
    mol_obabel.write("sdf", path, overwrite=True)
    mol = next(Chem.SDMolSupplier(path, removeHs=False, sanitize=False))
    assert mol is not None
    return mol


def full_structure_prep(
    folder: str, structure_id: str, output_basepath: str, cutoff: int = 6, dataset: str = "pdbbind"
):
    """Run full structure preparation starting from properly protonated ligand/protein
    obtained from https://pubs.acs.org/doi/10.1021/acs.jmedchem.2c00487
    (Volkov et al).
    Cuts protein at C_alpha/C_beta positions and in the backbone (similar to QM/MM) so that we can
    do xTB calculations later.

    Parameters
    ----------
    folder : str
        path to input folder
    output_basepath : str
        path to output folder basepath
    cutoff : int, optional
        cutoff distance in Angstrom, by default 6
    """

    print(f"{folder} --- {structure_id}")
    ligand, protein = read_protein_and_ligand(folder, structure_id, dataset=dataset)

    fix_phosphoric_acids(ligand)
    fix_phosphoric_esters(ligand)
    fix_guadinine_groups(ligand)
    fix_carboxylic_acids(ligand, dataset=dataset)
    fix_nitrogen_ion_groups(ligand)

    # remove overlying ligands if they exist
    remove_protein_overlap_with_ligand(ligand, protein)

    # preliminarily cut all C_alpha/C_beta bonds & backbone, determine which fragments are within cutoff of ligand
    idxs_within_cutoff, prop_assignment = get_idxs_within_cutoff(protein, ligand, cutoff=cutoff)

    transfer_props(protein, prop_assignment)

    # then truly cut all C_alpha/C_beta bonds for those fragments too far away
    cut_remote_fragments(protein, idxs_within_cutoff)

    # pad the cut bonds with hydrogen
    pad_with_protons(protein)

    # fix radical aldehydes
    fix_radical_aldehydes(protein)
    fix_guadinine_groups(protein)
    fix_carboxylic_acids(protein, dataset=dataset)
    fix_carbons_with_missing_hydrogens(protein)
    fix_nitrogens_with_missing_hydrogens(protein)
    fix_nitrogen_ion_groups(protein)

    # check for radical fragments
    fragments_have_radicals = check_radical_fragments(protein) or check_radical_fragments(ligand)

    write_to_file(output_basepath, structure_id, protein, ligand, fragments_have_radicals)

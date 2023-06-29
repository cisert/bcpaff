"""
© 2023, ETH Zurich
"""

import os

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.spatial.distance import cdist

from bcpaff.utils import ANGSTROM2BOHR, BOHR2ANGSTROM

SEP = "----------------"

ESP_OFFSET = -2
SIGMA_TO_NAME = {
    -3: "nucleus_critical_point",
    -1: "bond_critical_point",
    1: "ring_critical_point",
    3: "cage_critical_point",
}
QTAIM_UNITS = {
    "density": "electrons/Angstrom^3",
    "laplacian": "electrons/Angstrom^5",
    "elf": "-",
    "lol": "-",
    "rdg": "-",
    "sign_lap_rho": "electrons/Angstrom^3",
    "gradient_norm": "electrons/Angstrom^4",
    "hessian_eigenvalues": "-",
    "ellipticity": "-",
    "eta_index": "-",
    "esp": "J/C = eV/e",
    "position": "Angstrom",
}

ABBREVIATIONS = {
    "nucleus_critical_point": "NCP",
    "bond_critical_point": "BCP",
    "ring_critical_point": "RCP",
    "cage_critical_point": "CCP",
}


class QtaimProps(object):
    def __init__(
        self,
        basepath=None,
        cp_file=None,
        cpprop_file=None,
        paths_file=None,
        ligand_sdf=None,
        pl_complex_xyz=None,
        identifier=None,
    ):
        self.basepath = basepath
        self.identifier = identifier
        self.cp_file = cp_file
        self.paths_file = paths_file
        self.cpprop_file = cpprop_file
        self.pl_complex_xyz = pl_complex_xyz
        self.ligand_sdf = ligand_sdf
        if self.cp_file is None:
            self.identifier = os.path.basename(self.basepath)
            self.cp_file = os.path.join(self.basepath, "CPs.txt")
            self.paths_file = os.path.join(self.basepath, "paths.txt")
            self.cpprop_file = os.path.join(self.basepath, "CPprop.txt")
            self.pl_complex_xyz = os.path.join(self.basepath, "pl_complex.xyz")
            self.ligand_sdf = os.path.join(self.basepath, f"{self.identifier}_ligand_with_hydrogens.sdf")
        self.atom_ids, self.pl_complex_coords = self._get_pl_complex_info(self.pl_complex_xyz)
        self.ligand = next(Chem.SDMolSupplier(self.ligand_sdf, removeHs=False, sanitize=False))
        self.natoms_ligand = self.ligand.GetNumAtoms()
        self.critical_points = []
        self._read_critical_points()
        self.cp_positions = np.vstack([cp.position for cp in self.critical_points])
        self._read_paths()
        self.num_cps = len(self.critical_points)
        self._get_df()

    def _get_df(self):
        df = pd.DataFrame([cp.idx for cp in self.critical_points], columns=["idx"])
        df = pd.concat(
            [df, pd.DataFrame([ABBREVIATIONS[cp.name] for cp in self.critical_points], columns=["point_name"])], axis=1
        )
        df = pd.concat(
            [df, pd.DataFrame([cp.intermolecular for cp in self.critical_points], columns=["intermolecular"])], axis=1
        )
        df = pd.concat([df, pd.DataFrame([cp.props for cp in self.critical_points])], axis=1)
        df = pd.concat([df, pd.DataFrame(self.cp_positions, columns=["x", "y", "z"])], axis=1)
        atom_neighbors = []
        for cp in self.critical_points:
            if len(cp.atom_neighbors) == 0:
                atom_neighbors.append([float("NaN"), float("NaN")])
            elif len(cp.atom_neighbors) == 1:
                atom_neighbors.append(cp.atom_neighbors + [float("NaN")])
            elif len(cp.atom_neighbors) == 2:
                atom_neighbors.append(sorted(cp.atom_neighbors))
            else:
                raise ValueError("Incorrect number of atom neighbors")
        df = pd.concat([df, pd.DataFrame(atom_neighbors, columns=["atom_neighbor_1", "atom_neighbor_2"])], axis=1)
        self.df = df

    def _get_pl_complex_info(self, pl_complex_xyz):
        with open(pl_complex_xyz, "r") as f:
            lines = [l.rstrip("\n").split() for l in f.readlines()[2:]]
        atom_ids, coords = zip(*[(l[0], [float(x) for x in l[1:]]) for l in lines])
        return np.asarray(atom_ids), np.asarray(coords)

    def _read_critical_points(self):
        with open(self.cpprop_file, "r") as f:
            lines = [line.lstrip(" ").rstrip("\n") for line in f.readlines()]
        self.include_esp = True if any([line.startswith("Total ESP") for line in lines]) else False
        split_idx = [i for i, line in enumerate(lines) if (line.startswith(SEP) and line.endswith(SEP))]
        split_idx.remove(0)
        blocks = [lines[i:j] for i, j in zip([0] + split_idx, split_idx + [None])]
        self.num_paths = len(blocks)
        for block in blocks:
            cp = CriticalPoint(
                block, self.include_esp, pl_complex_coords=self.pl_complex_coords, atom_ids=self.atom_ids
            )
            if cp.name == "nucleus_critical_point" and cp.corresponding_atom_symbol == "Unknown":
                continue  #  don't append NCPs with unknown atom
            self.critical_points.append(cp)

    def _read_paths(self):
        with open(self.paths_file, "r") as f:
            lines = [line.lstrip(" ").rstrip("\n") for line in f.readlines()]
        split_idx = [i for i, line in enumerate(lines) if line == ""]
        split_idx.remove(1)
        blocks = [lines[i:j] for i, j in zip([1] + split_idx, split_idx + [None])]
        for block in blocks:
            path_positions = np.array([[x for x in line.split()] for line in block[3:]]).astype(float) * BOHR2ANGSTROM

            # path always starts at critical point (path_positions[0, None]) and ends at atom (path_positions[-1, None])
            atom_id = cdist(path_positions[-1, None], self.pl_complex_coords).argmin()
            cp_id = cdist(path_positions[0, None], self.cp_positions).argmin()
            self.critical_points[cp_id]._add_path(atom_id, self.atom_ids[atom_id], path_positions, self.natoms_ligand)


class CriticalPoint(object):
    def __init__(self, block, include_esp, pl_complex_coords, atom_ids):
        self.include_esp = include_esp
        self.esp_offset = 0 if include_esp else ESP_OFFSET
        self.pl_complex_coords = pl_complex_coords
        self.atom_ids = atom_ids
        self.atom_neighbors = []
        self.atom_neighbors_symbol = []
        self.path_positions = None
        self._read_block(block)
        self.intermolecular = False  # can be overwritten by _add_paths later

    def _read_block(self, block):
        for line in block:
            if line.startswith(SEP):
                self.idx = int(line.strip("- ").split("Type")[0].strip("CP, "))
                omega, sigma = line.strip("- ").split("Type")[1].strip("() ").split(",")
                self.omega = int(omega)
                self.sigma = int(sigma)
                self.name = SIGMA_TO_NAME[self.sigma]
            elif line.startswith("Corresponding nucleus:"):  # only for nucleus-critical points
                corresponding_nucleus = line.lstrip("Corresponding nucleus: ").split("(")[0]
                if corresponding_nucleus == "Unknown":  # no corresponding nucleus found
                    self.corresponding_atom_symbol = "Unknown"  # figured this out later from position
                else:
                    self.corresponding_atom_id = int(corresponding_nucleus) - 1  # zero indexing
                    self.corresponding_atom_symbol = line.lstrip("Corresponding nucleus: ").split("(")[-1].rstrip(" )")
            elif line.startswith("Position (Bohr):"):
                self.position = np.array([float(x) * BOHR2ANGSTROM for x in line.lstrip("Position (Bohr): ").split()])
                if (
                    self.name == "nucleus_critical_point" and self.corresponding_atom_symbol == "Unknown"
                ):  # need to fix
                    distance_matrix = cdist(np.expand_dims(self.position, axis=0), self.pl_complex_coords)
                    assert distance_matrix.min() < 0.3  # sanity check, 0.3 Angstrom = manually inspected cutoff
                    self.corresponding_atom_id = distance_matrix.argmin()
                    self.corresponding_atom_symbol = self.atom_ids[self.corresponding_atom_id]
            elif line.startswith("Density of all electrons:"):
                self.density = float(line.lstrip("Density of all electrons: ")) * (ANGSTROM2BOHR ** 3)
            elif line.startswith("Laplacian of electron density:"):
                self.laplacian = float(line.lstrip("Laplacian of electron density: ")) * (ANGSTROM2BOHR ** 5)
            elif line.startswith("Electron localization function (ELF):"):
                self.elf = float(line.lstrip("Electron localization function (ELF): "))
            elif line.startswith("Localized orbital locator (LOL):"):
                self.lol = float(line.lstrip("Localized orbital locator (LOL): "))
            elif line.startswith("Reduced density gradient (RDG):"):
                self.rdg = float(line.lstrip("Reduced density gradient (RDG): "))
            elif line.startswith("Sign(lambda2)*rho:"):
                self.sign_lap_rho = float(line.lstrip("Sign(lambda2)*rho: ")) * (ANGSTROM2BOHR ** 3)
            elif line.startswith("Total ESP:") and self.include_esp:
                self.esp = float(line.lstrip("Total ESP:  ").split("(")[0].split("a.u.")[0].strip(" "))
            elif line.startswith("ESP from nuclear charges:  ") and self.include_esp:
                self.esp_nuc = float(line.lstrip("ESP from nuclear charges:  "))
            elif line.startswith("ESP from electrons:") and self.include_esp:
                self.esp_ele = float(line.lstrip("ESP from electrons: "))
            elif line.startswith("Norm of gradient is:"):
                self.gradient_norm = float(line.lstrip("Norm of gradient is: ")) * (ANGSTROM2BOHR ** 4)
            elif line.startswith("Eigenvalues of Hessian:"):
                matrix = np.array([float(x) for x in line.lstrip("Eigenvalues of Hessian: ").split()])
                self.hessian_eigenvalues = matrix
            elif line.startswith("Ellipticity of electron density:"):
                self.ellipticity = float(line.lstrip("Ellipticity of electron density: "))
            elif line.startswith("eta index:"):
                self.eta_index = float(line.lstrip("eta index: "))

        self.props = {
            "density": self.density,
            "laplacian": self.laplacian,
            "elf": self.elf,
            "lol": self.lol,
            "rdg": self.rdg,
            "sign_lap_rho": self.sign_lap_rho,
            "gradient_norm": self.gradient_norm,
            "hessian_eigenvalues": self.hessian_eigenvalues,
            "ellipticity": self.ellipticity,
            "eta_index": self.eta_index,
        }
        if self.include_esp:
            self.props.update({"esp": self.esp, "esp_nuc": self.esp_nuc, "esp_ele": self.esp_ele})

    def _add_path(self, atom_id, atom_symbol, path_positions, natoms_ligand):
        self.atom_neighbors.append(atom_id)
        self.atom_neighbors_symbol.append(atom_symbol)
        if self.path_positions is None:
            self.path_positions = np.flip(
                path_positions, axis=0
            )  # so that complete array goes from atom1 --> BCP --> atom2
            # the bond path doesn't end perfectly at the atom, but just before (distance ca. 0.05 Angstrom)
        else:
            self.path_positions = np.vstack([self.path_positions, path_positions])
        if sum([a <= natoms_ligand - 1 for a in self.atom_neighbors]) == 1:
            # one and only one atom neighbor belongs to ligand --> intermolecular
            self.intermolecular = True
        else:
            # either none or both atom neighbors belong to ligand --> not intermolecular
            self.intermolecular = False


class CriticalPointCritic2(object):
    def __init__(self, i, row, pl_complex_coords, atom_ids):
        self.idx = i
        assert self.idx == row.cp_idx - 1  # zero indexing
        self.pl_complex_coords = pl_complex_coords
        self.atom_ids = atom_ids
        self.atom_neighbors = []
        self.atom_neighbors_symbol = []
        self.path_positions = None
        # self._read_block(block)
        self.intermolecular = False  # can be overwritten by _add_paths later
        self.density = row.edens * (ANGSTROM2BOHR ** 3)  # convert to electrons/Angstrom^3 etc.
        self.gradient_norm = row.grad * (ANGSTROM2BOHR ** 4)
        self.laplacian = row.lap * (ANGSTROM2BOHR ** 5)
        omega, sigma = row.type.strip("()").split(",")
        self.omega, self.sigma = int(omega), int(sigma)
        self.name = SIGMA_TO_NAME[self.sigma]
        if row.name == "nucleus_critical_point":
            self.corresponding_atom_id = self.idx  # already zero-indexed above
            self.corresponding_atom_symbol = row.type_name
            assert self.atom_ids[self.corresponding_atom_id] == self.corresponding_atom_symbol  # sanity check
        self.position = row[["x", "y", "z"]].to_numpy().astype(float)
        self.props = {
            "density": self.density,
            "laplacian": self.laplacian,
            "gradient_norm": self.gradient_norm,
        }

    def _add_path(self, atom_ids, atom_symbols, path_positions, natoms_ligand):
        self.atom_neighbors = atom_ids
        self.atom_neighbors_symbol = atom_symbols
        self.path_positions = path_positions

        if sum([a <= natoms_ligand - 1 for a in self.atom_neighbors]) == 1:
            # one and only one atom neighbor belongs to ligand --> intermolecular
            self.intermolecular = True
        else:
            # either none or both atom neighbors belong to ligand --> not intermolecular
            self.intermolecular = False


class QtaimPropsCritic2(object):
    def __init__(self, basepath=None, output_cri=None, pl_complex_xyz=None, ligand_sdf=None):
        self.basepath = basepath
        self.output_cri = output_cri if output_cri is not None else os.path.join(basepath, "output.cri")
        self.pl_complex_xyz = (
            pl_complex_xyz if pl_complex_xyz is not None else os.path.join(basepath, "pl_complex.xyz")
        )
        if ligand_sdf is not None:
            self.ligand_sdf = ligand_sdf
            self.ligand = next(Chem.SDMolSupplier(self.ligand_sdf, removeHs=False, sanitize=False))
        else:
            self.ligand_sdf = self.pl_complex_xyz  # everything is ligand
            self.ligand = Chem.rdmolfiles.MolFromXYZFile(self.ligand_sdf)
        self.natoms_ligand = self.ligand.GetNumAtoms()
        self.atom_ids, self.pl_complex_coords = self._get_pl_complex_info(self.pl_complex_xyz)

        self.critical_points = []
        self._read_critical_points()
        self.cp_positions = np.vstack([cp.position for cp in self.critical_points])
        self._read_paths()
        self.num_cps = len(self.critical_points)

    def _get_pl_complex_info(self, pl_complex_xyz):
        with open(pl_complex_xyz, "r") as f:
            lines = [l.rstrip("\n").split() for l in f.readlines()[2:]]
        atom_ids, coords = zip(*[(l[0], [float(x) for x in l[1:]]) for l in lines])
        return np.asarray(atom_ids), np.asarray(coords)

    def _read_critical_points(self):
        with open(self.output_cri, "r") as f:
            lines = [line.lstrip(" ").rstrip("\n") for line in f.readlines()]
        SEP_CRITIC2 = "Poincare-Hopf sum:"
        start_idx = [i for i, line in enumerate(lines) if line.startswith(SEP_CRITIC2)]
        assert len(start_idx) == 1
        start_idx = start_idx[0]
        self.poincare_hopf_sum = int(lines[start_idx].split(": ")[-1])
        self.found_all_cps = self.poincare_hopf_sum == 1
        if self.poincare_hopf_sum != 1:
            with open(os.path.join(os.path.dirname(self.output_cri), "missing_points"), "w") as f:
                f.write(f"Poincare-Hopf sum = {self.poincare_hopf_sum}")
        end_idx = lines.index("* Analysis of system bonds")
        df_lines = lines[start_idx + 2 : end_idx - 1]
        df_lines = [l.split() for l in df_lines]
        # sometimes splitting gets messed up because of "(3,1 )"
        for i, l in enumerate(df_lines):
            if len(l) == 11:  # instead of the usual 10
                l = [l[0]] + [l[1] + l[2]] + l[3:]  # string concat for elems 1 & 2
                df_lines[i] = l
        df = pd.DataFrame(
            df_lines, columns=["cp_idx", "type", "name", "x", "y", "z", "type_name", "edens", "grad", "lap"]
        )
        dtypes = {
            "cp_idx": int,
            "type": str,
            "name": str,
            "x": float,
            "y": float,
            "z": float,
            "type_name": str,
            "edens": float,
            "grad": float,
            "lap": float,
        }
        df = df.astype(dtypes)

        for i, row in df.iterrows():
            cp = CriticalPointCritic2(i, row, self.pl_complex_coords, self.atom_ids)
            self.critical_points.append(cp)

    def _read_paths(self):
        with open(self.output_cri, "r") as f:
            lines = [line.lstrip(" ").rstrip("\n") for line in f.readlines()]
        start_idx = lines.index("# ncp   End-1      End-2    r1(ang_)   r2(ang_)     r1/r2   r1-B-r2 (degree)")
        end_idx = lines.index("* Analysis of system rings")
        df_lines = lines[start_idx + 1 : end_idx - 1]
        df_lines = [l.split()[:5] for l in df_lines]
        df_lines = [[x.strip("()") for x in l] for l in df_lines]
        df = pd.DataFrame(
            df_lines, columns=["cp_idx", "neighbor_type_1", "neighbor_id_1", "neighbor_type_2", "neighbor_id_2"]
        )
        dtypes = {
            "cp_idx": int,
            "neighbor_type_1": str,
            "neighbor_id_1": int,
            "neighbor_type_2": str,
            "neighbor_id_2": int,
        }
        df = df.astype(dtypes)
        df.loc[:, "cp_idx"] = df.cp_idx - 1  # zero indexing
        df.loc[:, "neighbor_id_1"] = df.neighbor_id_1 - 1  # zero indexing
        df.loc[:, "neighbor_id_2"] = df.neighbor_id_2 - 1  # zero indexing

        for _, row in df.iterrows():
            path_positions = self.cp_positions[[row.neighbor_id_1, row.cp_idx, row.neighbor_id_2]]
            self.critical_points[row.cp_idx]._add_path(
                [row.neighbor_id_1, row.neighbor_id_2],
                [row.neighbor_type_1, row.neighbor_type_2],
                path_positions,
                self.natoms_ligand,
            )


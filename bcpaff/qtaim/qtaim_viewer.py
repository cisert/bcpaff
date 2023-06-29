"""
© 2023, ETH Zurich
"""

import os

import numpy as np
import py3Dmol
from rdkit import Chem


def get_radii(x):
    MAX_RADIUS = 0.8  # very transparent, visually implies low importance
    MIN_RADIUS = 0.2  # not very transparent (quite solid color), visually implies high importance
    x = np.abs(x)
    x = x - np.min(x)
    x = x / np.max(x)
    x = x * (MAX_RADIUS - MIN_RADIUS) + MIN_RADIUS
    return x


class QtaimViewer(object):
    def __init__(
        self,
        qtaim_props,
        only_intermolecular=True,
        detailed_paths=False,
        attributions_data=None,
        width=640,
        height=480,
    ):
        self.v = py3Dmol.view(width=width, height=height)
        with open(os.path.join(os.path.dirname(qtaim_props.ligand_sdf), "pl_complex.xyz"), "r") as f:
            xyz_str = f.read()
        self.v.addModel(xyz_str, "xyz")
        self.v.setStyle({"model": 0}, {"stick": {"colorscheme": "lightgrayCarbon", "radius": 0.1}})
        self.v.addModel(Chem.MolToMolBlock(qtaim_props.ligand, kekulize=False), "mol")
        self.v.setStyle({"model": 1}, {"stick": {"colorscheme": "blackCarbon", "radius": 0.2}})
        self.v.setBackgroundColor("white")
        self.v.zoomTo()
        for cp in qtaim_props.critical_points:
            if only_intermolecular:
                if not (cp.name == "bond_critical_point" and cp.intermolecular):
                    continue
            if cp.name != "bond_critical_point":
                continue
            if detailed_paths:
                points = cp.path_positions
            else:
                points = cp.path_positions[np.round(np.linspace(0, len(cp.path_positions) - 1, 5)).astype(int)]
                # first, last, and some in between (need first & last to have proper attachment to atoms)
            points = [{key: val for (key, val) in zip(["x", "y", "z"], pos)} for pos in points]

            self.v.addCurve({"points": points, "radius": 0.05, "color": "yellow"})
            self.v.addSphere(
                {
                    "center": {key: val for (key, val) in zip(["x", "y", "z"], cp.position)},
                    "radius": 0.1,
                    "color": "red",
                }
            )
        if attributions_data is not None:
            # map attribution values to transparency values and colors
            radii = get_radii(attributions_data["attributions"])
            colors = ["green" if x > 0 else "red" for x in attributions_data["attributions"]]

            # plot
            for xyz, c, r in zip(attributions_data["coords"], colors, radii):
                self.v.addSphere(
                    {
                        "center": {key: val for (key, val) in zip(["x", "y", "z"], xyz.tolist())},
                        "radius": float(r),
                        "color": c,
                    }
                )

    def show(self):
        return self.v.show()

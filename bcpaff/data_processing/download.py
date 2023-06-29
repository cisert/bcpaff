"""
© 2023, ETH Zurich
"""

import os
import shutil
import tarfile
import zipfile

import requests
from tqdm import tqdm

from bcpaff.utils import DATA_PATH

INDEX_URL = "http://www.pdbbind.org.cn/download/PDBbind_v2019_plain_text_index.tar.gz"

PDBBIND_STUCTURES_URL = "http://bioinfo-pharma.u-strasbg.fr/labwebsite/downloads/pdbbind.tgz"  # PDBbind processed by Volkov et al. (10.1021/acs.jmedchem.2c00487)
PDE10A_STRUCTURES_URL = (
    "https://figshare.com/ndownloader/files/37712256"  # Tosstorff et al (10.1007/s10822-022-00478-x)
)
PDE10A_AFFINITY_URL = "https://static-content.springer.com/esm/art%3A10.1007%2Fs10822-022-00478-x/MediaObjects/10822_2022_478_MOESM2_ESM.csv"


def download(src: str, dest: str):
    """Simple requests.get with a progress bar
    Parameters
    ----------
    src : str
        Remote path to be downloaded
    dest : str
        Local path for the download
    Returns
    -------
    None
    """
    r = requests.get(src, stream=True)
    tsize = int(r.headers.get("content-length", 0))
    progress = tqdm(total=tsize, unit="iB", unit_scale=True, position=0, leave=False)

    with open(dest, "wb") as handle:
        progress.set_description(os.path.basename(dest))
        for chunk in r.iter_content(chunk_size=1024):
            handle.write(chunk)
            progress.update(len(chunk))


def download_pdbbind_data():
    """
    Download PDBbind data as prepared by Volkov et al.
    (10.1021/acs.jmedchem.2c00487)
    """
    pdbbind_structure_path = os.path.join(DATA_PATH, "pdbbind")
    pdbbind_csv = os.path.join(
        pdbbind_structure_path, "PDBbind_v2019_plain_text_index/plain-text-index/index/INDEX_general_PL_data.2019"
    )
    if not os.path.exists(pdbbind_csv):
        os.makedirs(pdbbind_structure_path, exist_ok=True)

        # download affinity data
        dest_archive = os.path.join(pdbbind_structure_path, os.path.basename(INDEX_URL))
        dest_extract = dest_archive[: -len(".tar.gz")]
        download(INDEX_URL, dest_archive)
        with tarfile.open(dest_archive) as handle:
            handle.extractall(dest_extract)
        os.remove(dest_archive)

        # download structure data
        dest_archive = os.path.join(pdbbind_structure_path, os.path.basename(PDBBIND_STUCTURES_URL))
        dest_extract = dest_archive[: -len(".tgz")]
        download(PDBBIND_STUCTURES_URL, dest_archive)
        print("Extracting, takes a few minutes...")
        with tarfile.open(dest_archive) as handle:
            handle.extractall(dest_extract)
        shutil.move(os.path.join(dest_extract, "dataset"), os.path.join(pdbbind_structure_path, "dataset"))
        shutil.move(os.path.join(dest_extract, "pdb_ids"), os.path.join(pdbbind_structure_path, "pdb_ids"))
        os.remove(dest_archive)
        shutil.rmtree(dest_extract)


def download_pde10a_data():
    """
    Download PDE10A inhibitor data from Tosstorff et al.
    (10.1007/s10822-022-00478-x)
    """
    pde10a_structure_path = os.path.join(DATA_PATH, "pde10a")
    pde10a_csv = os.path.join(pde10a_structure_path, "10822_2022_478_MOESM2_ESM.csv")
    if not os.path.exists(pde10a_csv):
        os.makedirs(pde10a_structure_path, exist_ok=True)

        download(PDE10A_AFFINITY_URL, pde10a_csv)

        dest_archive = os.path.join(pde10a_structure_path, "pde-10_pdb_bind_format_blinded.zip")
        download(PDE10A_STRUCTURES_URL, dest_archive)
        print("Extracting, takes a few minutes...")

        with zipfile.ZipFile(dest_archive, "r") as handle:
            handle.extractall(pde10a_structure_path)

        os.remove(dest_archive)
        if os.path.exists(os.path.join(pde10a_structure_path, "__MACOSX")):
            shutil.rmtree(os.path.join(pde10a_structure_path, "__MACOSX"))


if __name__ == "__main__":
    download_pdbbind_data()
    download_pde10a_data()

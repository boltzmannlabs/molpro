from molpro.site_pred.model import SitePredModel
from molpro.utils.preprocess import make_3dgrid, Featurizer
import numpy as np
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing
import torch
from scipy.spatial.distance import cdist
import openbabel
from openbabel import pybel
from typing import List


def load_model(checkpoint_path: str):
    """Function for loading pytorch lightning module from checkpoint
        Parameters
        ----------
        checkpoint_path: str,
            Path to pytorch lightning checkpoint
    """
    model = SitePredModel.load_from_checkpoint(checkpoint_path)
    model.freeze()
    return model


def featurize(file_path: str, file_type: str):
    """Function for extracting features and coordinates from a given input file
        Parameters
        ----------
        file_path: str,
            Path to input file
        file_type: str,
            Format of input file ('pdb', 'mol2', 'smi')
    """
    protein_featurizer = Featurizer(file_path, file_type,
                                    named_props=['hyb', 'heavydegree', 'heterodegree', 'partialcharge'],
                                    smarts_labels=['hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring'],
                                    metal_halogen_encode=True)
    return protein_featurizer.coords, protein_featurizer.features, protein_featurizer.mol


def pocket_density_from_file(file_path: str, file_type: str, checkpoint_path: str, max_dist: int, grid_resolution: int):
    """Function for extracting features and coordinates from a given input file
        Parameters
        ----------
        file_path: str,
            Path to input file
        file_type: str,
            Format of input file ('pdb', 'mol2', 'smi')
        checkpoint_path: str,
            Path to pytorch lightning checkpoint
        grid_resolution: float,
            Resolution of a grid (in Angstroms).
        max_dist: float,
            Maximum distance between atom and box center. Resulting box has size of
            2*`max_dist`+1 Angstroms and atoms that are too far away are not
            included.
    """
    model = load_model(checkpoint_path)
    prot_coords, prot_features, mol = featurize(file_path, file_type)
    centroid = prot_coords.mean(axis=0)
    prot_coords -= centroid
    x = make_3dgrid(prot_coords, prot_features,
                    max_dist=max_dist,
                    grid_resolution=grid_resolution)
    x = x.transpose((0, 4, 1, 2, 3))
    grid = torch.tensor(x)
    density = model(grid)
    density = density.detach().numpy()
    density = density.transpose((0, 2, 3, 4, 1))
    origin = (centroid - max_dist)
    step = np.array([grid_resolution] * 3)
    return density, origin, step, mol


def get_pockets_segmentation(density: np.ndarray, threshold: float, min_size: int, grid_resolution: int):
    """Predict pockets using specified threshold on the probability density.
       Parameters
        ----------
       density: np.ndarray,
            Predicted probability density grid from input protein
       threshold: float,
            Extract voxels with specified threshold
       min_size: int,
            Filter out pockets smaller than min_size A^3
       grid_resolution: float,
            Resolution of a grid (in Angstroms).
    """
    voxel_size = grid_resolution ** 3
    bw = closing((density[0] > threshold).any(axis=-1))

    cleared = clear_border(bw)

    label_image, num_labels = label(cleared, return_num=True)
    for i in range(1, num_labels + 1):
        pocket_idx = (label_image == i)
        pocket_size = pocket_idx.sum() * voxel_size
        if pocket_size < min_size:
            label_image[np.where(pocket_idx)] = 0
    return label_image


def predict_pocket_atoms(file_path: str, file_type: str, checkpoint_path: str, max_dist: int = 35,
                         grid_resolution: int = 2, threshold: float = 0.5, min_size: int = 1,
                         dist_cutoff: float = 6, expand_residue: bool = True):
    """Predict pockets using specified threshold on the probability density. Output atoms based on predicted pockets
       Parameters
        ----------
       file_path: str,
            Path to input file
       file_type: str,
            Format of input file ('pdb', 'mol2', 'smi')
       checkpoint_path: str,
            Path to pytorch lightning checkpoint
       grid_resolution: float,
            Resolution of a grid (in Angstroms).
       max_dist: float,
            Maximum distance between atom and box center. Resulting box has size of
            2*`max_dist`+1 Angstroms and atoms that are too far away are not
            included.
       threshold: float,
            Extract voxels with specified threshold
       min_size: int,
            Filter out pockets smaller than min_size A^3
       dist_cutoff: float,
            Distance cutoff to include atoms around predicted pocket.
       expand_residue: bool,
            Whether to include residues around the predicted pocket.
    """
    density, origin, step, mol = pocket_density_from_file(file_path, file_type, checkpoint_path, max_dist,
                                                          grid_resolution)
    coords = np.array([a.coords for a in mol.atoms])
    atom2residue = np.array([a.residue.idx for a in mol.atoms])
    residue2atom = np.array([[a.idx - 1 for a in r.atoms]
                             for r in mol.residues])

    # predict pockets
    pockets = get_pockets_segmentation(density, threshold, min_size, grid_resolution)

    # find atoms close to pockets
    pocket_atoms = []
    for pocket_label in range(1, pockets.max() + 1):
        indices = np.argwhere(pockets == pocket_label).astype('float32')
        indices *= step
        indices += origin
        distance = cdist(coords, indices)
        close_atoms = np.where((distance < dist_cutoff).any(axis=1))[0]
        if len(close_atoms) == 0:
            continue
        if expand_residue:
            residue_ids = np.unique(atom2residue[close_atoms])
            close_atoms = np.concatenate(residue2atom[residue_ids])
        pocket_atoms.append([int(idx) for idx in close_atoms])

    pocket_mols = []

    for pocket in pocket_atoms:
        pocket_mol = mol.clone
        atoms_to_del = (set(range(len(pocket_mol.atoms)))
                        - set(pocket))
        pocket_mol.OBMol.BeginModify()
        for aidx in sorted(atoms_to_del, reverse=True):
            atom = pocket_mol.OBMol.GetAtom(aidx + 1)
            pocket_mol.OBMol.DeleteAtom(atom)
        pocket_mol.OBMol.EndModify(False)
        pocket_mols.append(pocket_mol)
    return pocket_mols


def write_to_file(mols: List[openbabel.pybel.Molecule], file_path: str, file_type: str):
    """Function for writing predicted pockets to required file format
        Parameters
        ----------
        mols: List[openbabel.pybel.Molecule],
            List containing predicted pockets
        file_path: str,
            Path to output file along with file name
        file_type: str,
            Format of input file ('pdb', 'mol2', 'smi')
    """
    for i, pocket in enumerate(mols):
        pocket.write(file_type, '%s/pocket_%i.pdb' % (file_path, i), overwrite=True)

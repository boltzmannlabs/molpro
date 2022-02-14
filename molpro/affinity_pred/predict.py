from molpro.affinity_pred.model import AffinityPredModel
from molpro.utils.preprocess import make_3dgrid, Featurizer
import numpy as np
import torch


def load_model(checkpoint_path: str):
    """Function for loading pytorch lightning module from checkpoint
        Parameters
        ----------
        checkpoint_path: str,
            Path to pytorch lightning checkpoint
    """
    model = AffinityPredModel.load_from_checkpoint(checkpoint_path)
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
    protein_featurizer = Featurizer(file_path, file_type, named_props=['partialcharge'],
                                    smarts_labels=['aromatic', 'acceptor', 'donor'],
                                    metal_halogen_encode=False)
    return protein_featurizer.coords, protein_featurizer.features


def predict_affinity(protein_file_path: str, protein_file_type: str, ligand_file_path: str, ligand_file_type: str,
                     checkpoint_path: str, max_dist: int = 24, grid_resolution: int = 1):
    model = load_model(checkpoint_path)
    prot_coords, prot_features = featurize(protein_file_path, protein_file_type)
    ligand_coords, ligand_features = featurize(ligand_file_path, ligand_file_type)
    centroid = prot_coords.mean(axis=0)
    prot_coords -= centroid
    ligand_coords -= centroid
    rec_grid = make_3dgrid(prot_coords, prot_features, max_dist=max_dist,
                           grid_resolution=grid_resolution)
    lig_grid = make_3dgrid(ligand_coords, ligand_features, max_dist=max_dist,
                           grid_resolution=grid_resolution)
    rec_grid = rec_grid.squeeze(0).transpose((3, 0, 1, 2))
    lig_grid = lig_grid.squeeze(0).transpose((3, 0, 1, 2))
    x = torch.tensor((np.concatenate((rec_grid, lig_grid))))
    pkd = float(model(x))
    return pkd

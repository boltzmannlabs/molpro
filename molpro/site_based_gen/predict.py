import torch
from molpro.utils.preprocess import make_3dgrid, Featurizer
from molpro.shape_based_gen.predict import unique_canonical, initialize_model, decode_smiles
from molpro.site_based_gen.model import SiteGenModel


def load_models(generator_checkpoint_path: str, captioning_checkpoint_path: str):
    """Function for loading pytorch lightning module from checkpoint
        Parameters
        ----------
        generator_checkpoint_path: str,
            Path to pytorch lightning checkpoint for generator model
        captioning_checkpoint_path: str,
            Path to pytorch lightning checkpoint for captioning model
    """
    gen_model = SiteGenModel.load_from_checkpoint(generator_checkpoint_path)
    cap_model = initialize_model(captioning_checkpoint_path)
    gen_model.freeze()
    cap_model.freeze()
    return gen_model, cap_model


def featurize(file_path: str, file_type: str):
    """Function for extracting features and coordinates from a given input file
        Parameters
        ----------
        file_path: str,
            Path to input file
        file_type: str,
            Format of input file ('pdb', 'mol2', 'smi')
    """
    protein_featurizer = Featurizer(file_path, file_type, named_props=['partialcharge', 'heavydegree'],
                                    smarts_labels=['hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring'],
                                    metal_halogen_encode=False)
    return protein_featurizer.coords, protein_featurizer.features


def generate_smiles(protein_file_path: str, protein_file_type: str, generator_checkpoint_path: str,
                    captioning_checkpoint_path: str, max_dist: int = 23, grid_resolution: int = 2,
                    generator_steps: int = 2, decoding_steps: int = 2):
    gen_model, cap_model = load_models(generator_checkpoint_path, captioning_checkpoint_path)
    prot_coords, prot_features = featurize(protein_file_path, protein_file_type)
    centroid = prot_coords.mean(axis=0)
    prot_coords -= centroid
    rec_grid = make_3dgrid(prot_coords, prot_features, max_dist=max_dist,
                           grid_resolution=grid_resolution)
    rec_grid = rec_grid.transpose((0, 4, 1, 2, 3))
    x = torch.tensor(rec_grid)
    generated_smiles = dict()
    for _ in range(generator_steps):
        z = gen_model.get_z_random(1)
        pred = gen_model(x, z)
        lig_grid1 = pred[:, :5, :, :, :]
        lig_grid2 = pred[:, 3:, :, :, :]
        lig_grid = torch.cat((lig_grid1, lig_grid2), axis=1)
        model_outputs = [[cap_model.prediction(lig_grid, sample_prob=True)] for i in range(decoding_steps)]
        decoded_smiles = decode_smiles(model_outputs=model_outputs)
        generated_smiles["smi"] = unique_canonical(decoded_smiles)
    return generated_smiles

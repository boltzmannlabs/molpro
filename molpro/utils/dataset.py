import numpy as np
from rdkit import Chem
from typing import List
import torch
from torch.utils.data import Dataset
from molpro.utils.preprocess import make_3dgrid, Featurizer, rotate_grid
from molpro.shape_based_gen.data import vocab_c2i_v1
from random import choice


class ShapeBasedGenDataset(Dataset):
    """ Class to featurize smile while training 

    Input Parameters :
    -------------------------

    smiles_list : List[str]
                 a list which contains smiles

    file_type : str 
                by which file format smiles are extracted .smi or other
    
    """

    def __init__(self,smiles_list: List[str],file_type:str="smi"):

        self.smiles_list = smiles_list
        self.file_type = file_type

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx: int):
        smi = self.smiles_list[idx]

        smiles_token = smi
        #featurizer = featurization.Featurizer(smiles_token,file_type= 'smi')
        fatures_list = ['hydrophobic', 'aromatic','acceptor', 'donor','ring']

        featurizer = Featurizer(input_file = smi , file_type= 'smi', named_props  = ["partialcharge"], smarts_labels = fatures_list, metal_halogen_encode = False)


        coords = featurizer.coords
        centroid = coords.mean(axis=0)
        coords -= centroid
        afeats = featurizer.features
        features1 = afeats[:,:5]
        features2 = afeats[:,3:]
        rot = choice(range(24))
        tr1 = 2 * np.random.rand(1, 3)
        tr2 = 0.5 * np.random.rand(1, 3)
        coords1 = rotate_grid(coords,rot)
        coords1 += tr1
        f1n = make_3dgrid(coords1,features1,23,2)

        coords2 = rotate_grid(coords,rot)
        coords2 += tr2
        f2n = make_3dgrid(coords2,features2,23,2)

        feats_final = np.concatenate([f1n,f2n],axis=4)


        vox = np.squeeze(feats_final, 0).transpose(3, 0, 1, 2)

        mol = Chem.MolFromSmiles(smiles_token)
        if not mol:
            raise ValueError(f"Failed to parse molecule '{mol}'")

        sstring = Chem.MolToSmiles(mol)  # Make the SMILES canonical.
        sstring = sstring.replace("Cl", "X").replace("[nH]", "Y") \
                                            .replace("Br", "Z")
        try:
            vals = [1] + \
                   [vocab_c2i_v1[xchar] for xchar in sstring] + \
                   [2]
        except KeyError:
            raise ValueError(
                ("Unkown SMILES tokens: {} in string '{}'."
                 .format(", ".join([x for x in sstring if
                                    x not in vocab_c2i_v1]),
                         sstring)))
        end_token = vals.index(2)
        return torch.Tensor(vox), torch.Tensor(vals), end_token + 1


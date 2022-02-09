from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
import numpy as np
import random
import torch
from typing import List
from geomol_utils import featurize_mol_from_smiles , construct_conformers
from torch_geometric.data import Batch
from model import Geomol_model_module
from omegaconf import OmegaConf

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def initialize_model(hparams_path,ckpt_path):
    params = OmegaConf.load(hparams_path)
    model = Geomol_model_module(params["hyper_parameters"],params["node_features"],params["edge_features"]).load_from_checkpoint(ckpt_path)
    return model


conformer_dict = {}
def generate_conformers(input_smiles:List[str],hparams_path:str=None,ckpt_path :str=None,
                                 n_conformers:int=10,dataset :str ="drugs",mmff: bool =False) -> dict :

    """ This function takes INPUT ARGS and returns generated conformers in this format 
    generated_conformers = {"smile_1":[gen_conf_1,gen_conf_2......,gen_conf_n],
                            "smile_2":[gen_conf_1,gen_conf_2......,gen_conf_n],
                            .
                            .
                            .
                            "smile_+str(len(input_smiles))": [gen_conf_1,gen_conf_2.......,gen_conf_n] }
        ----------------

        INPUT ARGS ::

        input_smiles : List[str]
                       those simles for which you want to generated 3d-conformer. input should be in ['smile_1,smiles_2,.....,smile_n] format
        hparams_path : str
                       path for the hparams.yaml file
        ckpt_path    : str 
                       path for the ckpt file of model
        n_conformers : int
                       how many number of conformers you want to generate per smile
        dataset : bool
                       By which model you want to generate the conformers ('drugs' or 'qm9')
        mmff : bool 
                       want to optimize the generated molecules by adding 'MMff94s' energy.?"""

    model = initialize_model(hparams_path,ckpt_path)
    for smi in input_smiles:
        data_obj = featurize_mol_from_smiles(smi, dataset=dataset)
        if not data_obj:
            print(f'failed to featurize SMILES: {smi}')
            continue
        data = Batch.from_data_list([data_obj])
        n_atoms = data.x.size(0)
        model_coords = model.prediction(data,n_confs= 10)
        mols = []
        for x in model_coords.split(1, dim=1):
            mol = Chem.AddHs(Chem.MolFromSmiles(smi))
            coords = x.squeeze(1).double().cpu().detach().numpy()
            mol.AddConformer(Chem.Conformer(n_atoms), assignId=True)
            for i in range(n_atoms):
                mol.GetConformer(0).SetAtomPosition(i, Geometry.Point3D(coords[i, 0], coords[i, 1], coords[i, 2]))

            if mmff:
                try:
                    AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')
                except Exception as e:
                    pass
            mols.append(mol)
        conformer_dict["smi"] = mols
    return conformer_dict

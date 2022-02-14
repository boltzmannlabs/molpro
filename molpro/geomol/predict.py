from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
import rdkit 
import numpy as np
import random
import torch
from typing import List
from geomol_utils import featurize_mol_from_smiles , construct_conformers
from torch_geometric.data import Batch
from model import GeomolModelModule
from omegaconf import OmegaConf

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def initialize_model(hparams_path: str = None, checkpoint_path: str = None):
    
    """ Function for loading pytorch lightning module from checkpoint,hyperparameters

        Parameters :
        --------------

        hparams_path : str
            path for the hparams.yaml file
        
        checkpoint_path: str,
            Path to pytorch lightning checkpoint

    """

    params = OmegaConf.load(hparams_path)
    model = GeomolModelModule(params["hyper_parameters"],params["node_features"],params["edge_features"]).load_from_checkpoint(checkpoint_path)
    return model


def optimize(mol: rdkit.Chem.rdchem.Mol = None ) -> None or rdkit.Chem.rdchem.Mol :
    
    """ Function for optimizing molecule by adding MMFF94 energy 
    
    Parameters :
    --------------

    mol : rdkit.Chem.rdchem.Mol
           a rdkit.Chem.rdchem.Mol type object to optimize
    """

    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')
        return mol
    except Exception as e:
        return None


def generate_mols_from_coords(generated_cords: torch.tensor = None, num_atoms: int = None, mmff: bool = False,smi: str = None) -> List[rdkit.Chem.rdchem.Mol]:

    """ Function for preparing conformers from generated cordinates by model 
    
    Parameters :
    --------------

    generated_cords : torch.tensor
                     generated cordinates by model
    num_atoms : int 
                number of atoms in the input molecule
    mmff  : bool
                want to optimize the generated molecules by adding 'MMff94s' energy.?
    smi  : str
                smile for which you are generating the conformers
    
    """

    mols = [],
    for x in generated_cords.split(1, dim=1):
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        coords = x.squeeze(1).double().cpu().detach().numpy()
        mol.AddConformer(Chem.Conformer(num_atoms), assignId=True)
        for i in range(num_atoms):
            mol.GetConformer(0).SetAtomPosition(i, Geometry.Point3D(coords[i, 0], coords[i, 1], coords[i, 2]))

        if mmff:
            mol = optimize(mol) if optimize(mol) is not None else mol
        mols.append(mol)
    return mols


def generate_conformers(input_smiles:List[str],hparams_path:str=None,checkpoint_path :str=None,
                                 n_conformers:int=10,dataset :str ="drugs",mmff: bool =False) -> dict :

    """ This function takes Parametrs and returns generated conformers in this format 
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

    conformer_dict = {}
    model = initialize_model(hparams_path,checkpoint_path)
    for smi in input_smiles:
        data_obj = featurize_mol_from_smiles(smi, dataset=dataset)
        if not data_obj:
            print(f'failed to featurize SMILES: {smi}')
            continue
        data = Batch.from_data_list([data_obj])
        model_coords = model.prediction(data,n_confs= n_conformers)
        generated_mols = generate_mols_from_coords(generated_cords = model_coords, num_atoms = data.x.size(0), mmff = False,smi = smi)
        conformer_dict[smi] = generated_mols
    
    return conformer_dict


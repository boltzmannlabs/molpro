
from importlib.resources import path
import numpy as np
from rdkit import Chem
import torch
from typing import Callable, List
from model import ShapeBasedGenModule
from data import vocab_i2c_v1,vocab_c2i_v1
from molpro.models.shape_captioning import ShapeEncoder, DecoderRNN, VAE
import molpro.utils.preprocess as featurization


def unique_canonical(smiles_list: List[str] = None) -> List[str]:
    """ Function to filter unique and valid smiles.

    Parameters :
    -------------
    smiles_list : List[str],
                a list that contains smiles which need to be filter
    
    Returns :
    -------------
    filterd_smiles : List[str],
                    a list that contains unique and valid smiles

    """
    filterd_smiles = list(set([Chem.MolToSmiles(x) for x in [Chem.MolFromSmiles(x) for x in smiles_list] if x is not None]))
    return filterd_smiles

def initialize_model(ckpt_path: str = None,device:str = "cpu"):
    
    """ Function for loading pytorch lightning module from checkpoint,hyperparameters.

        Parameters :
        --------------
        
        checkpoint_path: str,
                           Path to pytorch lightning checkpoint
        device : str,
                   device 
        
        Returns :
        --------------

        model : loaded model in a object

    """
    encoder = ShapeEncoder(9)
    decoder = DecoderRNN(512, 1024, 29, 1,device)
    vae_model = VAE(nc=9,device=device)
    ckpt = torch.load(ckpt_path)
    model = ShapeBasedGenModule(encoder, decoder, vae_model).load_state_dict(ckpt['state_dict'])
    return model 


def featurize_smile(smile:str = None) -> torch.tensor:

    """ Function to featurize smile before giving to model.
    
    Parameters :
    --------------
    
    smile : str,
             smile sequence that you want to featurize
    
    Returns :
    --------------

    vox : torch.tensor,
             a tensor which contains featurization(shape) of that smile
    """
    featurizer = featurization.Featurizer(smile)
    featurizer.generate_conformer()
    coords = featurizer.get_coords()
    centroid = coords.mean(axis=0)
    coords -= centroid
    afeats = featurizer.atom_features()
    vox = featurizer.make_3dgrid(coords, afeats, 23, 2)
    vox = torch.tensor(np.squeeze(vox, 0).transpose(3, 0, 1, 2)).unsqueeze(0)

    return vox


def decode_smiles(model_outputs:torch.tensor = None) -> List[str]:

    """ Function to decode smiles from their generated indexes. 

    Parameters :
    -------------

    model_outputs : torch.tensor,
                  a tensor of tensors that containes indices of each generated smile character.

    Returns :
    ------------
    decoded_smiles : List[str],
                   a list that contains decoded smiles.

    """

    decoded_smiles = []
    for op in model_outputs:
        op = op[0]
        smile = str()
        for i in op[1:]:
            char  = vocab_i2c_v1.get(int(i[0]))
            if char == "end":
                break
            smile = smile+str(char)
        decoded_smiles.append(smile)
    return decoded_smiles


def generate_smiles(input_smiles :List[str]= None, ckpt_path :str = None,
                               n_attempts :int= 20 , sample_prob :bool= False
                               ,unique_valid :bool= False,device: str= "cpu") -> dict:
    

    """ This function can be used for generate similiar molecules based on their shape.

        Parameters :
        ----------------

        input_smiles : List[str],
                       those simles for which you want to generated similar smiles. input should be in ['smile_1,smiles_2,.....,smile_n] format
        ckpt_path : str,
                       path for the trained lightning model
        n_attempts : int,
                       how many number of smiliar smiles you want to generate per smile
        sample_prob : bool,
                       samples smiles tockens for given shape features (probalistic picking)
        unique_valid : bool,
                       want to filter unique smiles from generated smiles or not.? 

        Returns  :
        ---------------
        
        generated_smiles : dict,           
                            {"smile_1":[gen_smi_1,gen_smi_2......,gen_smi_n],
                            "smile_2":[gen_smi_1,gen_smi_2......,gen_smi_n],
                            .
                            .
                            .
                            "smile_+str(len(input_smiles))": [gen_smi_1,gen_smi_2.......,gen_smi_n] }
        """

    if input_smiles or ckpt_path is None:
        raise TypeError('Please give right input_molecule and ckpt file.')

    model  = initialize_model(ckpt_path = ckpt_path,device = device)
    
    generated_smiles = dict()
    for smi in input_smiles:
        vox = featurize_smile(smi)
        model_outputs = [[model.prediction(vox,sample_prob=sample_prob)] for i in range(n_attempts)]
        decoded_smiles = decode_smiles(model_outputs = model_outputs)
        if unique_valid :
            generated_smiles["smi"] = unique_canonical(decoded_smiles)
        else :
            generated_smiles["smi"] = decoded_smiles

    return generated_smiles



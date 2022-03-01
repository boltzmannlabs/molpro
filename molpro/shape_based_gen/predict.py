
from importlib.resources import path
import numpy as np
from rdkit import Chem
import torch
from typing import Callable, List
from model import ShapeBasedGenModule
from data import vocab_i2c_v1,vocab_c2i_v1
from molpro.models.shape_captioning import ShapeEncoder, DecoderRNN, VAE
from molpro.utils.preprocess import make_3dgrid, Featurizer, rotate_grid
from random import choice


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
    encoder = ShapeEncoder(5)
    decoder = DecoderRNN(512, 1024, 29, 1,device)
    #decoder = DecoderRNN(512, 16, 29, 1,device)
    vae_model = VAE(nc=5,device=device)
    model = ShapeBasedGenModule(encoder, decoder, vae_model)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
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
    fatures_list = ['hydrophobic', 'aromatic','acceptor', 'donor','ring']

    featurizer = Featurizer(input_file = smile , file_type= 'smi', named_props  = ["partialcharge"], smarts_labels = fatures_list, metal_halogen_encode = False)


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
    vox = torch.tensor(vox).unsqueeze(0)

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

    model_outputs = torch.stack(model_outputs,1)
    decoded_smiles = []
    for tensor in model_outputs:
        smile = str()
        for i in tensor[1:]:
            char  = vocab_i2c_v1.get(int(i))
            if char == "end":
                break
            smile = smile+str(char)
        decoded_smiles.append(smile)
    return decoded_smiles


def generate_smiles(input_smiles :List[str]= None, ckpt_path :str = None,
                               n_attempts :int= 20 , sample_prob :bool= False,factor : float= 1.
                               ,unique_valid :bool= False, device: str= "cpu") -> dict:
    

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
        factor  : float,
                       variability factor
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

    """if not input_smiles or ckpt_path is None:
        raise TypeError('Please give right input_molecule and ckpt file.')"""

    model  = initialize_model(ckpt_path = ckpt_path,device = device)
    
    generated_smiles = dict()
    for smi in input_smiles:
        vox = featurize_smile(smi).repeat(n_attempts,1,1,1,1)
        model_outputs = model.prediction((vox,None,None),sample_prob=sample_prob,factor=factor)
        decoded_smiles = decode_smiles(model_outputs = model_outputs)
        if unique_valid :
            generated_smiles[smi] = unique_canonical(decoded_smiles)
        else :
            generated_smiles[smi] = decoded_smiles

    return generated_smiles

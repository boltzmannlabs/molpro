
from importlib.resources import path
from typing import Callable, List
from models.shape_captioning import ShapeEncoder, DecoderRNN, VAE
from model import BpModule
import torch
import utils.preprocess as bup
from rdkit import Chem
import numpy as np
from data import vocab_i2c_v1,vocab_c2i_v1


def unique_canonical(smiles):
    """
    :param in_mols - list of SMILES strings
    :return: list of uinique and valid SMILES strings in canonical form.
    """
    xresults = [Chem.MolFromSmiles(x) for x in smiles]  # Convert to RDKit Molecule
    xresults = [Chem.MolToSmiles(x) for x in xresults if x is not None]  # Filter out invalids
    #return [Chem.MolFromSmiles(x) for x in set(xresults)]  # Check for duplicates and filter out invalids
    return list(set(xresults))



def generate_smiles(input_smiles :List[str]= None, ckpt_path :str = None,
                               n_attempts :int= 20 , sample_prob :bool= False
                               ,unique_valid :bool= False) -> dict:
    """ This function takes INPUT ARGS and returns generated smiles in this format 
    generated_smiles = {"smile_1":[gen_smi_1,gen_smi_2......,gen_smi_n],
                        "smile_2":[gen_smi_1,gen_smi_2......,gen_smi_n],
                        .
                        .
                        .
                        "smile_+str(len(input_smiles))": [gen_smi_1,gen_smi_2.......,gen_smi_n] }
        ----------------

        INPUT ARGS ::

        input_smiles : List[str]
                       those simles for which you want to generated similar smiles. input should be in ['smile_1,smiles_2,.....,smile_n] format
        ckpt_path : str
                       path for the trained lightning model
        n_attempts : int
                       how many number of smiliar smiles you want to generate per smile
        sample_prob : bool
                       samples smiles tockens for given shape features (probalistic picking)
        unique_valid : bool 
                       want to filter unique smiles from generated smiles or not.? """
        
    if input_smiles or ckpt_path is None:
        raise TypeError('Please give right input_molecule and ckpt file.')
    encoder = ShapeEncoder(35)
    #decoder = DecoderRNN(512, 1024, 29, 1,params.device) # Original 
    decoder = DecoderRNN(512, 16, 29, 1,"cpu") # reduced no. of params just to check training on my system
    vae_model = VAE(nc=35,device="cpu")

    ckpt = torch.load(ckpt_path)
    modell = BpModule(encoder, decoder, vae_model)
    modell.load_state_dict(ckpt['state_dict'])


    generated_smiles = dict()
    for smi in input_smiles:
        featurizer = bup.Featurizer(smi)
        featurizer.generate_conformer()
        coords = featurizer.get_coords()
        centroid = coords.mean(axis=0)
        coords -= centroid
        afeats = featurizer.atom_features()
        vox = bup.make_3dgrid(coords, afeats, 23, 2)
        vox = torch.tensor(np.squeeze(vox, 0).transpose(3, 0, 1, 2)).unsqueeze(0)
        outputs = [[modell.prediction(vox,sample_prob=sample_prob)] for i in range(n_attempts)]
        output_smiles = []
        for op in outputs:
            op = op[0]
            smile = str()
            for i in op[1:]:
                a  = vocab_i2c_v1.get(int(i[0]))
                if a == "end":
                    break
                smile = smile+str(a)
            output_smiles.append(smile)
        if unique_valid :
            generated_smiles["smi"] = unique_canonical(output_smiles)
        else :
            generated_smiles["smi"] = output_smiles

    return generated_smiles



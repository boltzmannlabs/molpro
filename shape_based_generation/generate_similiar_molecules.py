
from importlib.resources import path
from shape_captioning import ShapeEncoder, DecoderRNN, VAE
from model import BpModule
import torch
import preprocess as bup
from rdkit import Chem
import numpy as np
from data import vocab_i2c_v1,vocab_c2i_v1


def unique_canonical(in_mols):
    """
    :param in_mols - list of SMILES strings
    :return: list of uinique and valid SMILES strings in canonical form.
    """
    xresults = [Chem.MolFromSmiles(x) for x in in_mols]  # Convert to RDKit Molecule
    xresults = [Chem.MolToSmiles(x) for x in xresults if x is not None]  # Filter out invalids
    #return [Chem.MolFromSmiles(x) for x in set(xresults)]  # Check for duplicates and filter out invalids
    return list(set(xresults))



def generate_molecules(smiles = None , ckpt_path = None, n_attempts = 20 , sample_prob = False , unique_valid = False):
    

    if smiles or ckpt_path is None:
        raise TypeError('Please give right smiles and ckpt file.')
    encoder = ShapeEncoder(35)
    #decoder = DecoderRNN(512, 1024, 29, 1,params.device) # Original 
    decoder = DecoderRNN(512, 16, 29, 1,"cpu") # reduced no. of params just to check training on my system
    vae_model = VAE(nc=35,device="cpu")

    ckpt = torch.load(ckpt_path)
    modell = BpModule(encoder, decoder, vae_model)
    modell.load_state_dict(ckpt['state_dict'])


    generated_smiles = dict()
    for smi in smiles:
        featurizer = bup.Featurizer(smi, 'smi', False, False,True, True, True)
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



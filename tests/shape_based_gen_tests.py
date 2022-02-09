import torch
from models.shape_captioning import ShapeEncoder, DecoderRNN, VAE
from model import BpModule
import utils.preprocess as bup
import numpy as np
import rdkit
from rdkit import Chem
from shape_based_generation.data import vocab_c2i_v1

encoder = ShapeEncoder(9)
decoder = DecoderRNN(512, 1024, 29, 1,"cpu")
vae_model = VAE(nc=9,device="cpu")

model = BpModule(encoder, decoder, vae_model)


mol_batch = torch.rand((1, 9, 24, 24, 24))
caption = torch.rand([1, 34])
lengths = (34,)
x = (mol_batch, caption, lengths)

def test_shape_gen():
    recon_batch,mu,logvar = model(x,only_vae=True)
    condition1 = recon_batch is not None and mu is not None and logvar is not None
    condition2 = type(recon_batch).__name__ == 'Tensor' and type(mu).__name__ == 'Tensor' and type(logvar).__name__ == 'Tensor'
    condition3 = recon_batch.size() == (1, 9, 24, 24, 24) and mu.size() == (1,512) and logvar.size() == (1,512)
    assert condition1 and condition2 and condition3


def test_shape_based_gen():
    output = model(x,only_vae=False)
    assert output is not None and type(output).__name__ == 'Tensor' and output.shape == (34,29)



print("Succesfully Done...")


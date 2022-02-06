import argparse
import numpy as np
import time
from datetime import datetime
import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import utils.preprocess as bup
from models.shape_captioning import ShapeEncoder, DecoderRNN, VAE
from data import SmilesDataset,BpDataModule, read_smi, read_csv




def parse_options():
    """method to parse the config options. modify config options
    by adding the corresponding key, val pairs"""

    parser = argparse.ArgumentParser('modify config options by adding \
                                      KEY VAL pairs', add_help=False)
    parser.add_argument("-i", "--input_path", required=True,
                                help="Path to input smi file.")
    parser.add_argument("--batch_size", default=32, type=int,
                                help="batch size for single gpu")
    parser.add_argument("--max_epochs", default=3, type=int,
                                help="max epochs to train for")
    parser.add_argument("--num_workers", type=int, default=8,
                                help="number of workers for pytorch dataloader")
    parser.add_argument("--device", default="cpu", type=str,
                                help="on which device you want to train the model (cpu or cuda)")
    parser.add_argument("--gpus", default=1, type=int,
                                help="numbers of gpus to train model")
    
    args, unparsed = parser.parse_known_args()
    return args


def loss_function(reconstruction_function, recon_x, x, mu, logvar):
    """custom loss function using binary cross entropy and KL divergence"""

    BCE = reconstruction_function(recon_x, x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return BCE + KLD


class BpModule(pl.LightningModule):
    """Lightning trainer module to handle training/validation for dataset"""

    def __init__(self, encoder, decoder, vae_model):
        super().__init__()


        self.encoder = encoder
        self.decoder = decoder
        self.vae_model = vae_model
        self.cap_loss = 0.
        ######################################################################################################################################################################
        self.caption_start = 4000  
        #######################################################################################################################################################################
        self.caption_criterion = nn.CrossEntropyLoss()
        self.reconstruction_function = nn.BCELoss()
        self.reconstruction_function.size_average = False
        self.rec_loss_func = loss_function
        self.automatic_optimization = False
        self.save_hyperparameters()


    def forward(self, x, only_vae = False):
        
        mol_batch, caption, lengths = x
        recon_batch, mu, logvar = self.vae_model(mol_batch)
        if only_vae :
            return recon_batch,mu,logvar
        else : 
            features = self.encoder(recon_batch)
            outputs = self.decoder(features, caption, lengths)
            return outputs


    def training_step(self, batch, batch_idx):
        caption_optimizer, vae_optimizer = self.optimizers()
        mol_batch, caption, lengths = batch
        vae_optimizer.zero_grad()
        x = batch
        recon_batch,mu,logvar = self(x,only_vae = True)

        
        vae_loss = self.rec_loss_func(self.reconstruction_function,
                                      recon_batch, mol_batch, mu, logvar)
        self.manual_backward(vae_loss, retain_graph=True
                             if batch_idx >= self.caption_start else False)
        p_loss = vae_loss.item()
        self.log("train_p_loss", p_loss,on_step=False, on_epoch=True, prog_bar=True, logger=True)
        vae_optimizer.step()

        cap_loss = torch.tensor([0])
        if batch_idx > self.caption_start:
            targets = pack_padded_sequence(caption,
                                            lengths, batch_first=True)[0]
            self.encoder.zero_grad()
            self.decoder.zero_grad()
            outputs = self(x)
            cap_loss = self.caption_criterion(outputs, targets)
            self.log("train_cap_loss",cap_loss,on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.manual_backward(cap_loss)
            caption_optimizer.step()
            

        if (batch_idx + 1) % 60000 == 0:
            # Command = "Reducing learning rate"
            # .format(i+1, float(loss.data.cpu().numpy()))
            self.log("Reducing LR\n")
            for param_group in caption_optimizer.param_groups:
                lr = param_group["lr"] / 2.
                param_group["lr"] = lr

    def validation_step(self, batch, batch_idx):
        mol_batch, caption, lengths = batch
        x = batch
        recon_batch,mu,logvar = self(x,only_vae = True)
        vae_loss = self.rec_loss_func(self.reconstruction_function,
                                      recon_batch, mol_batch, mu, logvar)
        
        p_loss = vae_loss.item()
        self.log("val_p_loss", p_loss,on_step=False, on_epoch=True, prog_bar=True, logger=True)

        cap_loss = 0
        if batch_idx > self.caption_start:
            captions = caption.to(self.device)
            targets = pack_padded_sequence(captions,
                                            lengths, batch_first=True)[0]
            self.decoder.zero_grad()
            self.encoder.zero_grad()
            outputs = self(x)
            cap_loss = self.caption_criterion(outputs, targets)
            self.log("val_cap_loss", cap_loss,on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        mol_batch, caption, lengths = batch
        x = batch
        recon_batch,mu,logvar = self(x,only_vae = True)
        
        vae_loss = self.rec_loss_func(self.reconstruction_function,
                                      recon_batch, mol_batch, mu, logvar)
        p_loss = vae_loss.item()
        self.log("test_p_loss", p_loss,on_step=False, on_epoch=True, prog_bar=True, logger=True)

        cap_loss = 0
        if batch_idx > self.caption_start:
            captions = caption.to(self.device)
            targets = pack_padded_sequence(captions,
                                            lengths, batch_first=True)[0]
            self.decoder.zero_grad()
            self.encoder.zero_grad()
            outputs = self(x)
            cap_loss = self.caption_criterion(outputs, targets)
            self.log("test_cap_loss", cap_loss,on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        caption_optimizer = None
        caption_params = list(self.decoder.parameters()) + \
            list(self.encoder.parameters())
        
        caption_optimizer = torch.optim.Adam(caption_params,lr=.001)
        vae_optimizer = None
        vae_optimizer = torch.optim.Adam(self.vae_model.parameters(),lr=1e-4)
        return caption_optimizer, vae_optimizer


    def prediction(self,data,sample_prob=False):

        recon_batch  = self.vae_model(data)
        features = self.encoder(recon_batch)
        if sample_prob :
            output = self.decoder.sample_prob(features)
        else :
            output = self.decoder.sample(features)
            return output



def main(params):
    print("Starting of main Func")
    st = time.perf_counter()

    bpdata = BpDataModule(smiles_path=params.input_path,
                          batch_size=params.batch_size,
                          read_func=read_smi if params.input_path.endswith("smi") else read_csv,
                          nworkers = params.nworkers)

    bpdata.prepare_data()
    print(f"time taken to process dataset: {time.perf_counter() - st} secs")
    encoder = ShapeEncoder(35)
    decoder = DecoderRNN(512, 1024, 29, 1,params.device)
    vae_model = VAE(nc=35,device=params.device)

    model = BpModule(encoder, decoder, vae_model)
    cur_time = datetime.now().strftime("%d%m%Y_%H:%M:%S")
    save_models_dir = "./trained-models/"
    os.makedirs(save_models_dir, exist_ok=True)

    checkpoint_callback1 = ModelCheckpoint(
        monitor="val_p_loss",
        dirpath=save_models_dir,
        filename="val-ligdream-{epoch:02d}-{val_p_loss:.2f}"+f"-{cur_time}",
        save_top_k=3,
        mode="min"
    )
    checkpoint_callback2 = ModelCheckpoint(
        monitor="val_cap_loss",
        dirpath=save_models_dir,
        filename="val-ligdream-{epoch:02d}-{val_cap_loss:.2f}"+f"-{cur_time}",
        save_top_k=3,
        mode="min"
    )

    print("Starting of trainers")
    trainer = pl.Trainer(max_epochs=int(params.max_epochs),
                         progress_bar_refresh_rate=20,
                         gpus = None if params.device == "cpu" else int(params.gpus),
                         callbacks=[checkpoint_callback1,checkpoint_callback2]
                         )
    trainer.fit(model, bpdata)
    trainer.test(model, bpdata)


if __name__ == "__main__":
    st = time.perf_counter()
    configs = parse_options()
    main(configs)
    print(f"Total time taken: {time.perf_counter() - st}")



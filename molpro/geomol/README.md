## Generating 3d structure of a molecule

#### 1. Overview ->
Prediction of a molecule’s 3D conformer ensemble from the molecular graph holds a key role in areas of cheminformatics and drug discovery. We are using a  machine learning approach to generate distributions of low-energy molecular 3D conformers. Leveraging the power of message passing neural networks (MPNNs) to capture local and global graph information, we predict local atomic 3D structures and torsion angles, and using these we are assembling the whole conformer for that molecule. The original work is done by Department of Chemical Engineering, MIT, Cambridge. For more details visit :

    https://arxiv.org/pdf/2106.07802.pdf


#### 2. For training the model -> 

##### Data for training:
GEOM is a dataset with over 37 million molecular conformations annotated by energy and statistical weight for over 450,000 molecules. Here we will use a subset of it. you can download the dataset by clicking the link bellow

    
    https://dataverse.harvard.edu/api/access/datafile/4327252
      

After downloading and unzipping the file you will get a folder name rdkit_folder under that 2 diffrent dataset will be there.

    rdkit_folder/
                 drugs/
                        smile_seq1.pickle
                        smile_seq2.pickle
                        .
                        .
                        304340 pickle files
                 qm9/
                        smile_name1.pickle
                        smile_name2.pickle
                        .
                        .
                        133232 pickle files

or if you just want to try on sample datset then we have created a sample dataset by randomly selecting datapoints from original dataset. You can download the sample dataset through the link : 

    https://drive.google.com/drive/folders/1Y7gktdSyq3iu6OmHXVdUfgANvtOq9GU_?usp=sharing

##### Training of model:
Once you have the dataset you can start training the model. For that can run the function: 

    from molpro.geomol.model import train_geomol
    train_geomol(data_dir:str = "/",dataset:str = "drugs", seed:int=0, n_epochs:int = 2, batch_size:int = 16,
                                      lr:float = 1e-3, num_workers:int = 6, num_gpus:int = 1, verbose:bool = False)
                                      
Input Parameters :
    
    data_dir : str 
             directory path where data is stored for training
    dataset : str
             on which dataset you want to train the model "drugs" or "qm9"
    seed : int
           the seed you want to fix
    n_epochs : int 
           number of epochs you want to train the model
    bath_size : int
           bath size for model training
    lr : float
           learning rate for optimizers
    num_workers : int
           number of workers to be used
    num_gpus : int
           on how many gpus you want to train then model
    verbose : bool
           verbose "True" or "False"

after executing you will get a new folder called "lightning_logs".

#### 3. Generating similiar molecules with trained model ->
After training of model you can start generating 3d representation of molecules. For this you can use generate_conformers function from predict.py script. This function takes a list of smiles and returns a dictionary containing smile as the key and values as a list of generated conformers.

    from molpro.geomol.predict import generate_conformers
    generated_comformers = generate_conformers(input_smiles:List[str],hparams_path:str=None,checkpoint_path :str=None,
                                 n_conformers:int=10,dataset :str ="drugs",mmff: bool =False) 
    
Input parameters :

        input_smiles : List[str]
                       those simles for which you want to generated 3d-conformer. input should be in ['smile_1,smiles_2,.....,smile_n] format
        hparams_path : str
                       path for the hparams.yaml file. which is under lightning_logs folder
        ckpt_path    : str 
                       path of the ckpt file. which is under lightning_logs/checkpoints directory
        n_conformers : int
                       how many number of conformers you want to generate per smile
        dataset : str
                       By which model you want to generate the conformers ('drugs' or 'qm9')
        mmff : bool 
                       want to optimize the generated conformers by adding 'MMff94s' energy.?

Returns : 

    generated_conformers = {"smile_1":[gen_conf_1,gen_conf_2......,gen_conf_n],
                            "smile_2":[gen_conf_1,gen_conf_2......,gen_conf_n],
                            .
                            .
                            .
                            "smile_+str(len(input_smiles))": [gen_conf_1,gen_conf_2.......,gen_conf_n] }

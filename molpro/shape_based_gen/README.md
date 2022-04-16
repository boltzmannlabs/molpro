### Generating similiar molecules based on their shape (Shape based generation)
#### 1. Overview ->
The generative design of novel scaffolds and functional groups can cover unexplored regions of chemical space that still possess lead-like properties.
Here we are using an AI approach to generate novel molecules starting from a seed compound, its three-dimensional (3D) shape. A variational autoencoder is used to generate the 3D representation of a compound, followed by a system of convolutional for encoding and recurrent neural networks that generate a sequence of SMILES tokens. The generative design of novel scaffolds and functional groups can cover unexplored regions of chemical space that still possess lead-like properties. Refrence: https://pubs.acs.org/doi/10.1021/acs.jcim.8b00706


#### 2. For training the model -> 

##### Data for training:

We will be using a subset of Zinc15 dataset for our model training. That will only have drug like smiles. you can download the dataset by clicking the link given bellow:       

    http://pub.htmd.org/zinc15_druglike_clean_canonical_max60.zip
      

After downloading unzipping the file you will get a .smi file as name "zinc15_druglike_clean_canonical_max60.smi". which will have smiles.

or if you just want to try on sample datset then we have created a sample dataset by randomly selecting datapoints from original dataset. You can download the sample dataset through the link :

    https://drive.google.com/drive/folders/1CBakMNrUu-mJdH6oJJzNT1dcDVj6ECJV?usp=sharing
    

##### Training of model:
Once you have the dataset you can start training the model. For that can execute the function : 

    from molrpro.dhape_based_gen.model import train_shape_based_gen
    train_shape_based_gen(input_path:str = "/", batch_size:int = 32, max_epochs:int = 3, 
                         num_workers:int = 6, device:str = "cpu",gpus:int = 1)
        
Input Parameters :
    
    input_path : str 
              Path to input smi file.
    batch_size : int
              batch size for single gpu
    max_epochs : int 
              max epochs to train for 
    num_workers : int
              number of workers for pytorch dataloader 
    device : str
              on which device you want to train the model (cpu or cuda)
    gpus : int 
              numbers of gpus to train model

after executing you will get a new folder called "lightning_logs".

#### 3. Generating similiar molecules with trained model ->
After training of model you can start generating similiar molecules based on their shape. For this you can use generate_smiles from predict.py script. This function takes a list of smiles and returns a dictionary containing smile as the key and values as a list of generated smiles.

    from molpro.shape_based_gen.predict import generate_smiles
    generated_smiles = generate_smiles(input_smiles = None, ckpt_path = None, n_attempts = 20, sample_prob = False, factor = 1.,  unique_valid = False) 
    
##### Input parameters :

    input_smiles : List[str]
                   those simles for which you want to generated similar smiles. input should be in ['smile_1,smiles_2,.....,smile_n] format
    ckpt_path : str
                   path of the ckpt file. which is under lightning_logs/checkpoints directory. 
    n_attempts : int
                   how many number of smiliar smiles you want to generate per smile
    sample_prob : bool
                   samples smiles tockens for given shape features (probalistic picking)
    factor : float
                   variability factor
    unique_valid : bool 
                   want to filter unique smiles from generated smiles or not.?

##### Returns : 

    generated_smiles = {"smile_1":[gen_smi_1,gen_smi_2......,gen_smi_n],
                        "smile_2":[gen_smi_1,gen_smi_2......,gen_smi_n],
                        .
                        .
                        .
                        "smile_+str(len(input_smiles))": [gen_smi_1,gen_smi_2.......,gen_smi_n] }

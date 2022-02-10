## Generating 3d structure of a molecule

#### 1. Overview ->
Prediction of a molecule’s 3D conformer ensemble from the molecular graph holds a key role in areas of cheminformatics and drug discovery. We are using a  machine learning approach to generate distributions of low-energy molecular 3D conformers. Leveraging the power of message passing neural networks (MPNNs) to capture local and global graph information, we predict local atomic 3D structures and torsion angles, and using these we are assembling the whole conformer for that molecule.


#### 2. For training the model -> 

##### Data for training:

We will be using a subset of Zinc15 dataset for our model training. That will only have drug like smiles. you can download the dataset by clicking the link given bellow: 
    
    http://pub.htmd.org/zinc15_druglike_clean_canonical_max60.zip
      

After downloading unzipping the file you will get a .smi file as name "zinc15_druglike_clean_canonical_max60.smi". which will have 66666 smiles.
##### Training of model:
Once you have the dataset you can start training the model. For that can execute model.py file with the following command : 

    python model.py --input_path {path_for_.smi_file} --batch_size {your_batch_size} --max_epochs {max_numnber_of_epochs} --num_workers {num_of_workers} --device     {'cpu'_or_'gpu'} --gpus {num_of_gpus_for_training}

after executing you will get a new folder called "lightning_logs".

#### 3. Generating similiar molecules with trained model ->
After training of model you can start generating similiar molecules based on their shape. For this you can use generate_smiles from predict.py script. This function takes a list of smiles and returns a dictionary containing smile as the key and values as a list of generated smiles.

    predict.generate_smiles(input_smiles :List[str]= None, ckpt_path :str = None,n_attempts :int= 20 , sample_prob :bool= False,unique_valid :bool= False) 
    
Input parameters :

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

Returns : 

    generated_smiles = {"smile_1":[gen_smi_1,gen_smi_2......,gen_smi_n],
                        "smile_2":[gen_smi_1,gen_smi_2......,gen_smi_n],
                        .
                        .
                        .
                        "smile_+str(len(input_smiles))": [gen_smi_1,gen_smi_2.......,gen_smi_n] }

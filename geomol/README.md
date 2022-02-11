## Generating 3d structure of a molecule

#### 1. Overview ->
Prediction of a molecule’s 3D conformer ensemble from the molecular graph holds a key role in areas of cheminformatics and drug discovery. We are using a  machine learning approach to generate distributions of low-energy molecular 3D conformers. Leveraging the power of message passing neural networks (MPNNs) to capture local and global graph information, we predict local atomic 3D structures and torsion angles, and using these we are assembling the whole conformer for that molecule.


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


##### Training of model:
Once you have the dataset you can start training the model. For that can execute model.py file with the following command : 

    python model.py --data_dir {pat_of_the_dataset_directory(drugs or qm9)}  --split_path {path_of_the_numpy_file_which_contain_indices_of_train_test_files} --n_epochs {max_number_of_epoch} --dataset {which_dataset_will_be_used_for_training_('drugs' or 'qm9')

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

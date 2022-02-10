### Generating similiar molecules based on their shape (Shape based generation)
#### 1. Overview ->
The generative design of novel scaffolds and functional groups can cover unexplored regions of chemical space that still possess lead-like properties.
Here we are using an AI approach to generate novel molecules starting from a seed compound, its three-dimensional (3D) shape. A variational autoencoder is used to generate the 3D representation of a compound, followed by a system of convolutional for encoding and recurrent neural networks that generate a sequence of SMILES tokens. The generative design of novel scaffolds and functional groups can cover unexplored regions of chemical space that still possess lead-like properties.


#### 2. For training the model -> 

###### Data for training:

We will be using a subset of Zinc15 dataset for our model training. That will only have drug like smiles. you can download the dataset by clicking the link given bellow:       http://pub.htmd.org/zinc15_druglike_clean_canonical_max60.zip
      

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

### Generating novel molecules accoding to a given protein binding site
#### 1. Overview:
A novel method was developed to generate focused virtual libraries of small molecules based on the protein structure using deep learning-based generative models. 
Structures of protein–ligand complexes obtained from ligand docking are used to train a generative adversarial model to generate compound structures that are complementary to protein but also maintain diversity among themselves. 


#### 2. Preparing dataset:
We will be using a subset of PDBBind dataset given in the sample data folder for training. 

    python data.py --data_path {path where pdb and mol2 files are stored} --hdf_path {path where processed dataset is set to be stored} --df_path {path to csv file containing pdb ids and associated smiles} 

#### 3. Training model: 
Once you have the dataset you can start training the model. For that can execute model.py file with the following command : 

    python model.py --hdf_path {path where dataset is stored} --train_ids_path {path where list of train ids is stored} --val_ids_path {path where list of validation ids is stored} --test_ids_path {path where list of test ids is stored} --batch_size {batch size for model training} --max_epochs {epochs to train for} --num_workers {number of workers for dataloader} --gpus {num_of_gpus_for_training: None for 'cpu'}

after executing you will get a new folder called "lightning_logs".

#### 4. Generate novel molecules:
After training the model the checkpoint file saved in lightning_logs can be used for generating novel molecules. The protein file should be protonated and not contain heteroatoms (water or ligand).

    from molpro.site_based_gen.predict import generate_smiles
    smiles = generate_smiles(protein_file_path, protein_file_type, generator_checkpoint_path, captioning_checkpoint_path, generator_steps, decoding_steps)
    
Input parameters :

    protein_file_path : str
                   Path to protein file
    protein_file_type : str
                   File format of protein (mol2,pdb,pdbqt)
    generator_checkpoint_path : str
                   Path to the checkpoint of saved generated model
    captioning_checkpoint_path : str
                   Path to the checkpoint of saved shape captioning model
    generator_steps : str 
                   The number of times a unique ligand shape is produced
    decoding_steps : str 
                   The number of times a unique ligand shape is decoded into SMILES

Returns : 

    A list of SMILES containing less than or equal to generator_steps x decoding_steps entries : List[str]


### Predicting the binding site of a protein ligand complex (Binding site prediction)
#### 1. Overview:
The aim of rational drug design is to discover new drugs faster and cheaper. Much of the effort is put into improving docking and scoring methodologies. However, most techniques assume that the exact location of binding sites – also referred to as pockets or binding cavities – is known. Such pockets can be located both on a surface of a single protein (and be used to modulate its activity) or at protein-protein interaction (PPI) interfaces (and be used to disrupt the interaction). This task is very challenging and we lack a method that would predict binding sites with high accuracy – most methods are able to detect only 30%–40% of pockets. In our case, the input is a 3D structure of a protein represented with a grid that can be analyzed in the same manner as 3D images, whereas the desired object is the binding pocket. Our model is based on U-Net – a state of the art model for image segmentation. The model takes protein structure as input, automatically converts it to a 3D grid with features, and outputs probability density – each point in the 3D space has assigned probability of being a part of a pocket.

#### 2. Preparing dataset:
We will be using a subset of scPDB dataset given in the sample data folder for training. 

    python data.py --data_path {path where pdb and mol2 files are stored} --hdf_path {path where processed dataset is set to be stored}

#### 3. Training model: 
Once you have the dataset you can start training the model. For that can execute model.py file with the following command : 

    python model.py --hdf_path {path where dataset is stored} --train_ids_path {path where list of train ids is stored} --val_ids_path {path where list of validation ids is stored} --test_ids_path {path where list of test ids is stored} --batch_size {batch size for model training} --max_epochs {epochs to train for} --num_workers {number of workers for dataloader} --gpus {num_of_gpus_for_training: None for 'cpu'}

after executing you will get a new folder called "lightning_logs".

#### 4. Binding site prediction:
After training the model the checkpoint file saved in lightning_logs can be used for predicting the affinity of protein ligand complex. Make sure that the ligand is docked before giving to the model as input. The protein file should be protonated and not contain heteroatoms (water or ligand).

    from molpro.site_pred.predict import predict_pocket_atoms, write_to_file
    mols = predict_pocket_atoms(protein_file_path, protein_file_type, model_checkpoint_path) # Returns openbabel mol objects
    write_to_file(mols, file_path, file_type)
    
Input parameters :

    protein_file_path : str
                   Path to protein file
    protein_file_type : str
                   File format of protein (mol2,pdb,pdbqt)
    file_path : str
                   Path to output file
    file_type : str
                   File format of output file (mol2,pdb,pdbqt)
    model_checkpoint_path : str 
                   Path to the checkpoint of saved model

Returns : 

    Output files are stored according to output format and file name
    
#### Sample Data Link: https://drive.google.com/drive/folders/1Z6WV3Pk6EQgUtWMEHn7xx1zhh2_dhFlh?usp=sharing

#### Reference:
    
Stepniewska-Dziubinska, M. M., Zielenkiewicz, P., & Siedlecki, P. (2020). Improving detection of protein-ligand binding sites with 3D segmentation. Scientific Reports, 10(1), 5035. https://doi.org/10.1038/s41598-020-61860-z


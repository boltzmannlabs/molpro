### Predicting affinity of a protein ligand complex (Pkd value prediction)
#### 1. Overview:
The worldwide increase and proliferation of drug resistant microbes, coupled with the lag in new drug development, represents a major threat to human health. In order to reduce the time and cost for exploring the chemical search space, drug discovery increasingly relies on computational biology approaches. One key step in these approaches is the need for the rapid and accurate prediction of the binding affinity for potential leads. Here, we present an ensemble of three-dimensional (3D) Convolutional Neural Networks (CNNs), which combines voxelized  molecular descriptors for predicting the absolute binding affinity of protein–ligand complexes.

#### 2. Preparing dataset:
We will be using a subset of PDBBind dataset given in the sample data folder for training. 

    python data.py --data_path {path where pdb and mol2 files are stored} --hdf_path {path where processed dataset is set to be stored} --df_path {path to csv     file containing pkd values and pdb ids} 

#### 3. Training model: 
Once you have the dataset you can start training the model. For that can execute model.py file with the following command : 

    python model.py --hdf_path {path where dataset is stored} --train_ids_path {path where list of train ids is stored} --val_ids_path {path where list of validation ids is stored} --test_ids_path {path where list of test ids is stored} --batch_size {batch size for model training} --max_epochs {epochs to train for} --num_workers {number of workers for dataloader} --gpus {num_of_gpus_for_training: None for 'cpu'}

after executing you will get a new folder called "lightning_logs".

#### 4. Pkd value prediction:
After training the model the checkpoint file saved in lightning_logs can be used for predicting the affinity of protein ligand complex. Make sure that the ligand is docked before giving to the model as input. The protein file should be protonated and not contain heteroatoms (water or ligand).

    from molpro.affinity_pred.predict import predict_affinity
    pkd = predict_affinity(protein_file_path, protein_file_type, ligand_file_path, ligand_file_type, model_checkpoint_path)
    print("The pkd of the protein ligand complex is %s" % pkd)
    
Input parameters :

    protein_file_path : str
                   Path to protein file
    protein_file_type : str
                   File format of protein (mol2,pdb,pdbqt)
    ligand_file_path : str
                   Path to ligand file
    ligand_file_type : str
                   File format of ligand (mol2,pdb,pdbqt)
    model_checkpoint_path : str 
                   Path to the checkpoint of saved model

Returns : 

    pkd value of the protein ligand complex : float

#### Sample Data Link: https://drive.google.com/drive/folders/15x-gLYOGfXYpGjVafmNLIv33e1pm7dUY?usp=sharing

#### Reference: 
    
Hassan-Harrirou, H., Zhang, C., & Lemmin, T. (2020). Rosenet: Improving binding affinity prediction by leveraging molecular mechanics energies with an ensemble of 3d convolutional neural networks. Journal of Chemical Information and Modeling, 60(6), 2791–2802. https://doi.org/10.1021/acs.jcim.0c00075


# Molpro
##### MolPro is a comprehensive python package for small molecule generation using protein active site or/and similar molecules using 3D information of molecules with in-silico validation of molecules by docking , pharmacophore hypothesis. Also off target prediction based on the binding site similarities. 


## How to use --

### Step-1 : Download the molpro package by given link :
    
    https://github.com/boltzmannlabs/molpro
    
### or you can clone the repo by using this git command :

    git clone https://github.com/boltzmannlabs/molpro
    
but before runnig git command make sure you have git installed in your system.

### Step-2 : Install the package
 
 Once you have package in your memory. Then structure will be of directory like this :
 
     /molpro/
            molpro/
                   affinity_pred/
                                 .
                                 .
                   geomol/
                          .
                          .
                   models/
                          .
                          .
                   shape_based_gen/
                                   .
                                   .
                   .
                   .
            install.sh
            License
            setup.py
 
###### Navigate to the parent molpro folder and run :
    
    pip install -e.
    
##### Or run the setup.py file :

    python setup.py
    
Step-2 will install molpro as a package in your current env.


### Step-3 Install all the Dependecies :

##### First make sure you have conda insatall in your system then run the bellow command to create a new eniroment:
    
    conda create --name molpro_env python=3.7
    
##### This command will create a new enviroment as "molpro_env" with python version 3.7. To activate that enviroment run:

    conda activate molpro_env
    
##### Now you can install all the dependecies by running the following command:

    sh install.sh
    

### Step-4 Start using diffrrent-diffrent features :

##### 1. affinity_pred :
The worldwide increase and proliferation of drug resistant microbes, coupled with the lag in new drug development, represents a major threat to human health. In order to reduce the time and cost for exploring the chemical search space, drug discovery increasingly relies on computational biology approaches. One key step in these approaches is the need for the rapid and accurate prediction of the binding affinity for potential leads. Here, we present an ensemble of three-dimensional (3D) Convolutional Neural Networks (CNNs), which combines voxelized molecular descriptors for predicting the absolute binding affinity of protein–ligand complexes. For whole code and how to use that feature visit that directory under molpro directory or you can click on the bellow link :
    
    https://github.com/boltzmannlabs/molpro/tree/main/molpro/affinity_pred
    
##### 2. geomol
Prediction of a molecule’s 3D conformer ensemble from the molecular graph holds a key role in areas of cheminformatics and drug discovery. We are using a  machine learning approach to generate distributions of low-energy molecular 3D conformers. Leveraging the power of message passing neural networks (MPNNs) to capture local and global graph information, we predict local atomic 3D structures and torsion angles, and using these we are assembling the whole conformer for that molecule. For whole code and how to use that feature visit that directory under molpro directory or you can click on the bellow link :

     https://github.com/boltzmannlabs/molpro/tree/main/molpro/geomol
     
 ##### 3. Shape based Generation :
Here we are using an AI approach to generate novel molecules starting from a seed compound, its three-dimensional (3D) shape. A variational autoencoder is used to generate the 3D representation of a compound, followed by a system of convolutional for encoding and recurrent neural networks that generate a sequence of SMILES tokens. The generative design of novel scaffolds and functional groups can cover unexplored regions of chemical space that still possess lead-like properties. For whole code and how to use that feature visit that directory under molpro directory or you can click on the bellow link :
 
     https://github.com/boltzmannlabs/molpro/tree/main/molpro/shape_based_generation
     
##### 4. site_based_gen :
A novel method was developed to generate focused virtual libraries of small molecules based on the protein structure using deep learning-based generative models. Structures of protein–ligand complexes obtained from ligand docking are used to train a generative adversarial model to generate compound structures that are complementary to protein but also maintain diversity among themselves. For whole code and how to use that feature visit that directory under molpro directory or you can click on the bellow link :

    https://github.com/boltzmannlabs/molpro/tree/main/molpro/site_based_gen
    
##### 5. site_pred : 
Task of predicting binding sites of protein is very challenging. Our model is based on U-Net (a state of the art model for image segmentation). The model takes protein structure as input, automatically converts it to a 3D grid with features, and outputs probability density – each point in the 3D space has assigned probability of being a part of a pocket. For whole code and how to use that feature visit that directory under molpro directory or you can click on the bellow link :

    https://github.com/boltzmannlabs/molpro/tree/main/molpro/site_pred
    
    
    
##### 6. Docking :
AutoDock Vina is one of the fastest and most widely used open-source docking engines. It is a turnkey computational docking program that is based on a simple scoring function and rapid gradient-optimization conformational search. It was originally designed and implemented by Dr. Oleg Trott in the Molecular Graphics Lab, and it is now being maintained and develop by the Forli Lab at The Scripps Research Institute.In this, we have created an End-to-End docking pipline, starting from docking, Protein-ligands complex generation and prediction of their interaction , then rescoring or Pkd prediction. For whole code and how to use that feature visit that directory under molpro directory or you can click on the bellow link : 

    https://github.com/boltzmannlabs/molpro/tree/main/molpro/docking
    
##### 7. Shape-Alignment
In drug discovery, common atomic level information of the small molecules / drugs aren't avaiable. In such cases, 3D arrangement (or superposition) of putative ligands have been utilized to conclude underlying necessities for organic movement. Various techniques are proposed for little atomic superposition or primary arrangement. These techniques can be ordered generally into two kinds, in particular point-based and property-based strategies. In point-based strategies, sets of molecules or pharmacophoric focuses are superposed by the least-squares fitting. Notwithstanding, in property-based techniques, different sub-atomic properties are used for superposition, including electron thickness, sub-atomic volume or shape, charge conveyance or sub-atomic electrostatic potential (MEP), hydrophobicity, hydrogen holding capacity, etc. For whole code and how to use that feature, visit that directory under molpro directory or you can click on the bellow link : 
    
    https://github.com/boltzmannlabs/molpro/tree/main/molpro/shape_alignment


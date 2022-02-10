### Generating similiar molecules based on their shape (Shape based generation)
The generative design of novel scaffolds and functional groups can cover unexplored regions of chemical space that still possess lead-like properties.
Here we are using an AI approach to generate novel molecules starting from a seed compound, its three-dimensional (3D) shape. A variational autoencoder is used to generate the 3D representation of a compound, followed by a system of convolutional for encoding and recurrent neural networks that generate a sequence of SMILES tokens. The generative design of novel scaffolds and functional groups can cover unexplored regions of chemical space that still possess lead-like properties.


## For training the model 
##### Data for Training ::

We will be using a subset of Zinc15 dataset for our model training. That will only have drug like smiles. you can download the dataset by clicking the link given bellow: 
    http://pub.htmd.org/zinc15_druglike_clean_canonical_max60.zip
    
After downloading unzipping the file you will get a .smi file as name "zinc15_druglike_clean_canonical_max60.smi". which will have 66666 smiles.

Once you have the dataset you can start training the model. For that can execute model.py file with the following command : 
 `` python model.py --input_path {path_for_.smi_file} --batch_size {your_batch_size} --max_epochs {max_numnber_of_epochs} --num_workers {num_of_workers} --device {'cpu'_or_'gpu'} --gpus {num_of_gpus_for_training}

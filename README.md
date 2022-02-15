# Molpro
###### MolPro is a comprehensive python package for small molecule generation using protein active site or/and similar molecules using 3D information of molecules with in-silico validation of molecules by docking , pharmacophore hypothesis. Also off target prediction based on the binding site similarities. 


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

For installing you can run install.sh file by given command :

    sh install.sh
    




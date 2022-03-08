### Performing molecular Docking using AutoDock Vina or Vinardo

### Overview :

AutoDock Vina is one of the fastest and most widely used open-source docking engines. It is a turnkey computational docking program that 
is based on a simple scoring function and rapid gradient-optimization conformational search. It was originally designed and implemented 
by Dr. Oleg Trott in the Molecular Graphics Lab, and it is now being maintained and develop by the Forli Lab at The Scripps Research Institute.

In this, we have created an End-to-End docking pipline, starting from docking, Protein-ligands  complex generation and prediction of their interaction 
, then rescoring or Pkd prediction.

### How to use?

## 1) perform docking

    from molpro.molpro.docking import dock_score
    binding_score=dock_score(docking_engine = None,protein = None,smile = None,out_path = None,split_out_file = None,grid_size=[40,40,40],center = [0,0,0],exhaustiveness = 32,n_poses = 10)
    
    
    Output: Binding affinity scores of n_poses like:..
    [-6.206, -6.143, -6.136, -6.054, -6.027, -5.999, -5.923, -5.802, -5.75, -5.593]
## Input parameters :

    docking_engine: Str
                Docking Engine to be used** vina or vinardo**
    protein: Str
            pdbqt protein file which need to be docked
    smile: Str
            Smile which need to be docked
    out_path: Str
            folder where the docked PDQT files to be saved
    split_out_file: boolean
            if yes, splits the pdbqt file conatining given number poses into multiple files with one pose each. examples: out_pose_1.pdbqt,out_pose_2.pdbqt,out_pose_3.pdbqt,
            out_pose_4.pdbqt, ...., out_pose_n.pdbqt
            if no, no splitting happens
    grid_size: list
            Box Size to be consider for rigid docking
    center: list
            Box center to be consider for rigid docking
    exhaustiveness: int
            flexibility of atoms to be given
    n_poses: int
            number of poses to be consider for docking
## Returns:
    
    Top_energies: list
                Binding affinity scores of n_poses 
    Generates PDBQT file [out.pdbqt, having the all poses informarion], if split_out_file=True, files will be geneated as mention above.
           
## 2) preparing the complex files for protein-Ligand interaction calculation

    from molpro.docking import complex_prep
    complex_prep(protein = None,ligand= None ,ligand_file_type = None)

## Input parameters :
    protein: str
            Path to protein ,file format PDB
    ligand: str
            Path to ligand file
    ligand_file_type: str
            File format of ligand (mol2,pdb,pdbqt)

## Returns:
    
    None
    
## 3) Protein ligand Interaction calculation

    from molpro.docking.dock import Pro_lig_int
    interaction=Pro_lig_int(complex=None)
    
    
    Output: PLI like:..
    
    {'hydrophobic': [{'RESNR': 212, 'RESTYPE': 'LEU', 'RESCHAIN': 'A', 'RESNR_LIG': 1, 'RESTYPE_LIG': 'UNL', 'RESCHAIN_LIG': 'Z', 'DIST': '3.96', 'LIGCARBONIDX': 1460, 'PROTCARBONIDX': 518, 'LIGCOO': (26.163, 31.117, 21.82), 'PROTCOO': (23.244, 29.905, 24.2)},
    {'RESNR': 261, 'RESTYPE': 'TYR', 'RESCHAIN': 'A', 'RESNR_LIG': 1, 'RESTYPE_LIG': 'UNL', 'RESCHAIN_LIG': 'Z', 'DIST': '3.49', 'LIGCARBONIDX': 1463, 'PROTCARBONIDX': 800, 'LIGCOO': (27.579, 28.735, 20.895), 'PROTCOO': (27.125, 25.332, 21.507)}], 
    'hbond': [{'RESNR': 241, 'RESTYPE': 'SER', 'RESCHAIN': 'A', 'RESNR_LIG': 1, 'RESTYPE_LIG': 'UNL', 'RESCHAIN_LIG': 'Z', 'SIDECHAIN': True, 'DIST_H-A': '3.59', 'DIST_D-A': '4.04', 'DON_ANGLE': '110.52', 'PROTISDON': True, 'DONORIDX': 649, 'DONORTYPE': 'O3', 
    'ACCEPTORIDX': 1485, 'ACCEPTORTYPE': 'Nar', 'LIGCOO': (29.472, 32.75, 21.679), 'PROTCOO': (31.738, 35.553, 23.5)}, {'RESNR': 243, 'RESTYPE': 'SER', 'RESCHAIN': 'A', 'RESNR_LIG': 1, 'RESTYPE_LIG': 'UNL', 
    'RESCHAIN_LIG': 'Z', 'SIDECHAIN': True, 'DIST_H-A': '3.14', 'DIST_D-A': '3.47', 'DON_ANGLE': '101.77', 'PROTISDON': True, 'DONORIDX': 663, 'DONORTYPE': 'O3', 'ACCEPTORIDX': 1486, 'ACCEPTORTYPE': 'Nar', 'LIGCOO': (29.412, 32.754, 23.0), 
    'PROTCOO': (27.882, 35.066, 25.094)}, {'RESNR': 260, 'RESTYPE': 'GLU', 'RESCHAIN': 'A', 'RESNR_LIG': 1, 'RESTYPE_LIG': 'UNL', 'RESCHAIN_LIG': 'Z', 'SIDECHAIN': True, 'DIST_H-A': '3.12', 'DIST_D-A': '3.54', 'DON_ANGLE': '109.13', 'PROTISDON': True, 'DONORIDX': 794, 
    'DONORTYPE': 'O3', 'ACCEPTORIDX': 1487, 'ACCEPTORTYPE': 'Nar', 'LIGCOO': (30.318, 31.87, 23.532), 'PROTCOO': (31.314, 30.247, 26.514)}]}
## Input parameters :
    
    complex: str
            path for the complex file, file format PDB
## Returns:
    
    interaction: Dict
                Returns the dictinary having the PLI 



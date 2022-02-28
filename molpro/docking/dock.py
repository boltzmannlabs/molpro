# importing libraries

## importing Docking packages
from vina import Vina
import oddt
import oddt.docking
from openbabel import openbabel
from PLI import plip
from complex_prep import Converter

## importing basic python packages
import os
import re

def dock_score(docking_engine,protein,smile,out_path,split_out_file,grid_size=[40,40,40],center=[0,0,0],exhaustiveness=32,n_poses=10):
    """
    Peforming Docking for smile and protein using given parameters 

    Parametrs:
    ----------
    docking_engine: Str
            Docking Engine to be used vina or vinardo
    protein: Str
            pdbqt protein file which need to be docked
    smile: Str
            Smile which need to be docked
    out_path: Str
            folder where the docked PDQT files to be saved
    split_out_file: boolean
            if yes, splits the pdbqt file conatining given number poses into multiple files with one pose each
            if no, no splitting happens
    grid_size: list
            Box Size to be consider for rigid docking
    center: list
            Box center to be consider for rigid docking
    exhaustiveness: int
            flexibility of atoms to be given
    n_poses: int
            number of poses to be consider for docking
    
    Returns:
    -------
    Top_energies: list
                Binding affinity scores fo n_poses 
    """
    
    print("started smile to pdbqt conversion")
    m = oddt.toolkit.readstring('smi',smile)
    if not m.OBMol.Has3D(): 
        m.make3D(forcefield='mmff94', steps=150)
    oddt.docking.AutodockVina.write_vina_pdbqt(m,os.getcwd(),name_id='temp')
    print(f"Done!! pdbqt conversion\n Setting up docking engine: {docking_engine}")
    
    v = Vina(sf_name=docking_engine,cpu=4)

    #set receptor
    v.set_receptor(protein)

    #set ligand
    v.set_ligand_from_file(os.path.join(os.getcwd(),"temp.pdbqt"))

    #generating the maps
    v.compute_vina_maps(center=center, box_size=grid_size)

    # Score the current pose
    energy = v.score()
    print('Score before minimization: %.3f (kcal/mol)' % energy[0])

    # Minimized locally the current pose
    energy_minimized = v.optimize()
    print('Score after minimization : %.3f (kcal/mol)' % energy_minimized[0])

    # Dock the ligand
    sc = v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
    e=v.energies(n_poses=n_poses)
    
    os.remove(os.path.join(os.getcwd(),"temp.pdbqt"))
    v.write_poses(os.path.join(out_path,'out.pdbqt'), n_poses=n_poses+1, overwrite=True)

    top_energy=[e[i][0] for i in range(len(e))]
    if split_out_file:
        #spliting the files
        path=out_path
        if not os.path.exists(path):
            os.mkdir(path)
        file = open(os.path.join(out_path,'out.pdbqt'), 'r')
        pose_data = file.read()
        pattern = "(MODEL)([A-Za-z\s0-9=_+:;()'\.\n\-]+)(ENDMDL)"
        pt = re.compile(pattern)
        finds = re.split(r'ENDMDL\n', pose_data)
        # print(finds)
        for i in range(len(finds)):
            finds[i] = finds[i] + 'ENDMDL'
            a = open(os.path.join(out_path,f'out_pose_{i+1}.pdbqt'), 'w')
            a.write(finds[i])
            a.close()
        os.remove(os.path.join(out_path,f'out_pose_{i+1}.pdbqt'))
    return top_energy

    

   

def complex_prep(protein,ligand,ligand_file_type):
    """
    Creating the complex file using the docked protein and ligand file 

    Parametrs:
    ----------
    protein: str
            Path to protein ,file format PDB
    ligand: str
            Path to ligand file
    ligand_file_type: str
            File format of ligand (mol2,pdb,pdbqt)
    Returns:
    -------
    None

    """
    if os.path.exists(os.path.join(os.getcwd(),'_complex_temp.pdb')):
        os.remove(os.path.join(os.getcwd(),'_complex_temp.pdb'))
    ligand_name=os.path.basename(ligand).split('.')[0]
    Converter(protein,ligand,ligand_name,ligand_file_type).convert()
    complex_file = open(os.path.join(os.getcwd(),'_complex_temp.pdb'), "r")
    lines = complex_file.readlines()
    complex_file.close()
    ind=lines.index('END                                                                             \n')
#     ind=search_string_in_file(f'{ligand_name}.pdb','END')
    print(ind)
    del lines[ind]
    new_file = open(os.path.join(os.getcwd(),f'{ligand_name}_complex_.pdb'), "w+")
    for line in lines:
        new_file.write(line)

    new_file.close()
    os.remove(os.path.join(os.getcwd(),'_complex_temp.pdb'))
    print("Successfully generated complex file:",os.path.join(os.getcwd(),f'{ligand_name}_complex_.pdb'))



def Pro_lig_int(complex):
    """
    Protein ligand interaction evaluation/prediction  

    Parametrs:
    ----------
    complex; str
            path for the complex file, file format PDB
    Returns:
    -------
    interaction: Dict
                Returns the dictinary having the PLI 
    """
    try:
        interaction=plip(complex)
    except:
        interaction="No interactions"

    return interaction

if __name__=='__main__':
        top_energy=dock_score('vina','/home/boltzmann/space/KF/affinity_update/sting.pdbqt','OC(CC1)CCC1CNC(CN2N=NN=C2C3=CC=C(S3)Br)=O',
        '/home/boltzmann/space/KF/affinity_update/temp_test',True,[40,40,40],[26.724,24.086,27.778],32,10)
        # complex_prep('/home/boltzmann/space/KF/affinity_update/protein.pdb','/home/boltzmann/space/KF/affinity_update/temp_test/out_pose_1.pdbqt','pdbqt')
        # interaction=Pro_lig_int("/home/boltzmann/space/KF/molpro/molpro/docking/out_pose_1_complex_.pdb")
        print(top_energy)
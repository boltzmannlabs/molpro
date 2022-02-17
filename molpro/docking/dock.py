from vina import Vina
import oddt
import oddt.docking
import os
import re
from openbabel import openbabel
import subprocess
from omegaconf import OmegaConf
import numpy as np
from PLI import plip
def dock_score(docking_engine,protein,smile,out_path,split_out_file,grid_size=[40,40,40],center=[0,0,0],exhaustiveness=32,n_poses=10):
    print("started smile to pdbqt conversion")
    m = oddt.toolkit.readstring('smi',smile)
    if not m.OBMol.Has3D(): 
        m.make3D(forcefield='mmff94', steps=150)
    oddt.docking.AutodockVina.write_vina_pdbqt(m,"/home/boltzmann/consumers/consumers/docking",name_id='ttt1')
    print(f"Done!! pdbqt conversion\n Setting up docking engine: {docking_engine}")
    
    v = Vina(sf_name=docking_engine,cpu=4)

    #set receptor
    v.set_receptor(protein)

    #set ligand
    v.set_ligand_from_file("temp.pdbqt")

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
    
    os.remove("temp.pdbqt")
    v.write_poses(os.path.join(out_path,'out.pdbqt'), n_poses=n_poses+1, overwrite=True)

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

def Pro_lig_int(complex):

    try:
        interaction=plip(complex)
    except:
        interaction="No interactions"

    return interaction
# importing libraries

## importing RDKIT packages
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import SDWriter
from rdkit.Chem.Subshape import SubshapeAligner, SubshapeBuilder, SubshapeObjects

## importing open drug dicovery packages
import oddt
import oddt.docking

## importing basic python packages
import pandas as pd
import numpy as np
import os
import traceback
import shutil
from multiprocessing import Pool
from itertools import repeat
import time

RDLogger.DisableLog('rdApp.info')   # to disable logs from rdkit

def library_screening(library_path,smile,thres=0.6):
    """
    Screening the database Chembl / Surechembl  or any huge libraries 

    Parametrs:
    ----------
    library_path : Str,
            Chembl / Surechembl or any huge libraries.Pickle File contains the ID,SMILES, and Fingerprint(
                morgan figerprint radius 2 and bits 1024) 
    smile: Str,
            Query smile whose similar smiles need to be screened
    path: Str,
            Path of folder where the aligned SDF files need to be saved
    thres: float
            Minimum 3D similarity score that smiles need to meet
    Returns:
    -------
    new_data: DataFrame
            DataFrame with smiles,taget_id(Library ID),fp_score(fingerprint score),Shape_similarity
            (3D shape similarity score),query_path(user smile's aligned SDF file path),hit_path(aligned SDF file path
            Smile from library which is similary to query smiles )  
    """
    df=pd.read_pickle(library_path) 
    df=df.dropna(axis=0)
    fp=AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile),2, nBits=1024)
    fps=list(df['fps'])
    sims = DataStructs.BulkTanimotoSimilarity(fp,fps)
    df['score']=sims
    df=df[df['score']>=0.6]   
    # df['Shape_similarity']=''
    df['query_path']=''
    df['hit_path']=''
    #path="sdfs"
    
    path=os.getcwd()
    df=df.rename(columns={'SMILES':'smiles','ID':'taget_id','score':'fp_score'})
    print(df.shape)
    if not os.path.exists(path):
        os.mkdir(path)
    query_mol = oddt.toolkit.readstring('smi',smile)
    if not query_mol.OBMol.Has3D(): 
        query_mol.make3D(forcefield='mmff94', steps=150)
    query_sdf=oddt.toolkits.ob.Outputfile(format='sdf',filename=os.path.join(path,"query.sdf"),overwrite=True)
    query_sdf.write(query_mol)
    query_sdf.close()
    pool = Pool()
    mols=df["smiles"].to_list()
    results = pool.starmap(align, zip(repeat(os.path.join(path,"query.sdf"), len(mols)),mols,list(range(len(mols))),repeat(path,len(mols))))
    if os.path.isdir(path):
        shutil.rmtree(path)
    
    print(results)
    df2=pd.DataFrame(results,columns=["Shape_similarity","ref","probe"])
    df2["smiles"]=mols
    
    df=pd.merge(df,df2,on='smiles')
    if not os.path.exists(path):
        os.mkdir(path)
    for i,r in df.iterrows():
        if r['Shape_similarity']!=0:
            r["ref"].SetProp("_Name","Query Molecule")
            writer=SDWriter(os.path.join(path,'ref{i}.sdf'))
            writer.write(r["ref"])
            writer.close()
            df.loc[i,'query_path']=os.path.join(path,'ref{i}.sdf')
            r["probe"].SetProp("_Name","Hit Molecule")
            w=SDWriter(os.path.join(path,'probe{i}.sdf'))
            w.write(r["probe"])
            w.close()
            df.loc[i,'hit_path']=os.path.join(path,'probe{i}.sdf')
        else:
            df.loc[i,'hit_path']="No files generated"
            df.loc[i,'query_path']="No files generated"
    new_data=df[df['Shape_similarity']>=thres]
    return new_data


def align(query,target,idx,path):
    """
    performing the actual 3D alignment for given to smiles 

    Parametrs:
    ----------
    query: Str
            Its the SDF file path of the Query smile
    target: Str
            Its the smile against which similarity is calculated
    idx: int
            It is the smile index from the library or the list
    Path: Str
            Path of folder where the aligned SDF files need to be saved
    Returns:
    -------
    score: float
            3D alignment score
    ref: 
            query smile aligned rdkit object 
    probe: 
            target smile aligned rdkit object 
    """
    path = path
    #creating temporary file for target
    target_mol = oddt.toolkit.readstring('smi',target)
    if not target_mol.OBMol.Has3D(): 
        target_mol.make3D(forcefield='mmff94', steps=150)
    target_sdf=oddt.toolkits.ob.Outputfile(format='sdf',filename=f"{path}/target_{idx}.sdf",overwrite=True)
    target_sdf.write(target_mol)
    target_sdf.close()

    querysdf_new=[m for m in Chem.SDMolSupplier(query)]
    targetsdf_new=[n for n in Chem.SDMolSupplier(f"{path}/target_{idx}.sdf")]

    ref = Chem.Mol(querysdf_new[0].ToBinary())
    probe = Chem.Mol(targetsdf_new[0].ToBinary())

    # creating multiple conformers and selecting the best one for alignment based on energy
    AllChem.CanonicalizeConformer(ref.GetConformer())
    AllChem.CanonicalizeConformer(probe.GetConformer())

    # setting up the alignment functions
    builder = SubshapeBuilder.SubshapeBuilder()
    builder.gridDims = (20.,20.,10)
    builder.gridSpacing=0.5
    builder.winRad = 4.

    # generating the subshapes for alignments
    refShape = builder.GenerateSubshapeShape(ref)
    probeShape = builder.GenerateSubshapeShape(probe)

    # aligning the smiles
    aligner = SubshapeAligner.SubshapeAligner()
    algs = aligner.GetSubshapeAlignments(ref, refShape, probe, probeShape, builder)

    if len(algs)!=0:
        alg = algs[0]
        AllChem.TransformMol(probe, alg.transform)
        newprobeShape = builder(probe)

        score=1.0-alg.shapeDist

        return score,ref,probe
    else:
        return 0,np.NaN,np.NaN


def UPA(query,mols):
    """
    Calculate 3D similarity of given smile against the given list of smiles 

    Parametrs:
    ---------- 
    query: Str,
            Query smile whose similar smiles need to be screened
    mols: List,
            List of smiles against which the query smile similarity to be checked
    Returns:
    -------
    new_data: DataFrame
            DataFrame with smiles,taget_id(Library ID),fp_score(fingerprint score),Shape_similarity
            (3D shape similarity score),query_path(user smile's aligned SDF file path),hit_path(aligned SDF file path
            Smile from library which is similary to query smiles )  
    """
    
    pool = Pool()
    
    path=os.getcwd()
    
    # calculating the Morgan fingerprints for query and given smiles list
    fp=AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(query),2, nBits=1024)
    fps=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(i),2, nBits=1024) for i in mols]

    # calculating the fingerprint tantimoto similarity of them
    sims = DataStructs.BulkTanimotoSimilarity(fp,fps)

    if not os.path.exists(path):
        os.mkdir(path)

    #creating temporay sdf file for query
    query_mol = oddt.toolkit.readstring('smi',query)
    if not query_mol.OBMol.Has3D(): 
        query_mol.make3D(forcefield='mmff94', steps=150)
    query_sdf=oddt.toolkits.ob.Outputfile(format='sdf',filename=os.path.join(path,"query.sdf"),overwrite=True)
    query_sdf.write(query_mol)
    query_sdf.close()

    # Calculating the similarity via multiprocessing
    results = pool.starmap(align, zip(repeat(os.path.join(path,"query.sdf"), len(mols)),mols,list(range(len(mols))),repeat(path,len(mols))))
    if os.path.isdir(path):
        shutil.rmtree(path)
    print(results)
    df=pd.DataFrame(results,columns=["Shape_similarity","ref","probe"])
    df["smiles"]=mols
    df['fps_score']=sims
    print(df)
    if not os.path.exists(path):
        os.mkdir(path)
    for i,r in df.iterrows():
        if r['Shape_similarity']!=0:
            r["ref"].SetProp("_Name","Query Molecule")
            writer=SDWriter(os.path.join(path,'ref{i}.sdf'))
            writer.write(r["ref"])
            writer.close()
            df.loc[i,'query_path']=os.path.join(path,'ref{i}.sdf')
            r["probe"].SetProp("_Name","Hit Molecule")
            w=SDWriter(os.path.join(path,'probe{i}.sdf'))
            w.write(r["probe"])
            w.close()
            df.loc[i,'hit_path']=os.path.join(path,'probe{i}.sdf')
        else:
            df.loc[i,'hit_path']="No files generated"
            df.loc[i,'query_path']="No files generated"
    
    return df

if __name__=='__main__':
    
    ## to run the library screening
    # data_out=library_screening(library_path='/home/boltzmann/space/KF/chembl/Chembl_29.pickle',smile='Cc1nc2ccccc2n1Cc3ccc(cc3F)C(=O)NO',thres=0.6)
    # print(data_out)

    ## to run the custom library screening
    mols=['Cc1nc2ccccc2n1Cc1c(F)cccc1F','Cc1nc2ccccc2n1Cc1ccc(Cl)cc1',
    'COC(=O)c1cccc2c1c(C(=O)c1ccc(Cn3c(C)nc4ccccc43)cc1)cn2C(=O)N(C)C',
    'CCCCc1nc2ccccc2n1Cc1ccc(C(=O)O)cc1Br',
    'Cc1nc2cnccc2n1Cc1ccc(C(=O)c2cn(C(=O)N(C)C)c3cccc(OC(=O)N(C)C)c23)cc1F']
    data_out=UPA('Cc1nc2ccccc2n1Cc1c(F)cccc1F',mols)
    print(data_out)
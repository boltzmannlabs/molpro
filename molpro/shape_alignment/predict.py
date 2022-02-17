from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PyMol
from rdkit.Chem.Subshape import SubshapeAligner, SubshapeBuilder, SubshapeObjects
import pandas as pd
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
import oddt
import oddt.docking
import os
import traceback
from rdkit.Chem.rdmolfiles import SDWriter
import shutil
from rdkit import RDLogger
import numpy as np
from multiprocessing import Pool
from itertools import repeat
import time
RDLogger.DisableLog('rdApp.info')

def library_screening(library_path,smile, thres=0.6):
    
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
    path="sdfs"
    df=df.rename(columns={'SMILES':'smiles','ID':'taget_id','score':'fp_score'})
    print(df.shape)
    if not os.path.exists(path):
        os.mkdir(path)
    query_mol = oddt.toolkit.readstring('smi',smile)
    if not query_mol.OBMol.Has3D(): 
        query_mol.make3D(forcefield='mmff94', steps=150)
    query_sdf=oddt.toolkits.ob.Outputfile(format='sdf',filename=f"{path}/query.sdf",overwrite=True)
    query_sdf.write(query_mol)
    query_sdf.close()
    pool = Pool()
    mols=df["smiles"].to_list()
    results = pool.starmap(align, zip(repeat(f"{path}/query.sdf", len(mols)),mols,list(range(len(mols))),repeat(path,len(mols))))
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
            writer=SDWriter(f'{path}/ref{i}.sdf')
            writer.write(r["ref"])
            writer.close()
            df.loc[i,'query_path']=f'{path}/ref{i}.sdf'
            r["probe"].SetProp("_Name","Hit Molecule")
            w=SDWriter(f'{path}/probe{i}.sdf')
            w.write(r["probe"])
            w.close()
            df.loc[i,'hit_path']=f'{path}/probe{i}.sdf'
        else:
            df.loc[i,'hit_path']="No files generated"
            df.loc[i,'query_path']="No files generated"
    new_data=df[df['Shape_similarity']>=thres]
    return new_data



def pharma_align(query,target):
    #creating temporay sdf file for query
    query_mol = oddt.toolkit.readstring('smi',query)
    if not query_mol.OBMol.Has3D(): 
        query_mol.make3D(forcefield='mmff94', steps=150)
    query_sdf=oddt.toolkits.ob.Outputfile(format='sdf',filename="query.sdf",overwrite=True)
    query_sdf.write(query_mol)
    query_sdf.close()

    #creating temporary file for target
    target_mol = oddt.toolkit.readstring('smi',target)
    if not target_mol.OBMol.Has3D(): 
        target_mol.make3D(forcefield='mmff94', steps=150)
    target_sdf=oddt.toolkits.ob.Outputfile(format='sdf',filename="target.sdf",overwrite=True)
    target_sdf.write(target_mol)
    target_sdf.close()

    #reading back the sdf files
    querysdf_new=[m for m in Chem.SDMolSupplier("query.sdf")]
    targetsdf_new=[n for n in Chem.SDMolSupplier("target.sdf")]

    #alignment

    ref = Chem.Mol(querysdf_new[0].ToBinary())
    probe = Chem.Mol(targetsdf_new[0].ToBinary())

    AllChem.CanonicalizeConformer(ref.GetConformer())
    AllChem.CanonicalizeConformer(probe.GetConformer())
    builder = SubshapeBuilder.SubshapeBuilder()
    builder.gridDims = (20.,20.,10)
    builder.gridSpacing=0.5
    builder.winRad = 4.

    refShape = builder.GenerateSubshapeShape(ref)
    probeShape = builder.GenerateSubshapeShape(probe)

    aligner = SubshapeAligner.SubshapeAligner()

    algs = aligner.GetSubshapeAlignments(ref, refShape, probe, probeShape, builder)

    alg = algs[0]
    AllChem.TransformMol(probe, alg.transform)
    newprobeShape = builder(probe)

    score=1.0-alg.shapeDist
    # os.remove('/home/bayeslabs/shape_aligment/query.sdf')
    # os.remove('/home/bayeslabs/shape_aligment/target.sdf')

    
    return score,ref,probe

def align(query,target,idx,path):
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

    AllChem.CanonicalizeConformer(ref.GetConformer())
    AllChem.CanonicalizeConformer(probe.GetConformer())
    builder = SubshapeBuilder.SubshapeBuilder()
    builder.gridDims = (20.,20.,10)
    builder.gridSpacing=0.5
    builder.winRad = 4.

    refShape = builder.GenerateSubshapeShape(ref)
    probeShape = builder.GenerateSubshapeShape(probe)
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
    pool = Pool()
    start=time.time()
    path="user_sdf"
    
    fp=AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(query),2, nBits=1024)
    fps=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(i),2, nBits=1024) for i in mols]
    sims = DataStructs.BulkTanimotoSimilarity(fp,fps)
    if not os.path.exists(path):
        os.mkdir(path)
    #creating temporay sdf file for query
    query_mol = oddt.toolkit.readstring('smi',query)
    if not query_mol.OBMol.Has3D(): 
        query_mol.make3D(forcefield='mmff94', steps=150)
    query_sdf=oddt.toolkits.ob.Outputfile(format='sdf',filename=f"{path}/query.sdf",overwrite=True)
    query_sdf.write(query_mol)
    query_sdf.close()
    results = pool.starmap(align, zip(repeat(f"{path}/query.sdf", len(mols)),mols,list(range(len(mols))),repeat(path,len(mols))))
    if os.path.isdir(path):
        shutil.rmtree(path)
    print(time.time()-start)
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
            writer=SDWriter(f'{path}/ref{i}.sdf')
            writer.write(r["ref"])
            writer.close()
            df.loc[i,'query_path']=f'{path}/ref{i}.sdf'
            r["probe"].SetProp("_Name","Hit Molecule")
            w=SDWriter(f'{path}/probe{i}.sdf')
            w.write(r["probe"])
            w.close()
            df.loc[i,'hit_path']=f'{path}/probe{i}.sdf'
        else:
            df.loc[i,'hit_path']="No files generated"
            df.loc[i,'query_path']="No files generated"
    
    return df

if __name__=='__main__':
    
    # data_out=library_screening(library_path='/home/boltzmann/space/KF/chembl/Chembl_29.pickle',smile='Cc1nc2ccccc2n1Cc3ccc(cc3F)C(=O)NO',thres=0.6)
    # print(data_out)

    mols=['Cc1nc2ccccc2n1Cc1c(F)cccc1F','Cc1nc2ccccc2n1Cc1ccc(Cl)cc1',
    'COC(=O)c1cccc2c1c(C(=O)c1ccc(Cn3c(C)nc4ccccc43)cc1)cn2C(=O)N(C)C',
    'CCCCc1nc2ccccc2n1Cc1ccc(C(=O)O)cc1Br',
    'Cc1nc2cnccc2n1Cc1ccc(C(=O)c2cn(C(=O)N(C)C)c3cccc(OC(=O)N(C)C)c23)cc1F']
    data_out=UPA('Cc1nc2ccccc2n1Cc1c(F)cccc1F',mols)
    print(data_out)
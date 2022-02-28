### To align small molecules and calculate their 3D similarity between them.

### 1. Overview:
In drug discovery, common atomic level information of the small molecules / drugs aren't avaiable. In such cases, 3D arrangement (or superposition) of putative ligands have been utilized to conclude underlying necessities for organic movement. Various techniques are proposed for little atomic superposition or primary arrangement. These techniques can be ordered generally into two kinds, in particular point-based and property-based strategies. In point-based strategies, sets of molecules or pharmacophoric focuses are superposed by the least-squares fitting. Notwithstanding, in property-based techniques, different sub-atomic properties are used for superposition, including electron thickness, sub-atomic volume or shape, charge conveyance or sub-atomic electrostatic potential (MEP), hydrophobicity, hydrogen holding capacity, etc.

### There are two function under this feature: 

### 1) To find similarities of one smile with user's small size custom library 

### How to use it?

    from molpro.molpro.shape_alignment import UPA

    input_smiles=[smile1,smile2,smile3,....,smileN]
    data_out=UPA(query_smile,input_smiles)
    
##### Input parameters :

    input_smiles : List[str]
              Simles for which you want to Find similarity with query smile. input should be in ['smile_1,smiles_2,.....,smile_n] format
    query_smile: str
        Smile for which you wnat to find the similarities. 
        
##### Returns : 

     data_out: pandas DataFrame
              DataFrame with smiles,taget_id(Library ID),fp_score(fingerprint score),Shape_similarity
              (3D shape similarity score),query_path(user smile's aligned SDF file path),hit_path(aligned SDF file path
              Smile from library which is similary to query smiles ) 
            

### 2) To screen a huge library which meets the similarity thresholds with given query smile.

### How to use it?

    from molpro.molpro.shape_alignment import library_screening

    library_path="chembl/surechembl path"
    data_out=UPA(library_path,smile)
    
##### Input parameters :
    
    library_path : Str,
            Chembl / Surechembl or any huge libraries.Pickle File contains the ID,SMILES, and Fingerprint(
                morgan figerprint radius 2 and bits 1024) 
    smile: Str,
            Query smile whose similar smiles need to be screened
    thres: float
            Minimum 3D similarity score that smiles need to meet. Default: 0.6
            
##### Returns : 

    data_out: pandas DataFrame
            DataFrame with smiles,taget_id(Library ID),fp_score(fingerprint score),Shape_similarity
            (3D shape similarity score),query_path(user smile's aligned SDF file path),hit_path(aligned SDF file path
            Smile from library which is similary to query smiles ) 



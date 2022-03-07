import numpy as np
from rdkit import Chem
from typing import List
import torch
from torch.utils.data import Dataset
from molpro.utils.preprocess import make_3dgrid, Featurizer, rotate_grid
from molpro.shape_based_gen.data import vocab_c2i_v1
from random import choice

import os.path as osp
import numpy as np
import glob
import pickle
import random
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Data, Dataset as tg_dataset
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from molpro.geomol.geomol_utils import one_k_encoding , get_dihedral_pairs , dihedral_pattern,chirality,qm9_types,drugs_types


class ShapeBasedGenDataset(Dataset):
    """ Class to featurize smile while training 

    Input Parameters :
    -------------------------

    smiles_list : List[str]
                 a list which contains smiles

    file_type : str 
                by which file format smiles are extracted .smi or other
    
    """

    def __init__(self,smiles_list: List[str],file_type:str="smi"):

        self.smiles_list = smiles_list
        self.file_type = file_type

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx: int):
        smi = self.smiles_list[idx]

        smiles_token = smi
        fatures_list = ['hydrophobic', 'aromatic','acceptor', 'donor','ring']

        featurizer = Featurizer(input_file = smi , file_type= 'smi', named_props  = ["partialcharge"], smarts_labels = fatures_list, metal_halogen_encode = False)


        coords = featurizer.coords
        centroid = coords.mean(axis=0)
        coords -= centroid
        afeats = featurizer.features
        features1 = afeats[:,:5]
        features2 = afeats[:,3:]
        rot = choice(range(24))
        tr1 = 2 * np.random.rand(1, 3)
        tr2 = 0.5 * np.random.rand(1, 3)
        coords1 = rotate_grid(coords,rot)
        coords1 += tr1
        f1n = make_3dgrid(coords1,features1,23,2)

        coords2 = rotate_grid(coords,rot)
        coords2 += tr2
        f2n = make_3dgrid(coords2,features2,23,2)

        feats_final = np.concatenate([f1n,f2n],axis=4)


        vox = np.squeeze(feats_final, 0).transpose(3, 0, 1, 2)

        mol = Chem.MolFromSmiles(smiles_token)
        if not mol:
            raise ValueError(f"Failed to parse molecule '{mol}'")

        sstring = Chem.MolToSmiles(mol)  # Make the SMILES canonical.
        sstring = sstring.replace("Cl", "X").replace("[nH]", "Y") \
                                            .replace("Br", "Z")
        try:
            vals = [1] + \
                   [vocab_c2i_v1[xchar] for xchar in sstring] + \
                   [2]
        except KeyError:
            raise ValueError(
                ("Unkown SMILES tokens: {} in string '{}'."
                 .format(", ".join([x for x in sstring if
                                    x not in vocab_c2i_v1]),
                         sstring)))
        end_token = vals.index(2)
        return torch.Tensor(vox), torch.Tensor(vals), end_token + 1




class geom_confs(tg_dataset):
    def __init__(self, dataset_path: str, indexes_array:np.array, mode:str, transform=None, pre_transform=None, max_confs=10):

        super(geom_confs, self).__init__(dataset_path, transform, pre_transform)
        self.dataset_path = dataset_path
        self.mode = mode
        self.indexes_array = indexes_array
        self.bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}


        self.dihedral_pairs = {} # for memoization
        all_files = sorted(glob.glob(osp.join(self.dataset_path, '*.pickle')))
        self.pickle_files = [f for i, f in enumerate(all_files) if i in self.indexes_array]
        self.max_confs = max_confs

    def len(self):
        ###################################################################################################################################################
        #return 40 if self.mode == "train" else 5 # for training in sample_data
        return 10000 if self.mode == 0 else 1000
        ###################################################################################################################################################

    def get(self, idx):
        data = None
        while not data:
            pickle_file = random.choice(self.pickle_files)
            mol_dic = self.open_pickle(pickle_file)
            data = self.featurize_mol(mol_dic)

        if idx in self.dihedral_pairs:
            data.edge_index_dihedral_pairs = self.dihedral_pairs[idx]
        else:
            data.edge_index_dihedral_pairs = get_dihedral_pairs(data.edge_index, data=data)

        return data

    def open_pickle(self, mol_path):
        with open(mol_path, "rb") as f:
            dic = pickle.load(f)
        return dic

    def featurize_mol(self, mol_dic):
        confs = mol_dic['conformers']
        random.shuffle(confs)  # shuffle confs
        name = mol_dic["smiles"]

        # filter mols rdkit can't intrinsically handle
        mol_ = Chem.MolFromSmiles(name)
        if mol_:
            canonical_smi = Chem.MolToSmiles(mol_)
        else:
            return None

        # skip conformers with fragments
        if '.' in name:
            return None

        # skip conformers without dihedrals
        N = confs[0]['rd_mol'].GetNumAtoms()
        if N < 4:
            return None
        if confs[0]['rd_mol'].GetNumBonds() < 4:
            return None
        if not confs[0]['rd_mol'].HasSubstructMatch(dihedral_pattern):
            return None

        pos = torch.zeros([self.max_confs, N, 3])
        pos_mask = torch.zeros(self.max_confs, dtype=torch.int64)
        
        k = 0
        for conf in confs:
            mol = conf['rd_mol']

            # skip mols with atoms with more than 4 neighbors for now
            n_neighbors = [len(a.GetNeighbors()) for a in mol.GetAtoms()]
            if np.max(n_neighbors) > 4:
                continue

            # filter for conformers that may have reacted
            try:
                conf_canonical_smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
            except Exception as e:
                continue

            if conf_canonical_smi != canonical_smi:
                continue

            pos[k] = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)
            pos_mask[k] = 1
            k += 1
            correct_mol = mol
            if k == self.max_confs:
                break

        # return None if no non-reactive conformers were found
        if k == 0:
            return None

        type_idx = []
        atomic_number = []
        atom_features = []
        chiral_tag = []
        neighbor_dict = {}
        ring = correct_mol.GetRingInfo()
        for i, atom in enumerate(correct_mol.GetAtoms()):
            type_idx.append(self.types[atom.GetSymbol()])
            n_ids = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(n_ids) > 1:
                neighbor_dict[i] = torch.tensor(n_ids)
            chiral_tag.append(chirality[atom.GetChiralTag()])
            atomic_number.append(atom.GetAtomicNum())
            atom_features.extend([atom.GetAtomicNum(),
                                  1 if atom.GetIsAromatic() else 0])
            atom_features.extend(one_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))
            atom_features.extend(one_k_encoding(atom.GetHybridization(), [
                                    Chem.rdchem.HybridizationType.SP,
                                    Chem.rdchem.HybridizationType.SP2,
                                    Chem.rdchem.HybridizationType.SP3,
                                    Chem.rdchem.HybridizationType.SP3D,
                                    Chem.rdchem.HybridizationType.SP3D2]))
            atom_features.extend(one_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]))
            atom_features.extend(one_k_encoding(atom.GetFormalCharge(), [-1, 0, 1]))
            atom_features.extend([int(ring.IsAtomInRingOfSize(i, 3)),
                                  int(ring.IsAtomInRingOfSize(i, 4)),
                                  int(ring.IsAtomInRingOfSize(i, 5)),
                                  int(ring.IsAtomInRingOfSize(i, 6)),
                                  int(ring.IsAtomInRingOfSize(i, 7)),
                                  int(ring.IsAtomInRingOfSize(i, 8))])
            atom_features.extend(one_k_encoding(int(ring.NumAtomRings(i)), [0, 1, 2, 3]))

        z = torch.tensor(atomic_number, dtype=torch.long)
        chiral_tag = torch.tensor(chiral_tag, dtype=torch.float)

        row, col, edge_type, bond_features = [], [], [], []
        for bond in correct_mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_type += 2 * [self.bonds[bond.GetBondType()]]
            bt = tuple(sorted([bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()])), bond.GetBondTypeAsDouble()
            bond_features += 2 * [int(bond.IsInRing()),
                                  int(bond.GetIsConjugated()),
                                  int(bond.GetIsAromatic())]

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        edge_attr = F.one_hot(edge_type, num_classes=len(self.bonds)).to(torch.float)

        perm = (edge_index[0] * N + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_type = edge_type[perm]
        edge_attr = edge_attr[perm]

        row, col = edge_index
        hs = (z == 1).to(torch.float)
        num_hs = scatter(hs[row], col, dim_size=N).tolist()

        x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(self.types))
        x2 = torch.tensor(atom_features).view(N, -1)
        x = torch.cat([x1.to(torch.float), x2], dim=-1)

        data = Data(x=x, z=z, pos=[pos], edge_index=edge_index, edge_attr=edge_attr, neighbors=neighbor_dict,
                    chiral_tag=chiral_tag, name=name, boltzmann_weight=conf['boltzmannweight'],
                    degeneracy=conf['degeneracy'], mol=correct_mol, pos_mask=pos_mask)
        return data



class geomol_qm9_confs_dataset(geom_confs):
    def __init__(self, dataset_path, indexes, mode, transform=None, pre_transform=None, max_confs=10):
        super(geomol_qm9_confs_dataset, self).__init__(dataset_path, indexes, mode, transform, pre_transform, max_confs)
        self.types = qm9_types


class geomol_drugs_confs_dataset(geom_confs):
    def __init__(self, dataset_path, indexes, mode, transform=None, pre_transform=None, max_confs=10):
        super(geomol_drugs_confs_dataset, self).__init__(dataset_path, indexes, mode, transform, pre_transform, max_confs)
        self.types = drugs_types



import os
import os.path as osp
import numpy as np
import glob
import pickle
import random
import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Dataset,Data,DataLoader
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType, BondType as BT, ChiralType
from molpro.geomol.geomol_utils import one_k_encoding , get_dihedral_pairs , dihedral_pattern,chirality,qm9_types,drugs_types
import pytorch_lightning as pl




class geom_confs(Dataset):
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
        return 40 if self.mode == "train" else 5
        #return 10000 if self.mode == 0 else 1000
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



class qm9_confs(geom_confs):
    def __init__(self, dataset_path, indexes, mode, transform=None, pre_transform=None, max_confs=10):
        super(qm9_confs, self).__init__(dataset_path, indexes, mode, transform, pre_transform, max_confs)
        self.types = qm9_types


class drugs_confs(geom_confs):
    def __init__(self, dataset_path, indexes, mode, transform=None, pre_transform=None, max_confs=10):
        super(drugs_confs, self).__init__(dataset_path, indexes, mode, transform, pre_transform, max_confs)
        self.types = drugs_types



class GeomolDataModule(pl.LightningDataModule):
    """Lightning datamodule to handle dataprep for dataloaders 
        
        Parameters :
        ---------------------

        dataset_path : str 
                    path of the dataset
        batch_size : int
                  batch_size for model training
        nworkers: int,
                number of workers for pytorch dataloader """

    def __init__(self,dataset_path: str = './',dataset:str="drugs",batch_size: int = 1,
                                                 nworkers: int = 6):


        super().__init__()
        self.dataset_path = dataset_path
        self.dataset = dataset
        self.batch_size = batch_size
        self.nworkers = nworkers

    def prepare_data(self):
        if not osp.exists(self.split_path):
            raise FileNotFoundError(f"file doesn't exist: {self.split_path}")
        np.random.seed(0)
        indexes = np.array(list(range(len(os.listdir(self.smiles_path)))))
        np.random.shuffle(indexes)
        self.train_indexes = indexes[:int(len(indexes)*80/100)]
        self.val_indexes = indexes[int(len(indexes)*80/100):int(len(indexes)*90/100)]
        self.test_indexes = indexes[int(len(indexes)*90/100):]
        print("Train_data_len:",len(self.train_indexes),"Val_data_len:", len(self.val_indexes),"Test_data_len:", len(self.test_indexes))

    def setup(self, stage=None):
        if self.dataset == "qm9":
            self.train_loader = qm9_confs(self.dataset_path,self.train_indexes,"train")
            self.val_loader = qm9_confs(self.dataset_path,self.val_indexes,"val")
            self.test_loader = qm9_confs(self.dataset_path,self.test_indexes,"test")
        

        if self.dataset == "drugs":
            self.train_loader = drugs_confs(self.dataset_path,self.train_indexes,"train")
            self.val_loader = drugs_confs(self.dataset_path,self.train_indexes,"train")
            self.test_loader = drugs_confs(self.dataset_path,self.train_indexes,"train")
        
    def train_dataloader(self):
        return DataLoader(self.train_loader, batch_size=self.batch_size,
                                 num_workers=self.nworkers)

    def val_dataloader(self):
        return DataLoader(self.val_loader, batch_size=self.batch_size, 
                               num_workers=self.nworkers)

    def test_dataloader(self):
        return DataLoader(self.test_loader, batch_size=self.batch_size,
                                  num_workers=self.nworkers)

from typing import List, Tuple
import torch
from torch.utils.data import Dataset
from random import choice
import h5py
import os.path as osp
import numpy as np
import glob
import pickle
import random
import torch.nn.functional as F
from scipy import ndimage
from skimage.draw import ellipsoid
from torch_scatter import scatter
from torch_geometric.data import Data, Dataset as tg_dataset
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from molpro.geomol.geomol_utils import one_k_encoding, get_dihedral_pairs, dihedral_pattern, \
    chirality, qm9_types, drugs_types
from molpro.utils.preprocess import make_3dgrid, Featurizer, rotate_grid
from molpro.shape_based_gen.data import vocab_c2i_v1


class AffinityPredDataset(Dataset):

    def __init__(self, hdf_path: str, max_dist: int, grid_resolution: int, id_file_path: str, augment: bool) -> None:
        """Pytorch dataset class for preparing 3d grid and labels
            Parameters
            ----------
            hdf_path: str,
                Path to save the HDF5 file
            grid_resolution: float
                Resolution of a grid (in Angstroms)
            max_dist: float
                Maximum distance between atom and box center. Resulting box has size of
                2*`max_dist`+1 Angstroms and atoms that are too far away are not
                included.
            id_file_path: str,
                Path to text file containing pdb ids
            augment: bool,
                Whether to augment the 3d grid or not
        """

        self.transform = augment
        self.max_dist = max_dist
        self.grid_resolution = grid_resolution
        self.hdf_path = hdf_path
        self.data_handle = None
        f = open(id_file_path)
        ids = f.read().splitlines()
        f.close()
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if self.data_handle is None:
            self.data_handle = h5py.File(self.hdf_path, 'r')
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pdb_id = self.ids[idx]
        pkd = self.data_handle[pdb_id]['pkd'][:]
        if self.transform:
            rot = choice(range(24))
            tr = 5 * np.random.rand(1, 3)
        else:
            rot = 0
            tr = (0, 0, 0)
        rec_grid, lig_grid = self.prepare_complex(pdb_id, rotation=rot, translation=tr)

        return np.concatenate((rec_grid, lig_grid)), pkd

    def prepare_complex(self, pdb_id: str, rotation: int = 0,
                        translation: Tuple[int, int, int] = (0, 0, 0)) -> Tuple[np.ndarray, np.ndarray]:
        """Transform coordinates and features to 3d probability density grid
            Parameters
            ----------
            pdb_id: str,
                PDB ID of complex to transform
            translation: tuple,
                distance for translation of 3d grid
            rotation: int,
                rotation integer that corresponds to certain axis and theta
        """

        prot_coords = self.data_handle[pdb_id]['prot_coords'][:]
        ligand_coords = self.data_handle[pdb_id]['ligand_coords'][:]
        ligand_features = self.data_handle[pdb_id]['ligand_features'][:]
        prot_features = self.data_handle[pdb_id]['prot_features'][:]
        prot_coords = rotate_grid(prot_coords, rotation)
        prot_coords += translation
        ligand_coords = rotate_grid(ligand_coords, rotation)
        ligand_coords += translation
        rec_grid = make_3dgrid(prot_coords, prot_features, max_dist=self.max_dist,
                               grid_resolution=self.grid_resolution)
        lig_grid = make_3dgrid(ligand_coords, ligand_features, max_dist=self.max_dist,
                               grid_resolution=self.grid_resolution)
        rec_grid = rec_grid.squeeze(0).transpose((3, 0, 1, 2))
        lig_grid = lig_grid.squeeze(0).transpose((3, 0, 1, 2))
        return rec_grid, lig_grid


class SiteGenDataset(Dataset):

    def __init__(self, hdf_path: str, max_dist: int, grid_resolution: int, id_file_path: str, augment: bool) -> None:
        """Pytorch dataset class for preparing 3d grid and labels
            Parameters
            ----------
            hdf_path: str,
                Path to save the HDF5 file
            grid_resolution: float
                Resolution of a grid (in Angstroms)
            max_dist: float
                Maximum distance between atom and box center. Resulting box has size of
                2*`max_dist`+1 Angstroms and atoms that are too far away are not
                included.
            id_file_path: str,
                Path to text file containing pdb ids
            augment: bool,
                Whether to augment the 3d grid or not
        """

        self.transform = augment
        self.max_dist = max_dist
        self.grid_resolution = grid_resolution
        self.hdf_path = hdf_path
        self.data_handle = None
        f = open(id_file_path)
        ids = f.read().splitlines()
        f.close()
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if self.data_handle is None:
            self.data_handle = h5py.File(self.hdf_path, 'r')
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pdb_id = self.ids[idx]
        if self.transform:
            rot = choice(range(24))
            tr = 5 * np.random.rand(1, 3)
        else:
            rot = 0
            tr = (0, 0, 0)
        rec_grid, lig_grid = self.prepare_complex(pdb_id, rotation=rot, translation=tr)

        return rec_grid, lig_grid

    def prepare_complex(self, pdb_id: str, rotation: int = 0,
                        translation: Tuple[int, int, int] = (0, 0, 0), v_min: int = 0,
                        v_max: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Transform coordinates and features to 3d probability density grid
            Parameters
            ----------
            pdb_id: str,
                PDB ID of complex to transform
            translation: tuple,
                distance for translation of 3d grid
            rotation: int,
                rotation integer that corresponds to certain axis and theta
            v_min: int,
                minimum density in the grid
            v_max: int,
                maximum density in the grid
        """

        prot_coords = self.data_handle[pdb_id]['prot_coords'][:]
        ligand_coords = self.data_handle[pdb_id]['ligand_coords'][:]
        ligand_features = self.data_handle[pdb_id]['ligand_features'][:]
        prot_features = self.data_handle[pdb_id]['prot_features'][:]
        prot_coords = rotate_grid(prot_coords, rotation)
        prot_coords += translation
        ligand_coords = rotate_grid(ligand_coords, rotation)
        ligand_coords += translation
        footprint = ellipsoid(2, 2, 2)
        footprint = footprint.reshape((1, *footprint.shape, 1))
        rec_grid = make_3dgrid(prot_coords, prot_features, max_dist=self.max_dist,
                               grid_resolution=self.grid_resolution)
        lig_grid = make_3dgrid(ligand_coords, ligand_features, max_dist=self.max_dist,
                               grid_resolution=1)
        margin = ndimage.maximum_filter(lig_grid, footprint=footprint)
        lig_grid += margin
        lig_grid = lig_grid.clip(v_min, v_max)

        zoom = rec_grid.shape[1] / lig_grid.shape[1]
        lig_grid = np.stack([ndimage.zoom(lig_grid[0, ..., i],
                                          zoom)
                             for i in range(ligand_features.shape[1])], -1)
        rec_grid = np.squeeze(rec_grid)
        rec_grid = rec_grid.transpose((3, 0, 1, 2))
        lig_grid = lig_grid.transpose((3, 0, 1, 2))
        return rec_grid, lig_grid


class SitePredDataset(Dataset):

    def __init__(self, hdf_path: str, max_dist: int, grid_resolution: int, id_file_path: str, augment: bool) -> None:
        """Pytorch dataset class for preparing 3d grid and labels
            Parameters
            ----------
            hdf_path: str,
                Path to save the HDF5 file
            grid_resolution: float
                Resolution of a grid (in Angstroms)
            max_dist: float
                Maximum distance between atom and box center. Resulting box has size of
                2*`max_dist`+1 Angstroms and atoms that are too far away are not
                included.
            id_file_path: str,
                Path to text file containing pdb ids
            augment: bool,
                Whether to augment the 3d grid or not
        """

        self.transform = augment
        self.max_dist = max_dist
        self.grid_resolution = grid_resolution
        self.hdf_path = hdf_path
        self.data_handle = None
        f = open(id_file_path)
        ids = f.read().splitlines()
        f.close()
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if self.data_handle is None:
            self.data_handle = h5py.File(self.hdf_path, 'r')
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pdb_id = self.ids[idx]
        if self.transform:
            rot = choice(range(24))
            tr = 5 * np.random.rand(1, 3)
        else:
            rot = 0
            tr = (0, 0, 0)
        rec_grid, pocket_dens = self.prepare_complex(pdb_id, rotation=rot, translation=tr)
        return rec_grid, pocket_dens

    def prepare_complex(self, pdb_id: str, rotation: int = 0,
                        translation: Tuple[int, int, int] = (0, 0, 0), v_min: int = 0,
                        v_max: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Transform coordinates and features to 3d probability density grid
            Parameters
            ----------
            pdb_id: str,
                PDB ID of complex to transform
            translation: tuple,
                distance for translation of 3d grid
            rotation: int,
                rotation integer that corresponds to certain axis and theta
            v_min: int,
                minimum density in the grid
            v_max: int,
                maximum density in the grid
        """

        prot_coords = self.data_handle[pdb_id]['coords'][:]
        poc_coords = self.data_handle[pdb_id]['pocket_coords'][:]
        poc_features = self.data_handle[pdb_id]['pocket_features'][:]
        prot_features = self.data_handle[pdb_id]['features'][:]
        prot_coords = rotate_grid(prot_coords, rotation)
        prot_coords += translation
        poc_coords = rotate_grid(poc_coords, rotation)
        poc_coords += translation
        footprint = ellipsoid(2, 2, 2)
        footprint = footprint.reshape((1, *footprint.shape, 1))
        rec_grid = make_3dgrid(prot_coords, prot_features, max_dist=self.max_dist,
                               grid_resolution=self.grid_resolution)
        pocket_dens = make_3dgrid(poc_coords, poc_features, max_dist=self.max_dist,
                                  grid_resolution=1)
        margin = ndimage.maximum_filter(pocket_dens, footprint=footprint)
        pocket_dens += margin
        pocket_dens = pocket_dens.clip(v_min, v_max)

        zoom = rec_grid.shape[1] / pocket_dens.shape[1]
        pocket_dens = np.stack([ndimage.zoom(pocket_dens[0, ..., i],
                                             zoom)
                                for i in range(poc_features.shape[1])], -1)
        rec_grid = np.squeeze(rec_grid)
        rec_grid = rec_grid.transpose((3, 0, 1, 2))
        pocket_dens = pocket_dens.transpose((3, 0, 1, 2))
        return rec_grid, pocket_dens


class ShapeBasedGenDataset(Dataset):
    """ Class to featurize smile while training 

    Input Parameters :
    -------------------------

    smiles_list : List[str]
                 a list which contains smiles

    file_type : str 
                by which file format smiles are extracted .smi or other
    
    """

    def __init__(self, smiles_list: List[str], file_type: str = "smi"):

        self.smiles_list = smiles_list
        self.file_type = file_type

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx: int):
        smi = self.smiles_list[idx]

        smiles_token = smi
        fatures_list = ['hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring']

        featurizer = Featurizer(input_file=smi, file_type='smi', named_props=["partialcharge"],
                                smarts_labels=fatures_list, metal_halogen_encode=False)

        coords = featurizer.coords
        centroid = coords.mean(axis=0)
        coords -= centroid
        afeats = featurizer.features
        features1 = afeats[:, :5]
        features2 = afeats[:, 3:]
        rot = choice(range(24))
        tr1 = 2 * np.random.rand(1, 3)
        tr2 = 0.5 * np.random.rand(1, 3)
        coords1 = rotate_grid(coords, rot)
        coords1 += tr1
        f1n = make_3dgrid(coords1, features1, 23, 2)

        coords2 = rotate_grid(coords, rot)
        coords2 += tr2
        f2n = make_3dgrid(coords2, features2, 23, 2)

        feats_final = np.concatenate([f1n, f2n], axis=4)

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
    def __init__(self, dataset_path: str, indexes_array: np.array, mode: str, transform=None, pre_transform=None,
                 max_confs=10):

        super(geom_confs, self).__init__(dataset_path, transform, pre_transform)
        self.dataset_path = dataset_path
        self.mode = mode
        self.indexes_array = indexes_array
        self.bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        self.dihedral_pairs = {}  # for memoization
        all_files = sorted(glob.glob(osp.join(self.dataset_path, '*.pickle')))
        self.pickle_files = [f for i, f in enumerate(all_files) if i in self.indexes_array]
        self.max_confs = max_confs

    def len(self):
        ###################################################################################################################################################
        # return 40 if self.mode == "train" else 5 # for training in sample_data
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
            bt = tuple(sorted(
                [bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()])), bond.GetBondTypeAsDouble()
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
        super(geomol_drugs_confs_dataset, self).__init__(dataset_path, indexes, mode, transform, pre_transform,
                                                         max_confs)
        self.types = drugs_types

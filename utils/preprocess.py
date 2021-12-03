
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence
from typing import List, Tuple
from math import sin, cos, sqrt


class Featurizer:
    """Calculates atomic features for molecules. Features can encode atom type,
    native rdkit properties and SMARTS strings"""

    def __init__(self, input_file: str, file_type: str, one_hot_metal: bool,
                 one_hot_halogen: bool, use_ring_size: bool, use_smarts: bool, use_gasteiger: bool) -> None:

        """Parameters
        ----------
        input_file: str,
            Path for input file
        file_type: str,
            Define input file type
        one_hot_metal: bool,
            Whether to one hot encode metal features
        one_hot_halogen: bool
            Whether to one hot encode halogen features
        use_ring_size: bool,
            Whether to use ring size as a feature
        use_smarts: bool,
            Whether to use encode SMARTS strings as a feature
        """

        self.input_file = input_file
        self.file_type = file_type
        self.one_hot_metal = one_hot_metal
        self.one_hot_halogen = one_hot_halogen
        self.use_ring_size = use_ring_size
        self.use_smarts = use_smarts
        self.use_gasteiger = use_gasteiger
        self.mol = self.parse_file()

    def parse_file(self) -> Chem.rdchem.Mol:
        """Parse the input file according to file type specified"""
        if self.file_type == 'smi':
            molecule = Chem.MolFromSmiles(self.input_file)
        if self.file_type == 'pdb':
            molecule = Chem.MolFromPDBFile(self.input_file)
        if self.file_type == 'mol2':
            molecule = Chem.MolFromMol2File(self.input_file)
        return molecule

    def get_coords(self) -> np.ndarray:
        """Get 3d cartesian coordinates for input file"""
        for conformer in self.mol.GetConformers():
            coords = conformer.GetPositions()
        return coords

    def generate_conformer(self) -> None:
        self.mol = Chem.AddHs(self.mol)
        AllChem.EmbedMolecule(self.mol)
        AllChem.MMFFOptimizeMolecule(self.mol)

    def encode_num(self, atomic_num: int) -> np.ndarray:
        """Encode metal and halogen features based on atomic number
            Parameters
            ----------
            atomic_num: int,
                Atomic number of the atom
        """

        ATOM_CODES = {}
        metals = ([3, 4, 11, 12, 13] + list(range(19, 32)) + list(range(37, 51)) + list(range(55, 84)) +
                  list(range(87, 104)))
        halogen = [9, 17, 35, 53]
        atom_classes = [5, 6, 7, 8, 15, 16, 32, halogen, metals]
        if not self.one_hot_metal and not self.one_hot_halogen:
            for code, atom in enumerate(atom_classes):
                if type(atom) is list:
                    for a in atom:
                        ATOM_CODES[a] = code
                else:
                    ATOM_CODES[atom] = code
            NUM_ATOM_CLASSES = len(atom_classes)
            encoding = np.zeros(NUM_ATOM_CLASSES)
            try:
                encoding[ATOM_CODES[atomic_num]] = 1.0
            except:
                pass
            return encoding
        if self.one_hot_metal and self.one_hot_halogen:
            atom_classes = [5, 6, 7, 8, 15, 16, 32, *halogen, *metals]
            for code, atom in enumerate(atom_classes):
                ATOM_CODES[atom] = code
            NUM_ATOM_CLASSES = len(atom_classes)
            encoding = np.zeros(NUM_ATOM_CLASSES)
            try:
                encoding[ATOM_CODES[atomic_num]] = 1.0
            except:
                pass
            return encoding

    @staticmethod
    def hybridization(atom_object: Chem.rdchem.Atom) -> np.ndarray:
        """Calculate hybridization state for atom
            Parameters
            ----------
            atom_object: Chem.rdchem.Atom,
                Rdkit atom object
        """

        hybridization_dict = {'SP': 0, 'SP2': 1, 'SP3': 2, 'SP3D': 3, 'SP3D2': 4, 'UNSPECIFIED': 5}
        NUM_HYBRID_CLASSES = len(hybridization_dict)
        encoding = np.zeros(NUM_HYBRID_CLASSES)
        try:
            encoding[hybridization_dict[str(atom_object.GetHybridization())]] = 1.0
        except KeyError:
            encoding[5] = 1.0
        return encoding

    @staticmethod
    def named_prop(atom_object: Chem.rdchem.Atom) -> List[int]:
        """Calculate native rdkit features
            Parameters
            ----------
            atom_object: Chem.rdchem.Atom,
                Rdkit atom object
        """

        properties = [atom_object.GetAtomicNum(), atom_object.GetDegree(),
                      atom_object.GetFormalCharge(), atom_object.GetTotalNumHs(), atom_object.GetImplicitValence(),
                      atom_object.GetNumRadicalElectrons(), int(atom_object.GetIsAromatic())]
        return properties



    @staticmethod
    def cip_rank(atom_object: Chem.rdchem.Atom) -> np.ndarray:
        """Calculate cip rank for atom
            Parameters
            ----------
            atom_object: Chem.rdchem.Atom,
                Rdkit atom object
        """

        cip_dict = {'R': 0, 'S': 1}
        rank = np.zeros(3)
        try:
            rank[cip_dict[atom_object.GetProp('_CIPCode')]] = 1
        except:
            rank[2] = 1
        return rank

    def compute_gasteiger_charge(self) -> None:
        AllChem.ComputeGasteigerCharges(self.mol)

    @staticmethod
    def ring_size(atom_object: Chem.rdchem.Atom, rings: Tuple[int, ...]) -> np.ndarray:
        """Calculate ring size for atom
            Parameters
            ----------
            atom_object: Chem.rdchem.Atom,
                Rdkit atom object
            rings: Tuple[int, ...],
                tuple containing index of all ring atoms,
        """

        one_hot = np.zeros(6)
        aid = atom_object.GetIdx()
        for ring in rings:
            if aid in ring and len(ring) <= 8:
                one_hot[len(ring) - 3] += 1
        return one_hot

    def smart_feats(self) -> np.ndarray:
        """Find hydrophobic, hydrogen bond donor and acceptor atoms using SMARTS strings"""
        PATTERNS = []
        SMARTS = [
            '[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]',
            '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
            '[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]'
        ]
        for smarts in SMARTS:
            PATTERNS.append(Chem.MolFromSmarts(smarts))

        features = np.zeros((self.mol.GetNumAtoms(), len(PATTERNS)))

        for (pattern_id, pattern) in enumerate(PATTERNS):
            atoms_with_prop = np.array(list(*zip(*self.mol.GetSubstructMatches(pattern))),
                                       dtype=int) - 1
            features[atoms_with_prop, pattern_id] = 1.0
        return features

    def atom_features(self) -> np.ndarray:
        """Generate atom features from rdkit mol object"""

        n = self.mol.GetNumAtoms()
        features = []
        if self.use_gasteiger:
            self.compute_gasteiger_charge()
        for j in range(n):
            atom_object = self.mol.GetAtomWithIdx(j)
            atomic_num = atom_object.GetAtomicNum()
            atom_features = np.concatenate((self.encode_num(atomic_num), self.named_prop(atom_object),
                                            self.hybridization(atom_object), self.cip_rank(atom_object)))
            if self.use_ring_size:
                ri = self.mol.GetRingInfo()
                rings = ri.AtomRings()
                atom_features = np.concatenate((atom_features, self.ring_size(atom_object, rings)))
            if self.use_gasteiger:
                atom_features = np.concatenate((atom_features,
                                                np.asarray(atom_object.GetDoubleProp("_GasteigerCharge")).reshape(1)))

            features.append(atom_features)
        if self.use_smarts:
            features = np.hstack([features, self.smart_feats()])
        else:
            features = np.hstack([features])
        return features

    @staticmethod
    def bond_features(self) -> None:
        """generate bond features from mol object"""
        return None


class Tokenizer:
    """Convert a list of smile strings into its respective tokens"""

    def __init__(self, smiles_list: List[str]) -> None:
        """Parameters
            ----------
            smiles_list: List[str],
                list containing smiles strings
        """
        self.stoi = {}
        self.itos = {}
        self.smiles_list = smiles_list

    def __len__(self) -> int:
        """Returns length of vocab list"""
        return len(self.stoi)

    def fit_on_smiles(self) -> None:
        """Generate vocab list from given list of smiles strings"""
        vocab = set()
        for text in self.smiles_list:
            vocab.update(list(text))
        vocab = sorted(vocab)
        vocab.append('<sos>')
        vocab.append('<eos>')
        vocab.append('<pad>')
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}

    def smile_to_sequence(self, text: str) -> List[int]:
        """Convert single smiles string to token
        Parameters
            ----------
            text: str,
                single smile string
        """
        sequence = list()
        sequence.append(self.stoi['<sos>'])
        for s in list(text):
            sequence.append(self.stoi[s])
        sequence.append(self.stoi['<eos>'])
        return sequence

    def smiles_to_sequences(self) -> List[torch.Tensor]:
        """Convert smiles to tokens"""
        sequences = []
        for text in self.smiles_list:
            sequence = self.smile_to_sequence(text)
            sequences.append(torch.tensor(sequence))
        return sequences

    def sequence_to_smile(self, sequence: List[int]) -> str:
        """Convert single token to smiles string
            Parameters
            ----------
            sequence: List[int],
                list containing tokens
        """
        return ''.join(list(map(lambda i: self.itos[i], sequence)))

    def sequences_to_smiles(self, sequences) -> List[str]:
        """Convert tokens to smiles"""
        texts = []
        for sequence in sequences:
            text = self.sequence_to_smile(sequence.numpy().tolist())
            texts.append(text)
        return texts

    @staticmethod
    def pad_sequences(sequences: List[torch.Tensor], pad_length: int, pad_value: int) -> torch.Tensor:
        """Pad sequences to specified length
            Parameters
            ----------
            sequences: List[torch.Tensor],
                list containing tokens
            pad_length: int
                maximum padding length
            pad_value: int,
                padding value used for padding
        """

        padded_sequences, len_sequences = pad_packed_sequence(pack_sequence(sequences, enforce_sorted=False),
                                                              batch_first=True, total_length=pad_length,
                                                              padding_value=pad_value)
        return padded_sequences


def rotation_matrix(axis: np.ndarray, theta: int) -> np.ndarray:
    """Create a rotation matrix for a given axis by theta radians
        Parameters
        ----------
        axis: np.ndarray,
            axis to rotate the 3d grid
        theta: int,
            angle to rotate the grid
    """

    if not isinstance(axis, (np.ndarray, list, tuple)):
        raise TypeError('axis must be an array of floats of shape (3,)')
    try:
        axis = np.asarray(axis, dtype=np.float)
    except ValueError:
        raise ValueError('axis must be an array of floats of shape (3,)')

    if axis.shape != (3,):
        raise ValueError('axis must be an array of floats of shape (3,)')

    if not isinstance(theta, (float, int)):
        raise TypeError('theta must be a float')

    axis = axis / sqrt(np.dot(axis, axis))
    a = cos(theta / 2.0)
    b, c, d = -axis * sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def rotate_grid(coords: np.ndarray, center: Tuple[int, int, int], axis: np.ndarray, theta: int) -> np.ndarray:
    """Rotate a selection of atoms by a given rotation around a center
        Parameters
        ----------
        coords: np.ndarray, shape (N, 3)
            Arrays with coordinates for each atoms.
        center: tuple, optional
            Center to rotate the 3d grid
        axis: np.ndarray,
            axis to rotate the 3d grid
        theta: int,
            angle to rotate the grid
    """
    rotMat = rotation_matrix(axis, theta)
    new_coords = coords - center
    rotated_coords = np.dot(new_coords, np.transpose(rotMat)) + center
    return rotated_coords


def make_3dgrid(coords: np.ndarray, features: np.ndarray, max_dist: int, grid_resolution: int) -> np.ndarray:
    """Convert atom coordinates and features represented as 2D arrays into a
    fixed-sized 3D box.
        Parameters
        ----------
        coords, features: array-likes, shape (N, 3) and (N, F)
            Arrays with coordinates and features for each atoms.
        grid_resolution: float, optional
            Resolution of a grid (in Angstroms).
        max_dist: float, optional
            Maximum distance between atom and box center. Resulting box has size of
            2*`max_dist`+1 Angstroms and atoms that are too far away are not
            included.
    """

    coords = np.asarray(coords, dtype=np.float)
    features = np.asarray(features, dtype=np.float)
    f_shape = features.shape
    num_features = f_shape[1]
    max_dist = float(max_dist)
    grid_resolution = float(grid_resolution)

    box_size = int(np.ceil(2 * max_dist / grid_resolution + 1))

    grid_coords = (coords + max_dist) / grid_resolution
    grid_coords = grid_coords.round().astype(int)

    in_box = ((grid_coords >= 0) & (grid_coords < box_size)).all(axis=1)
    grid = np.zeros((1, box_size, box_size, box_size, num_features), dtype=np.float32)
    for (x, y, z), f in zip(grid_coords[in_box], features[in_box]):
        grid[0, x, y, z] += f

    return grid


def smile_to_graph(smile) -> None:
    """convert smile to graph representation"""
    return None

import openbabel
from openbabel import pybel
import numpy as np
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence
from typing import List, Union
from math import sin, cos, sqrt, pi
from itertools import combinations

pybel.ob.obErrorLog.SetOutputLevel(0)


class Featurizer:
    """Calculates atomic features for molecules. Features can encode atom type,
    native openbabel properties and SMARTS strings"""

    def __init__(self, input_file: str = 'CCCC', file_type: str = 'smi', named_props=None,
                 smarts_labels=None, metal_halogen_encode: bool = True) -> None:

        """Parameters

        ----------
        input_file: str,
            Path for input file
        file_type: str,
            Define input file type
        named_props: List[str],
            Extract atom properties given a list of available properties. The available properties are
            'atomicmass', 'atomicnum', 'exactmass', 'formalcharge', 'heavydegree', 'heterodegree',
            'hyb', 'implicitvalence', 'isotope', 'partialcharge', 'spin' and 'degree'.
        smarts_labels: List[str],
            Extract patterns in molecule given the smarts strings. The available patterns are 'hydrophobic', 'aromatic',
            'acceptor', 'donor' and 'ring'.
        """

        self.mol = None
        self.coords = None
        self.features = None
        self.input_file = input_file
        self.file_type = file_type
        self.named_props = named_props
        self.smarts_labels = smarts_labels
        self.metal_halogen_encode = metal_halogen_encode
        self.parse_file()
        if self.file_type == 'smi':
            self.generate_conformer()
        self.get_coords()
        self.atom_features()

    def parse_file(self) -> None:
        """Parse the input file according to file type specified"""
        if self.file_type == 'smi':
            self.mol = pybel.readstring('smi', self.input_file)
        else:
            self.mol = next(pybel.readfile(self.file_type, self.input_file))

    def get_coords(self) -> None:
        """Get 3d cartesian coordinates for input file"""
        coords = []
        for a in self.mol.atoms:
            if a.atomicnum > 1:
                coords.append(a.coords)
        self.coords = np.array(coords)

    def generate_conformer(self) -> None:
        self.mol.make3D()
        self.mol.localopt()

    @staticmethod
    def encode_num(atomic_num: int) -> np.ndarray:
        """Encode metal and halogen features based on atomic number
            Parameters
            ----------
            atomic_num: int,
                Atomic number of the atom
        """

        ATOM_CODES = {}
        metals = ([3, 4, 11, 12, 13] + list(range(19, 32))
                  + list(range(37, 51)) + list(range(55, 84))
                  + list(range(87, 104)))
        atom_classes = [
            (5, 'B'),
            (6, 'C'),
            (7, 'N'),
            (8, 'O'),
            (15, 'P'),
            (16, 'S'),
            (34, 'Se'),
            ([9, 17, 35, 53], 'halogen'),
            (metals, 'metal')
        ]
        for code, (atom, name) in enumerate(atom_classes):
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

    def named_prop(self, atom: openbabel.pybel.Atom) -> np.ndarray:
        """Calculate native rdkit features
            Parameters
            ----------
            atom: openbabel.pybel.Atom,
                openbabel atom object
        """
        prop = [atom.__getattribute__(prop) for prop in self.named_props]
        return np.asarray(prop)

    def smart_feats(self) -> np.ndarray:
        """Find hydrophobic, hydrogen bond donor and acceptor atoms using SMARTS strings"""
        __PATTERNS = []
        smarts_dict = {'hydrophobic': '[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]', 'aromatic': '[a]',
                       'acceptor': '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
                       'donor': '[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]', 'ring': '[r]'}
        SMARTS = []
        for label in self.smarts_labels:
            SMARTS.append(smarts_dict[label])

        for smarts in SMARTS:
            __PATTERNS.append(pybel.Smarts(smarts))
        features = np.zeros((len(self.mol.atoms), len(__PATTERNS)))

        for (pattern_id, pattern) in enumerate(__PATTERNS):
            atoms_with_prop = np.array(list(*zip(*pattern.findall(self.mol))),
                                       dtype=int) - 1
            features[atoms_with_prop, pattern_id] = 1.0
        return features

    def atom_features(self) -> None:
        """Generate atom features from openbabel mol object"""

        features = []
        heavy_atoms = []
        if self.metal_halogen_encode or self.named_props is not None:
            for i, atom in enumerate(self.mol.atoms):
                atom_features = []
                if atom.atomicnum > 1:
                    heavy_atoms.append(i)
                    if self.metal_halogen_encode:
                        atom_features.append(self.encode_num(atom.atomicnum))
                    if self.named_props is not None:
                        atom_features.append(self.named_prop(atom))
                    features.append(np.concatenate(atom_features))
        if self.smarts_labels is not None:
            self.features = np.hstack([features, self.smart_feats()[heavy_atoms]])
        else:
            self.features = np.hstack([features])


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


def rotation_matrix(axis_rotate:  Union[List[int], np.ndarray], theta_rotate: Union[int, float]) -> np.ndarray:
    """Create a rotation matrix for a given axis by theta radians
        Parameters
        ----------
        axis_rotate: np.ndarray,
            axis to rotate the 3d grid
        theta_rotate: int,
            angle to rotate the grid
    """

    if not isinstance(axis_rotate, (np.ndarray, list, tuple)):
        raise TypeError('axis must be an array of floats of shape (3,)')
    try:
        axis_rotate = np.asarray(axis_rotate, dtype=np.float)
    except ValueError:
        raise ValueError('axis must be an array of floats of shape (3,)')

    if axis_rotate.shape != (3,):
        raise ValueError('axis must be an array of floats of shape (3,)')

    if not isinstance(theta_rotate, (float, int)):
        raise TypeError('theta must be a float')

    axis_rotate = axis_rotate / sqrt(np.dot(axis_rotate, axis_rotate))
    a = cos(theta_rotate / 2.0)
    b, c, d = -axis_rotate * sin(theta_rotate / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


ROTATIONS = [rotation_matrix([1, 1, 1], 0)]
for a1 in range(3):
    for t in range(1, 4):
        axis = np.zeros(3)
        axis[a1] = 1
        theta = t * pi / 2.0
        ROTATIONS.append(rotation_matrix(axis, theta))

for (a1, a2) in combinations(range(3), 2):
    axis = np.zeros(3)
    axis[[a1, a2]] = 1.0
    theta = pi
    ROTATIONS.append(rotation_matrix(axis, theta))
    axis[a2] = -1.0
    ROTATIONS.append(rotation_matrix(axis, theta))

for t in [1, 2]:
    theta = t * 2 * pi / 3
    axis = np.ones(3)
    ROTATIONS.append(rotation_matrix(axis, theta))
    for a1 in range(3):
        axis = np.ones(3)
        axis[a1] = -1
        ROTATIONS.append(rotation_matrix(axis, theta))


def rotate_grid(coords, rotation):
    global ROTATIONS
    return np.dot(coords, ROTATIONS[rotation])


def make_3dgrid(coords: np.ndarray, features: np.ndarray, max_dist: int, grid_resolution: int) -> np.ndarray:
    """Convert atom coordinates and features represented as 2D arrays into a
    fixed-sized 3D box.
        Parameters
        ----------
        coords, features: array-likes, shape (N, 3) and (N, F)
            Arrays with coordinates and features for each atom.
        grid_resolution: float, optional
            Resolution of a grid (in Angstroms).
        max_dist: float, optional
            Maximum distance between atom and box center. Resulting box has size of
            2*`max_dist`+1 Angstroms and atoms that are too far away are not
            included.
    """

    coords = np.asarray(coords, dtype=float)
    features = np.asarray(features, dtype=float)
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

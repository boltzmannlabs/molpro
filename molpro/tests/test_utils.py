from boltpro.utils.preprocess import Featurizer, Tokenizer, rotate_grid, make_3dgrid

featurizer = Featurizer('test.pdb', 'pdb', True, True, True)
smiles_list = ['Cc1ccccc1', 'CO(C)C', 'c1cc1']
tokenizer = Tokenizer(smiles_list)
coordinates = featurizer.get_coords()
axis = [1, 1, 1]
theta = 0
features = featurizer.atom_features()


def test_parse_file():
    mol_file = featurizer.parse_file()
    assert mol_file is not None and type(mol_file).__name__ == 'Mol'


def test_get_coords():
    coords = featurizer.get_coords()
    assert coords is not None and type(coords).__name__ == 'ndarray'


def test_encode_num():
    encoding = featurizer.encode_num(0)
    assert encoding is not None and type(encoding).__name__ == 'ndarray'


def test_hybridization():
    atom_object = featurizer.mol.GetAtomWithIdx(0)
    encoding = featurizer.hybridization(atom_object)
    assert encoding is not None and type(encoding).__name__ == 'ndarray'


def test_named_prop():
    atom_object = featurizer.mol.GetAtomWithIdx(0)
    encoding = featurizer.named_prop(atom_object)
    assert encoding is not None and type(encoding).__name__ == 'list'


def test_cip_rank():
    atom_object = featurizer.mol.GetAtomWithIdx(0)
    encoding = featurizer.cip_rank(atom_object)
    assert encoding is not None and type(encoding).__name__ == 'ndarray'


def test_ring_size():
    atom_object = featurizer.mol.GetAtomWithIdx(0)
    ri = featurizer.mol.GetRingInfo()
    rings = ri.AtomRings()
    encoding = featurizer.ring_size(atom_object, rings)
    assert encoding is not None and type(encoding).__name__ == 'ndarray'


def test_atom_features():
    encoding = featurizer.atom_features()
    assert encoding is not None and type(encoding).__name__ == 'ndarray'


def test_fit_on_smiles():
    tokenizer.fit_on_smiles()
    assert tokenizer.stoi is not None and tokenizer.itos is not None


def test_smile_to_sequence():
    sequence = tokenizer.smile_to_sequence(tokenizer.smiles_list[0])
    assert sequence is not None and type(sequence).__name__ == 'list' and type(sequence[0]).__name__ == 'int'


def test_smiles_to_sequences():
    sequences = tokenizer.smiles_to_sequences()
    assert sequences is not None and type(sequences).__name__ == 'list' and type(sequences[0]).__name__ == 'Tensor'


def test_sequence_to_smile():
    sequence = tokenizer.smile_to_sequence(tokenizer.smiles_list[0])
    smile = tokenizer.sequence_to_smile(sequence)
    assert smile is not None and type(smile).__name__ == 'str'


def test_sequences_to_smiles():
    sequences = tokenizer.smiles_to_sequences()
    smiles = tokenizer.sequences_to_smiles(sequences)
    assert smiles is not None and type(smiles).__name__ == 'list' and type(smiles[0]).__name__ == 'str'


def test_pad_sequences():
    padded = tokenizer.pad_sequences(tokenizer.smiles_to_sequences(), 20, 20)
    assert padded is not None and type(padded).__name__ == 'Tensor' and \
           padded.shape[0] == len(tokenizer.smiles_to_sequences()) and padded.shape[1] == 20


def test_rotate_grid():
    final_coords = rotate_grid(coordinates, (0, 0, 0), [1, 1, 1], 0)
    assert final_coords is not None and type(final_coords).__name__ == 'ndarray'


def test_make_3dgrid():
    grid = make_3dgrid(coordinates, features, 10, 2)
    assert grid is not None and type(grid).__name__ == 'ndarray'

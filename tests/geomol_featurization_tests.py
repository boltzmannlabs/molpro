
from geomol.data import drugs_confs as drugs_featuriter
from geomol.data import qm9_confs as qm9_featurizer


###### tests for drugs data featurization ###########

dataset = 'drugs'

files_path = r"molpro/geomol/sample_data/drugs"
split_path = r"molpro/geomol/sample_data/sample_data/drugs_split.npy"
featurizer = drugs_featuriter(files_path, split_path, "test")
pickle_files = featurizer.pickle_files

def test_open_pickle():
    data_dict = featurizer.open_pickle(pickle_files[0])
    assert data_dict is not None and type(data_dict).__name__ == 'dict'


data_dict = featurizer.open_pickle(pickle_files[0])
def test_featurize_mol():
    data_obj = featurizer.featurize_mol(data_dict)
    assert data_obj is not None and type(data_obj).__name__ == 'Data' and len(data_obj.keys) == 12

data_obj = featurizer.featurize_mol(data_dict)



def test_bolmann_weight():
    boltzmann_weight = data_obj.boltzmann_weight
    assert boltzmann_weight is not None and type(boltzmann_weight).__name__ == 'float'


def test_degeneracy():
    degeneracy = data_obj.degeneracy
    assert degeneracy is not None and type(degeneracy).__name__ == 'int'


def test_chiral_tag():
    chiral_tag = data_obj.chiral_tag
    assert chiral_tag is not None and type(chiral_tag).__name__ == 'Tensor'


def test_edge_attr():
    edge_attr = data_obj.edge_attr
    assert edge_attr is not None and type(edge_attr).__name__ == 'Tensor'


def test_edge_index():
    edge_index = data_obj.edge_index
    assert edge_index is not None and type(edge_index).__name__ == 'Tensor'


def test_mol():
    mol = data_obj.mol
    assert mol is not None and type(mol).__name__ == 'Mol'


def test_name():
    smile_name = data_obj.name
    assert smile_name is not None and type(smile_name).__name__ == 'str'


def test_neighbors():
    neighbors = data_obj.neighbors
    assert neighbors is not None and type(neighbors).__name__ == 'dict'


def test_neighbors():
    neighbors = data_obj.neighbors
    assert neighbors is not None and type(neighbors).__name__ == 'dict'


def test_pos():
    pos = data_obj.pos
    assert pos is not None and type(pos).__name__ == 'list' and type(pos[0]).__name__ == 'Tensor'


def test_pos_mask():
    pos_mask = data_obj.pos_mask
    assert pos_mask is not None and type(pos_mask).__name__ == 'Tensor'


def test_x():
    x = data_obj.x
    assert x is not None and type(x).__name__ == 'Tensor'


def test_z():
    z = data_obj.z
    assert z is not None and type(z).__name__ == 'Tensor'




###### tests for qm9 data featurization ###########
dataset = 'qm9'

files_path = r"molpro/geomol/sample_data/drugs"
split_path = r"molpro/geomol/sample_data/sample_data/drugs_split.npy"
featurizer = qm9_featurizer(files_path, split_path, "test")
pickle_files = featurizer.pickle_files


def test_open_pickle():
    data_dict = featurizer.open_pickle(pickle_files[0])
    assert data_dict is not None and type(data_dict).__name__ == 'dict'


data_dict = featurizer.open_pickle(pickle_files[0])
def test_featurize_mol():
    data_obj = featurizer.featurize_mol(data_dict)
    assert data_obj is not None and type(data_obj).__name__ == 'Data' and len(data_obj.keys) == 12


data_obj = featurizer.featurize_mol(data_dict)

def test_bolmann_weight():
    boltzmann_weight = data_obj.boltzmann_weight
    assert boltzmann_weight is not None and type(boltzmann_weight).__name__ == 'float'


def test_degeneracy():
    degeneracy = data_obj.degeneracy
    assert degeneracy is not None and type(degeneracy).__name__ == 'int'


def test_chiral_tag():
    chiral_tag = data_obj.chiral_tag
    assert chiral_tag is not None and type(chiral_tag).__name__ == 'Tensor'


def test_edge_attr():
    edge_attr = data_obj.edge_attr
    assert edge_attr is not None and type(edge_attr).__name__ == 'Tensor'


def test_edge_index():
    edge_index = data_obj.edge_index
    assert edge_index is not None and type(edge_index).__name__ == 'Tensor'


def test_mol():
    mol = data_obj.mol
    assert mol is not None and type(mol).__name__ == 'Mol'


def test_name():
    smile_name = data_obj.name
    assert smile_name is not None and type(smile_name).__name__ == 'str'


def test_neighbors():
    neighbors = data_obj.neighbors
    assert neighbors is not None and type(neighbors).__name__ == 'dict'


def test_neighbors():
    neighbors = data_obj.neighbors
    assert neighbors is not None and type(neighbors).__name__ == 'dict'


def test_pos():
    pos = data_obj.pos
    assert pos is not None and type(pos).__name__ == 'list' and type(pos[0]).__name__ == 'Tensor'


def test_pos_mask():
    pos_mask = data_obj.pos_mask
    assert pos_mask is not None and type(pos_mask).__name__ == 'Tensor'


def test_x():
    x = data_obj.x
    assert x is not None and type(x).__name__ == 'Tensor'


def test_z():
    z = data_obj.z
    assert z is not None and type(z).__name__ == 'Tensor'


print("Succesfully Done...")
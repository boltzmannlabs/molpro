import os.path as osp
import glob
import pickle
import random
import glob
import pickle
import torch
import os.path as osp
import numpy as np
import networkx as nx
import torch_geometric as tg
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_scatter import scatter
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
angle_mask_ref = torch.LongTensor([[0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0, 0],
                                   [1, 1, 1, 0, 0, 0],
                                   [1, 1, 1, 1, 1, 1]]).to(device)

angle_combos = torch.LongTensor([[0, 1],
                                 [0, 2],
                                 [1, 2],
                                 [0, 3],
                                 [1, 3],
                                 [2, 3]]).to(device)



def align_coords_Kabsch(p_cycle_coords, q_cycle_coords, p_mask, q_mask=None):
    """
    align p_cycle_coords with q_cycle_coords

    mask indicates which atoms to apply RMSD minimization over; these atoms are used to calculate the
    final rotation and translation matrices, which are applied to ALL atoms
    """
    if not q_mask:
        q_mask = p_mask

    q_cycle_coords_centered = q_cycle_coords[:, q_mask] - q_cycle_coords[:, q_mask].mean(dim=1, keepdim=True)
    p_cycle_coords_centered = p_cycle_coords[:, :, p_mask] - p_cycle_coords[:, :, p_mask].mean(dim=2, keepdim=True)

    H = torch.matmul(p_cycle_coords_centered.permute(0, 1, 3, 2), q_cycle_coords_centered.unsqueeze(0))
    u, s, v = torch.svd(H)
    d = torch.sign(torch.det(torch.matmul(v, u.permute(0, 1, 3, 2))))
    R_1 = torch.diag_embed(torch.ones([p_cycle_coords.size(0), q_cycle_coords.size(0), 3]))
    R_1[:, :, 2, 2] = d
    R = torch.matmul(v, torch.matmul(R_1, u.permute(0, 1, 3, 2)))
    b = q_cycle_coords[:, q_mask].mean(dim=1) - torch.matmul(R, p_cycle_coords[:, :, p_mask].mean(dim=2).unsqueeze(
        -1)).squeeze(-1)

    p_cycle_coords_aligned = torch.matmul(R, p_cycle_coords.permute(0, 1, 3, 2)).permute(0, 1, 3, 2) + b.unsqueeze(2)

    return p_cycle_coords_aligned


def get_neighbor_ids(data):
    """
    Takes the edge indices and returns dictionary mapping atom index to neighbor indices
    Note: this only includes atoms with degree > 1
    """
    neighbors = data.neighbors.pop(0)
    n_atoms_per_mol = data.batch.bincount()
    n_atoms_prev_mol = 0

    for i, n_dict in enumerate(data.neighbors):
        new_dict = {}
        n_atoms_prev_mol += n_atoms_per_mol[i].item()
        for k, v in n_dict.items():
            new_dict[k + n_atoms_prev_mol] = v + n_atoms_prev_mol
        neighbors.update(new_dict)
    return neighbors


def get_neighbor_bonds(edge_index, bond_type):
    """
    Takes the edge indices and bond type and returns dictionary mapping atom index to neighbor bond types
    Note: this only includes atoms with degree > 1
    """
    start, end = edge_index
    idxs, vals = torch.unique(start, return_counts=True)
    vs = torch.split_with_sizes(bond_type, tuple(vals))
    return {k.item(): v for k, v in zip(idxs, vs) if len(v) > 1}


def get_leaf_hydrogens(neighbors, x):
    """
    Takes the edge indices and atom features and returns dictionary mapping atom index to neighbors, indicating true
    for hydrogens that are leaf nodes
    Note: this only works because degree = 1 and hydrogen atomic number = 1 (checks when 1 == 1)
    Note: we use the 5th feature index bc this corresponds to the atomic number
    """
    leaf_hydrogens = {}
    h_mask = x[:, 0] == 1
    for k, v in neighbors.items():
        leaf_hydrogens[k] = h_mask[neighbors[k]]
    return leaf_hydrogens


def get_dihedral_pairs(edge_index, data):
    """
    Given edge indices, return pairs of indices that we must calculate dihedrals for
    """
    start, end = edge_index
    degrees = degree(end)
    dihedral_pairs_true = torch.nonzero(torch.logical_and(degrees[start] > 1, degrees[end] > 1))
    dihedral_pairs = edge_index[:, dihedral_pairs_true].squeeze(-1)

    # # first method which removes one (pseudo) random edge from a cycle
    dihedral_idxs = torch.nonzero(dihedral_pairs.sort(dim=0).indices[0, :] == 0).squeeze().detach().cpu().numpy()

    # prioritize rings for assigning dihedrals
    dihedral_pairs = dihedral_pairs.t()[dihedral_idxs]
    G = nx.to_undirected(tg.utils.to_networkx(data))
    cycles = nx.cycle_basis(G)
    keep, sorted_keep = [], []

    if len(dihedral_pairs.shape) == 1:
        dihedral_pairs = dihedral_pairs.unsqueeze(0)

    for pair in dihedral_pairs:
        x, y = pair

        if sorted(pair) in sorted_keep:
            continue

        y_cycle_check = [y in cycle for cycle in cycles]
        x_cycle_check = [x in cycle for cycle in cycles]

        if any(x_cycle_check) and any(y_cycle_check):  
            cycle_indices = get_current_cycle_indices(cycles, x_cycle_check, x)
            keep.extend(cycle_indices)

            sorted_keep.extend([sorted(c) for c in cycle_indices])
            continue

        if any(y_cycle_check):
            cycle_indices = get_current_cycle_indices(cycles, y_cycle_check, y)
            keep.append(pair)
            keep.extend(cycle_indices)

            sorted_keep.append(sorted(pair))
            sorted_keep.extend([sorted(c) for c in cycle_indices])
            continue

        keep.append(pair)

    keep = [t.to(device) for t in keep]
    return torch.stack(keep).t()


def batch_distance_metrics_from_coords(coords, mask):
    """
    Given coordinates of neighboring atoms, compute bond
    distances and 2-hop distances in local neighborhood
    """
    d_mat_mask = mask.unsqueeze(1) * mask.unsqueeze(2)

    if coords.dim() == 4:
        two_dop_d_mat = torch.square(coords.unsqueeze(1) - coords.unsqueeze(2) + 1e-10).sum(dim=-1).sqrt() * d_mat_mask.unsqueeze(-1)
        one_hop_ds = torch.linalg.norm(torch.zeros_like(coords[0]).unsqueeze(0) - coords, dim=-1)
    elif coords.dim() == 5:
        two_dop_d_mat = torch.square(coords.unsqueeze(2) - coords.unsqueeze(3) + 1e-10).sum(dim=-1).sqrt() * d_mat_mask.unsqueeze(-1).unsqueeze(1)
        one_hop_ds = torch.linalg.norm(torch.zeros_like(coords[0]).unsqueeze(0) - coords, dim=-1)

    return one_hop_ds, two_dop_d_mat


def batch_angle_between_vectors(a, b):
    """
    Compute angle between two batches of input vectors
    """
    inner_product = (a * b).sum(dim=-1)

    # norms
    a_norm = torch.linalg.norm(a, dim=-1)
    b_norm = torch.linalg.norm(b, dim=-1)

    # protect denominator during division
    den = a_norm * b_norm + 1e-10
    cos = inner_product / den

    return cos


def batch_angles_from_coords(coords, mask):
    """
    Given coordinates, compute all local neighborhood angles
    """
    if coords.dim() == 4:
        all_possible_combos = coords[:, angle_combos]
        v_a, v_b = all_possible_combos.split(1, dim=2)  
        angle_mask = angle_mask_ref[mask.sum(dim=1).long()]
        angles = batch_angle_between_vectors(v_a.squeeze(2), v_b.squeeze(2)) * angle_mask.unsqueeze(-1)
    elif coords.dim() == 5:
        all_possible_combos = coords[:, :, angle_combos]
        v_a, v_b = all_possible_combos.split(1, dim=3)  
        angle_mask = angle_mask_ref[mask.sum(dim=1).long()]
        angles = batch_angle_between_vectors(v_a.squeeze(3), v_b.squeeze(3)) * angle_mask.unsqueeze(-1).unsqueeze(-1)

    return angles


def batch_local_stats_from_coords(coords, mask):
    """
    Given neighborhood neighbor coordinates, compute bond distances,
    2-hop distances, and angles in local neighborhood (this assumes
    the central atom has coordinates at the origin)
    """
    one_hop_ds, two_dop_d_mat = batch_distance_metrics_from_coords(coords, mask)
    angles = batch_angles_from_coords(coords, mask)
    return one_hop_ds, two_dop_d_mat, angles


def batch_dihedrals(p0, p1, p2, p3, angle=False):

    s1 = p1 - p0
    s2 = p2 - p1
    s3 = p3 - p2

    sin_d_ = torch.linalg.norm(s2, dim=-1) * torch.sum(s1 * torch.cross(s2, s3, dim=-1), dim=-1)
    cos_d_ = torch.sum(torch.cross(s1, s2, dim=-1) * torch.cross(s2, s3, dim=-1), dim=-1)

    if angle:
        return torch.atan2(sin_d_, cos_d_ + 1e-10)

    else:
        den = torch.linalg.norm(torch.cross(s1, s2, dim=-1), dim=-1) * torch.linalg.norm(torch.cross(s2, s3, dim=-1), dim=-1) + 1e-10
        return sin_d_/den, cos_d_/den


def batch_vector_angles(xn, x, y, yn):
    uT = xn.view(-1, 3)
    uX = x.view(-1, 3)
    uY = y.view(-1, 3)
    uZ = yn.view(-1, 3)

    b1 = uT - uX
    b2 = uZ - uY

    num = torch.bmm(b1.view(-1, 1, 3), b2.view(-1, 3, 1)).squeeze(-1).squeeze(-1)
    den = torch.linalg.norm(b1, dim=-1) * torch.linalg.norm(b2, dim=-1) + 1e-10

    return (num / den).view(-1, 9)


def von_Mises_loss(a, b, a_sin=None, b_sin=None):
    """
    :param a: cos of first angle
    :param b: cos of second angle
    :return: difference of cosines
    """
    if torch.is_tensor(a_sin):
        out = a * b + a_sin * b_sin
    else:
        out = a * b + torch.sqrt(1-a**2 + 1e-5) * torch.sqrt(1-b**2 + 1e-5)
    return out


def rotation_matrix(neighbor_coords, neighbor_mask, neighbor_map, mu=None):
    """
    Given predicted neighbor coordinates from model, return rotation matrix

    :param neighbor_coords: neighbor coordinates for each edge as defined by dihedral_pairs
        (n_dihedral_pairs, 4, n_generated_confs, 3)
    :param neighbor_mask: mask describing which atoms are present (n_dihedral_pairs, 4)
    :param neighbor_map: mask describing which neighbor corresponds to the other central dihedral atom
        (n_dihedral_pairs, 4) each entry in neighbor_map should have one TRUE entry with the rest as FALSE
    :return: rotation matrix (n_dihedral_pairs, n_model_confs, 3, 3)
    """

    if not torch.is_tensor(mu):
        mu_num = neighbor_coords[~neighbor_map.bool()].view(neighbor_coords.size(0), 3, neighbor_coords.size(2), -1).sum(dim=1)
        mu_den = (neighbor_mask.sum(dim=-1, keepdim=True).unsqueeze(-1) - 1 + 1e-10)
        mu = mu_num / mu_den  
        mu = mu.squeeze(1)  
    p_Y = neighbor_coords[neighbor_map.bool(), :]
    h1 = p_Y / (torch.linalg.norm(p_Y, dim=-1, keepdim=True) + 1e-10)  
    h3_1 = torch.cross(p_Y, mu, dim=-1)
    h3 = h3_1 / (torch.linalg.norm(h3_1, dim=-1, keepdim=True) + 1e-10)  
    h2 = -torch.cross(h1, h3, dim=-1)  
    H = torch.cat([h1.unsqueeze(-2),
                   h2.unsqueeze(-2),
                   h3.unsqueeze(-2)], dim=-2)

    return H


def rotation_matrix_v2(neighbor_coords, neighbor_mask, neighbor_map):
    """
    Given predicted neighbor coordinates from model, return rotation matrix

    :param neighbor_coords: neighbor coordinates for each edge as defined by dihedral_pairs
        (n_dihedral_pairs, 4, n_generated_confs, 3)
    :param neighbor_mask: mask describing which atoms are present (n_dihedral_pairs, 4)
    :param neighbor_map: mask describing which neighbor corresponds to the other central dihedral atom
        (n_dihedral_pairs, 4) each entry in neighbor_map should have one TRUE entry with the rest as FALSE
    :return: rotation matrix (n_dihedral_pairs, n_model_confs, 3, 3)
    """

    p_Y = neighbor_coords[neighbor_map.bool(), :]

    eta_1 = torch.rand_like(p_Y)
    eta_2 = eta_1 - torch.sum(eta_1 * p_Y, dim=-1, keepdim=True) / (torch.linalg.norm(p_Y, dim=-1, keepdim=True)**2 + 1e-10) * p_Y
    eta = eta_2 / torch.linalg.norm(eta_2, dim=-1, keepdim=True)

    h1 = p_Y / (torch.linalg.norm(p_Y, dim=-1, keepdim=True) + 1e-10)  

    h3_1 = torch.cross(p_Y, eta, dim=-1)
    h3 = h3_1 / (torch.linalg.norm(h3_1, dim=-1, keepdim=True) + 1e-10)  

    h2 = -torch.cross(h1, h3, dim=-1)  

    H = torch.cat([h1.unsqueeze(-2),
                   h2.unsqueeze(-2),
                   h3.unsqueeze(-2)], dim=-2)

    return H


def signed_volume(local_coords):
    """
    Compute signed volume given ordered neighbor local coordinates

    :param local_coords: (n_tetrahedral_chiral_centers, 4, n_generated_confs, 3)
    :return: signed volume of each tetrahedral center (n_tetrahedral_chiral_centers, n_generated_confs)
    """
    v1 = local_coords[:, 0] - local_coords[:, 3]
    v2 = local_coords[:, 1] - local_coords[:, 3]
    v3 = local_coords[:, 2] - local_coords[:, 3]
    cp = v2.cross(v3, dim=-1)
    vol = torch.sum(v1 * cp, dim=-1)
    return torch.sign(vol)


def rotation_matrix_inf(neighbor_coords, neighbor_mask, neighbor_map):
    """
    Given predicted neighbor coordinates from model, return rotation matrix

    :param neighbor_coords: neighbor coordinates for each edge as defined by dihedral_pairs (4, n_model_confs, 3)
    :param neighbor_mask: mask describing which atoms are present (4)
    :param neighbor_map: mask describing which neighbor corresponds to the other central dihedral atom (4)
        each entry in neighbor_map should have one TRUE entry with the rest as FALSE
    :return: rotation matrix (3, 3)
    """

    mu = neighbor_coords.sum(dim=0, keepdim=True) / (neighbor_mask.sum(dim=-1, keepdim=True).unsqueeze(-1) + 1e-10)
    mu = mu.squeeze(0)
    p_Y = neighbor_coords[neighbor_map.bool(), :].squeeze(0)

    h1 = p_Y / (torch.linalg.norm(p_Y, dim=-1, keepdim=True) + 1e-10)

    h3_1 = torch.cross(p_Y, mu, dim=-1)
    h3 = h3_1 / (torch.linalg.norm(h3_1, dim=-1, keepdim=True) + 1e-10)

    h2 = -torch.cross(h1, h3, dim=-1)

    H = torch.cat([h1.unsqueeze(-2),
                   h2.unsqueeze(-2),
                   h3.unsqueeze(-2)], dim=-2)

    return H


def build_alpha_rotation_inf(alpha, n_model_confs):

    H_alpha = torch.FloatTensor([[[1, 0, 0], [0, 0, 0], [0, 0, 0]]]).repeat(n_model_confs, 1, 1)
    H_alpha[:, 1, 1] = torch.cos(alpha)
    H_alpha[:, 1, 2] = -torch.sin(alpha)
    H_alpha[:, 2, 1] = torch.sin(alpha)
    H_alpha[:, 2, 2] = torch.cos(alpha)

    return H_alpha


def random_rotation_matrix(dim):
    yaw = torch.rand(dim)
    pitch = torch.rand(dim)
    roll = torch.rand(dim)

    R = torch.stack([torch.stack([torch.cos(yaw) * torch.cos(pitch),
                                  torch.cos(yaw) * torch.sin(pitch) * torch.sin(roll) - torch.sin(yaw) * torch.cos(
                                      roll),
                                  torch.cos(yaw) * torch.sin(pitch) * torch.cos(roll) + torch.sin(yaw) * torch.sin(
                                      roll)], dim=-1),
                     torch.stack([torch.sin(yaw) * torch.cos(pitch),
                                  torch.sin(yaw) * torch.sin(pitch) * torch.sin(roll) + torch.cos(yaw) * torch.cos(
                                      roll),
                                  torch.sin(yaw) * torch.sin(pitch) * torch.cos(roll) - torch.cos(yaw) * torch.sin(
                                      roll)], dim=-1),
                     torch.stack([-torch.sin(pitch),
                                  torch.cos(pitch) * torch.sin(roll),
                                  torch.cos(pitch) * torch.cos(roll)], dim=-1)], dim=-2)

    return R


def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask



bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
qm9_types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
drugs_types = {'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7, 'Mg': 8, 'Al': 9, 'Si': 10,
               'P': 11, 'S': 12, 'Cl': 13, 'K': 14, 'Ca': 15, 'V': 16, 'Cr': 17, 'Mn': 18, 'Cu': 19, 'Zn': 20,
               'Ga': 21, 'Ge': 22, 'As': 23, 'Se': 24, 'Br': 25, 'Ag': 26, 'In': 27, 'Sb': 28, 'I': 29, 'Gd': 30,
               'Pt': 31, 'Au': 32, 'Hg': 33, 'Bi': 34}

dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')
chirality = {ChiralType.CHI_TETRAHEDRAL_CW: -1.,
             ChiralType.CHI_TETRAHEDRAL_CCW: 1.,
             ChiralType.CHI_UNSPECIFIED: 0,
             ChiralType.CHI_OTHER: 0}


def one_k_encoding(value, choices):
    """
    Creates a one-hot encoding with an extra category for uncommon values.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def featurize_mol_from_smiles(smile, dataset='drugs'):

    """ Prepares the data object from smile """

    if dataset == 'qm9':
        types = qm9_types
    elif dataset == 'drugs':
        types = drugs_types

    # filter fragments
    if '.' in smile:
        return None

    # filter mols rdkit can't intrinsically handle
    mol = Chem.MolFromSmiles(smile)
    if mol:
        mol = Chem.AddHs(mol)
    else:
        return None
    N = mol.GetNumAtoms()

    # filter out mols model can't make predictions for
    if not mol.HasSubstructMatch(dihedral_pattern):
        return None
    if N < 4:
        return None

    type_idx = []
    atomic_number = []
    atom_features = []
    chiral_tag = []
    neighbor_dict = {}
    ring = mol.GetRingInfo()
    for i, atom in enumerate(mol.GetAtoms()):
        type_idx.append(types[atom.GetSymbol()])
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
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]
        bt = tuple(
            sorted([bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()])), bond.GetBondTypeAsDouble()
        bond_features += 2 * [int(bond.IsInRing()),
                              int(bond.GetIsConjugated()),
                              int(bond.GetIsAromatic())]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]
    edge_attr = edge_attr[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=N).tolist()

    x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
    x2 = torch.tensor(atom_features).view(N, -1)
    x = torch.cat([x1.to(torch.float), x2], dim=-1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, neighbors=neighbor_dict, chiral_tag=chiral_tag,
                name=smile)
    data.edge_index_dihedral_pairs = get_dihedral_pairs(data.edge_index, data=data)

    return data

def construct_conformers(data, model):

    G = nx.to_undirected(tg.utils.to_networkx(data))
    cycles = nx.cycle_basis(G)

    new_pos = torch.zeros([data.batch.size(0), model.n_model_confs, 3])
    dihedral_pairs = model.dihedral_pairs.t().detach().numpy()

    Sx = []
    Sy = []
    in_cycle = 0

    for i, pair in enumerate(dihedral_pairs):

        x_index, y_index = pair
        cycle_added = False
        if in_cycle:
            in_cycle -= 1

        if in_cycle:
            continue

        y_cycle_check = [y_index in cycle for cycle in cycles]
        x_cycle_check = [x_index in cycle for cycle in cycles]

        if any(x_cycle_check) and any(y_cycle_check):  # both in new cycle

            cycle_indices = get_current_cycle_indices(cycles, x_cycle_check, x_index)
            cycle_avg_coords, cycle_avg_indices = smooth_cycle_coords(model, cycle_indices, new_pos, dihedral_pairs, i) # i instead of i+1

            # new graph
            if x_index not in Sx:
                new_pos[cycle_avg_indices] = cycle_avg_coords
                Sx = []

            else:
                assert sorted(Sx) == Sx
                p_mask = [True if a in Sx else False for a in sorted(cycle_avg_indices)]
                q_mask = [True if a in sorted(cycle_avg_indices) else False for a in Sx]
                p_reorder = sorted(range(len(cycle_avg_indices)), key=lambda k: cycle_avg_indices[k])
                aligned_cycle_coords = align_coords_Kabsch(cycle_avg_coords[p_reorder].permute(1, 0, 2).unsqueeze(0), new_pos[Sx].permute(1, 0, 2), p_mask, q_mask)
                aligned_cycle_coords = aligned_cycle_coords.squeeze(0).permute(1, 0, 2)
                cycle_avg_indices_reordered = [cycle_avg_indices[l] for l in p_reorder]

                # apply to all new coordinates?
                new_pos[cycle_avg_indices_reordered] = aligned_cycle_coords

            Sx.extend(cycle_avg_indices)
            Sx = sorted(list(set(Sx)))
            in_cycle = len(cycle_indices)  # one less than below bc 2 nodes are added to ring
            continue

        if any(y_cycle_check):
            cycle_indices = get_current_cycle_indices(cycles, y_cycle_check, y_index)
            cycle_added = True
            in_cycle = len(cycle_indices)+1

        # new graph
        p_coords = torch.zeros([4, model.n_model_confs, 3])
        p_idx = model.neighbors[x_index]

        if x_index not in Sx:
            Sx = []
            # set new immediate neighbor coords for X
            p_coords = model.p_coords[i]
            new_pos[p_idx] = p_coords[0:int(model.dihedral_x_mask[i].sum())]

        else:
            p_coords[0:p_idx.size(0)] = new_pos[p_idx] - new_pos[x_index]

        # update indices
        Sx.extend([x_index])
        Sx.extend(model.neighbors[x_index].detach().numpy())
        Sx = list(set(Sx))

        Sy.extend([y_index])
        Sy.extend(model.neighbors[y_index].detach().numpy())

        # set px
        p_X = new_pos[x_index]

        # translate current Sx
        new_pos_Sx = new_pos[Sx] - p_X

        # set Y
        if cycle_added:
            cycle_avg_coords, cycle_avg_indices = smooth_cycle_coords(model, cycle_indices, new_pos, dihedral_pairs, i+1)
            cycle_avg_coords = cycle_avg_coords - cycle_avg_coords[cycle_avg_indices == y_index] # move y to origin
            q_idx = model.neighbors[y_index]
            q_coords_mask = [True if a in q_idx else False for a in cycle_avg_indices]
            q_coords = torch.zeros([4, model.n_model_confs, 3])
            q_reorder = np.argsort([np.where(a == q_idx)[0][0] for a in torch.tensor(cycle_avg_indices)[q_coords_mask]])
            q_coords[0:sum(q_coords_mask)] = cycle_avg_coords[q_coords_mask][q_reorder]
            new_pos_Sy = cycle_avg_coords.clone()
            Sy = cycle_avg_indices

        else:
            q_coords = model.q_coords[i]
            q_idx = model.neighbors[y_index]
            new_pos[q_idx] = q_coords[0:int(model.dihedral_y_mask[i].sum())]
            new_pos[y_index] = torch.zeros_like(p_X)  # q_Y always at the origin
            new_pos_Sy = new_pos[Sy]

        # calculate rotation matrices
        H_XY = rotation_matrix_inf_v2(p_coords, model.x_map_to_neighbor_y[i])
        H_YX = rotation_matrix_inf_v2(q_coords, model.y_map_to_neighbor_x[i])

        # rotate
        new_pos_Sx_2 = torch.matmul(H_XY.unsqueeze(0), new_pos_Sx.unsqueeze(-1)).squeeze(-1)
        new_pos_Sy_2 = torch.matmul(H_YX.unsqueeze(0), new_pos_Sy.unsqueeze(-1)).squeeze(-1)

        # translate q
        new_p_Y = new_pos_Sx_2[Sx == y_index]
        transform_matrix = torch.diag(torch.tensor([-1., -1., 1.])).unsqueeze(0).unsqueeze(0)
        new_pos_Sy_3 = torch.matmul(transform_matrix, new_pos_Sy_2.unsqueeze(-1)).squeeze(-1) + new_p_Y

        # rotate by gamma
        H_gamma = calculate_gamma(model.n_model_confs, model.dihedral_mask[i], model.c_ij[i], model.v_star[i], Sx, Sy,
                                  p_idx, q_idx, x_index, y_index, new_pos_Sx_2, new_pos_Sy_3, new_p_Y)
        new_pos_Sx_3 = torch.matmul(H_gamma.unsqueeze(0), new_pos_Sx_2.unsqueeze(-1)).squeeze(-1)

        # update all coordinates
        new_pos[Sy] = new_pos_Sy_3
        new_pos[Sx] = new_pos_Sx_3

        # update indices
        Sx.extend(Sy)
        Sx = sorted(list(set(Sx)))
        Sy = []

    return new_pos


def smooth_cycle_coords(model, cycle_indices, new_pos, dihedral_pairs, cycle_start_idx):

    # find index of cycle starting position
    cycle_len = len(cycle_indices)

    # get dihedral pairs corresponding to current cycle
    cycle_pairs = dihedral_pairs[cycle_start_idx:cycle_start_idx+cycle_len]

    # create indices for cycle
    cycle_i = np.arange(cycle_start_idx, cycle_start_idx+cycle_len)

    # create ordered dihedral pairs and indices which each start at a different point in the cycle
    cycle_dihedral_pair_orders = np.stack([np.roll(cycle_pairs, -i, axis=0) for i in range(len(cycle_pairs))])[:-1]
    cycle_i_orders = np.stack([np.roll(cycle_i, -i, axis=0) for i in range(len(cycle_i))])[:-1]

    # intialize lists to track which indices have been added and cycle position vector
    Sx_cycle, Sy_cycle = [[] for i in range(cycle_len)], [[] for i in range(cycle_len)]
    cycle_pos = torch.zeros_like(new_pos).unsqueeze(0).repeat(cycle_len, 1, 1, 1)

    for ii, (pairs, ids) in enumerate(zip(cycle_dihedral_pair_orders, cycle_i_orders)):

        # ii is the enumeration index
        # pairs are the dihedral pairs for the cycle with shape (cycle_len-1, 2)
        # ids are the indices of the dihedral pairs corresponding to the model values with shape (cycle_len-1)

        x_indices, y_indices = pairs.transpose()

        p_coords = torch.zeros([cycle_len, 4, model.n_model_confs, 3])
        p_idx = [model.neighbors[x] for x in x_indices]
        if ii == 0:

            # set new immediate neighbor coords for X
            p_coords = model.p_coords[ids]
            for i, p_i in enumerate(p_idx):
                cycle_pos[i, p_i] = p_coords[i, 0:int(model.dihedral_x_mask[ids[i]].sum())]

        else:
            for i in range(cycle_len):
                p_coords[i, 0:p_idx[i].size(0)] = cycle_pos[i, p_idx[i]] - cycle_pos[i, x_indices[i]]

        # update indices
        q_idx = [model.neighbors[y] for y in y_indices]
        for i, (x_idx, p_idxs) in enumerate(zip(x_indices, p_idx)):

            Sx_cycle[i].extend([x_idx])
            Sx_cycle[i].extend(p_idxs.detach().cpu().numpy())
            Sx_cycle[i] = list(set(Sx_cycle[i]))

            Sy_cycle[i].extend([y_indices[i]])
            Sy_cycle[i].extend(q_idx[i].detach().cpu().numpy())

        # set px
        p_X = cycle_pos[torch.arange(cycle_len), x_indices]

        # translate current Sx
        new_pos_Sx = [cycle_pos[i, Sx_cycle[i]] - p_X[0].unsqueeze(0) for i in range(cycle_len)]

        # set Y
        q_coords = model.q_coords[ids]
        new_pos_Sy = []
        for i, q_i in enumerate(q_idx):
            cycle_pos[i, q_i] = q_coords[i, 0:int(model.dihedral_y_mask[ids[i]].sum())]
            cycle_pos[i, y_indices[i]] = torch.zeros_like(p_X[i])  # q_Y always at the origin
            new_pos_Sy.append(cycle_pos[i, Sy_cycle[i]])

        # calculate rotation matrices
        H_XY = list(map(rotation_matrix_inf_v2, p_coords, model.x_map_to_neighbor_y[ids]))
        H_YX = list(map(rotation_matrix_inf_v2, q_coords, model.y_map_to_neighbor_x[ids]))

        # rotate
        new_pos_Sx_2 = [torch.matmul(H_XY[i].unsqueeze(0), new_pos_Sx[i].unsqueeze(-1)).squeeze(-1) for i in range(cycle_len)]
        new_pos_Sy_2 = [torch.matmul(H_YX[i].unsqueeze(0), new_pos_Sy[i].unsqueeze(-1)).squeeze(-1) for i in range(cycle_len)]

        for i in range(cycle_len):

            # translate q
            new_p_Y = new_pos_Sx_2[i][Sx_cycle[i] == y_indices[i]].squeeze(-1)
            transform_matrix = torch.diag(torch.tensor([-1., -1., 1.])).unsqueeze(0).unsqueeze(0)
            new_pos_Sy_3 = torch.matmul(transform_matrix, new_pos_Sy_2[i].unsqueeze(-1)).squeeze(-1) + new_p_Y

            # rotate by gamma
            H_gamma = calculate_gamma(model.n_model_confs, model.dihedral_mask[ids[i]], model.c_ij[ids[i]],
                                      model.v_star[ids[i]], Sx_cycle[i], Sy_cycle[i], p_idx[i], q_idx[i], pairs[i][0],
                                      pairs[i][1], new_pos_Sx_2[i], new_pos_Sy_3, new_p_Y)
            new_pos_Sx_3 = torch.matmul(H_gamma.unsqueeze(0), new_pos_Sx_2[i].unsqueeze(-1)).squeeze(-1)

            # update all coordinates
            cycle_pos[i, Sy_cycle[i]] = new_pos_Sy_3
            cycle_pos[i, Sx_cycle[i]] = new_pos_Sx_3

            # update indices
            Sx_cycle[i].extend(Sy_cycle[i])
            Sx_cycle[i] = list(set(Sx_cycle[i]))

        # update y indices or create mask for last loop
        if not np.all(ids == cycle_i_orders[-1]):
            Sy_cycle = [[] for i in range(cycle_len)]
        else:
            cycle_mask = torch.ones([cycle_pos.size(0), cycle_pos.size(1)])
            for i in range(cycle_len):
                cycle_mask[i, y_indices[i]] = 0
                y_neighbor_ids = model.neighbors[y_indices[i]]
                y_neighbor_ids_not_x = y_neighbor_ids[~model.y_map_to_neighbor_x[ids[i]][0:len(y_neighbor_ids)].bool()]
                cycle_mask[i, y_neighbor_ids_not_x] = 0

    # extract unaligned coords
    final_cycle_coords_unaligned = cycle_pos[:, Sx_cycle[0]]
    q_cycle_coords = final_cycle_coords_unaligned[0].permute(1, 0, 2)  # target
    p_cycle_coords = final_cycle_coords_unaligned[1:].permute(0, 2, 1, 3)  # source

    # align coords with Kabsch algorithm
    q_cycle_coords_aligned = final_cycle_coords_unaligned[0]
    cycle_rmsd_mask = [True if a in np.unique(cycle_pairs) else False for a in Sx_cycle[0]]
    # cycle_rmsd_mask = [True for a in Sx_cycle[0]]
    p_cycle_coords_aligned = align_coords_Kabsch(p_cycle_coords, q_cycle_coords, cycle_rmsd_mask).permute(0, 2, 1, 3)

    # average aligned coords
    cycle_avg_coords_ = torch.vstack([q_cycle_coords_aligned.unsqueeze(0), p_cycle_coords_aligned]) * cycle_mask[:, Sx_cycle[0]].unsqueeze(-1).unsqueeze(-1)
    cycle_avg_coords = cycle_avg_coords_.sum(dim=0) / cycle_mask[:, Sx_cycle[0]].sum(dim=0).unsqueeze(-1).unsqueeze(-1)

    return cycle_avg_coords, Sx_cycle[0]


def construct_conformers_acyclic(data, n_true_confs, n_model_confs, dihedral_pairs, neighbors, model_p_coords,
                                 model_q_coords, dihedral_x_mask, dihedral_y_mask, x_map_to_neighbor_y,
                                 y_map_to_neighbor_x, dihedral_mask, c_ij, v_star):

    pos = torch.cat([torch.cat([p[0][i] for p in data.pos]).unsqueeze(1) for i in range(n_true_confs)], dim=1)
    new_pos = torch.zeros([pos.size(0), n_model_confs, 3]).to(device)
    dihedral_pairs = dihedral_pairs.t().detach().cpu().numpy()

    Sx = []
    Sy = []

    for i, pair in enumerate(dihedral_pairs):

        x_index, y_index = pair

        # skip cycles for now (check if all of y's neighbors are in Sx)
        if np.prod([n in Sx for n in neighbors[y_index]]):
            continue

        # new graph
        p_coords = torch.zeros([4, n_model_confs, 3]).to(device)
        p_idx = neighbors[x_index]

        if x_index not in Sx:
            Sx = []
            # set new immediate neighbor coords for X
            p_coords = model_p_coords[i]
            new_pos[p_idx] = p_coords[0:int(dihedral_x_mask[i].sum())]

        else:
            p_coords[0:p_idx.size(0)] = new_pos[p_idx] - new_pos[x_index]

        # update indices
        Sx.extend([x_index])
        Sx.extend(neighbors[x_index].detach().numpy())
        Sx = list(set(Sx))

        Sy.extend([y_index])
        Sy.extend(neighbors[y_index].detach().numpy())

        # set px
        p_X = new_pos[x_index]

        # translate current Sx
        new_pos_Sx = new_pos[Sx] - p_X

        # set y
        q_coords = model_q_coords[i]
        q_idx = neighbors[y_index]
        new_pos[q_idx] = q_coords[0:int(dihedral_y_mask[i].sum())]
        new_pos[y_index] = torch.zeros_like(p_X)  # q_Y always at the origin
        new_pos_Sy = new_pos[Sy]

        # calculate rotation matrices
        H_XY = rotation_matrix_inf_v2(p_coords, x_map_to_neighbor_y[i])
        H_YX = rotation_matrix_inf_v2(q_coords, y_map_to_neighbor_x[i])

        # rotate
        new_pos_Sx_2 = torch.matmul(H_XY.unsqueeze(0), new_pos_Sx.unsqueeze(-1)).squeeze(-1)
        new_pos_Sy_2 = torch.matmul(H_YX.unsqueeze(0), new_pos_Sy.unsqueeze(-1)).squeeze(-1)

        # translate q
        new_p_Y = new_pos_Sx_2[Sx == y_index]
        transform_matrix = torch.diag(torch.tensor([-1., -1., 1.])).unsqueeze(0).unsqueeze(0)
        new_pos_Sy_3 = torch.matmul(transform_matrix, new_pos_Sy_2.unsqueeze(-1)).squeeze(-1) + new_p_Y

        # rotate by gamma
        H_gamma = calculate_gamma(n_model_confs, dihedral_mask[i], c_ij[i], v_star[i], Sx, Sy, p_idx, q_idx, x_index,
                                  y_index, new_pos_Sx_2, new_pos_Sy_3, new_p_Y)
        new_pos_Sx_3 = torch.matmul(H_gamma.unsqueeze(0), new_pos_Sx_2.unsqueeze(-1)).squeeze(-1)

        # update all coordinates
        new_pos[Sy] = new_pos_Sy_3
        new_pos[Sx] = new_pos_Sx_3

        # update indices
        Sx.extend(Sy)
        Sx = sorted(list(set(Sx)))
        Sy = []

    return new_pos


pT_idx, qZ_idx = torch.cartesian_prod(torch.arange(3), torch.arange(3)).chunk(2, dim=-1)
pT_idx = pT_idx.squeeze(-1)
qZ_idx = qZ_idx.squeeze(-1)


def calculate_gamma(n_model_confs, dihedral_mask, c_ij, v_star, Sx, Sy, p_idx, q_idx, x_index, y_index,
                    new_pos_Sx_2, new_pos_Sy_3, new_p_Y):
    # calculate current dihedrals
    pT_prime = torch.zeros([3, n_model_confs, 3]).to(device)
    qZ_translated = torch.zeros([3, n_model_confs, 3]).to(device)

    pY_prime = new_p_Y.repeat(9, 1, 1)
    qX = torch.zeros_like(pY_prime)

    p_ids_in_Sx = [Sx.index(p.item()) for p in p_idx if p.item() != y_index]
    q_ids_in_Sy = [Sy.index(q.item()) for q in q_idx if q.item() != x_index]

    pT_prime[:len(p_ids_in_Sx)] = new_pos_Sx_2[p_ids_in_Sx]
    qZ_translated[:len(q_ids_in_Sy)] = new_pos_Sy_3[q_ids_in_Sy]

    XYTi_XYZj_curr_sin, XYTi_XYZj_curr_cos = batch_dihedrals(pT_prime[pT_idx], qX, pY_prime, qZ_translated[qZ_idx])
    A_ij = build_A_matrix_inf(XYTi_XYZj_curr_sin, XYTi_XYZj_curr_cos, n_model_confs) * dihedral_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # build A matrix
    A_curr = torch.sum(A_ij * c_ij.unsqueeze(-1), dim=0)
    determinants = torch.det(A_curr) + 1e-10
    A_curr_inv_ = A_curr.view(n_model_confs, 4)[:, [3, 1, 2, 0]] * torch.tensor([[1., -1., -1., 1.]])
    A_curr_inv = (A_curr_inv_ / determinants.unsqueeze(-1)).view(n_model_confs, 2, 2)
    A_curr_inv_v_star = torch.matmul(A_curr_inv, v_star.unsqueeze(-1)).squeeze(-1)

    # get gamma matrix
    v_gamma = A_curr_inv_v_star / (A_curr_inv_v_star.norm(dim=-1, keepdim=True) + 1e-10)
    gamma_cos, gamma_sin = v_gamma.split(1, dim=-1)
    H_gamma = build_gamma_rotation_inf(gamma_sin.squeeze(-1), gamma_cos.squeeze(-1), n_model_confs)

    return H_gamma


def rotation_matrix_inf_v2(neighbor_coords, neighbor_map):
    """
    Given predicted neighbor coordinates from model, return rotation matrix

    :param neighbor_coords: neighbor coordinates for each edge as defined by dihedral_pairs
        (n_dihedral_pairs, 4, n_generated_confs, 3)
    :param neighbor_mask: mask describing which atoms are present (n_dihedral_pairs, 4)
    :param neighbor_map: mask describing which neighbor corresponds to the other central dihedral atom
        (n_dihedral_pairs, 4) each entry in neighbor_map should have one TRUE entry with the rest as FALSE
    :return: rotation matrix (n_dihedral_pairs, n_model_confs, 3, 3)
    """

    p_Y = neighbor_coords[neighbor_map.bool(), :].squeeze(0)

    eta_1 = torch.rand_like(p_Y)
    eta_2 = eta_1 - torch.sum(eta_1 * p_Y, dim=-1, keepdim=True) / (torch.linalg.norm(p_Y, dim=-1, keepdim=True)**2 + 1e-10) * p_Y
    eta = eta_2 / torch.linalg.norm(eta_2, dim=-1, keepdim=True)

    h1 = p_Y / (torch.linalg.norm(p_Y, dim=-1, keepdim=True) + 1e-10)  # (n_dihedral_pairs, n_model_confs, 10)

    h3_1 = torch.cross(p_Y, eta, dim=-1)
    h3 = h3_1 / (torch.linalg.norm(h3_1, dim=-1, keepdim=True) + 1e-10)  # (n_dihedral_pairs, n_model_confs, 10)

    h2 = -torch.cross(h1, h3, dim=-1)  # (n_dihedral_pairs, n_model_confs, 10)

    H = torch.cat([h1.unsqueeze(-2),
                   h2.unsqueeze(-2),
                   h3.unsqueeze(-2)], dim=-2)

    return H


def build_A_matrix_inf(curr_sin, curr_cos, n_model_confs):

    A_ij = torch.FloatTensor([[[[0, 0], [0, 0]]]]).repeat(9, n_model_confs, 1, 1)
    A_ij[:, :, 0, 0] = curr_cos
    A_ij[:, :, 0, 1] = curr_sin
    A_ij[:, :, 1, 0] = curr_sin
    A_ij[:, :, 1, 1] = -curr_cos

    return A_ij


def build_gamma_rotation_inf(gamma_sin, gamma_cos, n_model_confs):
    H_gamma = torch.FloatTensor([[[1, 0, 0], [0, 0, 0], [0, 0, 0]]]).repeat(n_model_confs, 1, 1)
    H_gamma[:, 1, 1] = gamma_cos
    H_gamma[:, 1, 2] = -gamma_sin
    H_gamma[:, 2, 1] = gamma_sin
    H_gamma[:, 2, 2] = gamma_cos

    return H_gamma


def get_cycle_values(cycle_list, start_at=None):
    start_at = 0 if start_at is None else cycle_list.index(start_at)
    while True:
        yield cycle_list[start_at]
        start_at = (start_at + 1) % len(cycle_list)

def get_cycle_indices(cycle, start_idx):
    cycle_it = get_cycle_values(cycle, start_idx)
    indices = []

    end = 9e99
    start = next(cycle_it)
    a = start
    while start != end:
        b = next(cycle_it)
        indices.append(torch.tensor([a, b]))
        a = b
        end = b

    return indices

def get_current_cycle_indices(cycles, cycle_check, idx):
    c_idx = [i for i, c in enumerate(cycle_check) if c][0]
    current_cycle = cycles.pop(c_idx)
    current_idx = current_cycle[(np.array(current_cycle) == idx.item()).nonzero()[0][0]]
    return get_cycle_indices(current_cycle, current_idx)


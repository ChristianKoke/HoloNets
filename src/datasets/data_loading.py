import os
import numpy as np
import math
import pickle

import torch
import torch_geometric
from torch_geometric.data import download_url
from torch_geometric.datasets import (
    WikipediaNetwork,
    CitationFull,
    WebKB,
    LINKXDataset,
    Actor,   
)
from .amazon import Amazon



import torch_geometric.transforms as transforms
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from src.datasets.directed_heterophilous_graphs import DirectedHeterophilousGraphDataset
from src.datasets.telegram import Telegram
from src.datasets.data_utils import get_mask
from src.utils.third_party import (
    load_snap_patents_mat,
    load_pokec_mat,
    even_quantile_labels,
)
from src.datasets.synthetic import get_syn_dataset


def get_dataset(name: str, root_dir: str, homophily=None, undirected=False, self_loops=False, transpose=False):
    path = f"{root_dir}/"
    evaluator = None

    if name in ["chameleon", "squirrel"]:
        dataset = WikipediaNetwork(root=path, name=name, transform=transforms.NormalizeFeatures())
        dataset._data.y = dataset._data.y.unsqueeze(-1)

    elif name in ["actor"]:
        dataset = Actor(root=path, transform=transforms.NormalizeFeatures())
        dataset._data.y = dataset._data.y.unsqueeze(-1)

    elif name in ["computers", "photo"]:
        if name == "computers":
            name = "Computers"
        elif name == "photo":
            name = "Photo"
        dataset = Amazon(root=path, name = name, transform=transforms.NormalizeFeatures())
        dataset._data.y = dataset._data.y.unsqueeze(-1)


    elif name in ["cornell", "wisconsin", "texas"]:
        dataset = WebKB(root=path, name=name, transform=transforms.NormalizeFeatures())
        dataset._data.y = dataset._data.y.unsqueeze(-1)
    elif name in ["penn94", "reed98", "amherst41", "cornell115", "johnshopkins55", "genius"]:
        dataset = LINKXDataset(root=path, name=name, transform=transforms.NormalizeFeatures())
        dataset._data.y = dataset._data.y.unsqueeze(-1)
    




    elif name in ["ogbn-arxiv"]:
        dataset = PygNodePropPredDataset(name=name, transform=transforms.ToSparseTensor(), root=path)
        evaluator = Evaluator(name=name)
        split_idx = dataset.get_idx_split()
        dataset._data.train_mask = get_mask(split_idx["train"], dataset._data.num_nodes)
        dataset._data.val_mask = get_mask(split_idx["valid"], dataset._data.num_nodes)
        dataset._data.test_mask = get_mask(split_idx["test"], dataset._data.num_nodes)
    elif name in ["directed-roman-empire", "directed-amazon-ratings", "directed-questions"]:
        dataset = DirectedHeterophilousGraphDataset(name=name, transform=transforms.NormalizeFeatures(), root=path)
    elif name == "snap-patents":
        dataset = load_snap_patents_mat(n_classes=5, root=path)
    elif name == "pokec":
        dataset = load_pokec_mat(n_classes=2, root=path)
    elif name == "arxiv-year":
        # arxiv-year uses the same graph and features as ogbn-arxiv, but with different labels
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", transform=transforms.ToSparseTensor(), root=path)
        evaluator = Evaluator(name="ogbn-arxiv")
        y = even_quantile_labels(dataset._data.node_year.flatten().numpy(), nclasses=5, verbose=False)
        dataset._data.y = torch.as_tensor(y).reshape(-1, 1)
        # Tran, val and test masks are required during preprocessing. Setting them here to dummy values as
        # they are overwritten later for this dataset (see get_dataset_split function below)
        dataset._data.train_mask, dataset._data.val_mask, dataset._data.test_mask = 0, 0, 0
        # Create directory for this dataset
        os.makedirs(os.path.join(path, name.replace("-", "_"), "raw"), exist_ok=True)
    elif name == "syn-dir":
        dataset = get_syn_dataset(path)

    elif name in ["cora_ml", "citeseer_full"]:
        if name == "citeseer_full":
            name = "citeseer"
        
        dataset = Citation(path, name)
    elif name == "telegram":
        dataset = Telegram(path)
    else:
        raise Exception("Unknown dataset.")

    if undirected:
        dataset._data.edge_index = torch_geometric.utils.to_undirected(dataset._data.edge_index)
    if self_loops:
        dataset._data.edge_index, _ = torch_geometric.utils.add_self_loops(dataset._data.edge_index)
    if transpose:
        dataset._data.edge_index = torch.stack([dataset._data.edge_index[1], dataset._data.edge_index[0]])

    

    return dataset, evaluator


def get_dataset_split(name, data, root_dir, split_number):
    if name in ["snap-patents", "pokec", "chameleon", "squirrel", "telegram", "directed-roman-empire", "directed-amazon-ratings", "directed-questions"]:
        return (
            data["train_mask"][:, split_number],
            data["val_mask"][:, split_number],
            data["test_mask"][:, split_number],
        )
    elif name in ["cornell", "wisconsin", "texas"]:
        return (
            data["train_mask"][:, split_number],
            data["val_mask"][:, split_number],
            data["test_mask"][:, split_number],
        )
    


    elif name in ["penn94",  "genius"]:
        if name == "penn94":
            name = "fb100-Penn94"
        # Datasets from https://arxiv.org/pdf/2110.14446.pdf have five splits stored
        # in https://github.com/CUAI/Non-Homophily-Large-Scale/tree/82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6/data/splits
        num_nodes = data["y"].shape[0]
        github_url = f"https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data/splits/"
        split_file_name = f"{name}-splits.npy"
        local_dir = os.path.join(root_dir, name, "raw")

        download_url(os.path.join(github_url, split_file_name), local_dir, log=False)
        splits = np.load(os.path.join(local_dir, split_file_name), allow_pickle=True)
        split_idx = splits[split_number % len(splits)]

        train_mask = get_mask(split_idx["train"], num_nodes)
        val_mask = get_mask(split_idx["valid"], num_nodes)
        test_mask = get_mask(split_idx["test"], num_nodes)

        return train_mask, val_mask, test_mask



    elif name in ["actor"]:
        return (
            data["train_mask"][:, split_number],
            data["val_mask"][:, split_number],
            data["test_mask"][:, split_number],
        )


    elif name in ["ogbn-arxiv"]:
        # OGBN datasets have a single pre-assigned split
        return data["train_mask"], data["val_mask"], data["test_mask"]
    elif name in ["arxiv-year"]:
        # Datasets from https://arxiv.org/pdf/2110.14446.pdf have five splits stored
        # in https://github.com/CUAI/Non-Homophily-Large-Scale/tree/82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6/data/splits
        num_nodes = data["y"].shape[0]
        github_url = f"https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data/splits/"
        split_file_name = f"{name}-splits.npy"
        local_dir = os.path.join(root_dir, name.replace("-", "_"), "raw")
        

        download_url(os.path.join(github_url, split_file_name), local_dir, log=False)
        splits = np.load(os.path.join(local_dir, split_file_name), allow_pickle=True)
        split_idx = splits[split_number % len(splits)]

        train_mask = get_mask(split_idx["train"], num_nodes)
        val_mask = get_mask(split_idx["valid"], num_nodes)
        test_mask = get_mask(split_idx["test"], num_nodes)

        return train_mask, val_mask, test_mask
    


    
    elif name in ["syn-dir", "cora_ml", "citeseer_full"]:
       return (
            data["train_mask"],
            data["val_mask"],
            data["test_mask"],
        )
       
       
        # # Uniform 50/25/25 split
        # return set_uniform_train_val_test_split(split_number, data, train_ratio=0.5, val_ratio=0.25)


def set_uniform_train_val_test_split(seed, data, train_ratio=0.5, val_ratio=0.25):
    rnd_state = np.random.RandomState(seed)
    num_nodes = data.y.shape[0]

    # Some nodes have labels -1 (i.e. unlabeled), so we need to exclude them
    labeled_nodes = torch.where(data.y != -1)[0]
    num_labeled_nodes = labeled_nodes.shape[0]
    num_train = math.floor(num_labeled_nodes * train_ratio)
    num_val = math.floor(num_labeled_nodes * val_ratio)

    idxs = list(range(num_labeled_nodes))
    # Shuffle in place
    rnd_state.shuffle(idxs)

    train_idx = idxs[:num_train]
    val_idx = idxs[num_train : num_train + num_val]
    test_idx = idxs[num_train + num_val :]

    train_idx = labeled_nodes[train_idx]
    val_idx = labeled_nodes[val_idx]
    test_idx = labeled_nodes[test_idx]

    train_mask = get_mask(train_idx, num_nodes)
    val_mask = get_mask(val_idx, num_nodes)
    test_mask = get_mask(test_idx, num_nodes)

    return train_mask, val_mask, test_mask

####################################################################################################################



import os.path as osp
import numpy as np
import scipy.sparse as sp
import networkx as nx
import pandas as pd
import os
import torch
import sys
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, is_undirected, to_networkx
from networkx.algorithms.components import is_weakly_connected

from torch_geometric.utils import add_remaining_self_loops, add_self_loops, remove_self_loops
from torch_scatter import scatter_add
import scipy
from torch_geometric.data import InMemoryDataset, Dataset
# from get_adj import get_undirected_adj,get_pr_directed_adj,get_appr_directed_adj,get_second_directed_adj

class Citation(InMemoryDataset):
    r"""
    Nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"cora_ml"`,
            :obj:`"citeseer"`, :obj:`"pubmed"`), :obj:`"amazon_computer", :obj:`"amazon_photo", :obj:`"cora_full"`) .
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, root, name, alpha=0.1, adj_type='or', transform=None, pre_transform=None):
        self.name = name
        self.alpha = alpha
        self.adj_type = adj_type
        super(Citation, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return

    @property
    def processed_file_names(self):
        return 'data.pt'

    # def download(self):
    #     return

    def process(self):
        print('here')
        data = citation_datasets(self.raw_dir, self.name, self.alpha, self.adj_type)
        # data = read_planetoid_data(self.raw_dir, self.name)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)

def citation_datasets(path="./data", dataset='cora_ml', alpha=0.1, adj_type='or'):
    # path = os.path.join(save_path, dataset)
    os.makedirs(path, exist_ok=True)
    dataset_path = os.path.join(path, '{}.npz'.format(dataset))
    g = load_npz_dataset(dataset_path)
    adj, features, labels = g['A'], g['X'], g['z']
    
    # Set new random splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * the rest for testing

    mask = train_test_split(labels, seed=1020, train_examples_per_class=20, val_size=500, test_size=None)

    mask['train'] = torch.from_numpy(mask['train']).bool()
    mask['val'] = torch.from_numpy(mask['val']).bool()
    mask['test'] = torch.from_numpy(mask['test']).bool()

    coo = adj.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    indices = torch.from_numpy(indices).long()
    features = torch.from_numpy(features.todense()).float()
    labels = torch.from_numpy(labels).long()
    if adj_type == 'un':
        print("Processing to undirected adj")
        indices = to_undirected(indices)
        edge_index, edge_weight = get_undirected_adj(indices, features.shape[0], features.dtype)
        data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight, y=labels)
    elif adj_type == 'pr':
        print("Processing pagerank adj matrix")
        edge_index, edge_weight = get_pr_directed_adj(alpha, indices, features.shape[0],features.dtype)
        data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight, y=labels)
    elif adj_type == 'appr':
        print("Processing approximate personalized pagerank adj matrix")
        edge_index, edge_weight = get_appr_directed_adj(alpha, indices, features.shape[0],features.dtype)
        data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight, y=labels)
    elif adj_type == 'ib':
        print("Processing first and second-order adj matrix")
        edge_index, edge_weight = get_appr_directed_adj(alpha, indices, features.shape[0],features.dtype) 
        data = Data(x=features, edge_index=edge_index, eget_train_val_test_splitdge_weight=edge_weight, y=labels)
        edge_index, edge_weight = get_second_directed_adj(indices, features.shape[0],features.dtype)
        data.edge_index2 = edge_index
        data.edge_weight2 = edge_weight
    elif adj_type == 'or':
        print("Processing to original directed adj")
        data = Data(x=features, edge_index=indices, edge_weight=None, y=labels)
    else:
        print("Unsupported adj type.")
        sys.exit()
    
    data.train_mask = mask['train']
    data.val_mask = mask['val']
    data.test_mask = mask['test']
    return data

def load_npz_dataset(file_name):
    """Load a graph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    graph : dict
        Dictionary that contains:
            * 'A' : The adjacency matrix in sparse matrix format
            * 'X' : The attribute matrix in sparse matrix format
            * 'z' : The ground truth class labels
            * Further dictionaries mapping node, class and attribute IDs

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        edge_index = loader['adj_indices'].copy()
        A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                           loader['adj_indptr']), shape=loader['adj_shape'])

        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])

        z = loader.get('labels')

        graph = {
            'A': A,
            'X': X,
            'z': z
        }

        idx_to_node = loader.get('idx_to_node')
        if idx_to_node:
            idx_to_node = idx_to_node.tolist()
            graph['idx_to_node'] = idx_to_node

        idx_to_attr = loader.get('idx_to_attr')
        if idx_to_attr:
            idx_to_attr = idx_to_attr.tolist()
            graph['idx_to_attr'] = idx_to_attr

        idx_to_class = loader.get('idx_to_class')
        if idx_to_class:
            idx_to_class = idx_to_class.tolist()
            graph['idx_to_class'] = idx_to_class

        return graph

def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples = labels.shape[0]
    num_classes = labels.max()+1
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples = labels.shape[0]
    num_classes = labels.max()+1
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(
            random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(
            remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(
            random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(
            remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(
            remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)
               ) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)
               ) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate(
            (train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices

def train_test_split(labels, seed, train_examples_per_class=None, val_examples_per_class=None, test_examples_per_class=None, train_size=None, val_size=None, test_size=None):
    random_state = np.random.RandomState(seed)
    train_indices, val_indices, test_indices = get_train_val_test_split(
        random_state, labels, train_examples_per_class, val_examples_per_class, test_examples_per_class, train_size, val_size, test_size)

    #print('number of training: {}'.format(len(train_indices)))
    #print('number of validation: {}'.format(len(val_indices)))
    #print('number of testing: {}'.format(len(test_indices)))

    train_mask = np.zeros((labels.shape[0], 1), dtype=int)
    train_mask[train_indices, 0] = 1
    train_mask = np.squeeze(train_mask, 1)
    val_mask = np.zeros((labels.shape[0], 1), dtype=int)
    val_mask[val_indices, 0] = 1
    val_mask = np.squeeze(val_mask, 1)
    test_mask = np.zeros((labels.shape[0], 1), dtype=int)
    test_mask[test_indices, 0] = 1
    test_mask = np.squeeze(test_mask, 1)
    mask = {}
    mask['train'] = train_mask
    mask['val'] = val_mask
    mask['test'] = test_mask
    return mask



import os.path as osp
import numpy as np
import scipy.sparse as sp
import networkx as nx
import pandas as pd
import os
import torch
import shutil
import torch_geometric.transforms as T
from torch_geometric.data import Data
# from Citation import Citation
from torch_geometric.utils import to_undirected, is_undirected
from torch_geometric.data import InMemoryDataset


def get_citation_dataset(name, alpha=0.1, recache=False, normalize_features=False, adj_type=None, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)),'data')
    file_path = osp.join(path, name, 'processed')
    if recache == True:
        print("Delete old processed data cache...")
        if osp.exists(file_path):
            shutil.rmtree(file_path)
        os.mkdir(file_path)
        print('Finish cleaning.')
    dataset = Citation(path, name, alpha, adj_type=adj_type)
    print('Finish dataset preprocessing.')
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    return dataset




import os.path as osp
import numpy as np
import scipy.sparse as sp
import networkx as nx
import pandas as pd
import os
import torch

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, is_undirected, to_networkx
from networkx.algorithms.components import is_weakly_connected

from torch_geometric.utils import add_remaining_self_loops, add_self_loops, remove_self_loops
from torch_scatter import scatter_add
import scipy


def get_undirected_adj(edge_index, num_nodes, dtype):
    edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                    device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def get_pr_directed_adj(alpha, edge_index, num_nodes, dtype, edge_weight = None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
    else:
        edge_weight = torch.FloatTensor(edge_weight).to(edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)  
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight 
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes,num_nodes])).to_dense()
    
    # pagerank p
    p_pr = (1.0-alpha) * p_dense + alpha / num_nodes * torch.ones((num_nodes,num_nodes), dtype=dtype, device=p.device)


    eig_value, left_vector = scipy.linalg.eig(p_pr.numpy(),left=True,right=False)
    eig_value = torch.from_numpy(eig_value.real)
    left_vector = torch.from_numpy(left_vector.real)
    val, ind = eig_value.sort(descending=True)
    
    # assert val[0] == 1.0

    pi = left_vector[:,ind[0]] # choose the largest eig vector
    pi = pi/pi.sum()  # norm pi

    # Note that by scaling the vectors, even the sign can change. That's why positive and negative elements might get flipped.
    assert len(pi[pi<0]) == 0

    pi_inv_sqrt = pi.pow(-0.5)
    pi_inv_sqrt[pi_inv_sqrt == float('inf')] = 0
    pi_inv_sqrt = pi_inv_sqrt.diag()
    pi_sqrt = pi.pow(0.5)
    pi_sqrt[pi_sqrt == float('inf')] = 0
    pi_sqrt = pi_sqrt.diag()

    # L_pr
    L = (torch.mm(torch.mm(pi_sqrt, p_pr), pi_inv_sqrt) + torch.mm(torch.mm(pi_inv_sqrt, p_pr.t()), pi_sqrt)) / 2.0

    # make nan to 0
    L[torch.isnan(L)] = 0

    # # let little possbility connection to 0, make L sparse
    # L[ L < (1/num_nodes)] = 0
    # L[ L < 5e-4] = 0

    # transfer dense L to sparse
    L_indices = torch.nonzero(L,as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

def get_appr_directed_adj(alpha, edge_index, num_nodes, dtype, edge_weight=None):
    if edge_weight ==None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)  
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes) 
    deg_inv = deg.pow(-1) 
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight 

    # personalized pagerank p
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes,num_nodes])).to_dense()
    p_v = torch.zeros(torch.Size([num_nodes+1,num_nodes+1]))
    p_v[0:num_nodes,0:num_nodes] = (1-alpha) * p_dense
    p_v[num_nodes,0:num_nodes] = 1.0 / num_nodes
    p_v[0:num_nodes,num_nodes] = alpha
    p_v[num_nodes,num_nodes] = 0.0
    p_ppr = p_v 

    eig_value, left_vector = scipy.linalg.eig(p_ppr.numpy(),left=True,right=False)
    eig_value = torch.from_numpy(eig_value.real)
    left_vector = torch.from_numpy(left_vector.real)
    val, ind = eig_value.sort(descending=True)

    pi = left_vector[:,ind[0]] # choose the largest eig vector
    pi = pi[0:num_nodes]
    p_ppr = p_dense
    pi = pi/pi.sum()  # norm pi

    # Note that by scaling the vectors, even the sign can change. That's why positive and negative elements might get flipped.
    assert len(pi[pi<0]) == 0

    pi_inv_sqrt = pi.pow(-0.5)
    pi_inv_sqrt[pi_inv_sqrt == float('inf')] = 0
    pi_inv_sqrt = pi_inv_sqrt.diag()
    pi_sqrt = pi.pow(0.5)
    pi_sqrt[pi_sqrt == float('inf')] = 0
    pi_sqrt = pi_sqrt.diag()

    # L_appr
    L = (torch.mm(torch.mm(pi_sqrt, p_ppr), pi_inv_sqrt) + torch.mm(torch.mm(pi_inv_sqrt, p_ppr.t()), pi_sqrt)) / 2.0

    # make nan to 0
    L[torch.isnan(L)] = 0

    # transfer dense L to sparse
    L_indices = torch.nonzero(L,as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

def get_second_directed_adj(edge_index, num_nodes, dtype):

    edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                    device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight 
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes,num_nodes])).to_dense()
    
    L_in = torch.mm(p_dense.t(), p_dense)
    L_out = torch.mm(p_dense, p_dense.t())
    
    L_in_hat = L_in
    L_out_hat = L_out

    L_in_hat[L_out == 0] = 0
    L_out_hat[L_in == 0] = 0

    # L^{(2)}
    L = (L_in_hat + L_out_hat) / 2.0

    L[torch.isnan(L)] = 0
    L_indices = torch.nonzero(L,as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values
    
    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
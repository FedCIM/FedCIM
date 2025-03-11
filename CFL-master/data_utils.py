import numpy as np
from torch.utils.data import Subset

"""
def split_noniid(train_idcs, train_labels, alpha, n_clients):
    
    #Splits a list of data indices with corresponding labels into subsets according to a dirichlet distribution with parameter alpha
    
    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels[train_idcs]==y).flatten() 
           for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]
  
    return client_idcs
"""


# ************************************************* non-IID Partitioning Case *********************


# non-IID Partitioning Case 2
def split_noniid1(dataset, num_clients):
    """
    Create non-IID partition where each client holds data from only one class.
    """
    num_classes = 10
    class_indices = [
        np.where(np.array(dataset.targets) == i)[0] for i in range(num_classes)
    ]
    client_data_indices = []

    # Assign each client a single class
    for i in range(num_clients):
        client_class = i % num_classes
        client_data_indices.append(class_indices[client_class])

    return client_data_indices


# IID Partitioning
def iid_partition(dataset, num_clients):
    """
    Split the dataset into IID partitions where each client gets equal number of instances from all classes.
    """
    num_items_per_client = len(dataset) // num_clients
    indices = np.random.permutation(len(dataset))
    client_data_indices = np.array_split(indices, num_clients)
    return client_data_indices


# non-IID Partitioning Case 1
def split_noniid(dataset, num_clients, class_split_ratio):
    """
    Create non-IID partition where data is split into two parts per class in a given ratio.
    One part contains data from the same class and the other part contains data from different classes.
    """
    num_classes = 10
    class_indices = [
        np.where(np.array(dataset.targets) == i)[0] for i in range(num_classes)
    ]
    client_data_indices = [[] for _ in range(num_clients)]

    # Split each class into two parts based on the class_split_ratio
    for cls in range(num_classes):
        cls_data = class_indices[cls]
        np.random.shuffle(cls_data)
        split_point = int(class_split_ratio * len(cls_data))
        part1, part2 = cls_data[:split_point], cls_data[split_point:]

        # Assign part 1 to clients
        part1_split = np.array_split(part1, num_clients)
        for i in range(num_clients):
            client_data_indices[i].extend(part1_split[i])

        # Assign part 2 (mixed classes) to clients
        part2_split = np.array_split(part2, num_clients)
        for i in range(num_clients):
            client_data_indices[i].extend(
                part2_split[(i + 1) % num_clients]
            )  # Shift data

    return client_data_indices


# ************************************************* non-IID Partitioning Case 1 *********************close******
class CustomSubset(Subset):
    """A custom subset class with customizable data transformation"""

    def __init__(self, dataset, indices, subset_transform=None):
        super().__init__(dataset, indices)
        self.subset_transform = subset_transform

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]

        if self.subset_transform:
            x = self.subset_transform(x)

        return x, y

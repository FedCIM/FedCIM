import numpy as np


class Sampling:

    def create_iid_data(dataset, num_clients):
        """
        Split the dataset into IID partitions where each client gets equal number of instances from all classes.
        """
        num_items_per_client = len(dataset) // num_clients
        indices = np.random.permutation(len(dataset))
        client_data_indices = np.array_split(indices, num_clients)
        return client_data_indices

    def create_non_iid_data(dataset, num_clients=10, class_split_ratio=0.25):
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

    def create_non_iid_c2(dataset, num_clients):
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

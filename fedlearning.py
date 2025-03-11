import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from collections import defaultdict
import copy
from models import CNN, flatten_weights

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# import numpy as np
from datetime import datetime
import os

from sampling import Sampling
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class FederatedLearning:
    def __init__(
        self,
        args,
        num_clients,
        min_clusters=2,
        max_clusters=10,
        local_epochs=5,
        class_split_ratio=0.75,
    ):
        self.num_clients = num_clients
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.local_epochs = local_epochs
        self.class_split_ratio = class_split_ratio

        if args.gpu:
            torch.cuda.set_device(args.gpu)
        self.device = "cuda" if args.gpu else "cpu"
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize models
        self.global_model = CNN().to(self.device)
        self.client_models = {i: CNN().to(self.device) for i in range(num_clients)}

        # Setup data
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # Load train and test datasets
        self.train_dataset = torchvision.datasets.MNIST(
            "./data", train=True, download=True, transform=transform
        )
        self.test_dataset = torchvision.datasets.MNIST(
            "./data", train=False, download=True, transform=transform
        )

        # Create test loader
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=128, shuffle=False
        )

        # Create non-IID data distribution

        if args.iid:
            # Sample IID user data from Mnist
            self.client_data = Sampling.create_iid_data(self.train_dataset, num_clients)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                self.client_data = Sampling.create_non_iid_c2(
                    self.train_dataset, num_clients
                )
            else:
                # Chose euqal splits for every user
                self.client_data = Sampling.create_non_iid_data(
                    self.train_dataset, num_clients
                )

        # Initialize metrics tracking
        self.accuracy_history = []
        self.cluster_history = []
        self.silhouette_history = []
        self.train_accuracy_history = []
        self.test_accuracy_history = []

    def evaluate_model(self, model, dataset_type="test"):
        model.eval()
        correct = 0
        total = 0

        if dataset_type == "test":
            data_loader = self.test_loader
        else:  # train
            # Create a loader for the entire training dataset
            train_loader = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=128, shuffle=False
            )
            data_loader = train_loader

        with torch.no_grad():
            for data, labels in data_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    def find_optimal_clusters(self, client_weights):
        best_score = -1
        best_num_clusters = self.min_clusters
        best_labels = None
        best_medoids = None

        similarity_matrix = compute_similarity_matrix(client_weights)
        distance_matrix = 1 - similarity_matrix

        for n_clusters in range(
            self.min_clusters, min(self.max_clusters + 1, self.num_clients)
        ):
            try:
                kmedoids = KMedoids(n_clusters=n_clusters, random_state=0)
                cluster_labels = kmedoids.fit_predict(distance_matrix)
                score = silhouette_score(distance_matrix, cluster_labels)

                if score > best_score:
                    best_score = score
                    best_num_clusters = n_clusters
                    best_labels = cluster_labels
                    best_medoids = kmedoids.medoid_indices_
            except Exception as e:
                print(f"Clustering failed for n_clusters={n_clusters}: {e}")
                continue

        return best_labels, best_score, best_medoids, best_num_clusters

    def train_round(self):
        client_weights = {}

        # Train each client
        for client_id in range(self.num_clients):
            # Get client data
            indices = self.client_data[client_id]
            client_dataset = torch.utils.data.Subset(self.train_dataset, indices)
            client_loader = torch.utils.data.DataLoader(
                client_dataset, batch_size=32, shuffle=True
            )

            # Initialize client model with global weights
            self.client_models[client_id].load_state_dict(
                self.global_model.state_dict()
            )

            # If this is our designated adversarial client (e.g., client 0)
            if (
                client_id == 0
                and hasattr(self, "inject_adversary")
                and self.inject_adversary
            ):
                client_weights[client_id] = self.create_adversarial_client(client_id)
                continue

            # Train client model
            optimizer = torch.optim.SGD(
                self.client_models[client_id].parameters(),
                lr=0.01,
            )
            client_weights[client_id] = self.client_update(
                self.client_models[client_id], optimizer, client_loader
            )

        return client_weights

    def detect_outliers(self, client_weights, cluster_labels, threshold=0.3):
        outliers = []
        clusters = defaultdict(list)

        # Group clients by cluster
        for i, cluster in enumerate(cluster_labels):
            clusters[cluster].append(i)

        similarity_matrix = compute_similarity_matrix(client_weights)

        # Check each client's average similarity with its cluster
        for cluster_id, client_ids in clusters.items():
            for client_id in client_ids:
                cluster_similarities = [
                    similarity_matrix[client_id][other_id]
                    for other_id in client_ids
                    if other_id != client_id
                ]
                if cluster_similarities:  # Check if list is not empty
                    avg_similarity = np.mean(cluster_similarities)
                    if avg_similarity < threshold:
                        outliers.append((client_id, cluster_id))

        return outliers

    def client_update(self, client_model, optimizer, train_loader):
        client_model.train()
        for epoch in range(self.local_epochs):
            for data, labels in train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = client_model(data)
                loss = F.cross_entropy(output, labels)
                loss.backward()
                optimizer.step()
        return client_model.state_dict()

    def cluster_clients(self, client_weights):
        return self.find_optimal_clusters(client_weights)

    def train(self, num_rounds, inject_adversary_at_round=5):  # one function change
        """
        Modified training loop to inject adversarial client at specified round
        """
        best_sil_score = -1
        best_clustering = None
        self.inject_adversary = False

        for round in range(num_rounds):
            print(f"\nRound {round + 1}")

            # Inject adversarial client at specified round
            if round == inject_adversary_at_round:
                print("⚠️ Injecting adversarial client...")
                self.inject_adversary = True

            # Train clients
            client_weights = self.train_round()

            # Cluster and aggregate
            cluster_models, cluster_labels, sil_score, outliers, num_clusters = (
                self.cluster_and_aggregate(client_weights)
            )

            # Print detailed outlier information if found
            if outliers and round >= inject_adversary_at_round:
                print("\n🔍 Outlier Detection Results:")
                print(f"Found {len(outliers)} outlier(s)")
                for client_id, cluster_id in outliers:
                    if client_id == 0:  # Our injected adversarial client
                        print(
                            f"✅ Successfully detected adversarial client {client_id}"
                        )
                        print(f"This client was assigned to cluster {cluster_id}")

                        # Calculate similarity with other clients in the same cluster
                        similarity_matrix = compute_similarity_matrix(client_weights)
                        cluster_clients = [
                            i for i, c in enumerate(cluster_labels) if c == cluster_id
                        ]
                        avg_similarity = np.mean(
                            [similarity_matrix[0][i] for i in cluster_clients if i != 0]
                        )
                        print(f"Average similarity with cluster: {avg_similarity:.4f}")

            # Update best clustering if necessary
            if sil_score > best_sil_score:
                best_sil_score = sil_score
                best_clustering = (cluster_models, cluster_labels)

            # Update global model (excluding outliers)
            global_state_dict = {}
            valid_clients = [
                i
                for i in range(self.num_clients)
                if (i, cluster_labels[i]) not in outliers
            ]

            if valid_clients:
                for k in client_weights[0].keys():
                    global_state_dict[k] = torch.stack(
                        [client_weights[i][k] for i in valid_clients]
                    ).mean(0)

                self.global_model.load_state_dict(global_state_dict)

            # Evaluate and record metrics
            train_accuracy = self.evaluate_model(self.global_model, "train")
            test_accuracy = self.evaluate_model(self.global_model, "test")

            self.train_accuracy_history.append(train_accuracy)
            self.test_accuracy_history.append(test_accuracy)
            self.cluster_history.append(num_clusters)
            self.silhouette_history.append(sil_score)

            print(f"Global Model Train Accuracy: {train_accuracy:.2f}%")
            print(f"Global Model Test Accuracy: {test_accuracy:.2f}%")
            print(f"Number of Clusters: {num_clusters}")
            print(f"Silhouette Score: {sil_score:.4f}")

        # Plot final metrics
        self.plot_metrics()
        return best_clustering

    def cluster_and_aggregate(self, client_weights):
        # Perform clustering with optimal number of clusters
        cluster_labels, sil_score, medoids, num_clusters = self.cluster_clients(
            client_weights
        )

        # Detect outliers
        outliers = self.detect_outliers(client_weights, cluster_labels)

        # Aggregate within clusters
        cluster_models = {}
        for cluster_id in range(num_clusters):
            cluster_clients = [
                i for i, label in enumerate(cluster_labels) if label == cluster_id
            ]

            # Skip empty clusters
            if not cluster_clients:
                continue

            # Average the models in the cluster
            avg_state_dict = {}
            for k in client_weights[cluster_clients[0]].keys():
                avg_state_dict[k] = torch.stack(
                    [
                        client_weights[client_id][k].to(self.device)
                        for client_id in cluster_clients
                    ]
                ).mean(0)

            cluster_models[cluster_id] = avg_state_dict

        return cluster_models, cluster_labels, sil_score, outliers, num_clusters

    def create_adversarial_client(self, client_id):  # third function change
        """
        Create an adversarial client by significantly altering its model weights
        """
        # Get the original model state
        original_state = self.client_models[client_id].state_dict()

        # Create a corrupted state dictionary
        corrupted_state = {}
        for key in original_state:
            # Add significant random noise to weights
            noise = torch.randn_like(original_state[key]) * 4.0
            corrupted_state[key] = original_state[key] + noise

        return corrupted_state

    def plot_metrics(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

        # Plot accuracies
        ax1.plot(self.train_accuracy_history, label="Train Accuracy")
        ax1.plot(self.test_accuracy_history, label="Test Accuracy")
        ax1.set_title("Model Accuracy over Rounds")
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Accuracy (%)")
        ax1.legend()

        # Plot number of clusters
        ax2.plot(self.cluster_history)
        ax2.set_title("Number of Clusters over Rounds")
        ax2.set_xlabel("Round")
        ax2.set_ylabel("Number of Clusters")

        # Plot silhouette scores
        ax3.plot(self.silhouette_history)
        ax3.set_title("Silhouette Score over Rounds")
        ax3.set_xlabel("Round")
        ax3.set_ylabel("Silhouette Score")

        plt.tight_layout()
        plt.savefig("federated_learning_metrics.png")
        plt.close()


def compute_similarity_matrix(client_weights):
    n_clients = len(client_weights)
    similarity_matrix = np.zeros((n_clients, n_clients))

    # Convert weights to vectors
    weight_vectors = [flatten_weights(weights) for weights in client_weights.values()]

    # Compute similarity matrix
    for i in range(n_clients):
        for j in range(n_clients):
            similarity = F.cosine_similarity(
                weight_vectors[i].unsqueeze(0), weight_vectors[j].unsqueeze(0)
            ).item()
            similarity_matrix[i][j] = similarity

    return similarity_matrix

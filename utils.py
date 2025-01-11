import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import cdist


# Filter data by users
def filter_data_by_users(data, user_list):
    return data[data['user_id'].isin(user_list)]


# Ensure full anime coverage in training set
def ensure_full_coverage(source_data, target_data):
    source_anime_ids = set(source_data['anime_id'])
    target_anime_ids = set(target_data['anime_id'])
    missing_anime_ids = source_anime_ids - target_anime_ids

    if missing_anime_ids:
        print(f"Missing anime IDs in target set: {missing_anime_ids}")
        # Find users with missing anime IDs
        users_with_missing_anime = source_data[source_data['anime_id'].isin(missing_anime_ids)]['user_id'].unique()
        # Add those users' data to the target set
        extra_data = source_data[source_data['user_id'].isin(users_with_missing_anime)]
        target_data = pd.concat([target_data, extra_data]).drop_duplicates()
        # Remove extra data from source_data
        source_data = source_data[~source_data['user_id'].isin(users_with_missing_anime)]

    return target_data, source_data


def calculate_mbp_label_distribution(train_loader, num_classes):
    # Initialize a dictionary to count occurrences of each class
    label_counts = {i: 0 for i in range(num_classes)}

    # Iterate over the dataset
    for _, _, labels, _ in train_loader:
        for label in labels:
            label_counts[label.item()] += 1

    total_labels = sum(label_counts.values())

    # Calculate the proportion of each class
    label_distribution = {key: value / total_labels for key, value in label_counts.items()}
    return label_distribution


def calculate_nbp_label_distribution(train_loader, num_classes, K):
    # Initialize a dictionary to count occurrences of each class for each label position
    label_counts = [{i: 0 for i in range(num_classes)} for _ in range(K)]

    # Iterate over the dataset
    for _, _, labels, _ in train_loader:
        for k in range(K):
            for label in labels[:, k]:
                label_counts[k][label.item()] += 1

    # Calculate total labels for each position
    total_labels_per_position = [sum(counts.values()) for counts in label_counts]

    # Calculate the proportion of each class for each position
    label_distribution = [
        {key: value / total_labels_per_position[k] for key, value in counts.items()}
        for k, counts in enumerate(label_counts)
    ]

    label_distribution = {key: np.mean([label_distribution[0][key], label_distribution[1][key]]) for key in label_distribution[0].keys()}

    return label_distribution


def calculate_accuracy(scores, labels):

    probs = F.softmax(scores, dim=-1)
    predictions = torch.argmax(probs, dim=-1)

    correct = (predictions == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy


def plot_loss(train_loss, valid_loss):

    fig, ax = plt.subplots(figsize = (12, 7))

    ax.plot(train_loss, label = 'Train loss')
    ax.plot(valid_loss, label = 'Valid loss')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Train vs Validation losses')
    ax.set_xticks(np.arange(len(train_loss)), np.arange(1, len(train_loss)+1))

    ax.grid(True)
    ax.legend()

    plt.show()


def infer_single_user(model, user_history, candidate_animes, max_seq_len, device):
    """
    Make a prediction for a single user.

    Args:
        model (nn.Module): Trained model.
        user_history (list): List of anime IDs representing the user's watch history.
        candidate_animes (list): List of anime IDs to predict from.
        max_seq_len (int): Maximum sequence length for padding/truncation.
        device (torch.device): Device to run inference on.

    Returns:
        int: Index of the predicted anime in the candidate_animes list.
    """
    model.eval()

    # Prepare user history sequence
    padded_history = user_history[:max_seq_len] + [0] * max(0, max_seq_len - len(user_history))
    history_tensor = torch.tensor(padded_history, dtype=torch.long).unsqueeze(0).to(device)  # Shape: [1, seq_len]

    # Prepare candidate animes
    candidate_tensor = torch.tensor(candidate_animes, dtype=torch.long).unsqueeze(0).to(device)  # Shape: [1, K, num_candidates]

    # Create padding mask for user history
    padding_mask = torch.tensor(
        [1] * min(len(user_history), max_seq_len) + [0] * max(0, max_seq_len - len(user_history)),
        dtype=torch.bool
    ).unsqueeze(0).to(device)  # Shape: [1, seq_len]

    # Run inference
    with torch.no_grad():
        scores = model(history_tensor, candidate_tensor, padding_mask)  # Shape: [1, K, num_candidates]
        scores = scores.view(-1, scores.size(-1))
        probs = F.softmax(scores, dim=-1)
        predictions = probs.argmax(dim=-1)

    return predictions


def generate_user_embedding(model, user_history, max_seq_len, device):
    """
    Generate a single user's embedding.

    Args:
        model (nn.Module): Trained model.
        user_history (list): List of anime IDs representing the user's watch history.
        max_seq_len (int): Maximum sequence length for padding/truncation.
        device (torch.device): Device to run inference on.

    Returns:
        np.ndarray: User embedding as a 1D NumPy array.
    """
    model.eval()

    # Prepare user history sequence
    padded_history = user_history[:max_seq_len] + [0] * max(0, max_seq_len - len(user_history))
    history_tensor = torch.tensor(padded_history, dtype=torch.long).unsqueeze(0).to(device)  # Shape: [1, seq_len]

    # Create padding mask for user history
    padding_mask = torch.tensor(
        [1] * min(len(user_history), max_seq_len) + [0] * max(0, max_seq_len - len(user_history)),
        dtype=torch.bool
    ).unsqueeze(0).to(device)  # Shape: [1, seq_len]

    # Extract user embedding
    with torch.no_grad():
        behavior_embeddings = model.anime_embedding(history_tensor)
        user_embeddings = model.user_encoder(behavior_embeddings, src_key_padding_mask=~padding_mask)
        user_embedding = model.calculate_mean_embedding(user_embeddings, padding_mask)

    return user_embedding.squeeze(0).cpu().numpy()


def save_user_embeddings_to_dataframe(model, holdout_users, user_behaviors, max_seq_len, device):
    """
    Generate and save user embeddings for all users in the holdout set.

    Args:
        model (nn.Module): Trained model.
        holdout_users (list): List of user IDs in the holdout set.
        user_behaviors (dict): Dictionary mapping user IDs to their watch histories.
        max_seq_len (int): Maximum sequence length for padding/truncation.
        device (torch.device): Device to run inference on.

    Returns:
        pd.DataFrame: DataFrame containing user embeddings.
    """

    user_ids = []
    embeddings = []

    for user_id in holdout_users:
        user_history = user_behaviors.get(user_id, [])
        if len(user_history) == 0:
            continue  # Skip users with no history

        embedding = generate_user_embedding(model, user_history, max_seq_len, device)
        user_ids.append(user_id)
        embeddings.append(embedding)

    # Create DataFrame
    embedding_dim = len(embeddings[0])
    column_names = ['user_id'] + [f'dim_{i}' for i in range(embedding_dim)]
    data = [[user_id] + list(embedding) for user_id, embedding in zip(user_ids, embeddings)]

    return pd.DataFrame(data, columns=column_names)


def evaluate_with_clustering(embeddings, num_clusters=10):
    """
    Perform K-means clustering on embeddings and visualize using t-SNE.
    
    Args:
        embeddings (np.ndarray): User embeddings of shape (num_users, embedding_dim).
        num_clusters (int): Number of clusters for K-means.
        
    Returns:
        None (displays plot).
    """

    kmeans = KMeans(n_init = 'auto', init = 'k-means++', n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        hue=labels,
        palette=sns.color_palette("tab10", num_clusters),
        legend="full",
        s=50
    )
    plt.title("K-means Clustering Visualization with t-SNE")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()


def calculate_clustering_metrics(embeddings, num_clusters=10):
    """
    Calculate clustering metrics for the given embeddings.
    
    Args:
        embeddings (np.ndarray): User embeddings (num_users, embedding_dim).
        num_clusters (int): Number of clusters for K-means.
        
    Returns:
        dict: A dictionary containing clustering metrics.
    """

    kmeans = KMeans(n_init = 'auto', init = 'k-means++', n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    silhouette = silhouette_score(embeddings, labels)

    # Calinski-Harabasz Index
    calinski_harabasz = calinski_harabasz_score(embeddings, labels)

    # Davies-Bouldin Index
    davies_bouldin = davies_bouldin_score(embeddings, labels)

    # Intra-cluster distance
    intra_distances = [
        np.mean(cdist(embeddings[labels == cluster], [kmeans.cluster_centers_[cluster]], metric='euclidean'))
        for cluster in range(num_clusters)
    ]
    mean_intra_distance = np.mean(intra_distances)

    # Inter-cluster distance
    inter_distances = cdist(kmeans.cluster_centers_, kmeans.cluster_centers_, metric='euclidean')
    mean_inter_distance = np.mean(inter_distances[np.triu_indices(num_clusters, k=1)])

    # Dunn Index
    min_inter_cluster_distance = np.min(inter_distances[np.triu_indices(num_clusters, k=1)])
    max_intra_cluster_distance = np.max(intra_distances)
    dunn_index = min_inter_cluster_distance / max_intra_cluster_distance if max_intra_cluster_distance > 0 else 0

    metrics = {
        "Silhouette Score": silhouette,
        "Calinski-Harabasz Index": calinski_harabasz,
        "Davies-Bouldin Index": davies_bouldin,
        "Mean Intra-Cluster Distance": mean_intra_distance,
        "Mean Inter-Cluster Distance": mean_inter_distance,
        "Dunn Index": dunn_index
    }
    
    return metrics
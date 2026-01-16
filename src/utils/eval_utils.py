## COPYRIGHT NOTICE
# Parts of the following code cell was inspired by the following repository: https://github.com/luk27official/cryptoshow-benchmark/
# For further info see the LICENSE file.


import csv
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
sys.path.append('/home/skrhakv/cryptoshow-analysis/src/utils')
sys.path.append('/home/vit/Projects/cryptoshow-analysis/src/utils')
import cryptoshow_utils

DATA_PATH = '/home/skrhakv/cryptoshow-analysis/data/A-cluster-ligysis-data/cryptobench-clustered-binding-sites.txt'
SMOOTHING_DECISION_THRESHOLD = 0.4 # see src/C-optimize-smoother/classifier-for-cryptoshow.ipynb (https://github.com/skrhakv/cryptic-finetuning)
DCC_THRESHOLD = 4.0  # in Angstroms


def read_test_binding_residues(data_path=DATA_PATH, pocket_types=['CRYPTIC']) -> set[int]:
    cryptic_binding_residues = {}
    sequences = {}

    with open(data_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            chain_id = row[0][4:]
            protein_id = f'{row[0][:4]}_{chain_id}'
            sequence = row[4]

            if row[3] == '':
                continue

            binding_residue_indices = [f'{chain_id}_{int(i[1:])}'for i in row[3].split(' ')]
            if row[2] in pocket_types:
                if protein_id not in cryptic_binding_residues:
                    cryptic_binding_residues[protein_id] = []
                cryptic_binding_residues[protein_id].append(binding_residue_indices)
            sequences[protein_id] = sequence

    return cryptic_binding_residues, sequences

def print_plots(DCCs, coverages, dice_coefficients, binding_prediction_scores, number_of_pockets, model, dcc_threshold=DCC_THRESHOLD):
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

    _, axs = plt.subplots(1, 2, figsize=(16, 6))

    sns.histplot(DCCs, bins=20, color='skyblue', edgecolor='black', alpha=0.7, ax=axs[0])
    axs[0].set_title(f'{model}: Pocket Center Distance (DCC) distribution\n(median={np.median(DCCs):.2f} Å), DCC ≤ {dcc_threshold} Å: {np.sum(np.array(DCCs) < dcc_threshold)} / {number_of_pockets}', fontsize=12, fontweight='bold')
    axs[0].set_xlabel('DCC (Å)', fontsize=11)
    axs[0].set_ylabel('Count', fontsize=11)
    axs[0].set_xlim(0, 50)

    sns.histplot(coverages, bins=20, color='salmon', edgecolor='black', alpha=0.7, ax=axs[1])
    axs[1].set_title(f'{model}: Residues Covered Percentage\n(median={np.median(coverages):.1f}%)', fontsize=12, fontweight='bold')
    axs[1].set_xlabel('Residues Covered (%)', fontsize=11)
    axs[1].set_ylabel('Count', fontsize=11)

    plt.tight_layout()
    plt.show()

    _, axs = plt.subplots(1, 2, figsize=(16, 6))

    sns.histplot(dice_coefficients, bins=20, color='mediumseagreen', edgecolor='black', alpha=0.7, ax=axs[0])
    axs[0].set_title(f'{model}: Dice Coefficient Distribution\n(median={np.median(dice_coefficients):.2f})', fontsize=12, fontweight='bold')
    axs[0].set_xlabel('Dice Coefficient', fontsize=11)
    axs[0].set_ylabel('Count', fontsize=11)

    sns.histplot(binding_prediction_scores, bins=20, color='gold', edgecolor='black', alpha=0.7, ax=axs[1])
    axs[1].axvline(0.7, color='red', linestyle='--', linewidth=2, label='Threshold = 0.7')
    axs[1].set_title(f'{model}: Prediction Scores of Test Set CBSs\n(median={np.median(binding_prediction_scores):.2f})', fontsize=12, fontweight='bold')
    axs[1].set_xlabel('Prediction Score', fontsize=11)
    axs[1].set_ylabel('Count', fontsize=11)
    axs[1].legend(fontsize=10)

    plt.tight_layout()
    plt.show()


def compute_pocket_level_metrics(binding_residues, predicted_binding_sites, prediction_scores, coordinates_dir, output=False):
    DCCs = []
    coverages = []
    dice_coefficients = []
    binding_prediction_scores = []
    number_of_pockets = 0
    for protein_id in binding_residues.keys():
        number_of_pockets += len(binding_residues[protein_id])
        
        assert len(binding_residues[protein_id]) > 0, f"No binding residues for protein_id: {protein_id}"

        coordinates = np.load(f'{coordinates_dir}/{protein_id.replace("_", "")}.npy')

        # loop over each cryptic binding site
        for actual_cryptic_binding_residue_indices in binding_residues[protein_id]:
            dcc = float('inf')
            actual_cryptic_binding_residue_indices = [int(i.split('_')[1]) for i in actual_cryptic_binding_residue_indices]
    
            # loop over each predicted binding site and select the one with the lowest DCC
            for (predicted_cryptic_binding_residue_indices, tool) in predicted_binding_sites[protein_id]:
                this_dcc = cryptoshow_utils.compute_center_distance(coordinates, actual_cryptic_binding_residue_indices, predicted_cryptic_binding_residue_indices)
                if this_dcc < dcc:
                    dcc = this_dcc
                    tool_used = tool

            if dcc != float('inf'):
                DCCs.append(dcc)
                if output:
                    print(f'Protein ID: {protein_id}, DCC: {dcc:.2f} Å predicted by {tool_used}')
                    # if dcc < 4.0:
                    #     print(f'Protein ID: {protein_id}, DCC: {dcc:.2f} Å predicted by {tool_used}')

        concatenated_cryptic_binding_residues = np.array(np.unique(np.concatenate(binding_residues[protein_id])))
        concatenated_cryptic_binding_residues = [int(i.split('_')[1]) for i in concatenated_cryptic_binding_residues]
        concatenated_predicted_binding_residues = np.array(np.unique(np.concatenate([i[0]for i in predicted_binding_sites[protein_id]]))) if len(predicted_binding_sites[protein_id]) > 0 else np.array([])
        residues_covered = np.intersect1d(np.array(concatenated_cryptic_binding_residues), concatenated_predicted_binding_residues)

        this_coverage = len(residues_covered) / len(concatenated_cryptic_binding_residues) * 100
        this_dice_coefficient = 2 * len(residues_covered) / (len(concatenated_cryptic_binding_residues) + len(concatenated_predicted_binding_residues)) if (len(concatenated_cryptic_binding_residues) + len(concatenated_predicted_binding_residues)) > 0 else 0

        coverages.append(this_coverage)
        dice_coefficients.append(this_dice_coefficient)
        if protein_id in prediction_scores:
            binding_prediction_scores.extend(prediction_scores[protein_id][concatenated_cryptic_binding_residues])
    return DCCs, coverages, dice_coefficients, binding_prediction_scores, number_of_pockets


PREDICTED_RESIDUE_RADIUS_DISTANCE_THRESHOLD = 10 # in Angstroms; for each predicted binding residue, we consider all residues within this distance as candidate residues for inclusion into the binding site
CANDIDATE_RESIDUE_SURROUNDING_RADIUS_THRESHOLD = 15 # in Angstroms; for each candidate residue that is considered for inclusion into the binding site, we define the surrounding binding site respresentation as 
                                                    # the mean of embeddings of all predicted binding residues within this distance

def process_single_sequence(structure_name: str, 
                            chain_id: str, 
                            binding_residues_indices: np.ndarray, 
                            embedding_path: str, 
                            distance_matrix: np.ndarray,
                            candidate_radius_threshold=CANDIDATE_RESIDUE_SURROUNDING_RADIUS_THRESHOLD):
    id = structure_name.lower() + chain_id
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f'Embedding file for {id} not found in {embedding_path}')
    
    embedding = np.load(embedding_path)

    Xs = []
    idx = []
    
    candidate_residues_indices = set()
    
    for residue_idx in binding_residues_indices:
        close_residues_indices = np.where(distance_matrix[residue_idx] < PREDICTED_RESIDUE_RADIUS_DISTANCE_THRESHOLD)[0]
        close_binding_residues_indices = np.intersect1d(close_residues_indices, binding_residues_indices)

        candidate_residues_indices.update(set(list(close_residues_indices)) - set(list(binding_residues_indices)))

    for residue_idx in candidate_residues_indices:
        close_residues_indices = np.where(distance_matrix[residue_idx] < candidate_radius_threshold)[0]
        close_binding_residues_indices = np.intersect1d(close_residues_indices, binding_residues_indices)

        concatenated_embedding = np.concatenate((embedding[residue_idx], np.mean(embedding[close_binding_residues_indices], axis=0)))
        Xs.append(concatenated_embedding)
        idx.append(residue_idx)
        
    return np.array(Xs), np.array(idx)

def predict_single_sequence(Xs, idx, model_3):
    import torch
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Xs = torch.tensor(Xs, dtype=torch.float32).to(DEVICE)
    idx = torch.tensor(idx, dtype=torch.int64).to(DEVICE)

    test_logits = model_3(Xs).squeeze()
    test_pred = torch.sigmoid(test_logits)

    return {'predictions': test_pred.detach().cpu().numpy(), 'indices': idx.detach().cpu().numpy()}

def compute_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise distance matrix for a given set of coordinates.

    Args:
        coordinates (np.ndarray): A 2D array of shape (N, 3), where N is the number of points,
                                   and each row represents the (x, y, z) coordinates of a point.

    Returns:
        np.ndarray: A 2D array of shape (N, N), where the element at [i, j] represents the Euclidean
                    distance between the i-th and j-th points.
    """
    coordinates = np.array(coordinates)
    distance_matrix = np.linalg.norm(coordinates[:, np.newaxis] - coordinates[np.newaxis, :], axis=-1)
    return distance_matrix

DROPOUT = 0.5
LAYER_WIDTH = 2048
ESM2_DIM = 2560
INPUT_DIM  = ESM2_DIM * 2
import torch.nn as nn

class CryptoBenchClassifier(nn.Module):
    def __init__(self, dim=LAYER_WIDTH, dropout=DROPOUT):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=INPUT_DIM, out_features=dim)
        self.dropout1 = nn.Dropout(dropout)

        self.layer_2 = nn.Linear(in_features=dim, out_features=dim)
        self.dropout2 = nn.Dropout(dropout)

        self.layer_3 = nn.Linear(in_features=dim, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):
      # Intersperse the ReLU activation function between layers
       return self.layer_3(self.dropout2(self.relu(self.layer_2(self.dropout1(self.relu(self.layer_1(x)))))))

DECISION_THRESHOLD = 0.7  # Threshold to consider a point as high score; see src/decision-thresholds.ipynb (https://github.com/skrhakv/cryptic-finetuning)


EPS = 5.0  # Max distance for neighbors
MIN_SAMPLES = 3  # Min points to form a cluster

def compute_clusters(
    points: list[list[float]],
    prediction_scores: list[float],
    decision_threshold: float = DECISION_THRESHOLD,
    eps=EPS,
    min_samples=MIN_SAMPLES,
    method='dbscan',
    merge_threshold=None,
    embeddings_path=None,
):
    from sklearn.cluster import DBSCAN
    """
    Compute clusters based on the given points and prediction scores.

    Args:
        points (list[list[float]]): A list of points, where each point is a list of 3 coordinates [x, y, z].
        prediction_scores (list[float]): A list of prediction scores corresponding to each point.
        decision_threshold (float): The threshold above which points are considered as positive.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered

    Returns:
        np.ndarray: An array of cluster labels for each point. Points with no cluster are labeled as -1.
    """
    
    if method not in ['dbscan', 'meanshift', 'kmeans', 'gmm', 'pca+gmm+agglomerative']:
        raise ValueError(f"Unsupported clustering method: {method}. Supported methods are 'dbscan', 'meanshift', and 'kmeans'.")
    
    prediction_scores = prediction_scores.reshape(-1, 1)
    stacked = np.hstack((points, prediction_scores))  # Combine coordinates with scores

    high_score_mask = stacked[:, 3] > decision_threshold
    high_score_points = stacked[high_score_mask][:, :3]  # Extract only (x, y, z) coordinates

    # No pockets can be formed if there are not enough high score points.
    if len(high_score_points) < min_samples:
        return -1 * np.ones(len(points), dtype=int)

    if method == 'dbscan':
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
    elif method == 'meanshift':
        from sklearn.cluster import MeanShift
        clustering = MeanShift(bandwidth=eps)
    elif method == 'gmm':
        from sklearn.mixture import BayesianGaussianMixture

        # get the optimal number of components
        bgmm = BayesianGaussianMixture(
            n_components=max(len(high_score_points) - 1, 1), 
            random_state=42,
            covariance_type='spherical',
        )

        bgmm.fit(high_score_points)
        labels = bgmm.predict(high_score_points)

        active_clusters = sum(bgmm.weights_ > 0.1) # Check how many clusters are actually used - how many are composed of >10% of points
        clustering = BayesianGaussianMixture(
            n_components=active_clusters, 
            random_state=42,
            covariance_type='spherical',
        )
    elif method == 'pca+gmm+agglomerative':
        assert embeddings_path is not None, "embeddings_path must be provided for 'gmm+agglomerative' method"
        
        embeddings = np.load(embeddings_path)

        if embeddings.shape[0] != stacked.shape[0]:
            print('Mismatch between number of embeddings and number of points')
            return -1 * np.ones(len(points), dtype=int)
        
        high_score_points_embeddings = embeddings[high_score_mask]

        from sklearn.mixture import BayesianGaussianMixture
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.decomposition import PCA

        pca = PCA(n_components=min(5, high_score_points_embeddings.shape[0]), random_state=42)
        projections = pca.fit_transform(high_score_points_embeddings)

        concatenated_features =  np.concatenate((projections, high_score_points), axis=1)

        bgmm = BayesianGaussianMixture(
            n_components=len(concatenated_features), 
            random_state=42,
            covariance_type='spherical',
            weight_concentration_prior=1e-15,
        )   

        bgmm.fit(concatenated_features)    

        active_clusters = sum(bgmm.weights_ > 0.1)

        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(n_clusters=max(active_clusters, 1), linkage='single')


    labels = clustering.fit_predict(high_score_points)

    if merge_threshold is not None:
        from scipy.spatial.distance import cdist
        from scipy.sparse.csgraph import connected_components

        if method == 'gmm':
            centers = clustering.means_ 
        else:
            centers = []
            for label in np.unique(labels):
                if label == -1:
                    continue
                cluster_residues = np.where(labels == label)[0]
                assert len(labels) == len(high_score_points), "Labels length must match high_score_points length"
                centers.append(np.mean(high_score_points[cluster_residues], axis=0))
        
        # Calculate distance between all cluster centers
        # shape: (n_clusters, n_clusters)
        distances = cdist(centers, centers)
        
        # Create an adjacency matrix (True if close, False if far)
        # This creates a "graph" where clusters are nodes and closeness is an edge
        adjacency_matrix = distances < merge_threshold
        
        # Find connected components (groups of clusters to merge)
        # n_merged: total number of new super-clusters
        # new_mapping: array where index is old label, value is new label
        n_merged, new_mapping = connected_components(adjacency_matrix, directed=False)
        
        # Apply the new mapping to your data points
        final_labels = new_mapping[labels]
        labels = final_labels
    
    # Initialize all labels to -1
    all_labels = -1 * np.ones(len(points), dtype=int)
    # Assign cluster labels to high score points
    all_labels[high_score_mask] = labels
    labels = all_labels

    return labels

MAX_LENGTH = 1024
SEQUENCE_MAX_LENGTH = MAX_LENGTH - 2


def compute_prediction(sequence: str, emb_path: str, model, tokenizer) -> np.ndarray:
    """
    Compute the residue-level prediction using the CryptoBench model.

    Args:
        sequence (str): Sequence of amino acids to be predicted.

    Returns:
        np.ndarray: The predicted scores for each residue.
    """
    import torch
    from transformers import AutoTokenizer
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()

    sequence = str(sequence)
    all_embeddings = []
    final_output = []

    # Process sequence in chunks of SEQUENCE_MAX_LENGTH
    for i in range(0, len(sequence), SEQUENCE_MAX_LENGTH):
        processed_sequence = sequence[i : i + SEQUENCE_MAX_LENGTH]

        tokenized = tokenizer(
            processed_sequence, max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt"
        )
        tokenized = {k: v.to(DEVICE) for k, v in tokenized.items()}

        # embeddings
        with torch.no_grad():
            llm_output = model.llm(input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"])
            embeddings = llm_output.last_hidden_state  # shape: (1, seq_len, hidden_dim)

        embeddings_np = embeddings.squeeze(0).detach().cpu().numpy()
        mask = tokenized["attention_mask"].squeeze(0).detach().cpu().numpy().astype(bool)
        embeddings_np = embeddings_np[mask][1:-1]  # exclude [CLS], [SEP]
        all_embeddings.append(embeddings_np)

        # prediction
        with torch.no_grad():
            output = model(tokenized)
            
            if isinstance(output, tuple):
                output = output[0]
                
        output = output.squeeze(0)
        mask = tokenized["attention_mask"].squeeze(0).bool()
        output = output[mask][1:-1]  # exclude [CLS], [SEP]

        probabilities = torch.sigmoid(output).detach().cpu().numpy()
        final_output.extend(probabilities)

    # save the concatenated embeddings for the entire sequence
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    save_path = os.path.join(emb_path)
    np.save(save_path, final_embeddings)

    return np.array(final_output).flatten()


def compute_esm_embeddings(sequence: str, emb_path: str, device) -> np.ndarray:
    """
    Compute the ESM-2 embeddings for a given sequence.

    Args:
        sequence (str): Sequence of amino acids to be embedded.
        emb_path (str): Path to save the embeddings.
        device: Device to run the model on.

    Returns:
        np.ndarray: The computed embeddings for each residue.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = "facebook/esm2_t33_3B_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []

    # Process sequence in chunks of SEQUENCE_MAX_LENGTH
    for i in range(0, len(sequence), SEQUENCE_MAX_LENGTH):
        processed_sequence = sequence[i : i + SEQUENCE_MAX_LENGTH]

        tokenized = tokenizer(
            processed_sequence, max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt"
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        # embeddings
        with torch.no_grad():
            llm_output = model(input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"])
            embeddings = llm_output.last_hidden_state  # shape: (1, seq_len, hidden_dim)

        embeddings_np = embeddings.squeeze(0).detach().cpu().numpy()
        mask = tokenized["attention_mask"].squeeze(0).detach().cpu().numpy().astype(bool)
        embeddings_np = embeddings_np[mask][1:-1]  # exclude [CLS], [SEP]
        all_embeddings.append(embeddings_np)

    # save the concatenated embeddings for the entire sequence
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    save_path = os.path.join(emb_path)
    np.save(save_path, final_embeddings)

    return final_embeddings
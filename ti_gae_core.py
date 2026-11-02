"""TI-GAE: original encoder, decoder, reconstruction loss and k selection."""
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def choose_k_by_silhouette(X: np.ndarray, k_min: int=2, k_max: int=None, random_state: int=42):
    n = len(X)
    if n < 3:
        print('样本数 < 3，无法计算轮廓系数，回退到 k=2。')
        return (2, {})
    if k_max is None:
        k_max = min(10, n - 1)
    scores = {}
    best_k, best_score = (None, -1.0)
    for k in range(max(2, k_min), k_max + 1):
        if k >= n:
            break
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(X)
        if len(set(labels)) < 2 or len(set(labels)) >= n:
            continue
        score = silhouette_score(X, labels)
        scores[k] = score
        if score > best_score:
            best_k, best_score = (k, score)
    if best_k is None:
        best_k = 2
    return (best_k, scores)

def prepare_subset_data(raw: np.ndarray, indices: List[int], std_floor: float):
    """Historical per-station normalization, with statistics for checkpointing."""
    raw_combo = raw[:, :, indices].astype(np.float32)
    node_zero_mask = np.all(raw_combo == 0.0, axis=-1)
    flood_flag = (~node_zero_mask).astype(np.float32)[..., None]
    valid = (~node_zero_mask).astype(np.float64)[..., None]
    raw64 = raw_combo.astype(np.float64)
    count = valid.sum(axis=0)
    denominator = np.maximum(count, 1.0)
    mean = (raw64 * valid).sum(axis=0) / denominator
    difference = raw64 - mean[None, :, :]
    variance = ((difference * valid) ** 2).sum(axis=0) / denominator
    std = np.sqrt(variance)
    std = np.where((~np.isfinite(std)) | (std < std_floor), 1.0, std)
    normalized = ((raw64 - mean[None, :, :]) / (std[None, :, :] + 1e-8)).astype(np.float32)
    inputs = np.concatenate([normalized, flood_flag], axis=-1).astype(np.float32)
    return inputs, normalized.astype(np.float32), flood_flag.astype(np.float32), mean, std


class DirectedGraphAE(nn.Module):
    """
    连续特征：回归重建（MSE），训练时只在 flood_flag=1 的节点上计算 MSE；
    flag：BCEWithLogitsLoss（二分类）。
    """

    def __init__(self, node_dim: int, cont_dim: int, hidden_dim: int, graph_embed_dim: int):
        super().__init__()
        self.cont_dim = cont_dim
        self.conv1 = GATConv(node_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.graph_proj = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, graph_embed_dim))
        out_dim = 11 * (cont_dim + 1)
        self.decoder = nn.Sequential(nn.Linear(graph_embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        batch_vec = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        mean_pool = global_mean_pool(x, batch=batch_vec)
        max_pool = global_max_pool(x, batch=batch_vec)
        sum_pool = torch.sum(x, dim=0, keepdim=True)
        pooled = torch.cat([mean_pool, max_pool, sum_pool], dim=-1)
        graph_embed = self.graph_proj(pooled)
        return graph_embed

    def decode(self, graph_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.decoder(graph_embed)
        out = out.view(11, self.cont_dim + 1)
        cont_recon = out[:, :self.cont_dim]
        flag_logits = out[:, self.cont_dim:].view(11, 1)
        return (cont_recon, flag_logits)

def masked_mse_on_flag(cont_recon: torch.Tensor, y_cont: torch.Tensor, y_flag: torch.Tensor) -> torch.Tensor:
    """
    只在 flood_flag=1 的节点上计算连续特征 MSE。
    cont_recon: [11, K]
    y_cont:     [11, K]
    y_flag:     [11, 1]  (0/1)
    """
    mask = (y_flag > 0.5).float()
    mask = mask.expand_as(y_cont)
    diff2 = (cont_recon - y_cont) ** 2
    denom = mask.sum()
    loss = (diff2 * mask).sum() / (denom + 1e-08)
    return loss

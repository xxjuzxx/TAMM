from __future__ import annotations

import time
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

from src.data.label_policy import merged_cicids_label
from src.evaluation.metrics import classification_metrics, confusion, report_dict
from src.features.token_alias import canonical_token
from src.models.behavior_composer import BehaviorComposer, resolve_pooling_config
from src.utils.seed import set_seed


@dataclass
class TrainResult:
    metrics: dict[str, Any]
    report: dict[str, Any]
    confusion_matrix: list[list[int]]
    history: list[dict[str, float]]
    state_dict: dict[str, Any]
    predictions: list[dict[str, Any]]


@dataclass
class PrototypeBank:
    global_prototype: np.ndarray
    service_prototypes: dict[tuple[str, ...] | None, np.ndarray] = field(default_factory=dict)
    class_prototypes: dict[int, np.ndarray] = field(default_factory=dict)
    score_method: str = "cosine"
    normalize: str = "l2"
    include_special: bool = False
    prototype_scope: str = "benign_train"
    benign_label_id: int | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class StatFeatureNormalizer:
    names: list[str]
    mean: np.ndarray
    std: np.ndarray
    missing_names: list[str] = field(default_factory=list)


class ContextAwareClassifier(nn.Module):
    def __init__(
        self,
        encoder: BehaviorComposer,
        hidden_size: int,
        num_classes: int,
        context_size: int = 6,
        dropout: float = 0.1,
        anomaly_size: int = 0,
        anomaly_feature_dim: int = 0,
        use_hierarchical_classifier: bool = False,
        stat_size: int = 0,
        stat_mlp_dim: int = 0,
        use_app_projection: bool = False,
        app_projection_dim: int = 0,
        app_projection_input: str = "shared_encode",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.use_class_aware_pooling = bool(getattr(encoder, "class_aware_pooling", False))
        self.context_size = int(context_size)
        self.anomaly_size = int(anomaly_size)
        self.anomaly_feature_dim = int(anomaly_feature_dim)
        self.stat_size = int(stat_size)
        self.stat_mlp_dim = int(stat_mlp_dim)
        self.use_hierarchical_classifier = bool(use_hierarchical_classifier)
        self.use_app_projection = bool(use_app_projection)
        self.app_projection_dim = int(app_projection_dim) if int(app_projection_dim) > 0 else int(hidden_size)
        self.app_projection_input = str(app_projection_input)
        if self.app_projection_input not in {"shared_encode", "classification_embedding", "class_aware_summary", "residual_class_aware_pool"}:
            raise ValueError(f"Unsupported app_projection_input: {self.app_projection_input}")
        self.has_context = self.context_size > 0
        self.has_anomaly = self.anomaly_size > 0 and self.anomaly_feature_dim > 0
        self.has_stats = self.stat_size > 0 and self.stat_mlp_dim > 0
        if self.use_app_projection:
            self.app_projection = nn.Sequential(
                nn.Linear(hidden_size, self.app_projection_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.app_projection_dim, self.app_projection_dim),
                nn.LayerNorm(self.app_projection_dim),
            )
        if self.has_context:
            self.context_proj = nn.Sequential(
                nn.LayerNorm(self.context_size),
                nn.Linear(self.context_size, hidden_size),
                nn.GELU(),
            )
        if self.has_anomaly:
            self.anomaly_proj = nn.Sequential(
                nn.LayerNorm(self.anomaly_size),
                nn.Linear(self.anomaly_size, self.anomaly_feature_dim),
                nn.GELU(),
            )
        if self.has_stats:
            self.stat_proj = nn.Sequential(
                nn.LayerNorm(self.stat_size),
                nn.Linear(self.stat_size, self.stat_mlp_dim),
                nn.GELU(),
            )
        base_size = self.app_projection_dim if self.use_app_projection else hidden_size
        fusion_size = base_size
        if self.has_context:
            fusion_size += hidden_size
        if self.has_anomaly:
            fusion_size += self.anomaly_feature_dim
        if self.has_stats:
            fusion_size += self.stat_mlp_dim
        self.fusion_size = int(fusion_size)
        if self.use_class_aware_pooling and not self.use_app_projection:
            self.class_norm = nn.LayerNorm(fusion_size)
            self.class_dropout = nn.Dropout(dropout)
            self.class_logit_weight = nn.Parameter(torch.empty(num_classes, fusion_size))
            self.class_logit_bias = nn.Parameter(torch.zeros(num_classes))
            nn.init.normal_(self.class_logit_weight, mean=0.0, std=fusion_size**-0.5)
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(fusion_size),
                nn.Dropout(dropout),
                nn.Linear(fusion_size, num_classes),
            )
        if self.use_hierarchical_classifier:
            self.binary_classifier = nn.Sequential(
                nn.LayerNorm(fusion_size),
                nn.Dropout(dropout),
                nn.Linear(fusion_size, 2),
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        context_features: torch.Tensor | None = None,
        anomaly_features: torch.Tensor | None = None,
        stat_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_app_projection:
            features = self.fused_features(input_ids, attention_mask, token_type_ids, context_features, anomaly_features, stat_features)
            return self.classifier(features)
        if self.use_class_aware_pooling:
            features = self.class_fused_features(input_ids, attention_mask, token_type_ids, context_features, anomaly_features, stat_features)
            features = self.class_dropout(self.class_norm(features))
            return (features * self.class_logit_weight.unsqueeze(0)).sum(dim=-1) + self.class_logit_bias
        return self.classifier(
            self.fused_features(input_ids, attention_mask, token_type_ids, context_features, anomaly_features, stat_features)
        )

    def _app_projection_input(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        detach_shared: bool = False,
    ) -> torch.Tensor:
        if self.app_projection_input == "shared_encode":
            shared = self.encoder.encode(input_ids, attention_mask, token_type_ids)
        elif self.app_projection_input == "classification_embedding":
            shared = self.encoder.classification_embedding(input_ids, attention_mask, token_type_ids)
        elif self.app_projection_input == "class_aware_summary":
            token_hidden = self.encoder.encode_tokens(input_ids, attention_mask, token_type_ids)
            if not hasattr(self.encoder, "class_aware_summary"):
                raise RuntimeError("class_aware_summary app projection input requires a class-aware encoder")
            shared = self.encoder.class_aware_summary(token_hidden, attention_mask)
        elif self.app_projection_input == "residual_class_aware_pool":
            token_hidden = self.encoder.encode_tokens(input_ids, attention_mask, token_type_ids)
            if not hasattr(self.encoder, "residual_class_aware_pool"):
                raise RuntimeError("residual_class_aware_pool app projection input requires a compatible encoder")
            shared = self.encoder.residual_class_aware_pool(token_hidden, attention_mask)
        elif bool(getattr(self.encoder, "residual_class_aware", False)):
            shared = self.encoder.classification_embedding(input_ids, attention_mask, token_type_ids)
        else:
            shared = self.encoder.encode(input_ids, attention_mask, token_type_ids)
        return shared.detach() if detach_shared else shared

    def app_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        detach_shared: bool = False,
    ) -> torch.Tensor:
        shared = self._app_projection_input(input_ids, attention_mask, token_type_ids, detach_shared=detach_shared)
        if self.use_app_projection:
            return self.app_projection(shared)
        return shared

    def fused_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        context_features: torch.Tensor | None = None,
        anomaly_features: torch.Tensor | None = None,
        stat_features: torch.Tensor | None = None,
        detach_shared: bool = False,
    ) -> torch.Tensor:
        if self.use_app_projection:
            token_embed = self.app_features(input_ids, attention_mask, token_type_ids, detach_shared=detach_shared)
        else:
            token_embed = self.encoder.classification_embedding(input_ids, attention_mask, token_type_ids)
        pieces = [token_embed]
        if self.has_context:
            if context_features is None:
                context_features = torch.zeros((token_embed.shape[0], self.context_size), device=token_embed.device, dtype=token_embed.dtype)
            context_embed = self.context_proj(context_features.float())
            pieces.append(context_embed)
        if self.has_anomaly:
            if anomaly_features is None:
                anomaly_features = torch.zeros((token_embed.shape[0], self.anomaly_size), device=token_embed.device, dtype=token_embed.dtype)
            anomaly_embed = self.anomaly_proj(anomaly_features.float())
            pieces.append(anomaly_embed)
        if self.has_stats:
            if stat_features is None:
                stat_features = torch.zeros((token_embed.shape[0], self.stat_size), device=token_embed.device, dtype=token_embed.dtype)
            stat_embed = self.stat_proj(stat_features.float())
            pieces.append(stat_embed)
        return torch.cat(pieces, dim=-1)

    def class_fused_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        context_features: torch.Tensor | None = None,
        anomaly_features: torch.Tensor | None = None,
        stat_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.use_class_aware_pooling:
            raise RuntimeError("class_fused_features requires a class-aware encoder")
        token_hidden = self.encoder.encode_tokens(input_ids, attention_mask, token_type_ids)
        token_embed = self.encoder.class_aware_encode(token_hidden, attention_mask)
        pieces = [token_embed]
        batch_size, num_classes, _ = token_embed.shape

        def expand(feature: torch.Tensor) -> torch.Tensor:
            return feature.unsqueeze(1).expand(batch_size, num_classes, feature.shape[-1])

        if self.has_context:
            if context_features is None:
                context_features = torch.zeros((batch_size, self.context_size), device=token_embed.device, dtype=token_embed.dtype)
            pieces.append(expand(self.context_proj(context_features.float())))
        if self.has_anomaly:
            if anomaly_features is None:
                anomaly_features = torch.zeros((batch_size, self.anomaly_size), device=token_embed.device, dtype=token_embed.dtype)
            pieces.append(expand(self.anomaly_proj(anomaly_features.float())))
        if self.has_stats:
            if stat_features is None:
                stat_features = torch.zeros((batch_size, self.stat_size), device=token_embed.device, dtype=token_embed.dtype)
            pieces.append(expand(self.stat_proj(stat_features.float())))
        return torch.cat(pieces, dim=-1)

    def forward_heads(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        context_features: torch.Tensor | None = None,
        anomaly_features: torch.Tensor | None = None,
        stat_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_hierarchical_classifier:
            raise RuntimeError("forward_heads requires use_hierarchical_classifier=True")
        features = self.fused_features(input_ids, attention_mask, token_type_ids, context_features, anomaly_features, stat_features)
        if self.use_app_projection or self.use_class_aware_pooling:
            coarse_logits = self(input_ids, attention_mask, token_type_ids, context_features, anomaly_features, stat_features)
        else:
            coarse_logits = self.classifier(features)
        return self.binary_classifier(features), coarse_logits


def _split_indices(
    labels: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    split: str = "stratified",
    order_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    if split == "chronological":
        if order_values is not None:
            indices = indices[np.argsort(order_values)]
        train_end = int(len(indices) * (1.0 - val_ratio - test_ratio))
        val_end = int(len(indices) * (1.0 - test_ratio))
        return indices[:train_end], indices[train_end:val_end], indices[val_end:]
    if split == "temporal_stratified":
        train_parts: list[np.ndarray] = []
        val_parts: list[np.ndarray] = []
        test_parts: list[np.ndarray] = []
        order = order_values if order_values is not None else indices
        for label in sorted(set(labels.tolist())):
            label_idx = indices[labels == label]
            label_idx = label_idx[np.argsort(order[label_idx])]
            train_end = int(len(label_idx) * (1.0 - val_ratio - test_ratio))
            val_end = int(len(label_idx) * (1.0 - test_ratio))
            train_parts.append(label_idx[:train_end])
            val_parts.append(label_idx[train_end:val_end])
            test_parts.append(label_idx[val_end:])
        return np.concatenate(train_parts), np.concatenate(val_parts), np.concatenate(test_parts)
    train_idx, test_idx = train_test_split(indices, test_size=test_ratio, stratify=labels, random_state=seed)
    rel_val = val_ratio / (1.0 - test_ratio)
    train_labels = labels[train_idx]
    train_idx, val_idx = train_test_split(train_idx, test_size=rel_val, stratify=train_labels, random_state=seed)
    return train_idx, val_idx, test_idx


def _temporal_stratified_by_group_indices(
    labels: np.ndarray,
    group_labels: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    order_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    order = order_values if order_values is not None else indices
    for group in sorted(set(group_labels.tolist())):
        group_idx = indices[group_labels == group]
        group_idx = group_idx[np.argsort(order[group_idx])]
        train_end = int(len(group_idx) * (1.0 - val_ratio - test_ratio))
        val_end = int(len(group_idx) * (1.0 - test_ratio))
        train_parts.append(group_idx[:train_end])
        val_parts.append(group_idx[train_end:val_end])
        test_parts.append(group_idx[val_end:])
    return np.concatenate(train_parts), np.concatenate(val_parts), np.concatenate(test_parts)


def _dataset_from_parts(parts: list[tuple[dict[str, Any], torch.Tensor, np.ndarray]]) -> TensorDataset:
    input_ids = []
    attention_masks = []
    token_type_ids = []
    selected_labels = []
    for token_data, labels, indices in parts:
        idx = torch.tensor(indices, dtype=torch.long)
        input_ids.append(token_data["input_ids"][idx])
        attention_masks.append(token_data["attention_mask"][idx])
        token_type_ids.append(token_data["token_type_ids"][idx])
        selected_labels.append(labels[idx])
    return TensorDataset(
        torch.cat(input_ids, dim=0),
        torch.cat(attention_masks, dim=0),
        torch.cat(token_type_ids, dim=0),
        torch.cat(selected_labels, dim=0),
    )


def _loader(token_data: dict[str, Any], labels: torch.Tensor, indices: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return _loader_from_parts([(token_data, labels, indices)], batch_size=batch_size, shuffle=shuffle)


def _loader_from_parts(parts: list[tuple[dict[str, Any], torch.Tensor, np.ndarray]], batch_size: int, shuffle: bool) -> DataLoader:
    dataset = _dataset_from_parts(parts)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _context_matrix(token_data: dict[str, Any], indices: np.ndarray) -> np.ndarray:
    rows: list[list[float]] = []
    meta_rows = token_data.get("meta", [])
    for idx in indices.tolist():
        meta = meta_rows[int(idx)] if int(idx) < len(meta_rows) else {}
        context = meta.get("service_context") or {}
        rows.append(
            [
                float(context.get("recent_count", 0.0)),
                float(context.get("recent_short", 0.0)),
                float(context.get("recent_packets", 0.0)),
                float(context.get("short_ratio", 0.0)),
                float(context.get("last_gap") or 0.0),
                float(context.get("recent_count", 0.0) > 0),
            ]
        )
    return np.array(rows, dtype=np.float32)


def _skip_token_ids(vocab: dict[str, int], include_special: bool) -> set[int]:
    if include_special:
        return set()
    return {
        token_id
        for token in ("[PAD]", "[CLS]", "[SEP]", "[MASK]")
        if (token_id := vocab.get(token)) is not None
    }


def _normalize_feature_rows(features: np.ndarray, normalize: str) -> np.ndarray:
    if normalize == "none":
        return features
    if normalize == "l1":
        denom = np.sum(np.abs(features), axis=1, keepdims=True)
    elif normalize == "l2":
        denom = np.linalg.norm(features, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unsupported normalization: {normalize}")
    return np.divide(features, denom, out=np.zeros_like(features), where=denom > 0)


def _token_histograms(
    token_data: dict[str, Any],
    indices: np.ndarray,
    *,
    include_special: bool,
    normalize: str,
) -> np.ndarray:
    input_ids = token_data["input_ids"].cpu().numpy()
    attention_mask = token_data["attention_mask"].cpu().numpy()
    vocab_size = len(token_data["vocab"])
    skip_ids = _skip_token_ids(token_data["vocab"], include_special)
    features = np.zeros((len(indices), vocab_size), dtype=np.float32)
    for row_idx, data_idx in enumerate(indices.tolist()):
        active_ids = input_ids[data_idx][attention_mask[data_idx] > 0]
        if skip_ids:
            active_ids = np.array([token_id for token_id in active_ids if int(token_id) not in skip_ids], dtype=np.int64)
        if active_ids.size:
            features[row_idx] = np.bincount(active_ids, minlength=vocab_size)[:vocab_size]
    return _normalize_feature_rows(features, normalize)


def _cosine_distance(feature: np.ndarray, prototype: np.ndarray) -> float:
    proto_norm = float(np.linalg.norm(prototype))
    feature_norm = float(np.linalg.norm(feature))
    if proto_norm == 0.0 or feature_norm == 0.0:
        return 1.0
    similarity = float(feature @ prototype) / (proto_norm * feature_norm)
    return float(1.0 - similarity)


def _euclidean_distance(feature: np.ndarray, prototype: np.ndarray) -> float:
    return float(np.linalg.norm(feature - prototype))


def _distance(feature: np.ndarray, prototype: np.ndarray, score_method: str) -> float:
    if score_method == "cosine":
        return _cosine_distance(feature, prototype)
    if score_method == "euclidean":
        return _euclidean_distance(feature, prototype)
    raise ValueError(f"Unsupported score_method: {score_method}")


def _service_key(meta: dict[str, Any]) -> tuple[str, ...] | None:
    key = meta.get("service_key")
    if isinstance(key, (list, tuple)):
        return tuple(str(item) for item in key)
    if key is not None:
        return (str(key),)
    return None


def _benign_label_id(token_data: dict[str, Any], task: str, label_to_id: dict[str, int] | None) -> int:
    if task == "binary":
        binary_mapping = token_data.get("binary_label_to_id") or {}
        if "BENIGN" in binary_mapping:
            return int(binary_mapping["BENIGN"])
        return int(binary_mapping.get("Benign", 0))
    mapping = label_to_id or token_data.get("label_to_id") or {}
    for key in ("BENIGN", "Benign", "benign"):
        if key in mapping:
            return int(mapping[key])
    raise ValueError("Could not resolve benign label id for anomaly features")


def _maybe_benign_label_id(token_data: dict[str, Any], task: str, label_to_id: dict[str, int] | None) -> int | None:
    try:
        return _benign_label_id(token_data, task, label_to_id)
    except ValueError:
        return None


def _binary_targets(labels: torch.Tensor, benign_label_id: int) -> torch.Tensor:
    return (labels != int(benign_label_id)).long()


def _gated_multiclass_predictions(
    coarse_logits: torch.Tensor,
    binary_logits: torch.Tensor,
    benign_label_id: int,
) -> torch.Tensor:
    pred = torch.argmax(coarse_logits, dim=-1)
    binary_pred = torch.argmax(binary_logits, dim=-1)
    gated = pred.clone()
    benign_mask = binary_pred == 0
    gated[benign_mask] = int(benign_label_id)
    attack_mask = ~benign_mask
    if attack_mask.any() and coarse_logits.shape[1] > 1:
        attack_logits = coarse_logits.clone()
        attack_logits[:, int(benign_label_id)] = torch.finfo(coarse_logits.dtype).min
        gated[attack_mask] = torch.argmax(attack_logits[attack_mask], dim=-1)
    return gated


def _fit_prototype_bank(
    token_data: dict[str, Any],
    labels: torch.Tensor,
    train_idx: np.ndarray,
    task: str,
    label_to_id: dict[str, int] | None,
    *,
    include_special: bool,
    normalize: str,
    use_service_prototype_distance: bool,
    use_class_prototype_distance: bool,
    score_method: str = "cosine",
) -> PrototypeBank:
    train_features = _token_histograms(token_data, train_idx, include_special=include_special, normalize=normalize)
    train_labels = labels[torch.tensor(train_idx, dtype=torch.long)].cpu().numpy()
    warnings_list: list[str] = []
    benign_label_id = _maybe_benign_label_id(token_data, task, label_to_id)
    prototype_scope = "benign_train"
    if benign_label_id is None:
        warnings_list.append("No benign label found; using all train samples for prototype features.")
        benign_mask = np.ones_like(train_labels, dtype=bool)
        prototype_scope = "all_train_fallback"
    else:
        benign_mask = train_labels == benign_label_id
    if not benign_mask.any():
        warnings_list.append("No benign samples found in train split; using all train features for global prototype.")
        benign_mask = np.ones_like(train_labels, dtype=bool)
        prototype_scope = "all_train_fallback"
    global_prototype = train_features[benign_mask].mean(axis=0)

    service_prototypes: dict[tuple[str, ...] | None, np.ndarray] = {}
    if use_service_prototype_distance:
        meta_rows = token_data.get("meta", [])
        buckets: dict[tuple[str, ...] | None, list[np.ndarray]] = defaultdict(list)
        for local_pos, data_idx in enumerate(train_idx.tolist()):
            if not benign_mask[local_pos]:
                continue
            meta = meta_rows[int(data_idx)] if int(data_idx) < len(meta_rows) else {}
            buckets[_service_key(meta)].append(train_features[local_pos])
        for key, rows in buckets.items():
            service_prototypes[key] = np.stack(rows, axis=0).mean(axis=0)
        if not service_prototypes:
            warnings_list.append("No benign service prototypes were built; falling back to global prototype distances.")
    class_prototypes: dict[int, np.ndarray] = {}
    if use_class_prototype_distance:
        num_classes = int(labels.max().item()) + 1
        for class_id in range(num_classes):
            class_mask = train_labels == class_id
            if class_mask.any():
                class_prototypes[class_id] = train_features[class_mask].mean(axis=0)
            else:
                class_prototypes[class_id] = global_prototype
                warnings_list.append(f"Class prototype missing for label {class_id}; using global prototype fallback.")
    return PrototypeBank(
        global_prototype=global_prototype,
        service_prototypes=service_prototypes,
        class_prototypes=class_prototypes,
        score_method=score_method,
        normalize=normalize,
        include_special=include_special,
        prototype_scope=prototype_scope,
        benign_label_id=benign_label_id,
        warnings=warnings_list,
    )


def _anomaly_feature_names(
    num_classes: int,
    *,
    use_service_prototype_distance: bool,
    use_class_prototype_distance: bool,
) -> list[str]:
    names = ["global_anomaly_score", "nearest_benign_proto_distance"]
    if use_service_prototype_distance:
        names.insert(1, "service_anomaly_score")
    if use_class_prototype_distance:
        names.extend(f"class_proto_distance_{idx}" for idx in range(num_classes))
    return names


def _anomaly_feature_rows(
    token_data: dict[str, Any],
    indices: np.ndarray,
    bank: PrototypeBank,
    *,
    use_service_prototype_distance: bool,
    use_class_prototype_distance: bool,
) -> np.ndarray:
    if len(indices) == 0:
        width = 2 + int(use_service_prototype_distance) + (len(bank.class_prototypes) if use_class_prototype_distance else 0)
        return np.zeros((0, width), dtype=np.float32)
    features = _token_histograms(token_data, indices, include_special=bank.include_special, normalize=bank.normalize)
    meta_rows = token_data.get("meta", [])
    rows: list[list[float]] = []
    service_values = list(bank.service_prototypes.values())
    for row_idx, data_idx in enumerate(indices.tolist()):
        feature = features[row_idx]
        global_distance = _distance(feature, bank.global_prototype, bank.score_method)
        nearest_distance = global_distance
        if service_values:
            nearest_distance = min(_distance(feature, proto, bank.score_method) for proto in service_values)
        values = [global_distance]
        if use_service_prototype_distance:
            meta = meta_rows[int(data_idx)] if int(data_idx) < len(meta_rows) else {}
            service_key = _service_key(meta)
            service_proto = bank.service_prototypes.get(service_key, bank.global_prototype)
            values.append(_distance(feature, service_proto, bank.score_method))
        values.append(nearest_distance)
        if use_class_prototype_distance:
            for class_id in sorted(bank.class_prototypes):
                values.append(_distance(feature, bank.class_prototypes[class_id], bank.score_method))
        rows.append(values)
    return np.asarray(rows, dtype=np.float32)


DEFAULT_STAT_FEATURE_NAMES = [
    "flow_duration",
    "packet_count",
    "byte_count",
    "token_count",
    "active_token_count",
    "c2s_packet_count",
    "s2c_packet_count",
    "c2s_s2c_packet_ratio",
    "mean_packet_length",
    "std_packet_length",
    "min_packet_length",
    "max_packet_length",
    "iat_mean",
    "iat_std",
    "iat_quantile_25",
    "iat_quantile_50",
    "iat_quantile_75",
    "burst_count",
    "short_flow_flag",
    "same_length_flag",
    "repeat_count",
    "duplicate_count",
    "profile_short_count",
    "profile_same_count",
    "profile_repeat_count",
    "profile_duplicate_count",
]


def _token_prefix_count(active_tokens: list[str], prefixes: tuple[str, ...]) -> int:
    return sum(1 for token in active_tokens if canonical_token(token).startswith(prefixes))


def _stat_row_from_meta_tokens(
    meta: dict[str, Any],
    active_tokens: list[str],
    feature_names: list[str],
    missing: set[str],
) -> list[float]:
    packet_count = float(meta.get("packet_count") or 0.0)
    token_count = float(meta.get("token_count") or len(active_tokens))
    active_count = float(len(active_tokens))
    profile_short_count = float(_token_prefix_count(active_tokens, ("PRIM_PROFILE_SHORT_FLOW", "RHY_SHORT")))
    profile_same_count = float(_token_prefix_count(active_tokens, ("PRIM_PROFILE_SAME_LEN",)))
    profile_repeat_count = float(_token_prefix_count(active_tokens, ("PRIM_PROFILE_REPEAT_SEG", "RHY_PERIODIC")))
    profile_duplicate_count = float(_token_prefix_count(active_tokens, ("PRIM_PROFILE_DUP_SEG",)))
    burst_count_value = float(_token_prefix_count(active_tokens, ("BURST_START", "BURST_SINGLE")))
    values: dict[str, float] = {
        "flow_duration": 0.0,
        "packet_count": packet_count,
        "byte_count": 0.0,
        "token_count": token_count,
        "active_token_count": active_count,
        "c2s_packet_count": 0.0,
        "s2c_packet_count": 0.0,
        "c2s_s2c_packet_ratio": 0.0,
        "mean_packet_length": 0.0,
        "std_packet_length": 0.0,
        "min_packet_length": 0.0,
        "max_packet_length": 0.0,
        "iat_mean": 0.0,
        "iat_std": 0.0,
        "iat_quantile_25": 0.0,
        "iat_quantile_50": 0.0,
        "iat_quantile_75": 0.0,
        "burst_count": burst_count_value,
        "short_flow_flag": float(packet_count > 0.0 and packet_count < 6.0),
        "same_length_flag": float(profile_same_count > 0.0),
        "repeat_count": profile_repeat_count,
        "duplicate_count": profile_duplicate_count,
        "profile_short_count": profile_short_count,
        "profile_same_count": profile_same_count,
        "profile_repeat_count": profile_repeat_count,
        "profile_duplicate_count": profile_duplicate_count,
    }
    for name in feature_names:
        if name not in values:
            missing.add(name)
        elif values[name] == 0.0 and name in {
            "flow_duration",
            "byte_count",
            "c2s_packet_count",
            "s2c_packet_count",
            "c2s_s2c_packet_ratio",
            "mean_packet_length",
            "std_packet_length",
            "min_packet_length",
            "max_packet_length",
            "iat_mean",
            "iat_std",
            "iat_quantile_25",
            "iat_quantile_50",
            "iat_quantile_75",
        }:
            missing.add(name)
    return [float(values.get(name, 0.0)) for name in feature_names]


def _raw_stat_matrix(token_data: dict[str, Any], indices: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    input_ids = token_data["input_ids"].cpu().numpy()
    attention_mask = token_data["attention_mask"].cpu().numpy()
    meta_rows = token_data.get("meta", [])
    vocab = token_data.get("vocab", {})
    inv_vocab = {int(idx): str(token) for token, idx in vocab.items()}
    skip_ids = _skip_token_ids(vocab, include_special=False)
    rows: list[list[float]] = []
    missing: set[str] = set()
    for data_idx in indices.tolist():
        active_ids = input_ids[int(data_idx)][attention_mask[int(data_idx)] > 0]
        active_tokens = [inv_vocab.get(int(token_id), "[UNK]") for token_id in active_ids if int(token_id) not in skip_ids]
        meta = meta_rows[int(data_idx)] if int(data_idx) < len(meta_rows) else {}
        rows.append(_stat_row_from_meta_tokens(meta, active_tokens, feature_names, missing))
    return np.asarray(rows, dtype=np.float32), sorted(missing)


def _fit_stat_normalizer(token_data: dict[str, Any], train_idx: np.ndarray, feature_names: list[str]) -> StatFeatureNormalizer:
    raw, missing = _raw_stat_matrix(token_data, train_idx, feature_names)
    if raw.size == 0:
        mean = np.zeros((len(feature_names),), dtype=np.float32)
        std = np.ones((len(feature_names),), dtype=np.float32)
    else:
        mean = raw.mean(axis=0).astype(np.float32)
        std = raw.std(axis=0).astype(np.float32)
        std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return StatFeatureNormalizer(names=list(feature_names), mean=mean, std=std, missing_names=missing)


def _stat_feature_rows(token_data: dict[str, Any], indices: np.ndarray, normalizer: StatFeatureNormalizer) -> np.ndarray:
    raw, _ = _raw_stat_matrix(token_data, indices, normalizer.names)
    return ((raw - normalizer.mean.reshape(1, -1)) / normalizer.std.reshape(1, -1)).astype(np.float32)


def _loader_with_context(token_data: dict[str, Any], labels: torch.Tensor, indices: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    idx = torch.tensor(indices, dtype=torch.long)
    dataset = TensorDataset(
        token_data["input_ids"][idx],
        token_data["attention_mask"][idx],
        token_data["token_type_ids"][idx],
        torch.tensor(_context_matrix(token_data, indices), dtype=torch.float32),
        labels[idx],
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _dataset_from_parts_with_context(parts: list[tuple[dict[str, Any], torch.Tensor, np.ndarray]]) -> TensorDataset:
    input_ids = []
    attention_masks = []
    token_type_ids = []
    context_features = []
    selected_labels = []
    for token_data, labels, indices in parts:
        idx = torch.tensor(indices, dtype=torch.long)
        input_ids.append(token_data["input_ids"][idx])
        attention_masks.append(token_data["attention_mask"][idx])
        token_type_ids.append(token_data["token_type_ids"][idx])
        context_features.append(torch.tensor(_context_matrix(token_data, indices), dtype=torch.float32))
        selected_labels.append(labels[idx])
    return TensorDataset(
        torch.cat(input_ids, dim=0),
        torch.cat(attention_masks, dim=0),
        torch.cat(token_type_ids, dim=0),
        torch.cat(context_features, dim=0),
        torch.cat(selected_labels, dim=0),
    )


def _loader_from_parts_with_context(parts: list[tuple[dict[str, Any], torch.Tensor, np.ndarray]], batch_size: int, shuffle: bool) -> DataLoader:
    dataset = _dataset_from_parts_with_context(parts)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _dataset_from_parts_with_features(
    parts: list[tuple[dict[str, Any], torch.Tensor, np.ndarray]],
    *,
    use_service_context: bool,
    use_anomaly_features: bool,
    use_stat_fusion: bool = False,
    bank: PrototypeBank | None = None,
    stat_normalizer: StatFeatureNormalizer | None = None,
    use_service_prototype_distance: bool = False,
    use_class_prototype_distance: bool = False,
) -> TensorDataset:
    input_ids = []
    attention_masks = []
    token_type_ids = []
    context_features = []
    anomaly_features = []
    stat_features = []
    selected_labels = []
    for part_token_data, labels, indices in parts:
        idx = torch.tensor(indices, dtype=torch.long)
        input_ids.append(part_token_data["input_ids"][idx])
        attention_masks.append(part_token_data["attention_mask"][idx])
        token_type_ids.append(part_token_data["token_type_ids"][idx])
        if use_service_context:
            context_features.append(torch.tensor(_context_matrix(part_token_data, indices), dtype=torch.float32))
        if use_anomaly_features:
            if bank is None:
                raise ValueError("anomaly features requested without a prototype bank")
            anomaly_features.append(
                torch.tensor(
                    _anomaly_feature_rows(
                        part_token_data,
                        indices,
                        bank,
                        use_service_prototype_distance=use_service_prototype_distance,
                        use_class_prototype_distance=use_class_prototype_distance,
                    ),
                    dtype=torch.float32,
                )
            )
        if use_stat_fusion:
            if stat_normalizer is None:
                raise ValueError("stat fusion requested without a fitted normalizer")
            stat_features.append(torch.tensor(_stat_feature_rows(part_token_data, indices, stat_normalizer), dtype=torch.float32))
        selected_labels.append(labels[idx])
    tensors: list[torch.Tensor] = [torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0), torch.cat(token_type_ids, dim=0)]
    if use_service_context:
        tensors.append(torch.cat(context_features, dim=0))
    if use_anomaly_features:
        tensors.append(torch.cat(anomaly_features, dim=0))
    if use_stat_fusion:
        tensors.append(torch.cat(stat_features, dim=0))
    tensors.append(torch.cat(selected_labels, dim=0))
    return TensorDataset(*tensors)


def _loader_from_parts_with_features(
    parts: list[tuple[dict[str, Any], torch.Tensor, np.ndarray]],
    *,
    use_service_context: bool,
    use_anomaly_features: bool,
    use_stat_fusion: bool,
    bank: PrototypeBank | None,
    stat_normalizer: StatFeatureNormalizer | None,
    use_service_prototype_distance: bool,
    use_class_prototype_distance: bool,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = _dataset_from_parts_with_features(
        parts,
        use_service_context=use_service_context,
        use_anomaly_features=use_anomaly_features,
        use_stat_fusion=use_stat_fusion,
        bank=bank,
        stat_normalizer=stat_normalizer,
        use_service_prototype_distance=use_service_prototype_distance,
        use_class_prototype_distance=use_class_prototype_distance,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _class_weights_from_labels(selected: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(selected, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights


def _class_weights_for_parts(parts: list[tuple[dict[str, Any], torch.Tensor, np.ndarray]], num_classes: int) -> torch.Tensor:
    selected = []
    for _, labels, indices in parts:
        selected.append(labels[torch.tensor(indices, dtype=torch.long)])
    return _class_weights_from_labels(torch.cat(selected, dim=0), num_classes)


def _binary_class_weights_for_parts(parts: list[tuple[dict[str, Any], torch.Tensor, np.ndarray]], benign_label_id: int) -> torch.Tensor:
    selected = _labels_from_parts(parts)
    return _class_weights_from_labels(_binary_targets(selected, benign_label_id), 2)


def _class_weights(labels: torch.Tensor, indices: np.ndarray, num_classes: int) -> torch.Tensor:
    selected = labels[torch.tensor(indices, dtype=torch.long)]
    return _class_weights_from_labels(selected, num_classes)


def _labels_from_parts(parts: list[tuple[dict[str, Any], torch.Tensor, np.ndarray]]) -> torch.Tensor:
    selected = []
    for _, labels, indices in parts:
        selected.append(labels[torch.tensor(indices, dtype=torch.long)])
    return torch.cat(selected, dim=0) if selected else torch.tensor([], dtype=torch.long)


def _balanced_sampler_for_labels(labels: torch.Tensor, num_classes: int, seed: int) -> WeightedRandomSampler:
    counts = torch.bincount(labels, minlength=num_classes).float().clamp(min=1.0)
    sample_weights = (1.0 / counts)[labels]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        sample_weights.double(),
        num_samples=int(len(labels)),
        replacement=True,
        generator=generator,
    )


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, labels, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


class LogitAdjustedCrossEntropyLoss(nn.Module):
    def __init__(self, class_priors: torch.Tensor, tau: float = 1.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        priors = class_priors.float().clamp_min(1e-12)
        self.tau = float(tau)
        self.register_buffer("adjustment", self.tau * torch.log(priors))
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        adjusted = logits + self.adjustment.to(device=logits.device, dtype=logits.dtype).unsqueeze(0)
        return nn.functional.cross_entropy(adjusted, labels, weight=self.weight)


class LDAMLoss(nn.Module):
    def __init__(
        self,
        class_counts: torch.Tensor,
        max_m: float = 0.5,
        scale: float = 30.0,
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        counts = class_counts.float().clamp_min(1.0)
        margins = 1.0 / torch.sqrt(torch.sqrt(counts))
        margins = margins * (float(max_m) / margins.max().clamp_min(1e-12))
        self.scale = float(scale)
        self.register_buffer("margins", margins)
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        margins = self.margins.to(device=logits.device, dtype=logits.dtype)
        adjusted = logits.clone()
        adjusted[torch.arange(logits.shape[0], device=logits.device), labels] -= margins[labels]
        return nn.functional.cross_entropy(self.scale * adjusted, labels, weight=self.weight)


class ClassPrototypeRegularizer(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(num_classes, feature_dim))
        nn.init.normal_(self.prototypes, mean=0.0, std=feature_dim**-0.5)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = nn.functional.normalize(features, dim=-1)
        prototypes = nn.functional.normalize(self.prototypes.to(device=features.device, dtype=features.dtype), dim=-1)
        distances = torch.cdist(features, prototypes, p=2.0).pow(2)
        return nn.functional.cross_entropy(-distances, labels)


def shared_representation_preservation_loss(
    current: torch.Tensor,
    teacher: torch.Tensor,
    mode: str = "cosine",
) -> torch.Tensor:
    if current.shape[0] <= 1:
        return current.sum() * 0.0
    mode = str(mode).lower()
    if mode == "cosine":
        return 1.0 - nn.functional.cosine_similarity(current, teacher, dim=-1).mean()
    if mode == "mse":
        current_norm = nn.functional.normalize(current, dim=-1)
        teacher_norm = nn.functional.normalize(teacher, dim=-1)
        return nn.functional.mse_loss(current_norm, teacher_norm)
    raise ValueError(f"Unsupported shared representation preservation mode: {mode}")


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    if features.shape[0] <= 1:
        return features.sum() * 0.0
    features = nn.functional.normalize(features, dim=-1)
    logits = features @ features.T / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(labels.shape[0], device=labels.device, dtype=torch.bool)
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~self_mask
    valid_rows = positive_mask.any(dim=1)
    if not bool(valid_rows.any()):
        return features.sum() * 0.0
    logits_mask = ~self_mask
    exp_logits = torch.exp(logits) * logits_mask.to(dtype=logits.dtype)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_log_prob_pos = (positive_mask.to(dtype=log_prob.dtype) * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
    return -mean_log_prob_pos[valid_rows].mean()


def _class_counts_from_labels(selected: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.bincount(selected, minlength=num_classes).float()


def _class_counts_for_parts(parts: list[tuple[dict[str, Any], torch.Tensor, np.ndarray]], num_classes: int) -> torch.Tensor:
    return _class_counts_from_labels(_labels_from_parts(parts), num_classes)


def _effective_number_weights_from_counts(counts: torch.Tensor, beta: float) -> torch.Tensor:
    beta = float(beta)
    counts = counts.float().clamp_min(1.0)
    effective_num = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float32), counts)
    weights = (1.0 - beta) / effective_num.clamp_min(1e-12)
    return weights * (len(weights) / weights.sum().clamp_min(1e-12))


def _class_priors_from_counts(counts: torch.Tensor) -> torch.Tensor:
    counts = counts.float().clamp_min(1.0)
    return counts / counts.sum().clamp_min(1e-12)


def _validate_augment_token_data(base: dict[str, Any], augment: dict[str, Any]) -> None:
    if len(base["input_ids"]) != len(augment["input_ids"]):
        raise ValueError("augment token data must have the same number of rows as the clean token data")
    if tuple(base["input_ids"].shape[1:]) != tuple(augment["input_ids"].shape[1:]):
        raise ValueError("augment token data must use the same sequence length as the clean token data")
    if base.get("vocab") != augment.get("vocab"):
        raise ValueError("augment token data must use the same vocabulary as the clean token data")
    if base.get("binary_label_to_id") != augment.get("binary_label_to_id"):
        raise ValueError("augment token data must use the same binary label mapping as the clean token data")


def _validate_extra_train_token_data(base: dict[str, Any], extra: dict[str, Any]) -> None:
    if tuple(base["input_ids"].shape[1:]) != tuple(extra["input_ids"].shape[1:]):
        raise ValueError("extra train token data must use the same sequence length as the clean token data")
    if base.get("vocab") != extra.get("vocab"):
        raise ValueError("extra train token data must use the same vocabulary as the clean token data")


def _names_from_label_tensor(token_data: dict[str, Any], label_key: str, mapping_key: str) -> list[str]:
    labels = token_data[label_key]
    source_mapping = token_data.get(mapping_key)
    if source_mapping is None:
        raise ValueError(f"cannot remap {label_key} without {mapping_key}")
    inv = {idx: label for label, idx in source_mapping.items()}
    return [inv[int(label)] for label in labels.tolist()]


def _task_labels_with_mapping(token_data: dict[str, Any], task: str, target_label_to_id: dict[str, int] | None) -> torch.Tensor:
    if target_label_to_id is None:
        labels, _ = _task_labels(token_data, task)
        return labels
    if task == "binary":
        label_names = _names_from_label_tensor(token_data, "binary_labels", "binary_label_to_id")
    elif task == "multiclass":
        label_names = _names_from_label_tensor(token_data, "labels", "label_to_id")
    elif task == "multiclass_merged":
        label_names = [merged_cicids_label(str(meta.get("label", ""))) for meta in token_data.get("meta", [])]
        if len(label_names) != len(token_data["input_ids"]):
            raise ValueError("multiclass_merged extra train token data must include one meta row per token row")
    else:
        raise ValueError(f"Unsupported task: {task}")
    unknown = sorted({label for label in label_names if label not in target_label_to_id})
    if unknown:
        raise ValueError(f"extra train token data has labels not present in clean task mapping: {unknown}")
    return torch.tensor([target_label_to_id[label] for label in label_names], dtype=torch.long)


def _augment_indices(labels_np: np.ndarray, train_idx: np.ndarray, task: str, attack_only: bool, fraction: float, seed: int) -> np.ndarray:
    if fraction > 1.0:
        raise ValueError("augmentation fraction must be <= 1.0")
    selected = train_idx
    if attack_only:
        if task != "binary":
            raise ValueError("attack-only augmentation is currently supported only for binary classification")
        selected = selected[labels_np[selected] == 1]
    if fraction < 1.0:
        if fraction <= 0.0:
            return np.array([], dtype=np.int64)
        rng = np.random.default_rng(seed)
        keep = max(1, int(np.ceil(len(selected) * fraction))) if len(selected) else 0
        selected = np.sort(rng.choice(selected, size=keep, replace=False)) if keep else selected
    return selected.astype(np.int64, copy=False)


def _resolve_augment_fractions(num_sources: int, default_fraction: float, per_source_fractions: list[float] | None) -> list[float]:
    if per_source_fractions is None:
        return [float(default_fraction)] * num_sources
    if len(per_source_fractions) != num_sources:
        raise ValueError("augment_fractions length must match augment_token_data length")
    return [float(fraction) for fraction in per_source_fractions]


def _build_train_parts(
    token_data: dict[str, Any],
    labels: torch.Tensor,
    labels_np: np.ndarray,
    train_idx: np.ndarray,
    task: str,
    augment_token_data: list[dict[str, Any]] | None,
    augment_attack_only: bool,
    augment_fraction: float,
    seed: int,
    augment_fractions: list[float] | None = None,
) -> tuple[list[tuple[dict[str, Any], torch.Tensor, np.ndarray]], dict[str, Any]]:
    parts = [(token_data, labels, train_idx)]
    info: dict[str, Any] = {
        "enabled": bool(augment_token_data),
        "num_sources": 0,
        "attack_only": bool(augment_attack_only),
        "fraction": float(augment_fraction),
        "num_augmented_train": 0,
    }
    if not augment_token_data:
        return parts, info

    fractions = _resolve_augment_fractions(len(augment_token_data), augment_fraction, augment_fractions)
    counts: list[int] = []
    shared_idx = None
    if augment_fractions is None:
        shared_idx = _augment_indices(
            labels_np,
            train_idx,
            task=task,
            attack_only=augment_attack_only,
            fraction=augment_fraction,
            seed=seed,
        )
    for source_idx, (augment, fraction) in enumerate(zip(augment_token_data, fractions)):
        _validate_augment_token_data(token_data, augment)
        aug_labels, _ = _task_labels(augment, task)
        aug_idx = shared_idx
        if aug_idx is None:
            aug_idx = _augment_indices(
                labels_np,
                train_idx,
                task=task,
                attack_only=augment_attack_only,
                fraction=fraction,
                seed=seed + source_idx,
            )
        parts.append((augment, aug_labels, aug_idx))
        counts.append(int(len(aug_idx)))
    info["num_sources"] = len(augment_token_data)
    info["fractions"] = fractions
    info["num_augmented_train"] = int(sum(counts))
    info["num_augmented_train_per_source"] = counts[0] if len(set(counts)) == 1 else counts
    info["num_augmented_train_by_source"] = counts
    return parts, info


def _fraction_indices(num_rows: int, fraction: float, seed: int) -> np.ndarray:
    if fraction > 1.0:
        raise ValueError("extra train fraction must be <= 1.0")
    if fraction <= 0.0:
        return np.array([], dtype=np.int64)
    indices = np.arange(num_rows, dtype=np.int64)
    if fraction >= 1.0:
        return indices
    keep = max(1, int(np.ceil(num_rows * fraction))) if num_rows else 0
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=keep, replace=False)) if keep else np.array([], dtype=np.int64)


def _build_extra_train_parts(
    base_token_data: dict[str, Any],
    extra_train_token_data: list[dict[str, Any]] | None,
    task: str,
    target_label_to_id: dict[str, int] | None,
    fraction: float,
    seed: int,
) -> tuple[list[tuple[dict[str, Any], torch.Tensor, np.ndarray]], dict[str, Any]]:
    info: dict[str, Any] = {
        "enabled": bool(extra_train_token_data),
        "num_sources": 0,
        "fraction": float(fraction),
        "num_extra_train": 0,
    }
    if not extra_train_token_data:
        return [], info

    parts: list[tuple[dict[str, Any], torch.Tensor, np.ndarray]] = []
    counts: list[int] = []
    label_counts_by_source: list[dict[str, int]] = []
    id_to_label = {idx: label for label, idx in (target_label_to_id or {}).items()}
    for source_idx, extra in enumerate(extra_train_token_data):
        _validate_extra_train_token_data(base_token_data, extra)
        extra_labels = _task_labels_with_mapping(extra, task, target_label_to_id)
        extra_idx = _fraction_indices(len(extra_labels), fraction=fraction, seed=seed + source_idx)
        parts.append((extra, extra_labels, extra_idx))
        counts.append(int(len(extra_idx)))
        selected = extra_labels[torch.tensor(extra_idx, dtype=torch.long)] if len(extra_idx) else torch.tensor([], dtype=torch.long)
        if len(selected):
            unique_labels, unique_counts = torch.unique(selected, return_counts=True)
            label_counts_by_source.append(
                {
                    str(id_to_label.get(int(label.item()), int(label.item()))): int(count.item())
                    for label, count in sorted(zip(unique_labels, unique_counts), key=lambda item: int(item[0].item()))
                }
            )
        else:
            label_counts_by_source.append({})
    info["num_sources"] = len(extra_train_token_data)
    info["num_extra_train"] = int(sum(counts))
    info["num_extra_train_by_source"] = counts
    info["label_id_counts_by_source"] = label_counts_by_source
    return parts, info


def _merged_labels(token_data: dict[str, Any]) -> tuple[torch.Tensor, dict[str, int]]:
    labels = []
    for meta in token_data.get("meta", []):
        raw_label = str(meta.get("label", ""))
        labels.append(merged_cicids_label(raw_label))
    label_names = sorted(set(labels))
    if "BENIGN" in label_names:
        label_names = ["BENIGN"] + [label for label in label_names if label != "BENIGN"]
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    return torch.tensor([label_to_id[label] for label in labels], dtype=torch.long), label_to_id


def _task_labels(token_data: dict[str, Any], task: str) -> tuple[torch.Tensor, dict[str, int] | None]:
    if task == "binary":
        return token_data["binary_labels"], token_data.get("binary_label_to_id")
    if task == "multiclass":
        return token_data["labels"], token_data.get("label_to_id")
    if task == "multiclass_merged":
        return _merged_labels(token_data)
    raise ValueError(f"Unsupported task: {task}")


def _raw_label_group_ids(token_data: dict[str, Any]) -> np.ndarray:
    label_names = [str(meta.get("label", "")) for meta in token_data.get("meta", [])]
    if len(label_names) != len(token_data["input_ids"]):
        raise ValueError("raw-label temporal split requires one meta row per token row")
    mapping = {label: idx for idx, label in enumerate(sorted(set(label_names)))}
    return np.array([mapping[label] for label in label_names], dtype=np.int64)


def _subsample_train_indices(labels: np.ndarray, train_idx: np.ndarray, label_fraction: float, seed: int) -> np.ndarray:
    if label_fraction >= 1.0:
        return train_idx
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    for label in sorted(set(labels[train_idx].tolist())):
        label_idx = np.array([idx for idx in train_idx.tolist() if labels[idx] == label], dtype=np.int64)
        if label_idx.size == 0:
            continue
        keep = max(1, int(np.ceil(label_idx.size * label_fraction)))
        parts.append(np.sort(rng.choice(label_idx, size=keep, replace=False)))
    return np.concatenate(parts) if parts else train_idx


def _load_encoder_checkpoint(model: BehaviorComposer, checkpoint: str, device: torch.device, freeze_encoder: bool = False) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    model_state = model.state_dict()
    compatible = {}
    for key, value in state_dict.items():
        target_key = key
        if target_key not in model_state and target_key.startswith("encoder."):
            target_key = target_key.removeprefix("encoder.")
        if (
            target_key in model_state
            and tuple(model_state[target_key].shape) == tuple(value.shape)
            and not target_key.startswith("classifier.")
        ):
            compatible[target_key] = value
    model_state.update(compatible)
    model.load_state_dict(model_state)
    if freeze_encoder:
        encoder_names = _encoder_parameter_names(model)
        for name, parameter in model.named_parameters():
            if name in encoder_names:
                parameter.requires_grad = False
    return {"loaded_tensors": len(compatible), "checkpoint": checkpoint, "freeze_encoder": bool(freeze_encoder)}


def _encoder_parameter_names(model: nn.Module) -> set[str]:
    if isinstance(model, BehaviorComposer):
        encoder_prefixes = (
            "token_embedding.",
            "position_embedding.",
            "type_embedding.",
            "norm.",
            "encoder.",
        )
        return {name for name, _ in model.named_parameters() if name.startswith(encoder_prefixes)}
    return {name for name, _ in model.named_parameters() if name.startswith("encoder.")}


def _set_encoder_trainable(model: nn.Module, trainable: bool) -> int:
    encoder_names = _encoder_parameter_names(model)
    count = 0
    for name, parameter in model.named_parameters():
        if name in encoder_names:
            parameter.requires_grad = bool(trainable)
            count += 1
    return count


def _optimizer_parameter_groups(
    model: nn.Module,
    proto_regularizer: nn.Module | None,
    *,
    base_lr: float,
    encoder_lr: float | None,
    weight_decay: float,
    encoder_frozen: bool = False,
) -> list[dict[str, Any]]:
    encoder_names = _encoder_parameter_names(model)
    encoder_params: list[nn.Parameter] = []
    head_params: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if name in encoder_names:
            encoder_params.append(parameter)
        else:
            if parameter.requires_grad:
                head_params.append(parameter)
    if proto_regularizer is not None:
        head_params.extend(parameter for parameter in proto_regularizer.parameters() if parameter.requires_grad)
    groups: list[dict[str, Any]] = []
    if encoder_params:
        encoder_lr_resolved = 0.0 if encoder_frozen else float(encoder_lr if encoder_lr is not None else base_lr)
        encoder_weight_decay = 0.0 if encoder_frozen else weight_decay
        groups.append(
            {
                "params": encoder_params,
                "lr": encoder_lr_resolved,
                "weight_decay": encoder_weight_decay,
                "group_name": "encoder",
            }
        )
    if head_params:
        groups.append({"params": head_params, "lr": base_lr, "weight_decay": weight_decay, "group_name": "head"})
    return groups


def _set_optimizer_encoder_state(
    optimizer: torch.optim.Optimizer,
    *,
    lr: float,
    weight_decay: float,
) -> None:
    for group in optimizer.param_groups:
        if group.get("group_name") == "encoder":
            group["lr"] = float(lr)
            group["weight_decay"] = float(weight_decay)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_hierarchical_classifier: bool = False,
    use_hierarchical_binary_loss: bool = True,
    lambda_binary: float = 1.0,
    lambda_coarse: float = 1.0,
    benign_label_id: int = 0,
    gated_inference: bool = False,
) -> tuple[float, list[int], list[int], np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    scores: list[np.ndarray] = []
    has_context = bool(getattr(model, "has_context", False))
    has_anomaly = bool(getattr(model, "has_anomaly", False))
    has_stats = bool(getattr(model, "has_stats", False))
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, token_type_ids = batch[:3]
            labels = batch[-1]
            cursor = 3
            context_features = None
            anomaly_features = None
            stat_features = None
            if has_context:
                context_features = batch[cursor]
                cursor += 1
            if has_anomaly:
                anomaly_features = batch[cursor]
                cursor += 1
            if has_stats:
                stat_features = batch[cursor]
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            if use_hierarchical_classifier:
                context_arg = context_features.to(device) if context_features is not None else None
                anomaly_arg = anomaly_features.to(device) if anomaly_features is not None else None
                stat_arg = stat_features.to(device) if stat_features is not None else None
                binary_logits, logits = model.forward_heads(input_ids, attention_mask, token_type_ids, context_arg, anomaly_arg, stat_arg)
            elif context_features is None and anomaly_features is None and stat_features is None:
                logits = model(input_ids, attention_mask, token_type_ids)
                binary_logits = None
            else:
                context_arg = context_features.to(device) if context_features is not None else None
                anomaly_arg = anomaly_features.to(device) if anomaly_features is not None else None
                stat_arg = stat_features.to(device) if stat_features is not None else None
                logits = model(input_ids, attention_mask, token_type_ids, context_arg, anomaly_arg, stat_arg)
                binary_logits = None
            if use_hierarchical_classifier and binary_logits is not None:
                binary_labels = _binary_targets(labels, benign_label_id)
                loss = float(lambda_coarse) * criterion(logits, labels)
                if use_hierarchical_binary_loss:
                    loss = loss + float(lambda_binary) * criterion(binary_logits, binary_labels)
            else:
                loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=-1)
            losses.append(float(loss.item()))
            y_true.extend(labels.cpu().tolist())
            if use_hierarchical_classifier and binary_logits is not None and gated_inference:
                pred = _gated_multiclass_predictions(logits, binary_logits, benign_label_id)
            else:
                pred = torch.argmax(logits, dim=-1)
            y_pred.extend(pred.cpu().tolist())
            scores.append(probs.cpu().numpy())
    return float(np.mean(losses) if losses else 0.0), y_true, y_pred, np.concatenate(scores, axis=0)


def _prediction_rows(
    token_data: dict[str, Any],
    indices: np.ndarray,
    y_true: list[int],
    y_pred: list[int],
    y_score: np.ndarray,
    target_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    meta_rows = token_data.get("meta", [])
    for offset, idx in enumerate(indices.tolist()):
        meta = meta_rows[int(idx)] if int(idx) < len(meta_rows) else {}
        true_id = int(y_true[offset])
        pred_id = int(y_pred[offset])
        score_row = y_score[offset]
        rows.append(
            {
                "index": int(idx),
                "flow_id": meta.get("flow_id"),
                "label": meta.get("label"),
                "binary_label": meta.get("binary_label"),
                "true_label": target_names[true_id] if true_id < len(target_names) else str(true_id),
                "pred_label": target_names[pred_id] if pred_id < len(target_names) else str(pred_id),
                "pred_confidence": float(score_row[pred_id]),
                "start_ts": meta.get("start_ts"),
                "end_ts": meta.get("end_ts"),
                "duration": meta.get("duration"),
                "dataset_file": meta.get("dataset_file"),
                "src_ip": meta.get("src_ip"),
                "dst_ip": meta.get("dst_ip"),
                "src_port": meta.get("src_port"),
                "dst_port": meta.get("dst_port"),
                "protocol": meta.get("protocol"),
                "packet_count": meta.get("packet_count"),
                "token_count": meta.get("token_count"),
                "service_key": meta.get("service_key"),
                "scores": {target_names[class_idx]: float(score) for class_idx, score in enumerate(score_row[: len(target_names)])},
            }
        )
    return rows


def _best_binary_threshold(y_true: list[int], scores: np.ndarray) -> tuple[float, float]:
    if scores.shape[1] < 2:
        return 0.5, 0.0
    probs = scores[:, 1]
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.01, 0.99, 99):
        pred = (probs >= threshold).astype(int).tolist()
        cur_f1 = classification_metrics(y_true, pred, scores)["macro_f1"]
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_threshold = float(threshold)
    return best_threshold, float(best_f1)


def _shared_representation_embedding(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    token_type_ids: torch.Tensor | None,
) -> torch.Tensor:
    if isinstance(model, ContextAwareClassifier):
        encoder = model.encoder
    else:
        encoder = model
    if hasattr(encoder, "encode"):
        return encoder.encode(input_ids, attention_mask, token_type_ids)
    if hasattr(encoder, "classification_embedding"):
        return encoder.classification_embedding(input_ids, attention_mask, token_type_ids)
    raise AttributeError(f"{type(model).__name__} does not expose a shared representation encoder")


def train_classifier(
    token_data: dict[str, Any],
    config: dict[str, Any],
    task: str = "binary",
    split: str = "stratified",
    checkpoint: str | None = None,
    freeze_encoder: bool = False,
    label_fraction: float = 1.0,
    augment_token_data: list[dict[str, Any]] | None = None,
    augment_attack_only: bool = True,
    augment_fraction: float = 1.0,
    augment_fractions: list[float] | None = None,
    extra_train_token_data: list[dict[str, Any]] | None = None,
    extra_train_fraction: float = 1.0,
    train_seed: int | None = None,
    use_service_context: bool = False,
    use_anomaly_features: bool = False,
) -> TrainResult:
    started = time.perf_counter()
    seed = int(config.get("seed", 42))
    split_seed = int(config.get("split_seed", seed))
    resolved_train_seed = int(train_seed if train_seed is not None else config.get("train_seed", seed))
    set_seed(resolved_train_seed)
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    use_service_context = bool(use_service_context or model_cfg.get("use_service_context", False))
    use_anomaly_features = bool(use_anomaly_features or model_cfg.get("use_anomaly_features", False))
    use_stat_fusion = bool(model_cfg.get("use_stat_fusion", False))
    use_app_projection = bool(model_cfg.get("use_app_projection", train_cfg.get("use_app_projection", False)))
    app_projection_dim = int(model_cfg.get("app_projection_dim", train_cfg.get("app_projection_dim", model_cfg.get("hidden_size", 128))))
    app_projection_input = str(model_cfg.get("app_projection_input", train_cfg.get("app_projection_input", "shared_encode")))
    freeze_encoder_epochs = int(train_cfg.get("freeze_encoder_epochs", 0))
    pooling_params = resolve_pooling_config(model_cfg)
    pooling_strategy = str(pooling_params["pooling_strategy"])
    class_aware_pooling = bool(pooling_params["class_aware_pooling"])
    stat_feature_names = list(model_cfg.get("stat_feature_names") or DEFAULT_STAT_FEATURE_NAMES)
    stat_mlp_dim = int(model_cfg.get("stat_mlp_dim", 16))
    use_hierarchical_classifier = bool(train_cfg.get("use_hierarchical_classifier", False))
    hierarchical_gated_inference = bool(train_cfg.get("hierarchical_gated_inference", False))
    lambda_binary = float(train_cfg.get("lambda_binary", 1.0))
    lambda_coarse = float(train_cfg.get("lambda_coarse", 1.0))
    lambda_fine = float(train_cfg.get("lambda_fine", 0.0))
    use_supcon = bool(train_cfg.get("use_supcon", False))
    lambda_supcon = float(train_cfg.get("supcon_weight", train_cfg.get("lambda_supcon", 0.0)))
    supcon_temperature = float(train_cfg.get("supcon_temperature", 0.1))
    use_class_proto_loss = bool(train_cfg.get("use_app_proto_reg", train_cfg.get("use_class_proto_loss", False)))
    lambda_class_proto = float(train_cfg.get("app_proto_weight", train_cfg.get("lambda_class_proto", 0.0)))
    use_shared_preservation = bool(train_cfg.get("use_shared_representation_preservation", False))
    shared_preservation_weight = float(train_cfg.get("shared_representation_weight", 0.0))
    shared_preservation_mode = str(train_cfg.get("shared_representation_mode", "cosine"))
    shared_preservation_checkpoint = train_cfg.get("shared_representation_checkpoint", checkpoint)
    detach_shared_for_aux_loss = bool(train_cfg.get("detach_shared_for_aux_loss", False))
    if use_hierarchical_classifier and task == "binary":
        raise ValueError("hierarchical classifier is intended for multiclass tasks; use task=multiclass_merged")
    labels, label_to_id = _task_labels(token_data, task)
    labels_np = labels.numpy()
    num_classes = int(labels.max().item()) + 1
    benign_label_id_for_task = _maybe_benign_label_id(token_data, task, label_to_id)
    use_hierarchical_binary_loss = bool(use_hierarchical_classifier and lambda_binary > 0.0)
    requires_benign_label = bool(
        task == "binary"
        or (use_hierarchical_classifier and (use_hierarchical_binary_loss or hierarchical_gated_inference))
    )
    if benign_label_id_for_task is None and requires_benign_label:
        raise ValueError("Could not resolve benign label id for binary or BENIGN-gated hierarchical training")
    binary_benign_label_id = int(benign_label_id_for_task) if benign_label_id_for_task is not None else 0
    meta_rows = token_data.get("meta", [])
    if len(meta_rows) == len(labels_np):
        order_values = np.array([float(meta.get("start_ts") or idx) for idx, meta in enumerate(meta_rows)])
    else:
        order_values = np.arange(len(labels_np), dtype=float)
    split_mode = split
    if split == "temporal_stratified_raw_label":
        train_idx, val_idx, test_idx = _temporal_stratified_by_group_indices(
            labels_np,
            _raw_label_group_ids(token_data),
            val_ratio=float(train_cfg.get("val_ratio", 0.1)),
            test_ratio=float(train_cfg.get("test_ratio", 0.2)),
            order_values=order_values,
        )
    else:
        train_idx, val_idx, test_idx = _split_indices(
            labels_np,
            val_ratio=float(train_cfg.get("val_ratio", 0.1)),
            test_ratio=float(train_cfg.get("test_ratio", 0.2)),
            seed=split_seed,
            split=split,
            order_values=order_values,
        )
    train_idx_full = train_idx.copy()
    train_idx = _subsample_train_indices(labels_np, train_idx, label_fraction=float(label_fraction), seed=resolved_train_seed)
    batch_size = int(train_cfg.get("batch_size", 64))
    train_parts, augment_info = _build_train_parts(
        token_data,
        labels,
        labels_np,
        train_idx,
        task=task,
        augment_token_data=augment_token_data,
        augment_attack_only=augment_attack_only,
        augment_fraction=augment_fraction,
        augment_fractions=augment_fractions,
        seed=resolved_train_seed,
    )
    extra_train_parts, extra_train_info = _build_extra_train_parts(
        token_data,
        extra_train_token_data,
        task=task,
        target_label_to_id=label_to_id,
        fraction=extra_train_fraction,
        seed=resolved_train_seed,
    )
    train_parts.extend(extra_train_parts)
    train_sampling = str(train_cfg.get("sampling", "shuffle"))
    sampler = None
    if train_sampling == "class_balanced":
        sampler = _balanced_sampler_for_labels(_labels_from_parts(train_parts), num_classes, seed=resolved_train_seed)
    elif train_sampling != "shuffle":
        raise ValueError(f"Unsupported training sampling: {train_sampling}")
    anomaly_cfg = {
        **train_cfg.get("anomaly_features", {}),
        "use_service_prototype_distance": model_cfg.get("use_service_prototype_distance", train_cfg.get("anomaly_features", {}).get("use_service_prototype_distance", False)),
        "use_class_prototype_distance": model_cfg.get("use_class_prototype_distance", train_cfg.get("anomaly_features", {}).get("use_class_prototype_distance", False)),
        "anomaly_feature_dim": model_cfg.get("anomaly_feature_dim", train_cfg.get("anomaly_features", {}).get("anomaly_feature_dim", 0)),
    }
    use_service_prototype_distance = bool(anomaly_cfg.get("use_service_prototype_distance", False)) and bool(use_anomaly_features)
    use_class_prototype_distance = bool(anomaly_cfg.get("use_class_prototype_distance", False)) and bool(use_anomaly_features)
    anomaly_score_method = str(anomaly_cfg.get("score_method", "cosine"))
    anomaly_normalize = str(anomaly_cfg.get("normalize", "l2"))
    anomaly_include_special = bool(anomaly_cfg.get("include_special", False))
    anomaly_feature_dim = int(anomaly_cfg.get("anomaly_feature_dim", 0))
    anomaly_bank = None
    if use_anomaly_features:
        anomaly_bank = _fit_prototype_bank(
            token_data,
            labels,
            train_idx,
            task,
            label_to_id,
            include_special=anomaly_include_special,
            normalize=anomaly_normalize,
            use_service_prototype_distance=use_service_prototype_distance,
            use_class_prototype_distance=use_class_prototype_distance,
            score_method=anomaly_score_method,
        )
        if anomaly_bank.warnings:
            warnings.warn("; ".join(anomaly_bank.warnings))
    stat_normalizer = None
    if use_stat_fusion:
        stat_normalizer = _fit_stat_normalizer(token_data, train_idx, stat_feature_names)
    if use_service_context or use_anomaly_features or use_stat_fusion:
        train_dataset = _dataset_from_parts_with_features(
            train_parts,
            use_service_context=use_service_context,
            use_anomaly_features=use_anomaly_features,
            use_stat_fusion=use_stat_fusion,
            bank=anomaly_bank,
            stat_normalizer=stat_normalizer,
            use_service_prototype_distance=use_service_prototype_distance,
            use_class_prototype_distance=use_class_prototype_distance,
        )
        val_loader = _loader_from_parts_with_features(
            [(token_data, labels, val_idx)],
            use_service_context=use_service_context,
            use_anomaly_features=use_anomaly_features,
            use_stat_fusion=use_stat_fusion,
            bank=anomaly_bank,
            stat_normalizer=stat_normalizer,
            use_service_prototype_distance=use_service_prototype_distance,
            use_class_prototype_distance=use_class_prototype_distance,
            batch_size=batch_size,
            shuffle=False,
        )
        test_loader = _loader_from_parts_with_features(
            [(token_data, labels, test_idx)],
            use_service_context=use_service_context,
            use_anomaly_features=use_anomaly_features,
            use_stat_fusion=use_stat_fusion,
            bank=anomaly_bank,
            stat_normalizer=stat_normalizer,
            use_service_prototype_distance=use_service_prototype_distance,
            use_class_prototype_distance=use_class_prototype_distance,
            batch_size=batch_size,
            shuffle=False,
        )
    else:
        train_dataset = _dataset_from_parts(train_parts)
        val_loader = _loader(token_data, labels, val_idx, batch_size=batch_size, shuffle=False)
        test_loader = _loader(token_data, labels, test_idx, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = BehaviorComposer(
        vocab_size=len(token_data["vocab"]),
        num_classes=num_classes,
        max_seq_len=int(model_cfg.get("max_seq_len", token_data.get("max_len", 256))),
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        intermediate_size=int(model_cfg.get("intermediate_size", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        **pooling_params,
    )
    model: nn.Module
    context_size = 6 if use_service_context else 0
    anomaly_size = 0
    anomaly_feature_dim_resolved = 0
    stat_size = len(stat_feature_names) if use_stat_fusion else 0
    stat_mlp_dim_resolved = stat_mlp_dim if use_stat_fusion and stat_mlp_dim > 0 else 0
    if use_anomaly_features:
        anomaly_names = _anomaly_feature_names(
            num_classes,
            use_service_prototype_distance=use_service_prototype_distance,
            use_class_prototype_distance=use_class_prototype_distance,
        )
        anomaly_size = len(anomaly_names)
        anomaly_feature_dim_resolved = anomaly_feature_dim if anomaly_feature_dim > 0 else max(8, min(64, anomaly_size * 2))
    if use_service_context or use_anomaly_features or use_stat_fusion or use_hierarchical_classifier or use_app_projection:
        model = ContextAwareClassifier(
            encoder,
            hidden_size=int(model_cfg.get("hidden_size", 128)),
            num_classes=num_classes,
            context_size=context_size,
            dropout=float(model_cfg.get("dropout", 0.1)),
            anomaly_size=anomaly_size,
            anomaly_feature_dim=anomaly_feature_dim_resolved,
            use_hierarchical_classifier=use_hierarchical_classifier,
            stat_size=stat_size,
            stat_mlp_dim=stat_mlp_dim_resolved,
            use_app_projection=use_app_projection,
            app_projection_dim=app_projection_dim,
            app_projection_input=app_projection_input,
        ).to(device)
    else:
        model = encoder.to(device)
    proto_regularizer: ClassPrototypeRegularizer | None = None
    if use_class_proto_loss and lambda_class_proto > 0.0:
        if use_app_projection:
            feature_dim = int(app_projection_dim)
        else:
            feature_dim = int(getattr(model, "fusion_size", model_cfg.get("hidden_size", 128)))
        proto_regularizer = ClassPrototypeRegularizer(num_classes, feature_dim).to(device)
    shared_teacher: BehaviorComposer | None = None
    checkpoint_info = None
    if checkpoint:
        checkpoint_info = _load_encoder_checkpoint(encoder, checkpoint, device, freeze_encoder=freeze_encoder)
        if (use_service_context or use_anomaly_features or use_stat_fusion or use_hierarchical_classifier or use_app_projection) and hasattr(model, "encoder"):
            # Reload the wrapped encoder state after the base checkpoint has been applied.
            model.encoder.load_state_dict(encoder.state_dict())
    if use_shared_preservation and shared_preservation_weight > 0.0:
        if not shared_preservation_checkpoint:
            raise ValueError("shared representation preservation requires shared_representation_checkpoint or checkpoint")
        shared_teacher = BehaviorComposer(
            vocab_size=len(token_data["vocab"]),
            num_classes=num_classes,
            max_seq_len=int(model_cfg.get("max_seq_len", token_data.get("max_len", 256))),
            hidden_size=int(model_cfg.get("hidden_size", 128)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            num_heads=int(model_cfg.get("num_heads", 4)),
            intermediate_size=int(model_cfg.get("intermediate_size", 256)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            **pooling_params,
        ).to(device)
        _load_encoder_checkpoint(shared_teacher, str(shared_preservation_checkpoint), device, freeze_encoder=True)
        shared_teacher.eval()
        for parameter in shared_teacher.parameters():
            parameter.requires_grad = False
    learning_rate = float(train_cfg.get("learning_rate", 5e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    encoder_lr = train_cfg.get("encoder_learning_rate", train_cfg.get("encoder_lr"))
    encoder_lr_resolved = None if encoder_lr is None else float(encoder_lr)
    encoder_start_frozen = bool(freeze_encoder or freeze_encoder_epochs > 0)
    optimizer = torch.optim.AdamW(
        _optimizer_parameter_groups(
            model,
            proto_regularizer,
            base_lr=learning_rate,
            encoder_lr=encoder_lr_resolved,
            weight_decay=weight_decay,
            encoder_frozen=encoder_start_frozen,
        ),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_name = str(train_cfg.get("loss_type", train_cfg.get("loss", "ce")))
    if loss_name == "class_balanced":
        loss_name = "class_balanced_ce"
    loss_weights = _class_weights_for_parts(train_parts, num_classes).to(device)
    binary_loss_weights = _binary_class_weights_for_parts(train_parts, binary_benign_label_id).to(device)
    class_counts = _class_counts_for_parts(train_parts, num_classes).to(device)
    binary_class_counts = _class_counts_from_labels(_binary_targets(_labels_from_parts(train_parts), binary_benign_label_id), 2).to(device)
    if loss_name == "weighted_ce":
        criterion = nn.CrossEntropyLoss(weight=loss_weights)
        binary_criterion = nn.CrossEntropyLoss(weight=binary_loss_weights)
    elif loss_name == "focal":
        criterion = FocalLoss(gamma=float(train_cfg.get("focal_gamma", 2.0)))
        binary_criterion = FocalLoss(gamma=float(train_cfg.get("focal_gamma", 2.0)))
    elif loss_name == "weighted_focal":
        criterion = FocalLoss(gamma=float(train_cfg.get("focal_gamma", 2.0)), weight=loss_weights)
        binary_criterion = FocalLoss(gamma=float(train_cfg.get("focal_gamma", 2.0)), weight=binary_loss_weights)
    elif loss_name == "class_balanced_ce":
        cb_beta = float(train_cfg.get("cb_beta", 0.9999))
        criterion = nn.CrossEntropyLoss(weight=_effective_number_weights_from_counts(class_counts.cpu(), cb_beta).to(device))
        binary_criterion = nn.CrossEntropyLoss(weight=_effective_number_weights_from_counts(binary_class_counts.cpu(), cb_beta).to(device))
    elif loss_name == "class_balanced_focal":
        cb_beta = float(train_cfg.get("cb_beta", 0.9999))
        criterion = FocalLoss(
            gamma=float(train_cfg.get("focal_gamma", 2.0)),
            weight=_effective_number_weights_from_counts(class_counts.cpu(), cb_beta).to(device),
        )
        binary_criterion = FocalLoss(
            gamma=float(train_cfg.get("focal_gamma", 2.0)),
            weight=_effective_number_weights_from_counts(binary_class_counts.cpu(), cb_beta).to(device),
        )
    elif loss_name == "logit_adjusted_ce":
        tau = float(train_cfg.get("logit_adjust_tau", 1.0))
        criterion = LogitAdjustedCrossEntropyLoss(_class_priors_from_counts(class_counts.cpu()).to(device), tau=tau)
        binary_criterion = LogitAdjustedCrossEntropyLoss(_class_priors_from_counts(binary_class_counts.cpu()).to(device), tau=tau)
    elif loss_name == "weighted_logit_adjusted_ce":
        tau = float(train_cfg.get("logit_adjust_tau", 1.0))
        criterion = LogitAdjustedCrossEntropyLoss(_class_priors_from_counts(class_counts.cpu()).to(device), tau=tau, weight=loss_weights)
        binary_criterion = LogitAdjustedCrossEntropyLoss(_class_priors_from_counts(binary_class_counts.cpu()).to(device), tau=tau, weight=binary_loss_weights)
    elif loss_name == "ldam":
        criterion = LDAMLoss(
            class_counts.cpu(),
            max_m=float(train_cfg.get("ldam_max_m", 0.5)),
            scale=float(train_cfg.get("ldam_scale", 30.0)),
        )
        binary_criterion = LDAMLoss(
            binary_class_counts.cpu(),
            max_m=float(train_cfg.get("ldam_max_m", 0.5)),
            scale=float(train_cfg.get("ldam_scale", 30.0)),
        )
    elif loss_name == "weighted_ldam":
        criterion = LDAMLoss(
            class_counts.cpu(),
            max_m=float(train_cfg.get("ldam_max_m", 0.5)),
            scale=float(train_cfg.get("ldam_scale", 30.0)),
            weight=loss_weights,
        )
        binary_criterion = LDAMLoss(
            binary_class_counts.cpu(),
            max_m=float(train_cfg.get("ldam_max_m", 0.5)),
            scale=float(train_cfg.get("ldam_scale", 30.0)),
            weight=binary_loss_weights,
        )
    elif loss_name == "ce":
        criterion = nn.CrossEntropyLoss()
        binary_criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported training loss: {loss_name}")
    if encoder_start_frozen:
        _set_encoder_trainable(model, False)
    epochs = int(train_cfg.get("epochs", 3))
    best_state = None
    best_val = float("inf")
    history: list[dict[str, float]] = []
    encoder_unfreeze_epoch: int | None = None
    encoder_currently_frozen = bool(encoder_start_frozen)
    encoder_target_lr = float(encoder_lr_resolved if encoder_lr_resolved is not None else learning_rate)
    for epoch in range(1, epochs + 1):
        encoder_unfrozen_now = False
        if encoder_currently_frozen and not freeze_encoder and epoch > freeze_encoder_epochs:
            _set_encoder_trainable(model, True)
            _set_optimizer_encoder_state(optimizer, lr=encoder_target_lr, weight_decay=weight_decay)
            encoder_currently_frozen = False
            encoder_unfreeze_epoch = epoch
            encoder_unfrozen_now = True
        model.train()
        train_losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            batch_context = None
            batch_anomaly = None
            batch_stats = None
            input_ids, attention_mask, token_type_ids = batch[:3]
            batch_labels = batch[-1]
            cursor = 3
            if use_service_context:
                batch_context = batch[cursor]
                cursor += 1
            if use_anomaly_features:
                batch_anomaly = batch[cursor]
                cursor += 1
            if use_stat_fusion:
                batch_stats = batch[cursor]
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            if use_service_context or use_anomaly_features or use_stat_fusion or use_hierarchical_classifier or use_app_projection:
                context_arg = batch_context.to(device) if batch_context is not None else None
                anomaly_arg = batch_anomaly.to(device) if batch_anomaly is not None else None
                stat_arg = batch_stats.to(device) if batch_stats is not None else None
                if use_hierarchical_classifier:
                    binary_logits, logits = model.forward_heads(input_ids, attention_mask, token_type_ids, context_arg, anomaly_arg, stat_arg)
                else:
                    binary_logits = None
                    logits = model(input_ids, attention_mask, token_type_ids, context_arg, anomaly_arg, stat_arg)
            else:
                binary_logits = None
                logits = model(input_ids, attention_mask, token_type_ids)
            if use_hierarchical_classifier:
                loss = lambda_coarse * criterion(logits, batch_labels)
                if use_hierarchical_binary_loss:
                    binary_labels = _binary_targets(batch_labels, binary_benign_label_id)
                    loss = loss + lambda_binary * binary_criterion(binary_logits, binary_labels)
            else:
                loss = criterion(logits, batch_labels)
            if use_shared_preservation and shared_teacher is not None and shared_preservation_weight > 0.0:
                current_shared = _shared_representation_embedding(model, input_ids, attention_mask, token_type_ids)
                with torch.no_grad():
                    teacher_shared = _shared_representation_embedding(shared_teacher, input_ids, attention_mask, token_type_ids)
                loss = loss + shared_preservation_weight * shared_representation_preservation_loss(
                    current_shared,
                    teacher_shared,
                    mode=shared_preservation_mode,
                )
            if (use_supcon and lambda_supcon > 0.0) or (proto_regularizer is not None and lambda_class_proto > 0.0):
                if use_app_projection and hasattr(model, "app_features"):
                    reg_features = model.app_features(
                        input_ids,
                        attention_mask,
                        token_type_ids,
                        detach_shared=detach_shared_for_aux_loss,
                    )
                elif hasattr(model, "fused_features"):
                    reg_features = model.fused_features(input_ids, attention_mask, token_type_ids, context_arg, anomaly_arg, stat_arg)
                else:
                    reg_features = model.encode(input_ids, attention_mask, token_type_ids)
                if use_supcon and lambda_supcon > 0.0:
                    loss = loss + lambda_supcon * supervised_contrastive_loss(reg_features, batch_labels, temperature=supcon_temperature)
                if proto_regularizer is not None and lambda_class_proto > 0.0:
                    loss = loss + lambda_class_proto * proto_regularizer(reg_features, batch_labels)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_loss, val_true, val_pred, val_score = _evaluate(
            model,
            val_loader,
            device,
            use_hierarchical_classifier=use_hierarchical_classifier,
            use_hierarchical_binary_loss=use_hierarchical_binary_loss,
            lambda_binary=lambda_binary,
            lambda_coarse=lambda_coarse,
            benign_label_id=binary_benign_label_id,
            gated_inference=hierarchical_gated_inference,
        )
        val_metrics = classification_metrics(val_true, val_pred, val_score)
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            "val_macro_f1": val_metrics["macro_f1"],
            "encoder_frozen": bool(encoder_currently_frozen),
            "encoder_unfrozen_now": bool(encoder_unfrozen_now),
        }
        history.append(row)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    _, val_true, _, val_score = _evaluate(
        model,
        val_loader,
        device,
        use_hierarchical_classifier=use_hierarchical_classifier,
        use_hierarchical_binary_loss=use_hierarchical_binary_loss,
        lambda_binary=lambda_binary,
        lambda_coarse=lambda_coarse,
        benign_label_id=binary_benign_label_id,
        gated_inference=hierarchical_gated_inference,
    )
    test_loss, y_true, y_pred, y_score = _evaluate(
        model,
        test_loader,
        device,
        use_hierarchical_classifier=use_hierarchical_classifier,
        use_hierarchical_binary_loss=use_hierarchical_binary_loss,
        lambda_binary=lambda_binary,
        lambda_coarse=lambda_coarse,
        benign_label_id=binary_benign_label_id,
        gated_inference=hierarchical_gated_inference,
    )
    threshold = None
    if task == "binary":
        threshold, _ = _best_binary_threshold(val_true, val_score)
        y_pred = (y_score[:, 1] >= threshold).astype(int).tolist()
    metrics = classification_metrics(y_true, y_pred, y_score)
    metrics["test_loss"] = test_loss
    metrics["num_train"] = int(len(train_idx))
    metrics["num_train_full"] = int(len(train_idx_full))
    metrics["num_train_effective"] = int(len(train_loader.dataset))
    metrics["num_val"] = int(len(val_idx))
    metrics["num_test"] = int(len(test_idx))
    metrics["split"] = split
    metrics["task"] = task
    metrics["seed"] = seed
    metrics["split_seed"] = split_seed
    metrics["train_seed"] = resolved_train_seed
    metrics["label_fraction"] = float(label_fraction)
    metrics["train_seconds"] = float(time.perf_counter() - started)
    metrics["test_flows_per_second"] = float(len(test_idx) / max(metrics["train_seconds"], 1e-9))
    metrics["device"] = str(device)
    metrics["freeze_encoder_epochs"] = int(freeze_encoder_epochs)
    metrics["encoder_frozen_initially"] = bool(encoder_start_frozen)
    metrics["encoder_unfreeze_epoch"] = int(encoder_unfreeze_epoch) if encoder_unfreeze_epoch is not None else None
    metrics["encoder_frozen_finally"] = bool(encoder_currently_frozen)
    metrics["use_service_context"] = bool(use_service_context)
    metrics["use_anomaly_features"] = bool(use_anomaly_features)
    metrics["use_stat_fusion"] = bool(use_stat_fusion)
    metrics["use_hierarchical_classifier"] = bool(use_hierarchical_classifier)
    metrics["use_app_projection"] = bool(use_app_projection)
    if use_app_projection:
        metrics["app_projection"] = {
            "input": app_projection_input,
            "dim": int(app_projection_dim),
            "detach_shared_for_aux_loss": bool(detach_shared_for_aux_loss),
        }
    metrics["pooling_strategy"] = str(getattr(encoder, "pooling_strategy", pooling_strategy))
    metrics["pooling"] = str(getattr(encoder, "pooling_strategy", pooling_strategy))
    metrics["class_aware_pooling"] = bool(getattr(encoder, "class_aware_pooling", False))
    if getattr(encoder, "pooling_strategy", pooling_strategy) == "residual_class_aware":
        metrics["class_aware_alpha"] = float(getattr(encoder, "class_aware_alpha", model_cfg.get("class_aware_alpha", 0.5)))
        metrics["cls_beta"] = float(getattr(encoder, "cls_beta", model_cfg.get("cls_beta", 0.0)))
    metrics["use_supcon"] = bool(use_supcon)
    metrics["use_class_proto_loss"] = bool(use_class_proto_loss)
    metrics["use_app_proto_reg"] = bool(use_class_proto_loss)
    metrics["use_shared_representation_preservation"] = bool(use_shared_preservation)
    if use_shared_preservation:
        metrics["shared_representation"] = {
            "weight": float(shared_preservation_weight),
            "mode": shared_preservation_mode,
            "checkpoint": str(shared_preservation_checkpoint) if shared_preservation_checkpoint is not None else None,
            "teacher_loaded": bool(shared_teacher is not None),
        }
    if use_supcon:
        metrics["supcon"] = {
            "lambda_supcon": lambda_supcon,
            "supcon_weight": lambda_supcon,
            "temperature": supcon_temperature,
            "feature_space": "app_projection" if use_app_projection else "shared_or_fused",
        }
    if use_class_proto_loss:
        metrics["class_proto_loss"] = {
            "lambda_class_proto": lambda_class_proto,
            "app_proto_weight": lambda_class_proto,
            "feature_space": "app_projection" if use_app_projection else "shared_or_fused",
        }
    if use_stat_fusion:
        metrics["stat_fusion"] = {
            "feature_names": stat_feature_names,
            "feature_dim": stat_size,
            "projection_dim": stat_mlp_dim_resolved,
            "missing_or_unavailable_features": stat_normalizer.missing_names if stat_normalizer else [],
        }
    if use_hierarchical_classifier:
        metrics["hierarchical_classifier"] = {
            "lambda_binary": lambda_binary,
            "lambda_coarse": lambda_coarse,
            "lambda_fine": lambda_fine,
            "gated_inference": hierarchical_gated_inference,
            "binary_loss_enabled": use_hierarchical_binary_loss,
            "benign_label_id": int(benign_label_id_for_task) if benign_label_id_for_task is not None else None,
        }
    metrics["loss"] = loss_name
    metrics["sampling"] = train_sampling
    metrics["learning_rate"] = learning_rate
    if encoder_lr_resolved is not None:
        metrics["encoder_learning_rate"] = encoder_lr_resolved
        metrics["head_learning_rate"] = learning_rate
    if use_anomaly_features:
        metrics["anomaly_features"] = {
            "score_method": anomaly_score_method,
            "normalize": anomaly_normalize,
            "include_special": anomaly_include_special,
            "prototype_scope": anomaly_bank.prototype_scope if anomaly_bank else None,
            "benign_label_id": anomaly_bank.benign_label_id if anomaly_bank else None,
            "warnings": anomaly_bank.warnings if anomaly_bank else [],
            "use_service_prototype_distance": use_service_prototype_distance,
            "use_class_prototype_distance": use_class_prototype_distance,
            "feature_names": _anomaly_feature_names(
                num_classes,
                use_service_prototype_distance=use_service_prototype_distance,
                use_class_prototype_distance=use_class_prototype_distance,
            ),
            "feature_dim": anomaly_size,
            "projection_dim": anomaly_feature_dim_resolved,
        }
    if loss_name in {"focal", "weighted_focal", "class_balanced_focal"}:
        metrics["focal_gamma"] = float(train_cfg.get("focal_gamma", 2.0))
    if loss_name in {"class_balanced_ce", "class_balanced_focal"}:
        metrics["cb_beta"] = float(train_cfg.get("cb_beta", 0.9999))
    if loss_name in {"logit_adjusted_ce", "weighted_logit_adjusted_ce"}:
        metrics["logit_adjust_tau"] = float(train_cfg.get("logit_adjust_tau", 1.0))
    if loss_name in {"ldam", "weighted_ldam"}:
        metrics["ldam_max_m"] = float(train_cfg.get("ldam_max_m", 0.5))
        metrics["ldam_scale"] = float(train_cfg.get("ldam_scale", 30.0))
    if threshold is not None:
        metrics["threshold"] = threshold
    if checkpoint_info is not None:
        metrics["checkpoint"] = checkpoint_info
    if augment_info["enabled"]:
        metrics["augmentation"] = augment_info
    if extra_train_info["enabled"]:
        metrics["extra_train"] = extra_train_info
    target_names = None
    if task == "binary":
        target_names = ["BENIGN", "ATTACK"]
    else:
        inv = {idx: label for label, idx in (label_to_id or token_data["label_to_id"]).items()}
        target_names = [inv[idx] for idx in range(len(inv))]
    predictions = _prediction_rows(token_data, test_idx, y_true, y_pred, y_score, target_names)
    return TrainResult(
        metrics=metrics,
        report=report_dict(y_true, y_pred, target_names=target_names),
        confusion_matrix=confusion(y_true, y_pred),
        history=history,
        state_dict=best_state or {key: value.detach().cpu() for key, value in model.state_dict().items()},
        predictions=predictions,
    )

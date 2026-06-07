from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

from src.evaluation.metrics import classification_metrics, confusion, report_dict
from src.models.behavior_composer import BehaviorComposer, resolve_pooling_config
from src.training.classifier_trainer import (
    FocalLoss,
    LDAMLoss,
    LogitAdjustedCrossEntropyLoss,
    _class_counts_from_labels,
    _class_priors_from_counts,
    _class_weights_from_labels,
    _effective_number_weights_from_counts,
    _load_encoder_checkpoint,
    _split_indices,
    _task_labels,
)
from src.utils.seed import set_seed


@dataclass
class SessionTrainResult:
    metrics: dict[str, Any]
    report: dict[str, Any]
    confusion_matrix: list[list[int]]
    history: list[dict[str, float]]
    state_dict: dict[str, Any]
    predictions: list[dict[str, Any]]


def _safe_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    if value is None:
        return ("NONE",)
    return (str(value),)


def _group_key(meta: dict[str, Any], group_by: str) -> tuple[str, ...]:
    service_key = _safe_tuple(meta.get("service_key"))
    if group_by == "service_key":
        return ("service_key", *service_key)
    if group_by == "service_host_proto":
        host = service_key[0] if len(service_key) >= 1 else str(meta.get("dst_ip") or meta.get("src_ip") or "NONE")
        proto = service_key[2] if len(service_key) >= 3 else str(meta.get("protocol") or "NONE")
        return ("service_host_proto", host, proto)
    if group_by == "src_dst_pair":
        return ("src_dst_pair", str(meta.get("src_ip") or "NONE"), str(meta.get("dst_ip") or "NONE"))
    if group_by == "src_dst_proto":
        return (
            "src_dst_proto",
            str(meta.get("src_ip") or "NONE"),
            str(meta.get("dst_ip") or "NONE"),
            str(meta.get("protocol") or "NONE"),
        )
    if group_by == "global_time":
        return ("global_time",)
    raise ValueError(f"Unsupported session group_by: {group_by}")


def merge_flow_metadata(
    token_data: dict[str, Any],
    flow_rows: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fill non-label flow metadata in token_data["meta"] by flow_id."""
    if flow_rows is None:
        return token_data, {"metadata_flows": None, "matched_metadata_rows": 0, "missing_metadata_rows": 0}
    by_flow_id = {str(row.get("flow_id")): row for row in flow_rows if row.get("flow_id") is not None}
    fill_keys = [
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "protocol",
        "dataset_file",
        "start_ts",
        "end_ts",
        "duration",
        "service_key",
    ]
    matched = 0
    missing = 0
    merged_meta: list[dict[str, Any]] = []
    for row in token_data.get("meta", []):
        out = dict(row)
        flow = by_flow_id.get(str(row.get("flow_id")))
        if flow is None:
            missing += 1
        else:
            matched += 1
            for key in fill_keys:
                if out.get(key) is None and flow.get(key) is not None:
                    out[key] = flow[key]
        merged_meta.append(out)
    out_data = dict(token_data)
    out_data["meta"] = merged_meta
    total = len(merged_meta)
    return out_data, {
        "metadata_flow_rows": int(len(by_flow_id)),
        "matched_metadata_rows": int(matched),
        "missing_metadata_rows": int(missing),
        "metadata_match_rate": float(matched / total) if total else 0.0,
    }


def _sort_key(index: int, meta_rows: list[dict[str, Any]]) -> tuple[float, int, str]:
    meta = meta_rows[index] if index < len(meta_rows) else {}
    try:
        ts = float(meta.get("start_ts"))
    except (TypeError, ValueError):
        ts = float("inf")
    return ts, index, str(meta.get("flow_id") or "")


def _majority_label(ids: list[int]) -> tuple[int, float, dict[int, int]]:
    counts = Counter(int(item) for item in ids)
    label, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return int(label), float(count / max(1, len(ids))), {int(key): int(value) for key, value in counts.items()}


def _session_windows(indices: list[int], meta_rows: list[dict[str, Any]], *, window_n: int, stride: int) -> list[list[int]]:
    ordered = sorted(indices, key=lambda idx: _sort_key(idx, meta_rows))
    size = max(1, int(window_n))
    step = max(1, int(stride))
    if len(ordered) <= size:
        return [ordered]
    return [ordered[start : start + size] for start in range(0, len(ordered) - size + 1, step)]


def build_session_windows(
    token_data: dict[str, Any],
    labels: torch.Tensor,
    indices: np.ndarray,
    *,
    group_by: str,
    window_n: int,
    stride: int,
    min_flows: int,
    min_purity: float,
) -> tuple[TensorDataset, dict[str, Any], list[dict[str, Any]]]:
    meta_rows = list(token_data.get("meta", []))
    if len(meta_rows) != len(token_data["input_ids"]):
        raise ValueError("token_data meta row count must match input_ids row count")
    label_to_id = token_data.get("label_to_id") or {}
    id_to_label = {int(idx): str(label) for label, idx in label_to_id.items()}

    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for idx in indices.tolist():
        groups[_group_key(meta_rows[int(idx)], group_by)].append(int(idx))

    seq_len = int(token_data["input_ids"].shape[1])
    pad_id = int(token_data["vocab"].get("[PAD]", 0))
    pad_ids = torch.full((seq_len,), pad_id, dtype=torch.long)
    pad_mask = torch.zeros((seq_len,), dtype=torch.long)
    pad_types = torch.zeros((seq_len,), dtype=torch.long)

    input_ids: list[torch.Tensor] = []
    attention_masks: list[torch.Tensor] = []
    token_type_ids: list[torch.Tensor] = []
    flow_masks: list[torch.Tensor] = []
    session_labels: list[int] = []
    session_meta: list[dict[str, Any]] = []
    group_sizes: list[int] = []
    dropped_short = 0
    dropped_low_purity = 0

    for group_key, group_indices in groups.items():
        group_sizes.append(len(group_indices))
        for window in _session_windows(group_indices, meta_rows, window_n=window_n, stride=stride):
            if len(window) < int(min_flows):
                dropped_short += 1
                continue
            window_label_ids = [int(labels[idx].item()) for idx in window]
            majority_id, purity, label_counts = _majority_label(window_label_ids)
            if purity < float(min_purity):
                dropped_low_purity += 1
                continue

            window_len = len(window)
            padded_window = list(window) + [-1] * max(0, int(window_n) - window_len)
            flow_ids: list[str | None] = []
            flow_labels: list[str | None] = []
            flow_start_ts: list[float | None] = []
            stacked_ids: list[torch.Tensor] = []
            stacked_masks: list[torch.Tensor] = []
            stacked_types: list[torch.Tensor] = []
            flow_mask: list[int] = []
            for idx in padded_window[: int(window_n)]:
                if idx < 0:
                    stacked_ids.append(pad_ids.clone())
                    stacked_masks.append(pad_mask.clone())
                    stacked_types.append(pad_types.clone())
                    flow_ids.append(None)
                    flow_labels.append(None)
                    flow_start_ts.append(None)
                    flow_mask.append(0)
                    continue
                stacked_ids.append(token_data["input_ids"][idx].clone())
                stacked_masks.append(token_data["attention_mask"][idx].clone())
                stacked_types.append(token_data["token_type_ids"][idx].clone())
                flow_ids.append(str(meta_rows[idx].get("flow_id")))
                flow_labels.append(str(meta_rows[idx].get("label")))
                try:
                    flow_start_ts.append(float(meta_rows[idx].get("start_ts")))
                except (TypeError, ValueError):
                    flow_start_ts.append(None)
                flow_mask.append(1)

            input_ids.append(torch.stack(stacked_ids, dim=0))
            attention_masks.append(torch.stack(stacked_masks, dim=0))
            token_type_ids.append(torch.stack(stacked_types, dim=0))
            flow_masks.append(torch.tensor(flow_mask, dtype=torch.long))
            session_labels.append(int(majority_id))
            session_meta.append(
                {
                    "session_id": len(session_meta),
                    "group_key": list(group_key),
                    "group_by": group_by,
                    "source_indices": [int(idx) for idx in window],
                    "source_flow_ids": flow_ids,
                    "source_labels": flow_labels,
                    "source_start_ts": flow_start_ts,
                    "majority_label_id": int(majority_id),
                    "majority_label": id_to_label.get(int(majority_id), str(majority_id)),
                    "label_counts": {id_to_label.get(int(key), str(key)): int(value) for key, value in label_counts.items()},
                    "label_purity": float(purity),
                    "flow_count": int(window_len),
                    "padded_flow_count": int(window_n),
                }
            )

    if not input_ids:
        raise ValueError("No session windows were built. Lower --min_flows or check grouping metadata.")

    dataset = TensorDataset(
        torch.stack(input_ids, dim=0),
        torch.stack(attention_masks, dim=0),
        torch.stack(token_type_ids, dim=0),
        torch.stack(flow_masks, dim=0),
        torch.tensor(session_labels, dtype=torch.long),
    )
    stats = {
        "num_groups": int(len(groups)),
        "avg_group_size": float(np.mean(group_sizes)) if group_sizes else 0.0,
        "max_group_size": int(max(group_sizes)) if group_sizes else 0,
        "num_sessions": int(len(session_meta)),
        "avg_flows_per_session": float(np.mean([row["flow_count"] for row in session_meta])) if session_meta else 0.0,
        "max_flows_per_session": int(max([row["flow_count"] for row in session_meta], default=0)),
        "dropped_short_sessions": int(dropped_short),
        "dropped_low_purity_sessions": int(dropped_low_purity),
        "min_flows": int(min_flows),
        "min_purity": float(min_purity),
        "window_n": int(window_n),
        "stride": int(stride),
        "group_by": group_by,
    }
    return dataset, stats, session_meta


class SessionAggregationClassifier(nn.Module):
    def __init__(
        self,
        encoder: BehaviorComposer,
        hidden_size: int,
        num_classes: int,
        *,
        dropout: float = 0.1,
        flow_representation: str = "shared_encode",
        session_pooling_strategy: str = "mean",
        session_window_n: int = 8,
        session_transformer_layers: int = 1,
        session_transformer_heads: int = 4,
        session_transformer_intermediate_size: int | None = None,
        encoder_frozen: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.hidden_size = int(hidden_size)
        self.num_classes = int(num_classes)
        self.flow_representation = str(flow_representation)
        self.session_pooling_strategy = str(session_pooling_strategy)
        self.session_window_n = int(session_window_n)
        self.encoder_frozen = bool(encoder_frozen)
        if self.flow_representation not in {"shared_encode", "classification_embedding", "class_aware_summary", "residual_class_aware_pool"}:
            raise ValueError(f"Unsupported flow_representation: {self.flow_representation}")
        if self.session_pooling_strategy not in {"mean", "attentive", "transformer_mean", "transformer_attentive"}:
            raise ValueError(f"Unsupported session_pooling_strategy: {self.session_pooling_strategy}")
        self.use_session_transformer = self.session_pooling_strategy.startswith("transformer_")
        if self.session_pooling_strategy in {"attentive", "transformer_attentive"}:
            self.attention = nn.Linear(self.hidden_size, 1)
        if self.use_session_transformer:
            self.session_position_embedding = nn.Embedding(max(1, self.session_window_n), self.hidden_size)
            session_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=int(session_transformer_heads),
                dim_feedforward=int(session_transformer_intermediate_size or self.hidden_size * 2),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.session_encoder = nn.TransformerEncoder(session_layer, num_layers=max(1, int(session_transformer_layers)))
            self.session_norm = nn.LayerNorm(self.hidden_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_classes),
        )

    def _flow_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, window_n, seq_len = input_ids.shape
        flat_ids = input_ids.reshape(batch_size * window_n, seq_len)
        flat_mask = attention_mask.reshape(batch_size * window_n, seq_len) if attention_mask is not None else None
        flat_types = token_type_ids.reshape(batch_size * window_n, seq_len) if token_type_ids is not None else None
        padded_rows = None
        if flat_mask is not None:
            padded_rows = flat_mask.sum(dim=1) == 0
            if bool(padded_rows.any()):
                # Transformer attention can return NaNs when every token in a row
                # is masked. Let padded flow rows pass with one dummy visible token,
                # then zero their embeddings before session pooling.
                flat_mask = flat_mask.clone()
                flat_mask[padded_rows, 0] = 1
        grad_enabled = not self.encoder_frozen and any(parameter.requires_grad for parameter in self.encoder.parameters())
        with torch.set_grad_enabled(grad_enabled):
            if self.flow_representation == "classification_embedding":
                flow_embeddings = self.encoder.classification_embedding(flat_ids, flat_mask, flat_types)
            elif self.flow_representation == "class_aware_summary":
                token_hidden = self.encoder.encode_tokens(flat_ids, flat_mask, flat_types)
                if not hasattr(self.encoder, "class_aware_summary"):
                    raise RuntimeError("class_aware_summary requires a class-aware encoder")
                flow_embeddings = self.encoder.class_aware_summary(token_hidden, flat_mask)
            elif self.flow_representation == "residual_class_aware_pool":
                token_hidden = self.encoder.encode_tokens(flat_ids, flat_mask, flat_types)
                if not hasattr(self.encoder, "residual_class_aware_pool"):
                    raise RuntimeError("residual_class_aware_pool requires a compatible encoder")
                flow_embeddings = self.encoder.residual_class_aware_pool(token_hidden, flat_mask)
            else:
                flow_embeddings = self.encoder.encode(flat_ids, flat_mask, flat_types)
        flow_embeddings = torch.nan_to_num(flow_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        if padded_rows is not None and bool(padded_rows.any()):
            flow_embeddings = flow_embeddings.masked_fill(padded_rows.unsqueeze(-1), 0.0)
        return flow_embeddings.reshape(batch_size, window_n, -1)

    def session_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        flow_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flow_embeddings = self._flow_embeddings(input_ids, attention_mask, token_type_ids)
        if flow_mask is None:
            flow_mask = torch.ones(flow_embeddings.shape[:2], device=flow_embeddings.device, dtype=torch.long)
        valid = flow_mask.to(dtype=torch.bool)
        if self.use_session_transformer:
            positions = torch.arange(flow_embeddings.shape[1], device=flow_embeddings.device).clamp(max=self.session_window_n - 1)
            flow_embeddings = flow_embeddings + self.session_position_embedding(positions).unsqueeze(0)
            flow_embeddings = self.session_encoder(flow_embeddings, src_key_padding_mask=~valid)
            flow_embeddings = self.session_norm(torch.nan_to_num(flow_embeddings, nan=0.0, posinf=0.0, neginf=0.0))
            flow_embeddings = flow_embeddings.masked_fill(~valid.unsqueeze(-1), 0.0)
        if self.session_pooling_strategy in {"mean", "transformer_mean"}:
            mask = valid.to(dtype=flow_embeddings.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            return (flow_embeddings * mask).sum(dim=1) / denom
        scores = self.attention(flow_embeddings).squeeze(-1)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), flow_embeddings).squeeze(1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        flow_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = self.session_embedding(input_ids, attention_mask, token_type_ids, flow_mask)
        return self.classifier(pooled)


def _target_names(task: str, label_to_id: dict[str, int] | None) -> list[str]:
    if task == "binary":
        return ["BENIGN", "ATTACK"]
    if label_to_id is None:
        raise ValueError(f"{task} requires a label mapping")
    inv = {idx: label for label, idx in label_to_id.items()}
    return [inv[idx] for idx in range(len(inv))]


def _session_loader(dataset: TensorDataset, batch_size: int, shuffle: bool, sampler: WeightedRandomSampler | None = None) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False, sampler=sampler)


def _best_binary_threshold(y_true: list[int], scores: np.ndarray) -> tuple[float, float]:
    if scores.shape[1] < 2:
        return 0.5, 0.0
    probs = scores[:, 1]
    best_threshold = 0.5
    best_macro_f1 = -1.0
    for threshold in np.linspace(0.01, 0.99, 99):
        pred = (probs >= threshold).astype(int).tolist()
        cur_macro_f1 = classification_metrics(y_true, pred, scores)["macro_f1"]
        if cur_macro_f1 > best_macro_f1:
            best_macro_f1 = float(cur_macro_f1)
            best_threshold = float(threshold)
    return best_threshold, best_macro_f1


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, list[int], list[int], np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, token_type_ids, flow_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            flow_mask = flow_mask.to(device)
            labels = labels.to(device)
            logits = model(input_ids, attention_mask, token_type_ids, flow_mask)
            probs = torch.softmax(logits, dim=-1)
            losses.append(float(criterion(logits, labels).item()))
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            scores.append(probs.cpu().numpy())
    return float(np.mean(losses) if losses else 0.0), y_true, y_pred, np.concatenate(scores, axis=0)


def train_session_classifier(
    token_data: dict[str, Any],
    config: dict[str, Any],
    task: str = "multiclass",
    split: str = "temporal_stratified",
    checkpoint: str | None = None,
    freeze_encoder: bool = True,
) -> SessionTrainResult:
    started = time.perf_counter()
    seed = int(config.get("seed", 42))
    set_seed(seed)
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    session_group_by = str(model_cfg.get("session_group_by", "service_host_proto"))
    session_window_n = int(model_cfg.get("session_window_n", 8))
    session_stride = int(model_cfg.get("session_stride", 1))
    session_min_flows = int(model_cfg.get("session_min_flows", 2))
    session_min_purity = float(model_cfg.get("session_min_purity", 0.8))
    session_train_min_purity = float(model_cfg.get("session_train_min_purity", session_min_purity))
    session_eval_min_purity = float(model_cfg.get("session_eval_min_purity", session_min_purity))
    session_pooling_strategy = str(model_cfg.get("session_pooling_strategy", "mean"))
    session_flow_representation = str(model_cfg.get("session_flow_representation", "shared_encode"))
    flow_pooling_strategy = str(model_cfg.get("pooling", model_cfg.get("pooling_strategy", "cls")))
    session_transformer_layers = int(model_cfg.get("session_transformer_layers", 1))
    session_transformer_heads = int(model_cfg.get("session_transformer_heads", model_cfg.get("num_heads", 4)))
    session_transformer_intermediate_size = int(model_cfg.get("session_transformer_intermediate_size", int(model_cfg.get("hidden_size", 128)) * 2))

    labels, label_to_id = _task_labels(token_data, task)
    labels_np = labels.numpy()
    meta_rows = token_data.get("meta", [])
    if len(meta_rows) == len(labels_np):
        order_values = np.array([float(meta.get("start_ts") or idx) for idx, meta in enumerate(meta_rows)])
    else:
        order_values = np.arange(len(labels_np), dtype=float)
    train_idx, val_idx, test_idx = _split_indices(
        labels_np,
        val_ratio=float(train_cfg.get("val_ratio", 0.1)),
        test_ratio=float(train_cfg.get("test_ratio", 0.2)),
        seed=seed,
        split=split,
        order_values=order_values,
    )

    train_dataset, train_stats, train_sessions = build_session_windows(
        token_data,
        labels,
        train_idx,
        group_by=session_group_by,
        window_n=session_window_n,
        stride=session_stride,
        min_flows=session_min_flows,
        min_purity=session_train_min_purity,
    )
    val_dataset, val_stats, val_sessions = build_session_windows(
        token_data,
        labels,
        val_idx,
        group_by=session_group_by,
        window_n=session_window_n,
        stride=session_stride,
        min_flows=session_min_flows,
        min_purity=session_eval_min_purity,
    )
    test_dataset, test_stats, test_sessions = build_session_windows(
        token_data,
        labels,
        test_idx,
        group_by=session_group_by,
        window_n=session_window_n,
        stride=session_stride,
        min_flows=session_min_flows,
        min_purity=session_eval_min_purity,
    )

    batch_size = int(train_cfg.get("batch_size", 64))
    train_labels = train_dataset.tensors[-1]
    train_sampling = str(train_cfg.get("sampling", "shuffle"))
    sampler = None
    if train_sampling == "class_balanced":
        counts = torch.bincount(train_labels, minlength=int(labels.max().item()) + 1).float().clamp_min(1.0)
        sample_weights = (1.0 / counts)[train_labels]
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(sample_weights.double(), num_samples=int(len(train_labels)), replacement=True, generator=generator)
    elif train_sampling != "shuffle":
        raise ValueError(f"Unsupported session training sampling: {train_sampling}")

    train_loader = _session_loader(train_dataset, batch_size=batch_size, shuffle=True, sampler=sampler)
    val_loader = _session_loader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = _session_loader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = BehaviorComposer(
        vocab_size=len(token_data["vocab"]),
        num_classes=int(labels.max().item()) + 1,
        max_seq_len=int(model_cfg.get("max_seq_len", token_data.get("max_len", 256))),
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        intermediate_size=int(model_cfg.get("intermediate_size", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        **resolve_pooling_config(model_cfg),
    )
    checkpoint_info = None
    if checkpoint:
        checkpoint_info = _load_encoder_checkpoint(encoder, checkpoint, device, freeze_encoder=freeze_encoder)
    model = SessionAggregationClassifier(
        encoder,
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        num_classes=int(labels.max().item()) + 1,
        dropout=float(model_cfg.get("dropout", 0.1)),
        flow_representation=session_flow_representation,
        session_pooling_strategy=session_pooling_strategy,
        session_window_n=session_window_n,
        session_transformer_layers=session_transformer_layers,
        session_transformer_heads=session_transformer_heads,
        session_transformer_intermediate_size=session_transformer_intermediate_size,
        encoder_frozen=bool(freeze_encoder),
    ).to(device)

    if freeze_encoder:
        for name, parameter in model.named_parameters():
            if name.startswith("encoder."):
                parameter.requires_grad = False

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters remain. Disable freeze_encoder or provide a compatible checkpoint.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(train_cfg.get("learning_rate", 5e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    loss_name = str(train_cfg.get("loss_type", train_cfg.get("loss", "weighted_ce")))
    train_class_counts = _class_counts_from_labels(train_labels, int(labels.max().item()) + 1)
    train_weights = _class_weights_from_labels(train_labels, int(labels.max().item()) + 1).to(device)
    if loss_name == "weighted_ce":
        criterion = nn.CrossEntropyLoss(weight=train_weights)
    elif loss_name == "ce":
        criterion = nn.CrossEntropyLoss()
    elif loss_name == "focal":
        criterion = FocalLoss(gamma=float(train_cfg.get("focal_gamma", 2.0)))
    elif loss_name == "weighted_focal":
        criterion = FocalLoss(gamma=float(train_cfg.get("focal_gamma", 2.0)), weight=train_weights)
    elif loss_name == "class_balanced_ce":
        criterion = nn.CrossEntropyLoss(weight=_effective_number_weights_from_counts(train_class_counts, float(train_cfg.get("cb_beta", 0.9999))).to(device))
    elif loss_name == "class_balanced_focal":
        criterion = FocalLoss(
            gamma=float(train_cfg.get("focal_gamma", 2.0)),
            weight=_effective_number_weights_from_counts(train_class_counts, float(train_cfg.get("cb_beta", 0.9999))).to(device),
        )
    elif loss_name == "logit_adjusted_ce":
        criterion = LogitAdjustedCrossEntropyLoss(_class_priors_from_counts(train_class_counts).to(device), tau=float(train_cfg.get("logit_adjust_tau", 1.0)))
    elif loss_name == "weighted_logit_adjusted_ce":
        criterion = LogitAdjustedCrossEntropyLoss(
            _class_priors_from_counts(train_class_counts).to(device),
            tau=float(train_cfg.get("logit_adjust_tau", 1.0)),
            weight=train_weights,
        )
    elif loss_name == "ldam":
        criterion = LDAMLoss(train_class_counts, max_m=float(train_cfg.get("ldam_max_m", 0.5)), scale=float(train_cfg.get("ldam_scale", 30.0)))
    elif loss_name == "weighted_ldam":
        criterion = LDAMLoss(
            train_class_counts,
            max_m=float(train_cfg.get("ldam_max_m", 0.5)),
            scale=float(train_cfg.get("ldam_scale", 30.0)),
            weight=train_weights,
        )
    else:
        raise ValueError(f"Unsupported training loss: {loss_name}")

    epochs = int(train_cfg.get("epochs", 3))
    best_state = None
    best_val = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            input_ids, attention_mask, token_type_ids, flow_mask, batch_labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            flow_mask = flow_mask.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask, token_type_ids, flow_mask)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_loss, val_true, val_pred, val_score = _evaluate(model, val_loader, device)
        val_metrics = classification_metrics(val_true, val_pred, val_score)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(train_losses) if train_losses else 0.0),
                "val_loss": float(val_loss),
                "val_macro_f1": float(val_metrics["macro_f1"]),
            }
        )
        if val_loss < best_val:
            best_val = float(val_loss)
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, val_true, val_pred, val_score = _evaluate(model, val_loader, device)
    test_loss, y_true, y_pred, y_score = _evaluate(model, test_loader, device)
    threshold = None
    if task == "binary":
        threshold, _ = _best_binary_threshold(val_true, val_score)
        y_pred = (y_score[:, 1] >= threshold).astype(int).tolist()
    metrics = classification_metrics(y_true, y_pred, y_score)
    metrics.update(
        {
            "val_loss": float(val_loss),
            "test_loss": float(test_loss),
            "num_train": int(len(train_idx)),
            "num_val": int(len(val_idx)),
            "num_test": int(len(test_idx)),
            "num_session_train": int(len(train_sessions)),
            "num_session_val": int(len(val_sessions)),
            "num_session_test": int(len(test_sessions)),
            "split": split,
            "task": task,
            "seed": seed,
            "train_seconds": float(time.perf_counter() - started),
            "device": str(device),
            "flow_pooling_strategy": flow_pooling_strategy,
            "session_pooling_strategy": session_pooling_strategy,
            "session_transformer_layers": int(session_transformer_layers),
            "session_transformer_heads": int(session_transformer_heads),
            "session_transformer_intermediate_size": int(session_transformer_intermediate_size),
            "session_group_by": session_group_by,
            "session_window_n": int(session_window_n),
            "session_stride": int(session_stride),
            "session_min_flows": int(session_min_flows),
            "session_min_purity": float(session_min_purity),
            "session_train_min_purity": float(session_train_min_purity),
            "session_eval_min_purity": float(session_eval_min_purity),
            "session_flow_representation": session_flow_representation,
            "freeze_encoder": bool(freeze_encoder),
            "train_sampling": train_sampling,
            "loss": loss_name,
            "session_stats": {
                "train": train_stats,
                "val": val_stats,
                "test": test_stats,
            },
        }
    )
    if checkpoint_info is not None:
        metrics["checkpoint"] = checkpoint_info
    if threshold is not None:
        metrics["threshold"] = float(threshold)

    target_names = _target_names(task, label_to_id)
    predictions: list[dict[str, Any]] = []
    for offset, session_row in enumerate(test_sessions):
        source_indices = session_row.get("source_indices", [])
        meta = meta_rows[int(source_indices[0])] if source_indices else {}
        pred_id = int(y_pred[offset])
        true_id = int(y_true[offset])
        score_row = y_score[offset]
        predictions.append(
            {
                "index": int(source_indices[0]) if source_indices else -1,
                "flow_id": meta.get("flow_id"),
                "label": meta.get("label"),
                "true_label": target_names[true_id] if true_id < len(target_names) else str(true_id),
                "pred_label": target_names[pred_id] if pred_id < len(target_names) else str(pred_id),
                "pred_confidence": float(score_row[pred_id]),
                "start_ts": meta.get("start_ts"),
                "dataset_file": meta.get("dataset_file"),
                "service_key": meta.get("service_key"),
                "session_id": session_row.get("session_id"),
                "session_group_key": session_row.get("group_key"),
                "session_flow_ids": session_row.get("source_flow_ids"),
                "session_flow_labels": session_row.get("source_labels"),
                "session_label_purity": session_row.get("label_purity"),
                "scores": {target_names[class_idx]: float(score) for class_idx, score in enumerate(score_row[: len(target_names)])},
            }
        )
    return SessionTrainResult(
        metrics=metrics,
        report=report_dict(y_true, y_pred, target_names=target_names),
        confusion_matrix=confusion(y_true, y_pred),
        history=history,
        state_dict=best_state or {key: value.detach().cpu() for key, value in model.state_dict().items()},
        predictions=predictions,
    )

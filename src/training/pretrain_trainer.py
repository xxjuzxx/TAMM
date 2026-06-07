from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.behavior_composer import BehaviorComposer, resolve_pooling_config
from src.utils.seed import set_seed


@dataclass
class PretrainResult:
    metrics: dict[str, Any]
    history: list[dict[str, float]]
    state_dict: dict[str, Any]


class BehaviorComposerPretrainer(nn.Module):
    def __init__(self, encoder: BehaviorComposer, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.mlm_head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, vocab_size))
        self.rtd_head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))
        self.projection = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size))
        self.paragraph_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder.encode_tokens(input_ids, attention_mask, token_type_ids)
        return self.mlm_head(encoded), self.rtd_head(encoded).squeeze(-1), self.projection(encoded[:, 0, :])

    def contrastive_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.projection(self.encoder.encode(input_ids, attention_mask, token_type_ids))

    def paragraph_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.paragraph_projection(self.encoder.encode(input_ids, attention_mask, token_type_ids))


def _split_indices(n_items: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n_items)
    train_idx, val_idx = train_test_split(indices, test_size=val_ratio, random_state=seed, shuffle=True)
    return train_idx, val_idx


def _loader(token_data: dict[str, Any], indices: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    idx = torch.tensor(indices, dtype=torch.long)
    dataset = TensorDataset(
        idx,
        token_data["input_ids"][idx],
        token_data["attention_mask"][idx],
        token_data["token_type_ids"][idx],
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _service_labels(token_data: dict[str, Any], indices: np.ndarray) -> torch.Tensor:
    labels: list[str] = []
    for idx in indices.tolist():
        service_key = token_data.get("meta", [])[idx].get("service_key") if idx < len(token_data.get("meta", [])) else None
        labels.append("|".join(str(item) for item in service_key) if service_key else "GLOBAL")
    label_to_id = {label: pos for pos, label in enumerate(sorted(set(labels)))}
    return torch.tensor([label_to_id[label] for label in labels], dtype=torch.long)


def _context_group_key(meta: dict[str, Any], group_by: str) -> tuple[str, ...]:
    service_key = meta.get("service_key")
    if isinstance(service_key, (list, tuple)) and service_key:
        service_key = tuple(str(item) for item in service_key)
    elif service_key not in (None, ""):
        service_key = (str(service_key),)
    else:
        service_key = ("NONE",)
    if group_by == "service_key":
        return ("service_key", *service_key)
    if group_by == "service_host_proto":
        host = service_key[0] if len(service_key) >= 1 else "NONE"
        proto = service_key[2] if len(service_key) >= 3 else str(meta.get("protocol") or "NONE")
        return ("service_host_proto", host, proto)
    if group_by == "global_time":
        return ("global_time",)
    raise ValueError(f"Unsupported context group_by: {group_by}")


def _context_sort_key(index: int, meta_rows: list[dict[str, Any]]) -> tuple[float, int, str]:
    meta = meta_rows[index] if index < len(meta_rows) else {}
    try:
        ts = float(meta.get("start_ts"))
    except (TypeError, ValueError):
        ts = float("inf")
    return ts, index, str(meta.get("flow_id") or "")


def _build_context_positive_map(
    token_data: dict[str, Any],
    *,
    group_by: str,
    positive_mode: str = "next",
    indices: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    meta_rows = list(token_data.get("meta", []))
    positive_map = np.full(len(meta_rows), -1, dtype=np.int64)
    if not meta_rows:
        return positive_map, {
            "context_group_by": group_by,
            "context_positive_mode": positive_mode,
            "num_context_groups": 0,
            "num_context_pairs": 0,
            "num_context_flows_with_positive": 0,
        }

    selected = np.arange(len(meta_rows)) if indices is None else np.asarray(indices, dtype=np.int64)
    selected_set = {int(idx) for idx in selected.tolist()}
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for idx in selected.tolist():
        meta = meta_rows[int(idx)]
        groups[_context_group_key(meta, group_by)].append(idx)

    pair_count = 0
    for group_indices in groups.values():
        ordered = sorted(group_indices, key=lambda idx: _context_sort_key(idx, meta_rows))
        if len(ordered) < 2:
            continue
        for pos, idx in enumerate(ordered):
            if positive_mode in {"next", "forward"}:
                if pos + 1 < len(ordered):
                    positive = ordered[pos + 1]
                else:
                    positive = ordered[pos - 1]
            elif positive_mode in {"previous", "backward"}:
                if pos > 0:
                    positive = ordered[pos - 1]
                else:
                    positive = ordered[1]
            elif positive_mode in {"nearest", "bidirectional"}:
                left = ordered[pos - 1] if pos > 0 else None
                right = ordered[pos + 1] if pos + 1 < len(ordered) else None
                if left is None and right is None:
                    continue
                if left is None:
                    positive = right
                elif right is None:
                    positive = left
                else:
                    left_gap = abs(_context_sort_key(idx, meta_rows)[0] - _context_sort_key(left, meta_rows)[0])
                    right_gap = abs(_context_sort_key(right, meta_rows)[0] - _context_sort_key(idx, meta_rows)[0])
                    positive = left if left_gap <= right_gap else right
            else:
                raise ValueError(f"Unsupported context_positive_mode: {positive_mode}")
            if positive == idx:
                continue
            positive_map[idx] = int(positive)
            pair_count += 1

    return positive_map, {
        "context_group_by": group_by,
        "context_positive_mode": positive_mode,
        "num_context_groups": int(len(groups)),
        "num_context_pairs": int(pair_count),
        "num_context_flows_with_positive": int(np.sum(positive_map >= 0)),
    }


def _mask_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    vocab_size: int,
    mask_id: int,
    special_ids: set[int],
    mlm_probability: float,
    replace_probability: float,
    mask_strategy: str = "random",
    span_length: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = input_ids.device
    special = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in special_ids:
        special |= input_ids == token_id
    can_mask = (attention_mask > 0) & (~special)
    if mask_strategy == "random":
        rand = torch.rand(input_ids.shape, device=device)
        masked = (rand < mlm_probability) & can_mask
    elif mask_strategy == "span":
        masked = _span_mask(can_mask, mlm_probability=mlm_probability, span_length=span_length)
    else:
        raise ValueError(f"Unsupported mask_strategy: {mask_strategy}")
    labels = torch.full_like(input_ids, fill_value=-100)
    labels[masked] = input_ids[masked]

    corrupted = input_ids.clone()
    corrupted[masked] = mask_id
    replace_rand = torch.rand(input_ids.shape, device=device)
    replaced = (replace_rand < replace_probability) & can_mask & (~masked)
    random_tokens = torch.randint(low=0, high=vocab_size, size=input_ids.shape, device=device)
    corrupted[replaced] = random_tokens[replaced]
    return corrupted, labels, replaced.long()


def _span_mask(can_mask: torch.Tensor, *, mlm_probability: float, span_length: int) -> torch.Tensor:
    if can_mask.numel() == 0:
        return torch.zeros_like(can_mask, dtype=torch.bool)
    span = max(1, int(span_length))
    masked = torch.zeros_like(can_mask, dtype=torch.bool)
    seq_len = can_mask.shape[1]
    start_prob = min(1.0, float(mlm_probability) / float(span))
    starts = (torch.rand(can_mask.shape, device=can_mask.device) < start_prob) & can_mask
    for offset in range(span):
        if offset == 0:
            masked |= starts
        else:
            masked[:, offset:] |= starts[:, : seq_len - offset]
    masked &= can_mask
    return masked


def _contrastive_loss(embeddings: torch.Tensor, service_labels: torch.Tensor, temperature: float) -> torch.Tensor:
    if embeddings.shape[0] < 2:
        return embeddings.sum() * 0.0
    z = torch.nn.functional.normalize(embeddings, dim=1)
    logits = (z @ z.T) / temperature
    logits = logits - torch.eye(logits.shape[0], device=logits.device) * 1e9
    same = service_labels.unsqueeze(0) == service_labels.unsqueeze(1)
    same.fill_diagonal_(False)
    positive_counts = same.sum(dim=1)
    valid = positive_counts > 0
    if not torch.any(valid):
        return embeddings.sum() * 0.0
    log_probs = torch.log_softmax(logits, dim=1)
    per_sample = -(log_probs * same.float()).sum(dim=1) / torch.clamp(positive_counts, min=1)
    return per_sample[valid].mean()


def _flow_view(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    vocab: dict[str, int],
    token_drop_prob: float,
    span_drop_prob: float,
    span_drop_length: int,
    mask_prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    special_ids = {vocab[token] for token in ["[PAD]", "[CLS]", "[SEP]", "[MASK]"] if token in vocab}
    special = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in special_ids:
        special |= input_ids == token_id
    can_change = (attention_mask > 0) & (~special)
    view_ids = input_ids.clone()
    view_mask = attention_mask.clone()

    drop_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    if token_drop_prob > 0.0:
        drop_mask |= (torch.rand(input_ids.shape, device=input_ids.device) < float(token_drop_prob)) & can_change
    if span_drop_prob > 0.0:
        span_starts = (torch.rand(input_ids.shape, device=input_ids.device) < float(span_drop_prob)) & can_change
        seq_len = input_ids.shape[1]
        span_len = max(1, int(span_drop_length))
        for offset in range(span_len):
            if offset == 0:
                drop_mask |= span_starts
            else:
                drop_mask[:, offset:] |= span_starts[:, : seq_len - offset]
        drop_mask &= can_change
    if drop_mask.any():
        pad_id = int(vocab.get("[PAD]", 0))
        view_ids[drop_mask] = pad_id
        view_mask[drop_mask] = 0

    can_mask = (view_mask > 0) & (~special)
    if mask_prob > 0.0:
        mask_positions = (torch.rand(input_ids.shape, device=input_ids.device) < float(mask_prob)) & can_mask
        if mask_positions.any():
            view_ids[mask_positions] = int(vocab["[MASK]"])
    return view_ids, view_mask


def _nt_xent_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> torch.Tensor:
    batch_size = z_a.shape[0]
    if batch_size < 2:
        return (z_a.sum() + z_b.sum()) * 0.0
    z = torch.nn.functional.normalize(torch.cat([z_a, z_b], dim=0), dim=1)
    logits = (z @ z.T) / float(temperature)
    logits = logits - torch.eye(2 * batch_size, device=z.device, dtype=z.dtype) * 1e9
    targets = torch.arange(2 * batch_size, device=z.device)
    targets = (targets + batch_size) % (2 * batch_size)
    return torch.nn.functional.cross_entropy(logits, targets)


def _segment_views(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    vocab: dict[str, int],
    min_segment_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pad_id = int(vocab.get("[PAD]", 0))
    special_ids = {vocab[token] for token in ["[PAD]", "[CLS]", "[SEP]", "[MASK]"] if token in vocab}
    special = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in special_ids:
        special |= input_ids == token_id
    active_special = (attention_mask > 0) & special
    can_segment = (attention_mask > 0) & (~special)
    keep_a = active_special.clone()
    keep_b = active_special.clone()
    min_tokens = max(1, int(min_segment_tokens))
    for row_idx in range(input_ids.shape[0]):
        positions = torch.nonzero(can_segment[row_idx], as_tuple=False).flatten()
        count = int(positions.numel())
        if count == 0:
            continue
        if count == 1:
            keep_a[row_idx, positions] = True
            keep_b[row_idx, positions] = True
            continue
        if count >= 2 * min_tokens:
            split = max(min_tokens, min(count - min_tokens, count // 2))
        else:
            split = max(1, count // 2)
        keep_a[row_idx, positions[:split]] = True
        keep_b[row_idx, positions[split:]] = True
        if not keep_b[row_idx, positions].any():
            keep_b[row_idx, positions[-1:]] = True
    view_a = input_ids.clone()
    view_b = input_ids.clone()
    view_a[~keep_a] = pad_id
    view_b[~keep_b] = pad_id
    return view_a, keep_a.long(), view_b, keep_b.long()


def _paragraph_pair_views(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    vocab: dict[str, int],
    paragraph_sep_id: int,
    min_tokens_per_view: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pad_id = int(vocab.get("[PAD]", 0))
    special_ids = {vocab[token] for token in ["[PAD]", "[CLS]", "[SEP]", "[MASK]"] if token in vocab}
    special = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in special_ids:
        special |= input_ids == token_id
    active_special = (attention_mask > 0) & special
    active = (attention_mask > 0) & (~special)
    keep_a = active_special.clone()
    keep_b = active_special.clone()
    min_tokens = max(1, int(min_tokens_per_view))
    for row_idx in range(input_ids.shape[0]):
        positions = torch.nonzero(active[row_idx], as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        separators = torch.nonzero(input_ids[row_idx] == paragraph_sep_id, as_tuple=False).flatten().tolist()
        boundaries = [0, *[int(pos) + 1 for pos in separators], int(input_ids.shape[1])]
        chunks: list[tuple[int, int]] = []
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            if right > left:
                chunks.append((left, right))
        if not chunks:
            keep_a[row_idx, positions] = True
            keep_b[row_idx, positions] = True
            continue
        mid = max(1, len(chunks) // 2)
        a_chunks = chunks[:mid]
        b_chunks = chunks[mid:]
        if not b_chunks:
            b_chunks = [chunks[-1]]
        for left, right in a_chunks:
            keep_a[row_idx, left:right] = True
        for left, right in b_chunks:
            keep_b[row_idx, left:right] = True
        if int(keep_a[row_idx].sum().item()) < min_tokens:
            keep_a[row_idx, positions[: min_tokens]] = True
        if int(keep_b[row_idx].sum().item()) < min_tokens:
            keep_b[row_idx, positions[-min_tokens:]] = True
    view_a = input_ids.clone()
    view_b = input_ids.clone()
    view_a[~keep_a] = pad_id
    view_b[~keep_b] = pad_id
    return view_a, keep_a.long(), view_b, keep_b.long()


def _run_epoch(
    model: BehaviorComposerPretrainer,
    loader: DataLoader,
    service_labels_all: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    cfg: dict[str, Any],
    vocab: dict[str, int],
    token_data: dict[str, Any],
    context_positive_map: np.ndarray | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    vocab_size = len(vocab)
    special_ids = {vocab[token] for token in ["[PAD]", "[CLS]", "[SEP]", "[MASK]"] if token in vocab}
    mask_id = vocab["[MASK]"]
    mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    rtd_loss_fn = nn.BCEWithLogitsLoss()
    rows: list[dict[str, float]] = []
    objective = str(cfg.get("objective", "mlm_rtd_service_contrastive")).lower()
    use_flow_contrastive = bool(cfg.get("use_flow_view_contrastive", False)) or objective in {
        "flow_contrastive",
        "flow_view_contrastive",
        "flow_context_contrastive",
        "mlm_rtd_flow_contrastive",
        "span_mlm_rtd_flow_contrastive",
        "mlm_rtd_flow_context_contrastive",
        "span_mlm_rtd_flow_context_contrastive",
    }
    use_context_contrastive = bool(cfg.get("use_context_contrastive", False)) or float(cfg.get("context_contrastive_weight", 0.0)) > 0 or objective in {
        "context_contrastive",
        "flow_context_contrastive",
        "mlm_rtd_flow_context_contrastive",
        "span_mlm_rtd_flow_context_contrastive",
    }
    use_segment_contrastive = bool(cfg.get("use_segment_contrastive", False)) or objective in {
        "segment_contrastive",
        "adjacent_segment_contrastive",
        "same_flow_segment_contrastive",
        "span_mlm_rtd_segment_contrastive",
    }
    use_paragraph_contrastive = bool(cfg.get("use_paragraph_contrastive", False)) or objective in {
        "paragraph_contrastive",
        "paragraph_same_context_contrastive",
    }
    use_mlm_rtd = objective not in {
        "flow_contrastive",
        "flow_view_contrastive",
        "flow_context_contrastive",
        "segment_contrastive",
        "adjacent_segment_contrastive",
        "same_flow_segment_contrastive",
        "context_contrastive",
        "paragraph_contrastive",
        "paragraph_same_context_contrastive",
    }
    mask_strategy = "span" if objective.startswith("span_") else str(cfg.get("mask_strategy", "random")).lower()
    with torch.set_grad_enabled(training):
        for batch_indices, input_ids, attention_mask, token_type_ids in tqdm(loader, desc="pretrain", leave=False):
            service_labels = service_labels_all[batch_indices].to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            zero = input_ids.float().sum() * 0.0
            mlm_loss = zero
            rtd_loss = zero
            contrast_loss = zero
            flow_contrastive_loss = zero
            context_contrastive_loss = zero
            segment_contrastive_loss = zero
            paragraph_contrastive_loss = zero
            mlm_acc = 0.0
            rtd_acc = 0.0

            if use_mlm_rtd:
                corrupted, mlm_labels, replaced_labels = _mask_inputs(
                    input_ids,
                    attention_mask,
                    vocab_size=vocab_size,
                    mask_id=mask_id,
                    special_ids=special_ids,
                    mlm_probability=float(cfg.get("mlm_probability", 0.15)),
                    replace_probability=float(cfg.get("replace_probability", 0.10)),
                    mask_strategy=mask_strategy,
                    span_length=int(cfg.get("span_length", 3)),
                )
                mlm_logits, rtd_logits, embeddings = model(corrupted, attention_mask, token_type_ids)
                mlm_loss = mlm_loss_fn(mlm_logits.view(-1, vocab_size), mlm_labels.view(-1))
                rtd_mask = (attention_mask > 0) & (mlm_labels == -100)
                rtd_loss = rtd_loss_fn(rtd_logits[rtd_mask], replaced_labels.float()[rtd_mask]) if torch.any(rtd_mask) else rtd_logits.sum() * 0.0
                contrast_loss = _contrastive_loss(embeddings, service_labels, temperature=float(cfg.get("contrast_temperature", 0.2)))
                with torch.no_grad():
                    masked = mlm_labels != -100
                    mlm_acc = (
                        (torch.argmax(mlm_logits, dim=-1)[masked] == mlm_labels[masked]).float().mean().item()
                        if torch.any(masked)
                        else 0.0
                    )
                    rtd_pred = (torch.sigmoid(rtd_logits[rtd_mask]) >= 0.5).long() if torch.any(rtd_mask) else torch.tensor([], device=device)
                    rtd_acc = (rtd_pred == replaced_labels[rtd_mask]).float().mean().item() if torch.any(rtd_mask) else 0.0

            if use_flow_contrastive:
                view_a, mask_a = _flow_view(
                    input_ids,
                    attention_mask,
                    vocab=vocab,
                    token_drop_prob=float(cfg.get("view_token_drop_prob", 0.05)),
                    span_drop_prob=float(cfg.get("view_span_drop_prob", 0.02)),
                    span_drop_length=int(cfg.get("view_span_drop_length", 3)),
                    mask_prob=float(cfg.get("view_mask_prob", 0.10)),
                )
                view_b, mask_b = _flow_view(
                    input_ids,
                    attention_mask,
                    vocab=vocab,
                    token_drop_prob=float(cfg.get("view_token_drop_prob", 0.05)),
                    span_drop_prob=float(cfg.get("view_span_drop_prob", 0.02)),
                    span_drop_length=int(cfg.get("view_span_drop_length", 3)),
                    mask_prob=float(cfg.get("view_mask_prob", 0.10)),
                )
                z_a = model.contrastive_embedding(view_a, mask_a, token_type_ids)
                z_b = model.contrastive_embedding(view_b, mask_b, token_type_ids)
                flow_contrastive_loss = _nt_xent_loss(z_a, z_b, temperature=float(cfg.get("flow_contrastive_temperature", cfg.get("contrast_temperature", 0.2))))

            if use_context_contrastive and context_positive_map is not None:
                batch_index_list = batch_indices.detach().cpu().numpy()
                positive_indices = context_positive_map[batch_index_list]
                valid = positive_indices >= 0
                if np.any(valid):
                    anchor_idx = torch.tensor(batch_index_list[valid], dtype=torch.long)
                    positive_idx = torch.tensor(positive_indices[valid], dtype=torch.long)
                    anchor_input_ids = token_data["input_ids"][anchor_idx].to(device)
                    anchor_attention_mask = token_data["attention_mask"][anchor_idx].to(device)
                    anchor_token_type_ids = token_data["token_type_ids"][anchor_idx].to(device)
                    positive_input_ids = token_data["input_ids"][positive_idx].to(device)
                    positive_attention_mask = token_data["attention_mask"][positive_idx].to(device)
                    positive_token_type_ids = token_data["token_type_ids"][positive_idx].to(device)
                    z_anchor = model.contrastive_embedding(anchor_input_ids, anchor_attention_mask, anchor_token_type_ids)
                    z_positive = model.contrastive_embedding(positive_input_ids, positive_attention_mask, positive_token_type_ids)
                    context_contrastive_loss = _nt_xent_loss(
                        z_anchor,
                        z_positive,
                        temperature=float(cfg.get("context_contrastive_temperature", cfg.get("contrast_temperature", 0.2))),
                    )

            if use_segment_contrastive:
                seg_a, seg_mask_a, seg_b, seg_mask_b = _segment_views(
                    input_ids,
                    attention_mask,
                    vocab=vocab,
                    min_segment_tokens=int(cfg.get("min_segment_tokens", 2)),
                )
                z_seg_a = model.contrastive_embedding(seg_a, seg_mask_a, token_type_ids)
                z_seg_b = model.contrastive_embedding(seg_b, seg_mask_b, token_type_ids)
                segment_contrastive_loss = _nt_xent_loss(
                    z_seg_a,
                    z_seg_b,
                    temperature=float(cfg.get("segment_contrastive_temperature", cfg.get("contrast_temperature", 0.2))),
                )

            if use_paragraph_contrastive:
                paragraph_sep_id = int(vocab.get(str(cfg.get("paragraph_sep_token", "[FLOW_SEP]")), vocab.get("[SEP]", 2)))
                para_a, para_mask_a, para_b, para_mask_b = _paragraph_pair_views(
                    input_ids,
                    attention_mask,
                    vocab=vocab,
                    paragraph_sep_id=paragraph_sep_id,
                    min_tokens_per_view=int(cfg.get("min_paragraph_tokens", 4)),
                )
                z_para_a = model.paragraph_embedding(para_a, para_mask_a, token_type_ids)
                z_para_b = model.paragraph_embedding(para_b, para_mask_b, token_type_ids)
                paragraph_contrastive_loss = _nt_xent_loss(
                    z_para_a,
                    z_para_b,
                    temperature=float(cfg.get("paragraph_contrastive_temperature", cfg.get("contrast_temperature", 0.2))),
                )

            loss = (
                mlm_loss
                + float(cfg.get("rtd_weight", 0.5)) * rtd_loss
                + float(cfg.get("contrast_weight", 0.5)) * contrast_loss
                + float(cfg.get("flow_contrastive_weight", 1.0)) * flow_contrastive_loss
                + float(cfg.get("context_contrastive_weight", 0.0)) * context_contrastive_loss
                + float(cfg.get("segment_contrastive_weight", 1.0)) * segment_contrastive_loss
                + float(cfg.get("paragraph_contrastive_weight", 1.0)) * paragraph_contrastive_loss
            )
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            rows.append(
                {
                    "loss": float(loss.item()),
                    "mlm_loss": float(mlm_loss.item()),
                    "rtd_loss": float(rtd_loss.item()),
                    "contrast_loss": float(contrast_loss.item()),
                    "flow_contrastive_loss": float(flow_contrastive_loss.item()),
                    "context_contrastive_loss": float(context_contrastive_loss.item()),
                    "segment_contrastive_loss": float(segment_contrastive_loss.item()),
                    "paragraph_contrastive_loss": float(paragraph_contrastive_loss.item()),
                    "mlm_acc": float(mlm_acc),
                    "rtd_acc": float(rtd_acc),
                }
            )
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]} if rows else {}


def pretrain(token_data: dict[str, Any], config: dict[str, Any]) -> PretrainResult:
    started = time.perf_counter()
    seed = int(config.get("seed", 42))
    set_seed(seed)
    pre_cfg = config.get("pretraining", {})
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    train_idx, val_idx = _split_indices(
        len(token_data["input_ids"]),
        val_ratio=float(train_cfg.get("val_ratio", 0.1)),
        seed=seed,
    )
    service_labels_all = _service_labels(token_data, np.arange(len(token_data["input_ids"])))
    context_positive_map: np.ndarray | None = None
    train_context_positive_map: np.ndarray | None = None
    val_context_positive_map: np.ndarray | None = None
    context_stats: dict[str, Any] = {}
    pretrain_objective = str(pre_cfg.get("objective", "mlm_rtd_service_contrastive")).lower()
    if bool(pre_cfg.get("use_context_contrastive", False)) or float(pre_cfg.get("context_contrastive_weight", 0.0)) > 0 or pretrain_objective in {
        "context_contrastive",
        "flow_context_contrastive",
        "mlm_rtd_flow_context_contrastive",
        "span_mlm_rtd_flow_context_contrastive",
    }:
        train_context_positive_map, train_stats = _build_context_positive_map(
            token_data,
            group_by=str(pre_cfg.get("context_group_by", "service_host_proto")),
            positive_mode=str(pre_cfg.get("context_positive_mode", "next")),
            indices=train_idx,
        )
        val_context_positive_map, val_stats = _build_context_positive_map(
            token_data,
            group_by=str(pre_cfg.get("context_group_by", "service_host_proto")),
            positive_mode=str(pre_cfg.get("context_positive_mode", "next")),
            indices=val_idx,
        )
        context_positive_map = train_context_positive_map
        context_stats = {
            "train_context_group_by": train_stats.get("context_group_by"),
            "train_context_positive_mode": train_stats.get("context_positive_mode"),
            "train_num_context_groups": train_stats.get("num_context_groups"),
            "train_num_context_pairs": train_stats.get("num_context_pairs"),
            "train_num_context_flows_with_positive": train_stats.get("num_context_flows_with_positive"),
            "val_num_context_groups": val_stats.get("num_context_groups"),
            "val_num_context_pairs": val_stats.get("num_context_pairs"),
            "val_num_context_flows_with_positive": val_stats.get("num_context_flows_with_positive"),
        }
    batch_size = int(pre_cfg.get("batch_size", train_cfg.get("batch_size", 64)))
    train_loader = _loader(token_data, train_idx, batch_size=batch_size, shuffle=True)
    val_loader = _loader(token_data, val_idx, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = BehaviorComposer(
        vocab_size=len(token_data["vocab"]),
        num_classes=2,
        max_seq_len=int(model_cfg.get("max_seq_len", token_data.get("max_len", 256))),
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        intermediate_size=int(model_cfg.get("intermediate_size", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        **resolve_pooling_config(model_cfg),
    )
    model = BehaviorComposerPretrainer(encoder, hidden_size=int(model_cfg.get("hidden_size", 128)), vocab_size=len(token_data["vocab"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(pre_cfg.get("learning_rate", train_cfg.get("learning_rate", 5e-4))),
        weight_decay=float(pre_cfg.get("weight_decay", train_cfg.get("weight_decay", 0.01))),
    )
    epochs = int(pre_cfg.get("epochs", train_cfg.get("epochs", 3)))
    history: list[dict[str, float]] = []
    best_state = None
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        train_row = _run_epoch(
            model,
            train_loader,
            service_labels_all,
            optimizer,
            device,
            pre_cfg,
            token_data["vocab"],
            token_data,
            context_positive_map=context_positive_map,
        )
        val_row = _run_epoch(
            model,
            val_loader,
            service_labels_all,
            None,
            device,
            pre_cfg,
            token_data["vocab"],
            token_data,
            context_positive_map=val_context_positive_map,
        )
        row = {"epoch": float(epoch)}
        row.update({f"train_{key}": value for key, value in train_row.items()})
        row.update({f"val_{key}": value for key, value in val_row.items()})
        history.append(row)
        if val_row.get("loss", float("inf")) < best_val:
            best_val = val_row["loss"]
            best_state = {key: value.detach().cpu() for key, value in model.encoder.state_dict().items()}
    metrics = {
        "best_val_loss": float(best_val),
        "epochs": epochs,
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "train_seconds": float(time.perf_counter() - started),
        "device": str(device),
        "objective": str(pre_cfg.get("objective", "mlm_rtd_service_contrastive")),
        "rtd_weight": float(pre_cfg.get("rtd_weight", 0.5)),
        "contrast_weight": float(pre_cfg.get("contrast_weight", 0.5)),
        "flow_contrastive_weight": float(pre_cfg.get("flow_contrastive_weight", 0.0)),
        "flow_contrastive_temperature": float(pre_cfg.get("flow_contrastive_temperature", pre_cfg.get("contrast_temperature", 0.2))),
        "segment_contrastive_weight": float(pre_cfg.get("segment_contrastive_weight", 0.0)),
        "segment_contrastive_temperature": float(pre_cfg.get("segment_contrastive_temperature", pre_cfg.get("contrast_temperature", 0.2))),
        "paragraph_contrastive_weight": float(pre_cfg.get("paragraph_contrastive_weight", 0.0)),
        "paragraph_contrastive_temperature": float(pre_cfg.get("paragraph_contrastive_temperature", pre_cfg.get("contrast_temperature", 0.2))),
        "paragraph_sep_token": str(pre_cfg.get("paragraph_sep_token", "[FLOW_SEP]")),
        "min_paragraph_tokens": int(pre_cfg.get("min_paragraph_tokens", 4)),
        "context_contrastive_weight": float(pre_cfg.get("context_contrastive_weight", 0.0)),
        "context_contrastive_temperature": float(pre_cfg.get("context_contrastive_temperature", pre_cfg.get("contrast_temperature", 0.2))),
        "context_group_by": str(pre_cfg.get("context_group_by", "service_host_proto")),
        "context_positive_mode": str(pre_cfg.get("context_positive_mode", "next")),
        "mlm_probability": float(pre_cfg.get("mlm_probability", 0.15)),
        "replace_probability": float(pre_cfg.get("replace_probability", 0.10)),
        "mask_strategy": str(pre_cfg.get("mask_strategy", "span" if str(pre_cfg.get("objective", "")).startswith("span_") else "random")),
        "span_length": int(pre_cfg.get("span_length", 3)),
        "view_token_drop_prob": float(pre_cfg.get("view_token_drop_prob", 0.0)),
        "view_span_drop_prob": float(pre_cfg.get("view_span_drop_prob", 0.0)),
        "view_mask_prob": float(pre_cfg.get("view_mask_prob", 0.0)),
        "min_segment_tokens": int(pre_cfg.get("min_segment_tokens", 2)),
    }
    metrics.update(context_stats)
    return PretrainResult(metrics=metrics, history=history, state_dict=best_state or {key: value.detach().cpu() for key, value in model.encoder.state_dict().items()})

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


POOLING_STRATEGIES = {"cls", "mean", "attentive", "class_aware_attentive", "residual_class_aware"}


def resolve_pooling_config(model_cfg: dict) -> dict[str, object]:
    pooling = str(model_cfg.get("pooling", model_cfg.get("pooling_strategy", "cls")))
    class_aware_pooling = bool(model_cfg.get("class_aware_pooling", False))
    if pooling == "residual_class_aware":
        class_aware_pooling = False
    return {
        "pooling_strategy": pooling,
        "class_aware_pooling": class_aware_pooling,
        "class_aware_alpha": float(model_cfg.get("class_aware_alpha", 0.5)),
        "cls_beta": float(model_cfg.get("cls_beta", 0.0)),
    }


class BehaviorComposer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        max_seq_len: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        intermediate_size: int = 256,
        dropout: float = 0.1,
        pooling_strategy: str = "cls",
        class_aware_pooling: bool = False,
        class_aware_alpha: float = 0.5,
        cls_beta: float = 0.0,
    ) -> None:
        super().__init__()
        if class_aware_pooling and pooling_strategy not in {"residual_class_aware", "class_aware_attentive"}:
            pooling_strategy = "class_aware_attentive"
        if pooling_strategy not in POOLING_STRATEGIES:
            raise ValueError(f"Unsupported pooling_strategy: {pooling_strategy}")
        self.pooling_strategy = str(pooling_strategy)
        self.class_aware_pooling = self.pooling_strategy == "class_aware_attentive"
        self.residual_class_aware = self.pooling_strategy == "residual_class_aware"
        self.class_aware_alpha = float(class_aware_alpha)
        self.cls_beta = float(cls_beta)
        self.num_classes = int(num_classes)
        self.hidden_size = int(hidden_size)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.type_embedding = nn.Embedding(2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        if self.pooling_strategy == "attentive":
            self.attention_pool = nn.Linear(hidden_size, 1)
        if self.class_aware_pooling or self.residual_class_aware:
            self.class_attention = nn.Linear(hidden_size, hidden_size, bias=False)
            self.class_queries = nn.Parameter(torch.empty(num_classes, hidden_size))
            nn.init.normal_(self.class_queries, mean=0.0, std=hidden_size**-0.5)
            if self.class_aware_pooling:
                self.class_logit_weight = nn.Parameter(torch.empty(num_classes, hidden_size))
                self.class_logit_bias = nn.Parameter(torch.zeros(num_classes))
                nn.init.normal_(self.class_logit_weight, mean=0.0, std=hidden_size**-0.5)
            else:
                self.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, num_classes),
                )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_classes),
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.class_aware_pooling:
            token_hidden = self.encode_tokens(input_ids, attention_mask, token_type_ids)
            pooled = self.class_aware_encode(token_hidden, attention_mask)
            return (pooled * self.class_logit_weight.unsqueeze(0)).sum(dim=-1) + self.class_logit_bias
        token_hidden = self.encode_tokens(input_ids, attention_mask, token_type_ids)
        pooled = self.pool_tokens(token_hidden, attention_mask)
        return self.classifier(pooled)

    def classification_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_hidden = self.encode_tokens(input_ids, attention_mask, token_type_ids)
        if self.residual_class_aware:
            return self.residual_class_aware_pool(token_hidden, attention_mask)
        if self.class_aware_pooling:
            return self.mean_pool(token_hidden, attention_mask)
        return self.pool_tokens(token_hidden, attention_mask)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_hidden = self.encode_tokens(input_ids, attention_mask, token_type_ids)
        if self.class_aware_pooling or self.residual_class_aware:
            # A single embedding is still needed by anomaly and fusion wrappers.
            # Mean pooling keeps that interface stable while forward() uses per-class pooling.
            return self.mean_pool(token_hidden, attention_mask)
        return self.pool_tokens(token_hidden, attention_mask)

    def pool_tokens(self, token_hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.pooling_strategy == "cls":
            return token_hidden[:, 0, :]
        if self.pooling_strategy == "mean":
            return self.mean_pool(token_hidden, attention_mask)
        if self.pooling_strategy == "attentive":
            scores = self.attention_pool(token_hidden).squeeze(-1)
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1)
            return torch.bmm(weights.unsqueeze(1), token_hidden).squeeze(1)
        if self.residual_class_aware:
            return self.residual_class_aware_pool(token_hidden, attention_mask)
        raise ValueError(f"pool_tokens does not support {self.pooling_strategy}")

    def cls_pool(self, token_hidden: torch.Tensor) -> torch.Tensor:
        return token_hidden[:, 0, :]

    def mean_pool(self, token_hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is None:
            return token_hidden.mean(dim=1)
        mask = attention_mask.to(dtype=token_hidden.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (token_hidden * mask).sum(dim=1) / denom

    def class_aware_summary(
        self,
        token_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.class_aware_encode(token_hidden, attention_mask).mean(dim=1)

    def residual_class_aware_pool(
        self,
        token_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z_mean = self.mean_pool(token_hidden, attention_mask)
        z_cls = self.cls_pool(token_hidden)
        z_ca = self.class_aware_summary(token_hidden, attention_mask)
        combined = z_mean + self.class_aware_alpha * z_ca + self.cls_beta * z_cls
        return F.layer_norm(combined, (self.hidden_size,))

    def class_aware_encode(
        self,
        token_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        projected = self.class_attention(token_hidden)
        scores = torch.einsum("bsh,ch->bcs", projected, self.class_queries)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bcs,bsh->bch", weights, token_hidden)

    def encode_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions) + self.type_embedding(token_type_ids)
        hidden = self.dropout(self.norm(hidden))
        src_key_padding_mask = attention_mask == 0 if attention_mask is not None else None
        return self.encoder(hidden, src_key_padding_mask=src_key_padding_mask)

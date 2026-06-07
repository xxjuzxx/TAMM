#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("outputs/results/crossnet_unlabeled_pretrain_20260518")
BASELINE_DIR = Path("outputs/results/crossnet_label_stratified_80_20_20260518/class_aware_weighted_ce")
BASELINE_AGG_DIR = Path("outputs/results/crossnet_label_stratified_80_20_20260518/class_aware_weighted_ce_mean_n3")
OUT_CSV = Path("experiments/crossnet_unlabeled_pretrain_summary.csv")
OUT_MD = Path("experiments/crossnet_unlabeled_pretrain_results.md")


EXPERIMENTS = [
    {
        "experiment": "baseline_random_init",
        "kind": "baseline",
        "pretrain_dir": None,
        "finetune_dir": BASELINE_DIR,
        "aggregation_dir": BASELINE_AGG_DIR,
        "note": "随机初始化 class-aware baseline",
    },
    {
        "experiment": "pretrainB_normal_lr",
        "kind": "CrossNetB MLM+RTD+service contrastive",
        "pretrain_dir": ROOT / "pretrain_crossnetB_128",
        "finetune_dir": ROOT / "finetune_crossnetA_p4_normal_lr",
        "aggregation_dir": ROOT / "finetune_crossnetA_p4_normal_lr_mean_n3",
        "note": "早期 CrossNetB MLM+RTD+service contrastive",
    },
    {
        "experiment": "pretrainA_trainonly_normal_lr",
        "kind": "CrossNetA train-only MLM+RTD+service contrastive",
        "pretrain_dir": ROOT / "pretrain_crossnetA_trainonly_128",
        "finetune_dir": ROOT / "finetune_crossnetA_trainonly_p4_normal_lr",
        "aggregation_dir": ROOT / "finetune_crossnetA_trainonly_p4_normal_lr_mean_n3",
        "note": "早期 CrossNetA train-only MLM+RTD+service contrastive",
    },
    {
        "experiment": "pretrainA_trainonly_mlmrtd",
        "kind": "CrossNetA train-only MLM+RTD",
        "pretrain_dir": ROOT / "pretrain_crossnetA_trainonly_128_mlm_rtd",
        "finetune_dir": ROOT / "finetune_crossnetA_trainonly_p4_mlm_rtd_normal_lr",
        "aggregation_dir": ROOT / "finetune_crossnetA_trainonly_p4_mlm_rtd_normal_lr_mean_n3",
        "note": "随机 MLM+RTD",
    },
    {
        "experiment": "pretrainB_mlmrtd",
        "kind": "CrossNetB MLM+RTD",
        "pretrain_dir": ROOT / "pretrain_crossnetB_128_mlm_rtd",
        "finetune_dir": ROOT / "finetune_crossnetA_crossnetB_mlm_rtd_normal_lr",
        "aggregation_dir": ROOT / "finetune_crossnetA_crossnetB_mlm_rtd_normal_lr_mean_n3",
        "note": "跨域随机 MLM+RTD",
    },
    {
        "experiment": "pretrainA_trainonly_flow_contrastive",
        "kind": "CrossNetA train-only flow-view contrastive",
        "pretrain_dir": ROOT / "pretrain_crossnetA_trainonly_128_flow_contrastive",
        "finetune_dir": ROOT / "finetune_crossnetA_trainonly_flow_contrastive_direct",
        "aggregation_dir": ROOT / "finetune_crossnetA_trainonly_flow_contrastive_direct_mean_n3",
        "note": "同一 flow 两个扰动视图做 NT-Xent",
    },
    {
        "experiment": "pretrainB_flow_contrastive",
        "kind": "CrossNetB flow-view contrastive",
        "pretrain_dir": ROOT / "pretrain_crossnetB_128_flow_contrastive",
        "finetune_dir": ROOT / "finetune_crossnetA_crossnetB_flow_contrastive_direct",
        "aggregation_dir": ROOT / "finetune_crossnetA_crossnetB_flow_contrastive_direct_mean_n3",
        "gate_global": ROOT / "anomaly_gate_flowB_global.json",
        "gate_service": ROOT / "anomaly_gate_flowB_service.json",
        "note": "聚合 Macro-F1 最优，anomaly 指标改善",
    },
    {
        "experiment": "pretrainA_trainonly_span_mlm_rtd",
        "kind": "CrossNetA train-only span MLM+RTD",
        "pretrain_dir": ROOT / "pretrain_crossnetA_trainonly_128_span_mlm_rtd",
        "finetune_dir": ROOT / "finetune_crossnetA_trainonly_span_mlm_rtd_direct",
        "aggregation_dir": ROOT / "finetune_crossnetA_trainonly_span_mlm_rtd_direct_mean_n3",
        "note": "连续 span mask 重构，分类下降",
    },
    {
        "experiment": "pretrainB_span_mlm_rtd",
        "kind": "CrossNetB span MLM+RTD",
        "pretrain_dir": ROOT / "pretrain_crossnetB_128_span_mlm_rtd",
        "finetune_dir": ROOT / "finetune_crossnetA_crossnetB_span_mlm_rtd_direct",
        "aggregation_dir": ROOT / "finetune_crossnetA_crossnetB_span_mlm_rtd_direct_mean_n3",
        "note": "跨域 span MLM+RTD，分类下降",
    },
    {
        "experiment": "pretrainA_trainonly_segment_contrastive",
        "kind": "CrossNetA train-only segment contrastive",
        "pretrain_dir": ROOT / "pretrain_crossnetA_trainonly_128_segment_contrastive",
        "finetune_dir": ROOT / "finetune_crossnetA_trainonly_segment_contrastive_direct",
        "aggregation_dir": ROOT / "finetune_crossnetA_trainonly_segment_contrastive_direct_mean_n3",
        "note": "同一 flow 前后段对比，分类下降",
    },
    {
        "experiment": "pretrainB_segment_contrastive",
        "kind": "CrossNetB segment contrastive",
        "pretrain_dir": ROOT / "pretrain_crossnetB_128_segment_contrastive",
        "finetune_dir": ROOT / "finetune_crossnetA_crossnetB_segment_contrastive_direct",
        "aggregation_dir": ROOT / "finetune_crossnetA_crossnetB_segment_contrastive_direct_mean_n3",
        "note": "跨域前后段对比，AUROC 高但 Macro-F1 下降",
    },
    {
        "experiment": "pretrainB_span_flow_contrastive",
        "kind": "CrossNetB span MLM+RTD + flow-view contrastive",
        "pretrain_dir": ROOT / "pretrain_crossnetB_128_span_flow_contrastive",
        "finetune_dir": ROOT / "finetune_crossnetA_crossnetB_span_flow_contrastive_direct",
        "aggregation_dir": ROOT / "finetune_crossnetA_crossnetB_span_flow_contrastive_direct_mean_n3",
        "gate_global": ROOT / "anomaly_gate_span_flowB_global.json",
        "gate_service": ROOT / "anomaly_gate_span_flowB_service.json",
        "note": "单流 Macro-F1 最优，anomaly 指标改善",
    },
]


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric(metrics: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metrics:
            return metrics[key]
    return None


def row_for(spec: dict[str, Any], base_single: float, base_agg: float) -> dict[str, Any]:
    pre = read_json(Path(spec["pretrain_dir"]) / "metrics.json") if spec.get("pretrain_dir") else {}
    fin = read_json(Path(spec["finetune_dir"]) / "metrics.json")
    agg = read_json(Path(spec["aggregation_dir"]) / "metrics.json")
    agg_level = agg.get("aggregated_flow_level", {})
    gate_global = read_json(spec.get("gate_global"))
    gate_service = read_json(spec.get("gate_service"))
    return {
        "experiment": spec["experiment"],
        "kind": spec["kind"],
        "pretrain_objective": pre.get("objective"),
        "pretrain_best_val_loss": pre.get("best_val_loss"),
        "accuracy": fin.get("accuracy"),
        "macro_f1": fin.get("macro_f1"),
        "delta_macro_f1": None if fin.get("macro_f1") is None else float(fin["macro_f1"]) - base_single,
        "weighted_f1": fin.get("weighted_f1"),
        "auroc_ovr": fin.get("auroc_ovr"),
        "auprc_ovr": fin.get("auprc_ovr"),
        "minority_macro_f1": fin.get("minority_macro_f1"),
        "aggregated_macro_f1": agg_level.get("macro_f1"),
        "delta_aggregated_macro_f1": None if agg_level.get("macro_f1") is None else float(agg_level["macro_f1"]) - base_agg,
        "aggregated_accuracy": agg_level.get("accuracy"),
        "aggregated_weighted_f1": agg_level.get("weighted_f1"),
        "aggregated_auroc_ovr": agg_level.get("auroc_ovr"),
        "aggregated_auprc_ovr": agg_level.get("auprc_ovr"),
        "global_anomaly_delta_auroc": gate_global.get("delta_auroc"),
        "global_anomaly_delta_auprc": gate_global.get("delta_auprc"),
        "global_anomaly_delta_fpr95": gate_global.get("delta_fpr95"),
        "service_anomaly_delta_auroc": gate_service.get("delta_auroc"),
        "service_anomaly_delta_auprc": gate_service.get("delta_auprc"),
        "service_anomaly_delta_fpr95": gate_service.get("delta_fpr95"),
        "anomaly_gate_status": "/".join(status for status in [gate_global.get("status"), gate_service.get("status")] if status),
        "train_seconds": fin.get("train_seconds"),
        "result_dir": str(spec["finetune_dir"]),
        "aggregation_dir": str(spec["aggregation_dir"]),
        "note": spec["note"],
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    baseline_metrics = read_json(BASELINE_DIR / "metrics.json")
    baseline_agg = read_json(BASELINE_AGG_DIR / "metrics.json").get("aggregated_flow_level", {})
    base_single = float(baseline_metrics["macro_f1"])
    base_agg = float(baseline_agg["macro_f1"])
    rows = [row_for(spec, base_single, base_agg) for spec in EXPERIMENTS if Path(spec["finetune_dir"]).exists()]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best_single = max(rows, key=lambda item: item.get("macro_f1") or -1)
    best_agg = max(rows, key=lambda item: item.get("aggregated_macro_f1") or -1)
    lines = [
        "# CrossNet 无标签预训练重构实验结果",
        "",
        "本轮目标是验证重构预训练任务能否提升 CrossNet 应用分类，同时不破坏 anomaly-friendly encoder 表示。下游分类统一使用 `class_aware_attentive + weighted_ce`，CrossNetA label-stratified 0.8/0.1/0.1 split，seed=42。聚合统一使用 `mean_softmax + service_host_proto + window_n=3`。",
        "",
        "## 总表",
        "",
        "| 实验 | 预训练目标 | 单流 Macro-F1 | Δ单流 | 聚合 Macro-F1 | Δ聚合 | AUROC | AUPRC | anomaly gate | 说明 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['experiment']}`",
                    str(row.get("pretrain_objective") or row["kind"]),
                    fmt(row.get("macro_f1")),
                    fmt(row.get("delta_macro_f1")),
                    fmt(row.get("aggregated_macro_f1")),
                    fmt(row.get("delta_aggregated_macro_f1")),
                    fmt(row.get("auroc_ovr")),
                    fmt(row.get("auprc_ovr")),
                    row.get("anomaly_gate_status") or "-",
                    row["note"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 关键结论",
            "",
            f"- 单流最优：`{best_single['experiment']}`，Macro-F1={fmt(best_single['macro_f1'])}，相对 baseline Δ={fmt(best_single['delta_macro_f1'])}。",
            f"- 聚合最优：`{best_agg['experiment']}`，聚合 Macro-F1={fmt(best_agg['aggregated_macro_f1'])}，相对 baseline Δ={fmt(best_agg['delta_aggregated_macro_f1'])}。",
            "- 单纯 MLM/RTD 或 span MLM/RTD 会降低应用分类，说明 token 重构能力没有直接转化为 app 类间边界。",
            "- flow-view contrastive 是最稳定的正向模块；跨域 CrossNetB 预训练比 CrossNetA train-only 更有效，说明它更像分布扩展/不变性学习，而不是记住当前训练集。",
            "- span MLM/RTD + flow-view contrastive 提升单流 Macro-F1 最明显，但聚合后不如纯 flow-view contrastive，说明它提高了单样本边界，却没有让同上下文窗口内预测更一致。",
            "- 已跑 anomaly gate 的两个候选都没有 anomaly 下降；gate 状态为 WARN 仅表示 encoder 权重不同，需要实际 anomaly 评估。指标本身均为改善。",
            "",
            "## 推荐",
            "",
            "- `Safe default`：如果默认评估包含应用上下文聚合，推荐 `pretrainB_flow_contrastive + class_aware_attentive + weighted_ce + mean_softmax(window_n=3)`。",
            "- `Single-flow best`：如果只报告单流分类，推荐 `pretrainB_span_flow_contrastive + class_aware_attentive + weighted_ce`。",
            "- `Do not use by default`：MLM/RTD-only、span MLM/RTD-only、segment contrastive-only。",
            "",
            "## 新增配置",
            "",
            "- `configs/model_crossnet_pretrain_128_flow_contrastive.yaml`",
            "- `configs/model_crossnet_pretrain_128_span_mlm_rtd.yaml`",
            "- `configs/model_crossnet_pretrain_128_segment_contrastive.yaml`",
            "- `configs/model_crossnet_pretrain_128_span_mlm_rtd_flow_contrastive.yaml`",
            "",
            "## 验证",
            "",
            "- `pytest -q` 已运行，结果写入本轮对话输出。",
            "- 当前工作区不是 git 仓库，无法输出 `git diff`；代码改动可直接查看上述文件。",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()

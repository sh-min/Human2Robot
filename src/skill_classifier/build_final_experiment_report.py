"""Build one presentation-ready overview and a reproducible Korean report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read(path):
    return json.loads(path.read_text())


def _percent(value):
    return f"{100.0 * value:.2f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    output = root / "final_report"
    output.mkdir(parents=True, exist_ok=True)

    ablations = _read(root / "ablations/summary/ablation_results.json")
    multiseed = _read(root / "multiseed_300e/summary/aggregate.json")
    holdout = _read(root / "date_holdout_300e/summary/date_holdout_paired.json")
    diagnostics = _read(root / "selected_model_diagnostics/diagnostics.json")
    ensemble_w4 = _read(root / "ensemble_w4/ensemble_summary.json")
    ensemble_w8 = _read(root / "ensemble_w8/ensemble_summary.json")
    holdout_ensembles = {
        date: _read(root / f"date_holdout_300e/ensemble_holdout_{date}/ensemble_summary.json")
        for date in ("0724", "0727", "0728")
    }

    fig, axes = plt.subplots(2, 2, figsize=(17, 13))
    multiseed_map = {row["group"]: row for row in multiseed}
    order = ("global_w4", "object_w4", "global_w8", "object_w8")
    labels = ("Global W4", "Object W4", "Global W8", "Object W8")
    values = [100 * multiseed_map[key]["accuracy_mean"] for key in order]
    errors = [100 * multiseed_map[key]["accuracy_std"] for key in order]
    axes[0, 0].bar(labels, values, yerr=errors, capsize=5,
                   color=("#4c78a8", "#f58518", "#4c78a8", "#f58518"))
    axes[0, 0].set_ylim(80, 100)
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_title("A. Fixed split, three seeds (mean ± std)")
    axes[0, 0].tick_params(axis="x", rotation=18)
    axes[0, 0].grid(axis="y", alpha=0.2)
    for index, value in enumerate(values):
        axes[0, 0].text(index, value + errors[index] + 0.4, f"{value:.1f}", ha="center")

    dates = [row["holdout"] for row in holdout["paired"]]
    x = np.arange(len(dates))
    width = 0.25
    global_values = [100 * row["global_accuracy"] for row in holdout["paired"]]
    object_values = [100 * row["object_accuracy"] for row in holdout["paired"]]
    ensemble_values = [100 * holdout_ensembles[date]["fixed_50_50"]["accuracy"] for date in dates]
    axes[0, 1].bar(x - width, global_values, width, label="Global", color="#4c78a8")
    axes[0, 1].bar(x, object_values, width, label="Object", color="#f58518")
    axes[0, 1].bar(x + width, ensemble_values, width, label="50:50", color="#54a24b")
    axes[0, 1].set_xticks(x, dates)
    axes[0, 1].set_ylim(60, 90)
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_title("B. Leave-one-date-out generalization")
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.2)

    per_class = holdout["sample_weighted_per_class"]
    class_names = list(per_class)
    delta = [100 * per_class[name]["delta"] for name in class_names]
    colors = ["#54a24b" if value >= 0 else "#e45756" for value in delta]
    axes[1, 0].barh(class_names, delta, color=colors)
    axes[1, 0].axvline(0, color="black", linewidth=1)
    axes[1, 0].set_xlabel("Object − global recall (percentage points)")
    axes[1, 0].set_title("C. Object grounding effect by class (date holdout)")
    axes[1, 0].grid(axis="x", alpha=0.2)
    for index, value in enumerate(delta):
        axes[1, 0].text(value + (0.5 if value >= 0 else -0.5), index,
                        f"{value:+.1f}", va="center",
                        ha="left" if value >= 0 else "right")

    for result in diagnostics:
        axes[1, 1].scatter(100 * result["accuracy"], 100 * result["ece"], s=100)
        axes[1, 1].annotate(
            f"{result['name']}\n{result['sample_latency_ms']:.3f} ms/sample",
            (100 * result["accuracy"], 100 * result["ece"]),
            xytext=(6, 6), textcoords="offset points", fontsize=8,
        )
    axes[1, 1].set_xlabel("Accuracy (%) → better")
    axes[1, 1].set_ylabel("ECE (%) → better")
    axes[1, 1].set_title("D. Accuracy–calibration–latency trade-off")
    axes[1, 1].grid(alpha=0.2)
    fig.suptitle("V-JEPA 2.1 + object semantics: complete comparison", fontsize=18)
    fig.tight_layout()
    fig.savefig(output / "experiment_overview.png", dpi=200)
    plt.close(fig)

    best_fixed = multiseed_map["global_w4"]
    global_holdout = holdout["aggregate"]["global"]
    object_holdout = holdout["aggregate"]["object"]
    component_rows = sorted(ablations, key=lambda row: row["accuracy"], reverse=True)
    selected_components = [
        next(row for row in component_rows if row["experiment"] == experiment)
        for experiment in (
            "global_baseline_window_4",
            "window_4",
            "full_object_semantics_300e",
            "global_baseline_300e",
            "global_plus_masked_no_scores",
            "bbox_instead_of_sam",
            "zero_context_control",
            "channel_shuffle_control",
            "temporal_shuffle_control",
            "clip_text_prototype",
            "no_color_jitter",
        )
    ]
    report = [
        "# V-JEPA 2.1 + 객체 의미 결합 비교실험 최종 보고서",
        "",
        "## 최종 결론",
        "",
        f"- 현재 고정 분할의 최종 기본 모델은 **Global W4**입니다. 3개 시드 평균 정확도 {_percent(best_fixed['accuracy_mean'])} ± {_percent(best_fixed['accuracy_std'])}, macro-F1 {_percent(best_fixed['macro_f1_mean'])} ± {_percent(best_fixed['macro_f1_std'])}입니다.",
        f"- 날짜 홀드아웃의 날짜별 단순 평균은 Global {_percent(global_holdout['accuracy']['mean'])}, Object {_percent(object_holdout['accuracy']['mean'])}입니다. macro-F1도 Global {_percent(global_holdout['macro_f1']['mean'])}, Object {_percent(object_holdout['macro_f1']['mean'])}로 전역 모델이 더 안정적입니다.",
        f"- 다만 샘플 수로 가중한 날짜 홀드아웃 정확도는 Global {_percent(global_holdout['accuracy']['sample_weighted_mean'])}, Object {_percent(object_holdout['accuracy']['sample_weighted_mean'])}입니다. 07/27의 큰 테스트셋에서 객체 정보가 도움을 준 영향입니다.",
        "- 객체 결합은 모든 라벨에 동일하게 유리하지 않습니다. 컵·락앤락·우유·스윕은 개선됐지만 과자와 전환은 저하되어, 다음 모델은 전역 특징을 기본으로 유지하고 객체 분기를 클래스별/샘플별로 게이트하는 방식이 적합합니다.",
        "",
        "## 실험 설계",
        "",
        "- 입력 표현: 동결된 V-JEPA 2.1 dense patch 특징",
        "- 객체 정보: Grounding DINO로 7개 공통 객체 후보를 검출하고 SAM2로 얻은 마스크를 V-JEPA patch에 정렬",
        "- 언어 정보: 한국어 행동 설명을 포함한 프롬프트를 CLIP ViT-B/32 임베딩으로 만든 prototype head",
        "- 라벨: Cup, Lock, Milk, Snack, Sweep, Trans의 저장 ID는 유지하고 표시용 행동 설명만 별도 관리",
        "- 증강: 원본 + color jitter, choco 포함 데이터 제외",
        "- 검증: 구성요소 ablation, 3개 시드, 날짜 전체 홀드아웃, 확률 앙상블, calibration, 군집도, 추론 지연",
        "",
        "## 주요 구성요소 실험",
        "",
        "| 실험 | 정확도 | Macro-F1 | 의미 |",
        "|---|---:|---:|---|",
    ]
    for row in selected_components:
        report.append(
            f"| {row['experiment']} | {_percent(row['accuracy'])} | {_percent(row['macro_f1'])} | {row['hypothesis']} |"
        )
    report.extend([
        "",
        "핵심 해석:",
        "",
        "- W=8에서는 객체 결합이 동일 조건 전역 모델보다 평균 정확도와 macro-F1을 개선했습니다.",
        "- W=4에서는 짧은 시간 문맥 자체의 이득이 매우 커서 전역 모델이 객체 모델보다 좋았습니다.",
        "- SAM2 마스크가 bbox보다 좋았고, 객체 점수만 쓰는 것보다 마스크 내부 V-JEPA 시각 특징을 쓰는 편이 좋았습니다.",
        "- 객체 채널/시간 순서를 섞으면 성능이 낮아져 객체 종류와 시간 정렬이 실제 정보를 제공한다는 negative control이 성립했습니다.",
        "- color jitter 제거 시 성능이 크게 하락해 색상 변화 증강은 계속 유지하는 것이 좋습니다.",
        "- CLIP 행동 prototype은 단독 주분류기보다 약하지만 Snack 재현율이 높아 향후 보조 loss나 게이트 입력으로 쓸 가치가 있습니다.",
        "",
        "## 날짜 홀드아웃",
        "",
        "| 날짜 | Global | Object | 고정 50:50 앙상블 |",
        "|---|---:|---:|---:|",
    ])
    for row in holdout["paired"]:
        ensemble = holdout_ensembles[row["holdout"]]["fixed_50_50"]
        report.append(
            f"| {row['holdout']} | {_percent(row['global_accuracy'])} | {_percent(row['object_accuracy'])} | {_percent(ensemble['accuracy'])} |"
        )
    report.extend([
        "",
        "| 라벨 | Global recall | Object recall | 차이 |",
        "|---|---:|---:|---:|",
    ])
    for name, metric in per_class.items():
        report.append(
            f"| {name} | {_percent(metric['global'])} | {_percent(metric['object'])} | {100 * metric['delta']:+.1f}%p |"
        )
    report.extend([
        "",
        "## 고정 분할 진단",
        "",
        "| 모델 | 정확도 | ECE↓ | NLL↓ | ARI↑ | NMI↑ | 지연/ms·sample | 파라미터 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in diagnostics:
        report.append(
            f"| {row['name']} | {_percent(row['accuracy'])} | {row['ece']:.3f} | {row['nll']:.3f} | "
            f"{row['cluster_ari']:.3f} | {row['cluster_nmi']:.3f} | {row['sample_latency_ms']:.3f} | {row['trainable_parameters']:,} |"
        )
    report.extend([
        "",
        "- 위 지연 시간은 저장된 V-JEPA/객체 sidecar를 입력받는 **분류기 head만** 측정한 값입니다. 영상에서 Grounding DINO·SAM2·V-JEPA를 실행하는 전처리 시간은 포함하지 않습니다.",
        "- 객체 모델은 전역 모델보다 파라미터와 지연이 늘지만, 네 모델 모두 head 자체는 매우 가볍습니다.",
        "- Global W4는 정확도와 군집 분리도가 가장 높고, Object 계열은 ECE가 더 낮아 확률 신뢰도가 상대적으로 낫습니다.",
        "",
        "## 앙상블",
        "",
        f"- W4 고정 50:50: 정확도 {_percent(ensemble_w4['fixed_50_50']['accuracy'])}; Global W4 단독 {_percent(ensemble_w4['global_only']['accuracy'])}보다 낮습니다.",
        f"- W8 고정 50:50: 정확도 {_percent(ensemble_w8['fixed_50_50']['accuracy'])}; 양쪽 단일 모델보다 높습니다.",
        "- 검증셋에서 고른 최적 가중치는 편향된 탐색 결과이므로 최종 성능으로 보고하지 않았습니다. 실제 게이트는 별도의 calibration split 또는 새로운 촬영일 데이터로 학습해야 합니다.",
        "",
        "## 권장 다음 단계",
        "",
        "1. 배포/후속 정책 입력 기본값은 Global W4 체크포인트로 둡니다.",
        "2. 객체 분기는 클래스별 게이트를 둡니다. 현재 근거상 Cup/Lock/Milk/Sweep에는 양의 가중치, Snack에는 낮은 가중치가 타당합니다.",
        "3. 날짜별 촬영 환경 편차가 크므로 새 날짜와 새 배경을 포함한 독립 test set을 먼저 확보합니다.",
        "4. 조기 종료를 적용합니다. 날짜 홀드아웃 최고점은 대체로 73–172 epoch였고 후반에는 과적합이 나타났습니다.",
        "5. 실시간 사용 시에는 Grounding DINO+SAM2를 매 프레임 돌리지 말고 주기적 검출 + mask tracking/캐시를 사용합니다.",
        "",
        "## 해석 시 주의사항",
        "",
        "- 고정 분할 정확도는 같은 촬영 분포 안의 결과이며, 날짜 홀드아웃보다 낙관적입니다.",
        "- 날짜별 테스트 표본 수가 101/262/734로 불균형하므로 날짜 단순 평균과 샘플 가중 평균을 모두 제시했습니다.",
        "- 3개 시드 반복은 고정 분할에 수행했고, 날짜 홀드아웃은 seed 42 한 번씩 수행했습니다.",
        "- 객체 sidecar는 오프라인으로 생성되어 학습 시 GT 라벨로 프롬프트를 선택하지 않습니다. 모든 샘플에 같은 7개 프롬프트 bank를 사용합니다.",
        "",
        "## 결과 파일",
        "",
        "- `experiment_overview.png`: 발표용 4분할 핵심 요약",
        "- `../ablations/summary/ablation_dashboard.png`: 전체 구성요소 실험",
        "- `../multiseed_300e/summary/multiseed_comparison.png`: 3개 시드 평균/표준편차",
        "- `../date_holdout_300e/summary/date_holdout_comparison.png`: 날짜 홀드아웃",
        "- `../date_holdout_300e/summary/date_holdout_per_class_delta.png`: 날짜 홀드아웃 클래스별 변화",
        "- `../selected_model_diagnostics/reliability_diagram.png`: 신뢰도 보정",
        "- `../selected_model_diagnostics/embedding_clusters.png`: 임베딩 t-SNE 및 군집 지표",
        "- `../ensemble_w4/ensemble_sweep.png`, `../ensemble_w8/ensemble_sweep.png`: 앙상블 민감도",
    ])
    (output / "README.md").write_text("\n".join(report) + "\n")

    machine_summary = {
        "recommended_checkpoint": diagnostics[0]["checkpoint"],
        "fixed_split_multiseed": multiseed,
        "date_holdout": holdout,
        "diagnostics": diagnostics,
        "ensemble_w4": ensemble_w4,
        "ensemble_w8": ensemble_w8,
    }
    (output / "summary.json").write_text(json.dumps(machine_summary, indent=2) + "\n")
    print(output.resolve())


if __name__ == "__main__":
    main()

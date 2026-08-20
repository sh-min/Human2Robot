"""Create a Korean, non-specialist-friendly HTML guide for all experiments."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


KOREAN_EXPERIMENTS = {
    "global_baseline_300e": ("전역 영상만 사용 (W8, 300 epoch)", "객체 검출 없이 전체 화면의 V-JEPA 특징만 사용"),
    "full_object_semantics_300e": ("전역+객체 의미 (W8, 300 epoch)", "Grounding DINO+SAM2 객체 마스크와 전역 특징을 함께 사용"),
    "full_object_150": ("전역+객체 기준 (W8)", "다른 ablation과 같은 150 epoch 객체 결합 기준"),
    "global_baseline_150": ("전역 기준 (W8)", "다른 ablation과 같은 150 epoch 전역 기준"),
    "global_baseline_window_4": ("전역 영상, 짧은 문맥 W4", "최근 4개 토큰만 보고 행동 분류"),
    "global_plus_scores": ("전역+검출 점수", "객체 내부 영상 특징은 빼고 검출 신뢰도·면적만 추가"),
    "global_plus_masked_no_scores": ("전역+객체 내부 영상 특징", "검출 점수 없이 SAM 영역 안의 V-JEPA 특징만 추가"),
    "object_only_full": ("객체 분기만 사용", "전체 화면을 빼고 객체 영역 정보로만 분류"),
    "detector_metadata_only": ("검출 숫자만 사용", "검출 신뢰도와 마스크 면적만으로 분류하는 shortcut 점검"),
    "bbox_instead_of_sam": ("SAM 대신 사각형 bbox", "정밀 마스크가 정말 필요한지 확인"),
    "zero_context_control": ("객체 입력을 모두 0으로", "파라미터가 많아진 효과인지 실제 객체 정보 효과인지 확인"),
    "channel_shuffle_control": ("객체 이름 섞기", "컵·우유·스펀지 등 객체 정체성이 실제로 중요한지 확인"),
    "temporal_shuffle_control": ("객체 시간 순서 섞기", "객체가 등장하는 시점이 실제로 중요한지 확인"),
    "clip_text_prototype": ("행동 문장 CLIP 분류", "한국어 행동 의미를 반영한 문장 prototype과 영상 특징을 비교"),
    "clip_text_hybrid": ("일반 분류기+CLIP 문장", "학습형 head와 행동 문장 유사도를 함께 사용"),
    "no_color_jitter": ("색상 증강 제거", "color jitter가 일반화에 주는 효과 확인"),
    "window_4": ("객체 결합 W4", "객체 결합 모델의 문맥을 최근 4토큰으로 축소"),
    "window_12": ("객체 결합 W12", "더 긴 12토큰 문맥이 좋은지 확인"),
    "dropout_03": ("Dropout 0.3", "객체 결합 head의 dropout을 낮춤"),
    "dropout_05": ("Dropout 0.5", "객체 결합 head의 dropout을 높임"),
    "window4_masked_no_scores": ("W4+객체 영상 특징만", "짧은 문맥과 score 없는 객체 시각 특징 결합"),
    "window4_dropout_03": ("W4+Dropout 0.3", "짧은 문맥과 낮은 dropout 결합"),
    "window4_masked_no_scores_dropout_03": ("W4+객체 영상 특징+Dropout 0.3", "좋아 보인 요인을 동시에 결합"),
    "window4_clip_text_prototype": ("W4+CLIP 행동 문장", "짧은 문맥에서 언어 의미가 도움 되는지 확인"),
}


def _load(path):
    return json.loads(path.read_text())


def _pct(value):
    return f"{100 * value:.2f}%"


def _pp(value):
    return f"{100 * value:+.1f}%p"


def _rel(output, target):
    return Path("../" + str(target.relative_to(output.parent))).as_posix()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "readable_report"
    output.mkdir(parents=True, exist_ok=True)

    ablations = _load(root / "ablations/summary/ablation_results.json")
    multiseed = _load(root / "multiseed_300e/summary/aggregate.json")
    holdout = _load(root / "date_holdout_300e/summary/date_holdout_paired.json")
    diagnostics = _load(root / "selected_model_diagnostics/diagnostics.json")
    multiseed_by_name = {row["group"]: row for row in multiseed}
    ablation_by_name = {row["experiment"]: row for row in ablations}

    rows = []
    for row in sorted(ablations, key=lambda item: item["accuracy"], reverse=True):
        title, meaning = KOREAN_EXPERIMENTS.get(row["experiment"], (row["experiment"], row["hypothesis"]))
        rows.append(
            f"<tr><td><code>{html.escape(row['experiment'])}</code><br><strong>{html.escape(title)}</strong></td>"
            f"<td>{html.escape(meaning)}</td><td>{_pct(row['accuracy'])}</td>"
            f"<td>{_pct(row['macro_f1'])}</td><td>{row['best_epoch']}</td></tr>"
        )

    class_rows = []
    for name, metric in holdout["sample_weighted_per_class"].items():
        tone = "good" if metric["delta"] >= 0 else "bad"
        interpretation = {
            "Cup": "컵과 컵 홀더의 객체 구성이 날짜가 바뀌어도 강한 단서였습니다.",
            "Lock": "두 통의 관계를 객체 영역이 보완했습니다.",
            "Milk": "우유팩·쓰레기통 정보가 소폭 도움이 됐지만 절대 recall은 아직 낮습니다.",
            "Snack": "객체 분기가 과자와 배경/쓰레기통 단서를 불안정하게 학습했습니다. 최우선 개선 대상입니다.",
            "Sweep": "스펀지 객체가 강한 단서지만 원래도 쉬운 클래스라 ceiling 효과가 있습니다.",
            "Trans": "객체 정체성보다 손·운동 변화가 중요한 라벨이라 객체 정보가 방해됐습니다.",
        }[name]
        class_rows.append(
            f"<tr><td><strong>{name}</strong></td><td>{_pct(metric['global'])}</td>"
            f"<td>{_pct(metric['object'])}</td><td class='{tone}'>{_pp(metric['delta'])}</td>"
            f"<td>{interpretation}</td></tr>"
        )

    diag_rows = []
    for row in diagnostics:
        diag_rows.append(
            f"<tr><td><strong>{html.escape(row['name'])}</strong></td><td>{_pct(row['accuracy'])}</td>"
            f"<td>{row['ece']:.3f}</td><td>{row['cluster_ari']:.3f}</td>"
            f"<td>{row['sample_latency_ms']:.3f}</td><td>{row['trainable_parameters']:,}</td></tr>"
        )

    global_w4 = multiseed_by_name["global_w4"]
    object_w4 = multiseed_by_name["object_w4"]
    global_w8 = multiseed_by_name["global_w8"]
    object_w8 = multiseed_by_name["object_w8"]
    no_jitter = ablation_by_name["no_color_jitter"]
    full_150 = ablation_by_name["full_object_150"]
    bbox = ablation_by_name["bbox_instead_of_sam"]
    mask_visual = ablation_by_name["global_plus_masked_no_scores"]
    score_only = ablation_by_name["global_plus_scores"]

    overview = _rel(output, root / "final_report/experiment_overview.png")
    ablation_plot = _rel(output, root / "ablations/summary/ablation_dashboard.png")
    multiseed_plot = _rel(output, root / "multiseed_300e/summary/multiseed_comparison.png")
    holdout_plot = _rel(output, root / "date_holdout_300e/summary/date_holdout_comparison.png")
    class_plot = _rel(output, root / "date_holdout_300e/summary/date_holdout_per_class_delta.png")
    cluster_plot = _rel(output, root / "selected_model_diagnostics/embedding_clusters.png")
    calibration_plot = _rel(output, root / "selected_model_diagnostics/reliability_diagram.png")
    checkpoint = diagnostics[0]["checkpoint"]

    document = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V-JEPA 비교실험 이해하기</title>
<style>
:root{{--ink:#172033;--muted:#64748b;--line:#dbe3ee;--blue:#2563eb;--green:#15803d;--red:#c2413b;--amber:#b45309;--bg:#f4f7fb}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:Pretendard,"Noto Sans KR",Arial,sans-serif;line-height:1.68}}
main{{max-width:1180px;margin:auto;padding:36px 24px 80px}} h1{{font-size:38px;line-height:1.25;margin:0 0 10px}} h2{{margin-top:48px;font-size:27px;border-bottom:2px solid var(--line);padding-bottom:10px}} h3{{margin-top:28px}}
.lead{{font-size:18px;color:#475569}} .card,.callout{{background:white;border:1px solid var(--line);border-radius:16px;padding:22px;margin:18px 0;box-shadow:0 5px 20px #16335b0d}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}} .metric{{background:white;border:1px solid var(--line);border-radius:14px;padding:18px}} .metric b{{font-size:29px;display:block;color:var(--blue)}}
.decision{{border-left:6px solid var(--green)}} .warn{{border-left:6px solid var(--amber)}} .bad{{color:var(--red);font-weight:700}} .good{{color:var(--green);font-weight:700}}
table{{width:100%;border-collapse:collapse;background:white;font-size:14px}} th,td{{padding:11px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}} th{{position:sticky;top:0;background:#eef4ff}} tr:hover{{background:#f8fbff}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:14px}} img{{max-width:100%;height:auto;border:1px solid var(--line);border-radius:14px;background:white}} code{{font-size:12px;color:#334155}} .tag{{display:inline-block;padding:3px 9px;border-radius:999px;background:#dbeafe;color:#1d4ed8;font-size:12px;font-weight:700}}
.priority{{display:grid;grid-template-columns:110px 1fr 115px;gap:12px;align-items:start;padding:16px 0;border-bottom:1px solid var(--line)}} .p1{{color:#b91c1c;font-weight:800}} .p2{{color:#b45309;font-weight:800}} .p3{{color:#1d4ed8;font-weight:800}}
details{{background:white;border:1px solid var(--line);border-radius:12px;padding:12px 16px;margin:10px 0}} summary{{cursor:pointer;font-weight:700}} a{{color:#1d4ed8}} ul,ol{{padding-left:23px}} @media(max-width:760px){{.grid{{grid-template-columns:1fr}}h1{{font-size:30px}}.priority{{grid-template-columns:1fr}}}}
</style></head><body><main>
<span class="tag">한눈에 보는 최종판</span><h1>V-JEPA 2.1 비교실험,<br>무엇을 했고 무엇이 부족한가</h1>
<p class="lead">숫자를 많이 보여주는 대신, 실험 질문 → 결과 → 결정 → 다음 작업 순서로 정리했습니다. 전체 정확도만 보지 않고 새 촬영 날짜와 행동별 실패를 중심으로 해석합니다.</p>

<div class="grid">
 <div class="metric"><span>현재 권장 모델</span><b>Global W4</b><small>객체 모델 없이 최근 4토큰 사용</small></div>
 <div class="metric"><span>고정 분할 3시드</span><b>{_pct(global_w4['accuracy_mean'])}</b><small>± {_pct(global_w4['accuracy_std'])}, Macro-F1 {_pct(global_w4['macro_f1_mean'])}</small></div>
 <div class="metric"><span>새 날짜 단순 평균</span><b>{_pct(holdout['aggregate']['global']['accuracy']['mean'])}</b><small>고정 분할보다 약 15%p 낮음</small></div>
</div>

<div class="callout decision"><h3>결론부터 말씀드리면</h3>
<p><strong>현재 체크포인트로 후속 작업을 진행할 때는 Global W4가 가장 안전합니다.</strong> 객체 결합은 W8에서는 평균 {_pp(object_w8['accuracy_mean']-global_w8['accuracy_mean'])} 개선됐지만, W4에서는 {_pp(object_w4['accuracy_mean']-global_w4['accuracy_mean'])} 낮았습니다. 객체 정보 자체가 무의미한 것은 아니며, 행동마다 도움이 되는 정도가 달라 무조건 합치는 구조가 문제입니다.</p></div>

<h2>1. 이 실험을 아주 간단히 설명하면</h2>
<div class="card"><ol>
<li>사람 손이 나오는 주방 영상을 V-JEPA 2.1의 dense patch 특징으로 바꿨습니다.</li>
<li>기본 모델은 화면 전체에서 중요한 위치를 attention으로 모아 6개 행동을 분류합니다.</li>
<li>객체 모델은 Grounding DINO로 컵·우유팩·스펀지 등을 찾고 SAM2 마스크 안쪽 V-JEPA 특징을 추가합니다.</li>
<li>CLIP 행동 문장, dropout, 시간 window, bbox/SAM, 잘못 섞은 객체 정보까지 비교했습니다.</li>
<li>마지막에는 3개 random seed와 촬영 날짜 전체 홀드아웃으로 재확인했습니다.</li>
</ol></div>
<p><strong>W4/W8</strong>은 모델이 현재 판단할 때 최근 V-JEPA 토큰을 4개/8개 보는 뜻입니다. 현재 영상의 토큰 주기 설정에 종속되므로 초 단위 문맥은 feature sampling contract와 함께 관리해야 합니다.</p>

<h2>2. 가장 중요한 결과 다섯 가지</h2>
<div class="card"><ol>
<li><strong>시간 문맥:</strong> W4가 가장 좋았습니다. 객체 W12는 {_pct(ablation_by_name['window_12']['accuracy'])}까지 떨어졌습니다. 긴 문맥이 오히려 이전 행동과 Trans를 섞었습니다.</li>
<li><strong>객체 정보:</strong> 동일 W8/3시드에서 Global {_pct(global_w8['accuracy_mean'])} → Object {_pct(object_w8['accuracy_mean'])}로 좋아졌습니다. 하지만 더 강한 W4 전역 모델은 넘지 못했습니다.</li>
<li><strong>SAM 마스크:</strong> full SAM 기준 {_pct(full_150['accuracy'])}, bbox {_pct(bbox['accuracy'])}였습니다. 정밀한 객체 경계는 실제로 도움이 됐습니다.</li>
<li><strong>검출 숫자보다 객체 안의 영상:</strong> 객체 내부 시각 특징 {_pct(mask_visual['accuracy'])}, 점수·면적 위주 {_pct(score_only['accuracy'])}였습니다. detector confidence shortcut에 의존하면 안 됩니다.</li>
<li><strong>Color jitter:</strong> 제거 시 {_pct(no_jitter['accuracy'])}로 크게 하락했습니다. 특히 Trans recall이 45.7%까지 떨어져 색상 증강은 유지해야 합니다.</li>
</ol></div>

<a href="{overview}"><img src="{overview}" alt="전체 비교 요약"></a>

<h2>3. 날짜가 바뀌면 어떻게 됐나</h2>
<p>원본 45개 녹화본은 07/24 10개, 07/27 30개, 07/28 5개입니다. 한 날짜 전체를 학습에서 제외하고 시험했습니다. 같은 촬영 분포의 고정 검증보다 훨씬 현실적인 지표입니다.</p>
<div class="table-wrap"><table><thead><tr><th>처음 보는 날짜</th><th>Global W4</th><th>Object W4</th><th>객체 효과</th><th>해석</th></tr></thead><tbody>
{''.join(f"<tr><td>{r['holdout']}</td><td>{_pct(r['global_accuracy'])}</td><td>{_pct(r['object_accuracy'])}</td><td class='{'good' if r['accuracy_delta']>=0 else 'bad'}'>{_pp(r['accuracy_delta'])}</td><td>{'객체가 도움' if r['accuracy_delta']>=0 else '전역 모델이 안정적'}</td></tr>" for r in holdout['paired'])}
</tbody></table></div>
<div class="callout warn"><strong>읽을 때 주의:</strong> 날짜별 데이터 수가 10/30/5로 불균형합니다. 날짜 단순 평균은 Global {_pct(holdout['aggregate']['global']['accuracy']['mean'])}, Object {_pct(holdout['aggregate']['object']['accuracy']['mean'])}이지만, 샘플 가중 평균은 07/27의 영향으로 Global {_pct(holdout['aggregate']['global']['accuracy']['sample_weighted_mean'])}, Object {_pct(holdout['aggregate']['object']['accuracy']['sample_weighted_mean'])}입니다. 둘 중 하나만 골라 말하면 오해가 생깁니다.</div>
<a href="{holdout_plot}"><img src="{holdout_plot}" alt="날짜 홀드아웃"></a>

<h2>4. 어떤 행동에 객체 정보가 도움이 됐나</h2>
<div class="table-wrap"><table><thead><tr><th>행동</th><th>Global recall</th><th>Object recall</th><th>차이</th><th>해석</th></tr></thead><tbody>{''.join(class_rows)}</tbody></table></div>
<a href="{class_plot}"><img src="{class_plot}" alt="클래스별 객체 효과"></a>

<h2>5. 모든 ablation을 쉬운 이름으로 보기</h2>
<p>아래 150 epoch 실험끼리는 공정하게 직접 비교할 수 있습니다. 이름에 300e가 붙은 기존 reference와 150 epoch 결과는 학습 budget이 다르므로 단순 순위로만 비교하면 안 됩니다.</p>
<div class="table-wrap" style="max-height:720px"><table><thead><tr><th>실험</th><th>무엇을 바꿨나</th><th>정확도</th><th>Macro-F1</th><th>최고 epoch</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p><a href="{ablation_plot}">전체 ablation 그림 크게 보기</a> · <a href="{multiseed_plot}">3시드 비교 그림 보기</a></p>

<h2>6. 점수 외 진단</h2>
<div class="table-wrap"><table><thead><tr><th>모델</th><th>정확도</th><th>ECE↓</th><th>군집 ARI↑</th><th>Head 지연 ms/sample</th><th>파라미터</th></tr></thead><tbody>{''.join(diag_rows)}</tbody></table></div>
<ul><li>Global W4는 정확도와 임베딩 군집 분리가 가장 좋습니다.</li><li>Object 모델은 ECE가 더 낮아 confidence는 상대적으로 잘 맞지만 정확도 손실이 있습니다.</li><li>속도는 <strong>분류 head만</strong> 측정했습니다. 실제 Grounding DINO·SAM2·V-JEPA 전처리 시간은 포함되지 않았습니다.</li></ul>
<p><a href="{cluster_plot}">임베딩 군집 보기</a> · <a href="{calibration_plot}">신뢰도 그래프 보기</a></p>

<h2>7. 현재 부족한 부분</h2>
<div class="card">
<div class="priority"><div class="p1">가장 큼</div><div><strong>독립 테스트 환경이 없습니다.</strong><br>날짜 홀드아웃은 했지만 같은 공간·카메라·물체·사람일 가능성이 높습니다. 새 배경, 조명, 사람, 카메라 위치, 물체 배치가 바뀐 데이터가 필요합니다.</div><div>결론 신뢰도</div></div>
<div class="priority"><div class="p1">가장 큼</div><div><strong>Snack과 Milk 표본 및 성능이 부족합니다.</strong><br>고정 검증 support는 Snack 17, Milk 19뿐입니다. 날짜 홀드아웃 recall도 Snack 55.3%/35.3%, Milk 57.3%/61.8%로 낮습니다.</div><div>소수 클래스</div></div>
<div class="priority"><div class="p1">가장 큼</div><div><strong>객체 결합 방식이 정적입니다.</strong><br>모든 행동에 같은 방식으로 객체 정보를 넣어 Snack과 Trans에 역효과가 생깁니다. 행동별 또는 샘플별 gate가 필요합니다.</div><div>모델 구조</div></div>
<div class="priority"><div class="p2">중요</div><div><strong>날짜 홀드아웃은 seed 42 한 번뿐입니다.</strong><br>고정 분할은 3시드지만 더 중요한 날짜 일반화 분산은 아직 모릅니다.</div><div>통계 안정성</div></div>
<div class="priority"><div class="p2">중요</div><div><strong>과적합과 조기 종료가 정식 적용되지 않았습니다.</strong><br>날짜별 최고 epoch가 73~172였고 이후 검증 성능이 내려갔습니다. 300 epoch 마지막 모델을 사용하면 안 됩니다.</div><div>학습 절차</div></div>
<div class="priority"><div class="p2">중요</div><div><strong>행동 경계 평가는 없습니다.</strong><br>토큰 정확도는 높아도 실제 연속 영상에서 행동 전환 시점이 늦거나 흔들릴 수 있습니다. segment F1, edit score, boundary tolerance 평가가 필요합니다.</div><div>실사용 지표</div></div>
<div class="priority"><div class="p3">추가</div><div><strong>언어 분기의 공정한 활용이 아직 약합니다.</strong><br>CLIP 문장 head를 교체했지만 보조 loss, prompt ensemble, 객체-행동 relation attention은 아직 비교하지 않았습니다.</div><div>VLM 활용</div></div>
<div class="priority"><div class="p3">추가</div><div><strong>온라인 비용을 측정하지 않았습니다.</strong><br>현재 속도는 feature가 이미 준비됐다는 가정입니다. 영상→검출/분할→V-JEPA→분류 전체 FPS와 GPU 메모리를 재야 합니다.</div><div>배포 가능성</div></div>
</div>

<h2>8. 다음에 할 실험 우선순위</h2>
<div class="card"><ol>
<li><strong>P1 — 독립 test set 수집:</strong> 새 날짜 2개 이상, 가능하면 다른 사람·조명·카메라 위치로 각 행동 10회 이상 촬영합니다. 기존 validation은 건드리지 않고 최종 한 번만 평가합니다.</li>
<li><strong>P1 — Snack/Milk 집중 보강:</strong> 배경과 쓰레기통 위치를 바꾸고, 들기·이동·버리기 전 구간을 균형 있게 추가합니다. 객체 크기·색상 shortcut을 막기 위해 위치와 방향을 다양화합니다.</li>
<li><strong>P1 — Gated fusion:</strong> Global W4 logits를 기본으로 유지하고 객체 branch의 출력을 학습 가능한 gate로 더합니다. 먼저 class-wise gate, 다음 sample-wise gate를 비교합니다.</li>
<li><strong>P2 — 날짜 홀드아웃 3시드:</strong> 6개 모델이 아니라 우선 Global W4, Object W4, gated W4 세 모델만 seed 7/42/123으로 반복합니다.</li>
<li><strong>P2 — 조기 종료:</strong> patience 25~30, 최소 개선폭 0.2%p를 넣고 epoch/성능/시간을 기존 300 epoch와 비교합니다.</li>
<li><strong>P2 — 연속 영상 평가:</strong> frame/token accuracy 외에 segmental F1@10/25/50, edit score, boundary ±0.5초 정확도를 추가합니다.</li>
<li><strong>P3 — VLM relation attention:</strong> 단순 문장 prototype 대신 `행동 ↔ source object ↔ target object` 관계를 query로 사용합니다. GT 행동 문장을 선택해 넣으면 label leakage이므로 모든 행동 query를 동시에 사용해야 합니다.</li>
<li><strong>P3 — 전체 파이프라인 속도:</strong> detector를 매 프레임 실행하는 방식과 1초마다 검출+tracking 방식의 FPS/메모리/성능을 비교합니다.</li>
</ol></div>

<h2>9. 바로 사용하면 되는 파일</h2>
<div class="card"><ul>
<li><strong>권장 체크포인트:</strong> <code>{html.escape(checkpoint)}</code></li>
<li><a href="../final_report/README.md">수치가 포함된 상세 Markdown 보고서</a></li>
<li><a href="../final_report/summary.json">프로그램에서 읽을 수 있는 전체 JSON</a></li>
<li><a href="../ablations/summary/README.md">Ablation 원본 요약</a></li>
<li><a href="../date_holdout_300e/summary/README.md">날짜 홀드아웃 원본 요약</a></li>
</ul></div>

<details><summary>정확도, Macro-F1, Recall, ECE가 무슨 뜻인가요?</summary><ul>
<li><strong>정확도:</strong> 전체 토큰 중 맞힌 비율입니다. Sweep처럼 많은 클래스의 영향이 큽니다.</li>
<li><strong>Macro-F1:</strong> 클래스별 F1을 동일 비중으로 평균합니다. 소수 클래스가 중요할 때 정확도보다 낫습니다.</li>
<li><strong>Recall:</strong> 실제 해당 행동을 얼마나 놓치지 않았는지입니다.</li>
<li><strong>ECE:</strong> 모델 confidence와 실제 정답률 차이입니다. 낮을수록 confidence를 믿기 쉽습니다.</li>
</ul></details>
</main></body></html>"""
    (output / "index.html").write_text(document)

    next_steps = f"""# 다음 비교실험 우선순위

## P1: 바로 해야 할 것

1. 새 날짜·사람·조명·카메라 위치의 독립 test set을 확보합니다.
2. Snack/Milk를 각 조건별로 보강하고 클래스 균형을 맞춥니다.
3. Global W4를 기본으로 두는 class-wise/object-confidence gated fusion을 구현합니다.

## P2: 결론을 단단하게 할 것

1. Global W4 / Object W4 / Gated W4만 날짜 홀드아웃 3시드로 반복합니다.
2. patience 기반 조기 종료를 넣어 과적합과 학습 시간을 비교합니다.
3. 연속 영상에서 segmental F1, edit score, boundary tolerance를 평가합니다.

## P3: 연구 확장

1. 모든 행동-객체 관계 query를 동시에 쓰는 relation attention을 비교합니다.
2. 전체 온라인 파이프라인의 FPS·GPU 메모리·정확도 trade-off를 측정합니다.

현재 배포 기본 체크포인트:

`{checkpoint}`
"""
    (output / "NEXT_EXPERIMENTS.md").write_text(next_steps)
    print(output / "index.html")
    print(output / "NEXT_EXPERIMENTS.md")


if __name__ == "__main__":
    main()

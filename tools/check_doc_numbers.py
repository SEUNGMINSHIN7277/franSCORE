"""문서에 적힌 수치가 실제 산출물과 일치하는지 자동 대조한다.

왜 필요한가 (정직성):
    파이프라인을 다시 돌리면 지표가 바뀐다. 문서의 숫자는 사람이 손으로 고치므로
    어딘가 하나는 반드시 낡은 채로 남는다. "문서 수치가 산출물과 일치한다"는 주장은
    주장으로 두지 말고 **기계가 매번 확인**해야 한다.

동작:
    ① 산출물(outputs/, data/processed/)에서 핵심 수치를 다시 계산한다.
       — 기대값을 이 파일에 적어두지 않는다. 적어두면 그 자체가 또 낡는다.
    ② 그 수치를 문서 표기 형태(예: `2.456`, `3,635`, `9.2%`)로 만들어,
       지정한 문서 파일에 그 문자열이 실제로 들어 있는지 확인한다.
    ③ 폐기된 표기(STALE)가 문서 어디에도 남아 있지 않은지 확인한다.
       (공급자 교체·지표 변경 후 잔존물을 잡는다.)

한계(명시): 문자열 포함 검사이므로 "숫자가 문서에 있다"는 것까지만 보증하고
    그 숫자가 올바른 문맥에 쓰였는지는 사람이 본다. 그래도 낡은 수치는 확실히 잡는다.

실행: python tools/check_doc_numbers.py      (종료코드 0 = 전부 일치)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
PROC = ROOT / "data" / "processed"

DOCS = {
    "README": ROOT / "README.md",
    "IMPL": ROOT / "docs" / "IMPLEMENTATION.md",
    "IFACE": ROOT / "docs" / "INTERFACES.md",
    "AIUSE": ROOT / "docs" / "AI_USAGE.md",
    "LEAK": ROOT / "docs" / "LEAKAGE_CHECKLIST.md",
}

# 폐기된 표기 — 문서 어디에도 남아 있으면 안 된다 (교체 누락 탐지).
STALE: list[tuple[str, str]] = [
    ("claude-opus-5", "런타임 LLM은 Gemini로 교체됨"),
    ("ANTHROPIC_API_KEY", "런타임 LLM 키 환경변수는 GEMINI_API_KEY"),
    ("anthropic SDK", "Gemini는 SDK 없이 REST 호출"),
    ("모의 Anthropic", "테스트는 HTTP 경계 모의로 변경"),
    ("walkforward_pooled_oos", "워크포워드 주 지표는 macro_avg_folds"),
    # 실제로 호출하는 모델은 config.yaml 의 llm.model (gemini-2.5-flash) 이다.
    # 문서에만 다른 이름이 남으면 '쓰지 않는 모델을 썼다'고 적는 셈이라 잡아낸다.
    ("gemini-3.6", "실제 호출 모델은 config.yaml llm.model 값"),
    ("src/ui.py", "화면 코드는 src/views/ 로 분리됨"),
    ("모델 근거", "해당 화면은 AI 상담으로 대체됨"),
]

_fails: list[str] = []
_checked = 0


def _txt(key: str) -> str:
    return DOCS[key].read_text(encoding="utf-8")


def need(label: str, value: str, *doc_keys: str) -> None:
    """value 문자열이 지정한 문서들에 모두 있어야 한다."""
    global _checked
    _checked += 1
    missing = [k for k in doc_keys if value not in _txt(k)]
    if missing:
        _fails.append(f"{label}: 실측 표기 '{value}' 가 {', '.join(missing)} 에 없음")
        print(f"  [MISS] {label:46s} '{value}'  → {', '.join(missing)}")
    else:
        print(f"  [OK  ] {label:46s} '{value}'  ({', '.join(doc_keys)})")


def _checked_inc() -> None:
    global _checked
    _checked += 1


def info(label: str, value: object) -> None:
    print(f"  [INFO] {label:46s} {value}")


def head(t: str) -> None:
    print("\n" + "=" * 92)
    print(t)
    print("=" * 92)


def thou(n: float) -> str:
    return f"{round(n):,}"


def main() -> int:
    head("① 패널 · 엔티티 정합")
    pf = pd.read_parquet(PROC / "panel_full.parquet")
    need("panel_full 행수", thou(len(pf)), "IMPL")
    n_mnno = pf.loc[pf["id_source"] == "MNNO", "brand_id"].nunique()
    n_name = pf.loc[pf["id_source"] == "NAME", "brand_id"].nunique()
    need("고유 브랜드 수", thou(pf["brand_id"].nunique()), "IMPL")
    need("관리번호 기반 브랜드", thou(n_mnno), "IMPL")
    need("명칭 기반 브랜드", thou(n_name), "IMPL")

    head("② 라벨 · 피처")
    lab = pd.read_parquet(PROC / "labels.parquet")
    need("라벨 표본 행수", thou(len(lab)), "README", "IMPL")
    need("라벨 양성률", f"{100 * lab['label'].mean():.1f}%", "README", "IMPL", "AIUSE")
    fs = pd.read_csv(OUT / "feature_summary.csv")
    fs_ext = pd.read_csv(OUT / "extended" / "feature_summary.csv")
    need("피처 수(기본 트랙)", str(len(fs)), "README", "IMPL", "IFACE")
    info("피처 수(확장 트랙)", len(fs_ext))
    dead = int((fs["n"] == 0).sum()) if "n" in fs.columns else 0
    info("전량 결측(죽은) 피처 수", dead)
    if dead:
        _fails.append(f"죽은 피처 {dead}개 — 문서의 '죽은 피처 0개' 주장과 불일치")

    head("③ 단일 시간분할 test (기본 트랙)")
    te = pd.read_csv(OUT / "metrics.csv")
    te = te[te["split"] == "test"].set_index("model")
    need("test 표본수", thou(te["n"].max()), "README", "IMPL")
    need("test 양성률", f"{100 * te['base_rate'].max():.1f}%", "README")
    for m in ("persistence", "single", "logistic", "lgbm"):
        need(f"{m} Lift@10", f"{te.loc[m, 'lift_at_10']:.3f}", "README")
    need("보정 Brier", f"{te.loc['lgbm_calibrated', 'brier']:.3f}", "README")

    head("④ 워크포워드 (주 지표 = fold 평균)")
    wm = pd.read_csv(OUT / "walkforward_metrics.csv")
    macro = wm[wm["scope"] == "macro_avg_folds"].set_index("model")
    need("WF 표본수(풀링 OOS)", thou(macro["n"].max()), "README")
    for m in ("persistence", "single", "logistic", "lgbm"):
        need(f"WF fold평균 {m} Lift@10", f"{macro.loc[m, 'lift_at_10']:.3f}", "README")
    # ⚠️ 예전에는 walkforward_delta_ci.csv(bootstrap_unit=year_block, 블록 3개)를
    #    기준으로 삼았다. 블록이 3개뿐인 백분위 부트스트랩은 t-구간의 약 1/3.4 폭밖에
    #    나오지 않아 없는 유의성을 만들어낸다 — 같은 저장소의 lift_delta_bootstrap.csv
    #    는 같은 비교를 비유의로 낸다. 두 산출물이 모순되므로, fold 값에서 직접 계산한
    #    t-구간을 문서 기준으로 삼는다.
    folds = wm[wm["valid_year"] > 0].pivot_table(
        index="valid_year", columns="model", values="lift_at_10")
    for base in ("persistence", "single", "logistic"):
        d = (folds["lgbm"] - folds[base]).to_numpy()
        m, n = float(d.mean()), len(d)
        se = float(d.std(ddof=1) / np.sqrt(n))
        t = float(stats.t.ppf(0.975, n - 1))
        need(f"lgbm - {base} Δ(t)", f"{m:+.3f}", "README")
        need(f"lgbm - {base} CI(t)", f"[{m - t * se:+.3f}, {m + t * se:+.3f}]", "README")
    bias = pd.read_csv(OUT / "walkforward_pool_bias.csv")
    worst = bias.loc[bias["bias_ratio"].sub(1.0).abs().idxmax()]
    need("원점수 풀링 최대 편향배수", f"{worst['bias_ratio']:.2f}", "IMPL")

    head("⑤ 두 트랙 비교")
    tc = pd.read_csv(OUT / "track_comparison.csv")
    ext_ss = tc[(tc["track"] == "extended") & (tc["eval"] == "single_split_test")].set_index("model")
    ext_wf = tc[(tc["track"] == "extended")
                & (tc["eval"] == "walkforward_macro_avg_folds")].set_index("model")
    ext_lab = pd.read_parquet(OUT / "extended" / "processed" / "labels.parquet")
    need("확장 라벨 표본", thou(len(ext_lab)), "README")
    need("확장 단일분할 lgbm", f"{ext_ss.loc['lgbm', 'lift_at_10']:.3f}", "README")
    need("확장 단일분할 logistic", f"{ext_ss.loc['logistic', 'lift_at_10']:.3f}", "README")
    need("확장 WF lgbm", f"{ext_wf.loc['lgbm', 'lift_at_10']:.3f}", "README")
    need("확장 WF logistic", f"{ext_wf.loc['logistic', 'lift_at_10']:.3f}", "README")

    head("⑥ 포트폴리오 (합성 예시 여신)")
    port = pd.read_csv(OUT / "portfolio.csv")
    s = json.loads((OUT / "portfolio_summary.json").read_text(encoding="utf-8"))
    need("총 익스포저(억원)", thou(port["exposure_mkrw"].sum() / 100), "README", "IMPL")
    need("EL LGD45(억원)", f"{port['el_lgd45_mkrw'].sum() / 100:.1f}", "README", "IMPL")
    need("스트레스 EL LGD45(억원)", f"{port['stress_el_lgd45_mkrw'].sum() / 100:.1f}",
         "README", "IMPL")
    need("상위10 집중도", f"{100 * s['concentration']['top10_share']:.1f}%", "README", "IMPL")
    need("HHI", f"{s['concentration']['hhi']:.3f}", "README", "IMPL")
    need("High 등급 브랜드 수", f"{int(s['risk_grades']['counts']['High'])}/60", "IMPL")

    head("⑥-2 가맹본부 재무 (DART) · 정보공개서 원문")
    dp = OUT / "dart_match_report.json"
    if not dp.exists():
        print("  [SKIP] dart_match_report.json 없음 — `--step dart` 실행 필요")
    else:
        dm = json.loads(dp.read_text(encoding="utf-8"))
        v = dm["verdicts"]
        need("DART 확정 법인", thou(v.get("확정", 0)), "IMPL")
        need("DART 불일치 법인", thou(v.get("불일치", 0)), "IMPL")
        # 이름 매칭만 믿었을 때의 오매칭 비율 — 이 층의 존재 이유를 숫자로 고정한다
        tot = sum(v.values()) or 1
        need("이름 매칭 오매칭률", f"{100 * v.get('불일치', 0) / tot:.0f}%", "IMPL")
        el = dm["eligible"]
        need("본부재무 커버리지(자격·가맹점가중)",
             f"{100 * el['with_financials_store_weighted']:.1f}%", "README", "IMPL")
        info("자격 브랜드 확정 커버리지(가맹점가중)",
             f"{100 * el['confirmed_store_weighted']:.1f}%")
        q = dm.get("quality") or {}
        if q.get("balance_pass_rate") is not None:
            info("회계 항등식 통과율", f"{100 * q['balance_pass_rate']:.1f}% "
                                       f"({q['balance_checked']:,}건 검증)")
        info("계속기업 불확실성 기재", q.get("going_concern_flagged"))

    ip = OUT / "ifrmp_status.json"
    if not ip.exists():
        print("  [SKIP] ifrmp_status.json 없음")
    else:
        st = json.loads(ip.read_text(encoding="utf-8"))
        pv = st.get("parser_validation") or {}
        need("정보공개서 표준 섹션 수", str(pv.get("sections_found", "?")), "IMPL")
        info("데모 고정응답 탐지", st.get("demo_check", {}).get("is_demo"))
        info("브랜드별 수집 건수", st.get("collected_brands"))
        # 데모 상태인데 '전량 수집했다'고 적혀 있으면 즉시 실패시킨다
        if st.get("demo_check", {}).get("is_demo"):
            for k in ("IMPL", "README"):
                if "정식 키" not in _txt(k):
                    _fails.append(f"정보공개서가 데모 상태인데 {k} 에 키 대기 사실이 없음")

    head("⑦ 브랜드 공통요인 상관 (전제 검증)")
    bc = json.loads((OUT / "brand_correlation.json").read_text(encoding="utf-8"))
    dec = bc["decomposition"]
    need("상관 통제없음", f"{dec['no_control']['rho_asset']:.3f}", "README")
    need("상관 연도통제", f"{dec['year_controlled']['rho_asset']:.3f}", "README")
    need("상관 연도+업종통제(headline)", f"{bc['rho_asset']:.3f}", "README", "IMPL")
    need("상관 95% CI", f"[{bc['rho_asset_ci_lo']:.3f}, {bc['rho_asset_ci_hi']:.3f}]", "README")
    need("분석 브랜드-연도", thou(bc["n_brand_years"]), "README")
    need("분석 지역쌍", thou(bc["n_region_pairs"]), "README")
    need("전지역 동시감소 실측", f"{100 * bc['all_regions_decline_observed']:.2f}%", "README")
    need("전지역 동시감소 배수", f"{bc['all_regions_decline_ratio']:.1f}배", "README")

    btw = bc["between_brand"]
    need("브랜드 간 상관 rho_B", f"{btw['rho_between']:.3f}", "README")
    need("rho_B CI", f"[{btw['rho_between_ci_lo']:.3f}, {btw['rho_between_ci_hi']:.3f}]", "README")

    ci_ = json.loads((OUT / "correlation_impact.json").read_text(encoding="utf-8"))
    need("차주(가맹점) 수", thou(ci_["n_franchisees"]), "README")
    for lvl, _lab in (("p95", "95%"), ("p99", "99%"), ("p999", "99.9%")):
        need(f"{lvl} 독립가정 손실(억)", f"{ci_[f'independent_{lvl}_mkrw'] / 100:.1f}", "README")
        need(f"{lvl} 상관반영 손실(억)", f"{ci_[f'brand_correlated_{lvl}_mkrw'] / 100:.1f}", "README")
        need(f"{lvl} 과소추정(억)", f"{ci_[f'understatement_{lvl}_mkrw'] / 100:.1f}", "README")
    need("UL99 배수", f"{ci_['ul99_multiple']:.2f}배", "README")
    need("UL99 독립(억)", f"{ci_['independent_ul99_mkrw'] / 100:.1f}", "README")
    need("UL99 상관(억)", f"{ci_['brand_correlated_ul99_mkrw'] / 100:.1f}", "README")

    head("⑦-3 LLM 정량 평가 (규칙기반 대비)")
    ep = OUT / "llm_eval.json"
    if not ep.exists():
        print("  [SKIP] llm_eval.json 없음 — `--step eval_llm` 실행 필요")
    else:
        ev = json.loads(ep.read_text(encoding="utf-8"))
        if not ev.get("llm_evaluated"):
            _fails.append(f"LLM 평가가 무효 상태다: {ev.get('llm_invalidated_reason')}")
            print(f"  [FAIL] LLM 지표 무효 — {ev.get('llm_invalidated_reason')}")
        else:
            for scope in ("real", "synthetic", "all"):
                if scope not in ev:
                    continue
                for who in ("rules", "llm"):
                    m = ev[scope][who]
                    need(f"{scope}/{who} 정확도", f"{m['accuracy']:.3f}", "README")
                    need(f"{scope}/{who} macro-F1", f"{m['macro_f1']:.3f}", "README")
                    need(f"{scope}/{who} 위험F1", f"{m['risk_f1']:.3f}", "README")
            need("평가 대상 실수집 건수", f"{ev['real']['rules']['n']}건", "README")
            need("평가 프로브 건수", f"{ev['synthetic']['rules']['n']}건", "README")
            info("평가 모델", ev.get("model"))

    head("⑦-2 폐기값 잔존 검사 (같은 자리에 옛 수치가 남아 있는가)")
    # need() 는 '정답이 문서에 있는가'만 본다 — 옛 값이 함께 남아 있어도 통과한다.
    # 실제로 이 사각지대 때문에 헤드라인에 옛 수치가 남는 사고가 있었다(자체 감사 검출).
    # 그래서 '같은 패턴에 매칭되는 다른 값'이 남아 있으면 실패시킨다.
    import re as _re
    for label, pattern, truth in (
        ("과소추정 억원", r"과소추정\**\s*\+?([\d,]+\.\d)억",
         f"{ci_['understatement_p99_mkrw'] / 100:.1f}"),
        ("몬테카를로 횟수", r"몬테카를로\s*(\d+)만\s*회", str(ci_["n_sims"] // 10000)),
        ("UL 배수", r"([\d.]+)배\*{0,2}다", f"{ci_['ul99_multiple']:.2f}"),
    ):
        txt = _txt("README")
        found = set(_re.findall(pattern, txt))
        stale = {v for v in found if v.replace(",", "") != truth}
        _checked_inc()
        if stale:
            _fails.append(f"{label}: 폐기값 {sorted(stale)} 이 README에 남아 있음 (정답 {truth})")
            print(f"  [STALE] {label:28s} 잔존 {sorted(stale)} (정답 {truth})")
        else:
            print(f"  [OK  ] {label:28s} 정답 '{truth}' 외 다른 값 없음")

    head("⑧ 폐기 표기 잔존 검사")
    for token, why in STALE:
        hits = [k for k, p in DOCS.items() if token in p.read_text(encoding="utf-8")]
        if hits:
            _fails.append(f"폐기 표기 '{token}' 잔존: {', '.join(hits)} ({why})")
            print(f"  [STALE] '{token}' → {', '.join(hits)}  ({why})")
        else:
            print(f"  [OK  ] '{token}' 잔존 없음")

    head("결과")
    if _fails:
        print(f"불일치 {len(_fails)}건 / 검사 {_checked}건")
        for f in _fails:
            print(f"   - {f}")
        return 1
    print(f"문서 수치 {_checked}건 전부 산출물 실측과 일치, 폐기 표기 잔존 없음")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())

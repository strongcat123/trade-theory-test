"""
트레이드 이론 적중률 실험 프로그램 (실시간 기록 방식)
====================================================
연구: "트레이드 이론은 미래를 맞출 수 있는가?"

흐름
  1) 이론 조건을 충족하는 봉 발견
  2) 그 순간 "오른다 / 내린다" 예측을 기록 (기록 시각, 기준가와 함께)
  3) 정해진 기간이 지나면 결과를 참 / 거짓으로 판정해서 채워 넣음

한 번 기록된 예측은 절대 수정하지 않습니다. 결과 칸만 나중에 채워집니다.

이론
  단기 (신호가 나온 봉의 종가 → 1거래일 뒤 종가로 판정)
    - RSI(14): 30 이하 → 오른다, 70 이상 → 내린다
    - 볼린저밴드(20, 2σ): 하단 아래 → 오른다, 상단 위 → 내린다
  중기 (매주 첫 거래일 종가로 예측 → 1주(5거래일) 뒤 종가로 판정)
    - 이동평균선: 정배열(5>20>60) → 오른다, 역배열(5<20<60) → 내린다
    - MACD(12,26,9): MACD > 시그널 → 오른다, 아니면 → 내린다

실행
  pip install yfinance pandas
  python trade_theory_tracker.py          # 계속 실행 (Ctrl+C로 종료)
  python trade_theory_tracker.py --once   # 한 번만 실행

결과 (results 폴더, 엑셀로 열림)
  단기_예측기록.csv   중기_예측기록.csv   적중률_요약.csv   실행로그.txt
"""

import argparse
import time
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ─────────────────────────── 설정 ───────────────────────────
START = date(2026, 11, 18)            # 연구 시작일
END = START + timedelta(days=21)      # 연구 종료일 (2026-12-09)
HORIZONS = [1]                        # 단기 판정: 몇 거래일 뒤 (예: [1, 3]으로 늘릴 수 있음)
MID_DAYS = 5                          # 중기 판정: 1주 = 5거래일 뒤
MID_LAST = END - timedelta(days=7)    # 중기 예측 마지막 날 (판정이 연구 기간 안에 끝나도록)
INTERVAL_MIN = 10                     # 확인 주기(분)

# 단기 신호를 언제 판단할지
#   "마감": 장이 끝나 봉이 완성된 뒤 판단 (기준가 = 종가). 일봉 이론의 정석.
#   "장중": 장중에 실시간으로 판단 (기준가 = 발견 시점 가격). 장 마감 때 조건이 사라져도 기록은 남음.
MODE = "마감"

OUT_DIR = Path(__file__).resolve().parent / "results"
SHORT_LOG = OUT_DIR / "단기_예측기록.csv"
MID_LOG = OUT_DIR / "중기_예측기록.csv"
SUMMARY = OUT_DIR / "적중률_요약.csv"
RUN_LOG = OUT_DIR / "실행로그.txt"

# (이름, yfinance 코드, 시장, 유형)
TICKERS = [
    ("코스피", "^KS11", "한국", "지수·ETF"),
    ("코스닥", "^KQ11", "한국", "지수·ETF"),
    ("삼성전자", "005930.KS", "한국", "개별종목"),
    ("SK하이닉스", "000660.KS", "한국", "개별종목"),
    ("현대차", "005380.KS", "한국", "개별종목"),
    ("SPY", "SPY", "미국", "지수·ETF"),
    ("QQQ", "QQQ", "미국", "지수·ETF"),
    ("버크셔해서웨이B", "BRK-B", "미국", "개별종목"),
    ("엔비디아", "NVDA", "미국", "개별종목"),
    ("테슬라", "TSLA", "미국", "개별종목"),
]

MARKETS = {  # 시간대, 개장, 마감 (미국 서머타임 자동 반영)
    "한국": (ZoneInfo("Asia/Seoul"), dtime(9, 0), dtime(15, 30)),
    "미국": (ZoneInfo("America/New_York"), dtime(9, 30), dtime(16, 0)),
}
CLOSE_BUFFER_MIN = 20  # 마감 후 데이터가 확정될 때까지 기다리는 시간
KST = ZoneInfo("Asia/Seoul")
UP, DOWN = "오른다", "내린다"

SHORT_COLS = (["예측ID", "기록시각(KST)", "기록방식", "종목", "코드", "시장", "유형", "이론", "근거", "예측",
               "봉날짜", "봉상태", "기준가"]
              + [c for k in HORIZONS for c in (f"{k}일뒤_날짜", f"{k}일뒤_종가", f"{k}일뒤_수익률(%)", f"{k}일뒤_결과")])
MID_COLS = ["예측ID", "기록시각(KST)", "기록방식", "종목", "코드", "시장", "유형", "이론", "근거", "예측",
            "봉날짜", "기준가", "판정일", "판정일_종가", "수익률(%)", "결과"]


# ─────────────────────────── 데이터 ───────────────────────────
def fetch(symbol: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.Ticker(symbol).history(start=(START - timedelta(days=200)).isoformat(), auto_adjust=False)
    if df.empty:
        raise RuntimeError("데이터를 받지 못했습니다")
    df.index = pd.to_datetime([d.date() for d in df.index])  # 거래소 현지 날짜
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])


def split_partial(df: pd.DataFrame, market: str, now: datetime):
    """(완성된 봉들, 장중 봉 또는 None)"""
    tz, _, close_t = MARKETS[market]
    local = now.astimezone(tz)
    cutoff = datetime.combine(local.date(), close_t, tz) + timedelta(minutes=CLOSE_BUFFER_MIN)
    if len(df) and df.index[-1].date() == local.date() and local < cutoff:
        return df.iloc[:-1], df.iloc[-1:]
    return df, None


def bar_close_time(d: date, market: str) -> datetime:
    tz, _, close_t = MARKETS[market]
    return datetime.combine(d, close_t, tz) + timedelta(minutes=CLOSE_BUFFER_MIN)


def record_type(d: date, market: str, now: datetime) -> str:
    """봉이 끝난 지 하루 안에 기록했으면 실시간, 아니면 소급(프로그램이 꺼져 있어 늦게 기록)"""
    return "실시간" if now <= bar_close_time(d, market) + timedelta(hours=24) else "소급"


# ─────────────────────────── 지표/이론 ───────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    diff = c.diff()
    gain = diff.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-diff.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss)
    mid, sd = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
    df["BB_상단"], df["BB_하단"] = mid + 2 * sd, mid - 2 * sd
    for n in (5, 20, 60):
        df[f"MA{n}"] = c.rolling(n).mean()
    df["MACD"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["MACD_시그널"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df


def short_theories(r):
    out = []
    if pd.notna(r["RSI"]):
        if r["RSI"] <= 30:
            out.append(("RSI", UP, f"RSI {r['RSI']:.1f} ≤ 30"))
        elif r["RSI"] >= 70:
            out.append(("RSI", DOWN, f"RSI {r['RSI']:.1f} ≥ 70"))
    if pd.notna(r["BB_하단"]):
        if r["Close"] < r["BB_하단"]:
            out.append(("볼린저밴드", UP, f"가격 {r['Close']:.2f} < 하단 {r['BB_하단']:.2f}"))
        elif r["Close"] > r["BB_상단"]:
            out.append(("볼린저밴드", DOWN, f"가격 {r['Close']:.2f} > 상단 {r['BB_상단']:.2f}"))
    return out


def mid_theories(r):
    out = []
    if pd.notna(r["MA60"]):
        if r["MA5"] > r["MA20"] > r["MA60"]:
            out.append(("이동평균선", UP, "정배열 (5>20>60)"))
        elif r["MA5"] < r["MA20"] < r["MA60"]:
            out.append(("이동평균선", DOWN, "역배열 (5<20<60)"))
        # 둘 다 아니면 이론상 예측이 없으므로 기록하지 않음
    if pd.notna(r["MACD"]):
        if r["MACD"] > r["MACD_시그널"]:
            out.append(("MACD", UP, f"MACD {r['MACD']:.2f} > 시그널 {r['MACD_시그널']:.2f}"))
        else:
            out.append(("MACD", DOWN, f"MACD {r['MACD']:.2f} ≤ 시그널 {r['MACD_시그널']:.2f}"))
    return out


def judge(pred: str, base: float, later: float) -> str:
    ret = later / base - 1
    return "참" if (pred == UP and ret > 0) or (pred == DOWN and ret < 0) else "거짓"


# ─────────────────────────── 기록장 ───────────────────────────
def load_log(path: Path, cols) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        return df[cols]
    return pd.DataFrame(columns=cols, dtype=str)


def save_csv(df: pd.DataFrame, path: Path) -> bool:
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return True
    except PermissionError:
        print(f"  ! {path.name} 저장 실패: 엑셀에서 파일을 닫아 주세요. 다음 확인 때 다시 저장합니다.")
        return False


def fmt(x: float) -> str:
    return f"{x:.2f}"


# ─────────────────────────── 한 번의 확인 ───────────────────────────
def run_once(now: datetime | None = None, fetcher=fetch):
    OUT_DIR.mkdir(exist_ok=True)
    now = now or datetime.now(KST)
    stamp = now.astimezone(KST).strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{stamp}] 확인 시작 (단기 판단 방식: {MODE})")

    short = load_log(SHORT_LOG, SHORT_COLS)
    mid = load_log(MID_LOG, MID_COLS)
    short_keys = set(zip(short["봉날짜"], short["코드"], short["이론"]))
    mid_keys = set(zip(mid["봉날짜"], mid["코드"], mid["이론"]))
    new_short, new_mid, n_judged, baselines, errors = [], [], 0, {}, []

    for name, sym, market, kind in TICKERS:
        try:
            raw = fetcher(sym)
        except Exception as e:
            errors.append(name)
            print(f"  ! {name}({sym}) 데이터 실패: {e}")
            continue
        closed, partial = split_partial(raw, market, now)
        ind = add_indicators(closed)
        cdates = list(closed.index.date)
        closes = closed["Close"].to_numpy()
        pos = {d: i for i, d in enumerate(cdates)}
        info = dict(종목=name, 코드=sym, 시장=market, 유형=kind)

        # ── 1) 단기: 조건 충족 봉 찾기 → 예측 기록
        cands = [(d, ind.loc[pd.Timestamp(d)], "완성") for d in cdates if START <= d <= END]
        if MODE == "장중" and partial is not None and START <= partial.index[0].date() <= END:
            live = add_indicators(pd.concat([closed, partial])).iloc[-1]
            cands.append((partial.index[0].date(), live, "장중"))
        for d, r, state in cands:
            for theory, pred, why in short_theories(r):
                key = (d.isoformat(), sym, theory)
                if key in short_keys:
                    continue  # 이미 기록됨 → 절대 덮어쓰지 않음
                short_keys.add(key)
                rec = {c: "" for c in SHORT_COLS}
                rec.update(info, 예측ID=f"S{len(short) + len(new_short) + 1:04d}", **{"기록시각(KST)": stamp},
                           기록방식=record_type(d, market, now), 이론=theory, 근거=why, 예측=pred,
                           봉날짜=d.isoformat(), 봉상태=state, 기준가=fmt(r["Close"]))
                new_short.append(rec)
                print(f"  [새 예측] {name} {theory}: {why} → {pred} (기준가 {fmt(r['Close'])}, {rec['기록방식']})")

        # ── 2) 중기: 매주 첫 거래일 → 예측 기록
        seen_weeks = set()
        for d in cdates:
            if not (START <= d <= MID_LAST):
                continue
            wk = d.isocalendar()[:2]
            if wk in seen_weeks:
                continue
            seen_weeks.add(wk)
            r = ind.loc[pd.Timestamp(d)]
            for theory, pred, why in mid_theories(r):
                key = (d.isoformat(), sym, theory)
                if key in mid_keys:
                    continue
                mid_keys.add(key)
                rec = {c: "" for c in MID_COLS}
                rec.update(info, 예측ID=f"M{len(mid) + len(new_mid) + 1:04d}", **{"기록시각(KST)": stamp},
                           기록방식=record_type(d, market, now), 이론=theory, 근거=why, 예측=pred,
                           봉날짜=d.isoformat(), 기준가=fmt(r["Close"]))
                new_mid.append(rec)
                print(f"  [새 예측·중기] {name} {theory}: {why} → {pred} (기준가 {fmt(r['Close'])})")

        # ── 비교 기준: 연구 기간 중 아무 날에 샀을 때 k일 뒤 오른/내린 비율
        sp = [i for i, d in enumerate(cdates) if START <= d <= END]
        baselines[sym] = {}
        for k in sorted(set(HORIZONS + [MID_DAYS])):
            rets = np.array([closes[i + k] / closes[i] - 1 for i in sp if i + k < len(closes)])
            if len(rets):
                baselines[sym][k] = {UP: (rets > 0).mean(), DOWN: (rets < 0).mean()}

        # ── 3) 결과 판정 (빈 칸만 채움)
        def fill_short(df):
            nonlocal n_judged
            for idx in df.index[df["코드"] == sym]:
                d = date.fromisoformat(df.at[idx, "봉날짜"])
                if d not in pos:
                    continue  # 장중 기록 봉이 아직 완성 전
                base, pred = float(df.at[idx, "기준가"]), df.at[idx, "예측"]
                for k in HORIZONS:
                    j = pos[d] + k
                    if df.at[idx, f"{k}일뒤_결과"] == "" and j < len(closes):
                        df.at[idx, f"{k}일뒤_날짜"] = cdates[j].isoformat()
                        df.at[idx, f"{k}일뒤_종가"] = fmt(closes[j])
                        df.at[idx, f"{k}일뒤_수익률(%)"] = fmt((closes[j] / base - 1) * 100)
                        df.at[idx, f"{k}일뒤_결과"] = judge(pred, base, closes[j])
                        n_judged += 1
                        print(f"  [판정] {name} {df.at[idx, '이론']} {df.at[idx, '봉날짜']} {pred} "
                              f"→ {k}일 뒤 {df.at[idx, f'{k}일뒤_결과']}")

        def fill_mid(df):
            nonlocal n_judged
            for idx in df.index[df["코드"] == sym]:
                d = date.fromisoformat(df.at[idx, "봉날짜"])
                j = pos.get(d, -1) + MID_DAYS
                if d not in pos or df.at[idx, "결과"] != "" or j >= len(closes):
                    continue
                base, pred = float(df.at[idx, "기준가"]), df.at[idx, "예측"]
                df.at[idx, "판정일"] = cdates[j].isoformat()
                df.at[idx, "판정일_종가"] = fmt(closes[j])
                df.at[idx, "수익률(%)"] = fmt((closes[j] / base - 1) * 100)
                df.at[idx, "결과"] = judge(pred, base, closes[j])
                n_judged += 1
                print(f"  [판정·중기] {name} {df.at[idx, '이론']} {df.at[idx, '봉날짜']} {pred} "
                      f"→ 1주 뒤 {df.at[idx, '결과']}")

        if new_short:
            short = pd.concat([short, pd.DataFrame(new_short)], ignore_index=True)
            new_short = []
        if new_mid:
            mid = pd.concat([mid, pd.DataFrame(new_mid)], ignore_index=True)
            new_mid = []
        fill_short(short)
        fill_mid(mid)

    added_s = len(short) - len(load_log(SHORT_LOG, SHORT_COLS))
    added_m = len(mid) - len(load_log(MID_LOG, MID_COLS))
    save_csv(short, SHORT_LOG)
    save_csv(mid, MID_LOG)
    summary = summarize(short, mid, baselines)
    save_csv(summary, SUMMARY)

    fail = ("\t실패: " + ",".join(errors)) if errors else ""
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(f"{stamp}\t새 예측 단기 {added_s}건 중기 {added_m}건\t판정 {n_judged}건{fail}\n")

    print(f"  새 예측 단기 {added_s}건, 중기 {added_m}건 / 새 판정 {n_judged}건")
    if len(summary):
        top = summary[(summary["시장"] == "전체") & (summary["유형"] == "전체")]
        print("\n  ── 전체 적중률 ──")
        print(top[["구분", "이론", "판정", "예측수", "적중률(%)", "비교기준(%)", "차이(%p)"]].to_string(index=False))
    print(f"  저장 위치: {OUT_DIR}")


# ─────────────────────────── 요약 ───────────────────────────
GROUPINGS = [[], ["시장"], ["유형"], ["시장", "유형"]]


def _groups(df, keys):
    if not keys:
        yield "전체", "전체", df
        return
    for vals, g in df.groupby(keys):
        vals = vals if isinstance(vals, tuple) else (vals,)
        d = dict(zip(keys, vals))
        yield d.get("시장", "전체"), d.get("유형", "전체"), g


def summarize(short, mid, baselines) -> pd.DataFrame:
    rows = []
    for k in HORIZONS:
        done = short[short[f"{k}일뒤_결과"].isin(["참", "거짓"])].copy()
        if done.empty:
            continue
        done["_기준"] = [baselines.get(s, {}).get(k, {}).get(p, np.nan) for s, p in zip(done["코드"], done["예측"])]
        for theory, t in done.groupby("이론"):
            for keys in GROUPINGS:
                for m, ty, g in _groups(t, keys):
                    hit = (g[f"{k}일뒤_결과"] == "참").mean() * 100
                    bl = g["_기준"].mean() * 100
                    rows.append(dict(구분="단기", 이론=theory, 시장=m, 유형=ty, 판정=f"{k}일 뒤", 예측수=len(g),
                                     참=int((g[f"{k}일뒤_결과"] == "참").sum()),
                                     **{"적중률(%)": round(hit, 1), "비교기준(%)": round(bl, 1),
                                        "차이(%p)": round(hit - bl, 1)}))
    done = mid[mid["결과"].isin(["참", "거짓"])].copy()
    if len(done):
        done["_기준"] = [baselines.get(s, {}).get(MID_DAYS, {}).get(p, np.nan) for s, p in zip(done["코드"], done["예측"])]
        for theory, t in done.groupby("이론"):
            for keys in GROUPINGS:
                for m, ty, g in _groups(t, keys):
                    hit = (g["결과"] == "참").mean() * 100
                    bl = g["_기준"].mean() * 100
                    rows.append(dict(구분="중기", 이론=theory, 시장=m, 유형=ty, 판정="1주 뒤", 예측수=len(g),
                                     참=int((g["결과"] == "참").sum()),
                                     **{"적중률(%)": round(hit, 1), "비교기준(%)": round(bl, 1),
                                        "차이(%p)": round(hit - bl, 1)}))
    return pd.DataFrame(rows)


# ─────────────────────────── 실행 ───────────────────────────
def main():
    ap = argparse.ArgumentParser(description="트레이드 이론 적중률 실험 (실시간 기록)")
    ap.add_argument("--once", action="store_true", help="한 번만 확인하고 종료")
    args = ap.parse_args()
    print(f"연구 기간: {START} ~ {END}  /  확인 주기: {INTERVAL_MIN}분")
    while True:
        try:
            run_once()
        except Exception as e:
            print(f"  ! 확인 중 오류: {e}")
        if args.once:
            break
        print(f"  다음 확인: {datetime.now(KST) + timedelta(minutes=INTERVAL_MIN):%H:%M} (종료: Ctrl+C)")
        time.sleep(INTERVAL_MIN * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n종료합니다.")

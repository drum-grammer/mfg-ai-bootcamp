"""실습 2 — Pandas로 설비 로그를 '모델에 넣을 표'로 만든다

왜 이 실습이 제일 중요한가
  강의는 모델 이야기에 시간을 쓰지만, 실제 과제에서 시간을 먹는 곳은 여기다.
  센서 테이블과 고장 이력 테이블을 붙여 '라벨이 달린 표' 하나를 만드는 작업 —
  지도학습 데이터셋은 하늘에서 떨어지지 않고 이 단계에서 만들어진다.

무엇을 느끼는가
  (1) DataFrame = 열마다 dtype이 다른 2차원 배열. 시계열은 인덱스를 시간으로 둔다
  (2) 결측은 지우는 게 답이 아니다 — 왜 비었는지에 따라 처리가 갈린다
  (3) resample = 1분 데이터를 1시간으로 줄이는 '집계' (제조 데이터는 대개 과샘플링돼 있다)
  (4) merge = 센서 + 고장 이력 → 라벨. 지도학습의 y가 여기서 생긴다
  (5) rolling = 실습 1의 이동통계를 표 위에서 (2단계 시계열 특징의 실제 작성법)

실행: python3 lab2_pandas_equipment_log.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 120)

rng = np.random.default_rng(7)

# ── (0) 데이터 만들기 — 현장 CSV를 흉내 낸다 ──────────────────────────────
# 설비 3대 × 7일 × 1분 간격. 일부러 지저분하게 만든다:
#   - 통신 끊김으로 값이 비는 구간
#   - 설비마다 기록 시작 시각이 조금씩 다름
#   - 온도 센서가 가끔 -999 (센서 고장 코드)를 뱉음
EQUIP = ["EQ-01", "EQ-02", "EQ-03"]
start = pd.Timestamp("2026-09-01 00:00")
idx = pd.date_range(start, periods=7 * 24 * 60, freq="1min")

rows = []
for k, eq in enumerate(EQUIP):
    n = len(idx)
    hour = idx.hour + idx.minute / 60
    temp = 55 + 4 * k + 2.5 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 0.6, n)
    vib = 0.30 + 0.05 * k + rng.normal(0, 0.03, n)
    cur = 11 + 0.8 * k + rng.normal(0, 0.25, n)
    df = pd.DataFrame({"time": idx, "equip": eq, "temp": temp, "vib": vib, "current": cur})

    # 고장 전조: 각 설비마다 한 번, 6시간에 걸쳐 진동·온도가 서서히 오른다
    fail_at = start + pd.Timedelta(days=2 + 1.5 * k, hours=9)
    ramp = (df["time"] > fail_at - pd.Timedelta(hours=6)) & (df["time"] <= fail_at)
    w = np.linspace(0, 1, int(ramp.sum()))
    df.loc[ramp, "vib"] += 0.18 * w
    df.loc[ramp, "temp"] += 6.0 * w

    # 통신 끊김 2회 (40분, 15분) → 값이 NaN
    for gap_start, mins in [(start + pd.Timedelta(days=1, hours=3 + k), 40),
                            (start + pd.Timedelta(days=4, hours=17), 15)]:
        gap = (df["time"] >= gap_start) & (df["time"] < gap_start + pd.Timedelta(minutes=mins))
        df.loc[gap, ["temp", "vib", "current"]] = np.nan

    # 온도 센서 오류 코드 -999 를 0.2% 확률로
    bad = rng.random(n) < 0.002
    df.loc[bad, "temp"] = -999.0
    rows.append(df)

raw = pd.concat(rows, ignore_index=True)
csv_path = os.path.join(OUT, "equipment_log.csv")
raw.to_csv(csv_path, index=False)

events = pd.DataFrame({
    "equip": EQUIP,
    "failed_at": [start + pd.Timedelta(days=2 + 1.5 * k, hours=9) for k in range(3)],
    "code": ["BRG-01", "BRG-01", "MTR-07"],
})
ev_path = os.path.join(OUT, "failure_events.csv")
events.to_csv(ev_path, index=False)
print(f"== (0) 예제 CSV 생성 ==\n  {os.path.relpath(csv_path, HERE)}  ({len(raw):,}행)")
print(f"  {os.path.relpath(ev_path, HERE)}  ({len(events)}행)")

# ── (1) 읽고, 제일 먼저 훑어본다 ──────────────────────────────────────────
df = pd.read_csv(csv_path, parse_dates=["time"])
print("\n== (1) 읽자마자 하는 3가지 ==")
print(f"  shape   : {df.shape}")
print("  dtypes  :")
for name, dt in df.dtypes.items():
    print(f"    {name:10s} {dt}")
print("  head(3) :")
print(df.head(3).to_string(index=False))
print("\n  describe() — 이상한 최솟값이 보이는가:")
print(df[["temp", "vib", "current"]].describe().round(2).to_string())
print("  → temp 의 min 이 -999. 센서 고장 코드가 숫자로 섞여 들어왔다. 평균을 그대로 믿으면 안 된다")

# ── (2) 결측 — 두 종류를 구분한다 ─────────────────────────────────────────
df["temp"] = df["temp"].replace(-999.0, np.nan)   # 오류 코드를 '없음'으로 바꾼다
print("\n== (2) 결측 ==")
print(df[["temp", "vib", "current"]].isna().sum().to_string())
print("  두 종류가 섞여 있다:")
print("   - 통신 끊김(수십 분 연속) → 값을 지어내면 위험. 구간째로 빼거나 '결측이었음' 플래그를 남긴다")
print("   - 센서 오류 코드(한두 점) → 앞뒤로 메우는 것이 합리적")

df = df.sort_values(["equip", "time"])
# 설비별로 따로 처리해야 한다. 설비가 섞인 채 ffill 하면 EQ-01 값이 EQ-02로 넘어간다.
g = df.groupby("equip", sort=False)
short = df[["temp", "vib", "current"]].copy()
filled = g[["temp", "vib", "current"]].transform(lambda s: s.interpolate(limit=5).ffill(limit=5))
df[["temp", "vib", "current"]] = filled
df["was_missing"] = short.isna().any(axis=1).astype(int)
print(f"\n  limit=5 (5분까지만) 메운 뒤 남은 결측: {int(df[['temp','vib','current']].isna().sum().sum())}개")
print("  → 긴 끊김은 일부러 남겼다. 이 행을 모델에 쓸지 말지는 '정답'이 아니라 선택이고, 근거를 적어 둬야 한다")

# ── (3) resample — 1분을 1시간으로 ────────────────────────────────────────
hourly = (df.set_index("time")
            .groupby("equip")
            .resample("1h")
            .agg(temp_mean=("temp", "mean"),
                 temp_max=("temp", "max"),
                 vib_mean=("vib", "mean"),
                 vib_std=("vib", "std"),
                 current_mean=("current", "mean"),
                 n=("temp", "size"))
            .reset_index())
print("\n== (3) 1분 → 1시간 집계 ==")
print(f"  {len(df):,}행 → {len(hourly):,}행")
num = hourly.select_dtypes("number").columns
print(hourly.head(3).round({c: 3 for c in num}).to_string(index=False))
print("  집계 함수 선택이 곧 특징 설계다: 평균은 추세, 최댓값은 피크, 표준편차는 흔들림을 남긴다")

# ── (4) merge — 고장 이력을 붙여 라벨을 만든다 ────────────────────────────
ev = pd.read_csv(ev_path, parse_dates=["failed_at"])
m = hourly.merge(ev, on="equip", how="left")          # 설비 단위로 붙인다
m["hours_to_failure"] = (m["failed_at"] - m["time"]).dt.total_seconds() / 3600
# 예지보전 라벨: "앞으로 6시간 안에 고장이 나는가"
m["label"] = ((m["hours_to_failure"] > 0) & (m["hours_to_failure"] <= 6)).astype(int)

print("\n== (4) merge 로 라벨 만들기 ==")
print(f"  라벨 분포: {m['label'].value_counts().to_dict()}   (불량/고장 라벨은 늘 이렇게 희귀하다)")
print(f"  양성 비율: {m['label'].mean() * 100:.2f}%  → 실습 4에서 다루는 '불균형 데이터' 그 자체")
peek = m.loc[m["label"] == 1, ["equip", "time", "temp_mean", "vib_mean", "hours_to_failure"]].head(4)
print(peek.round({"temp_mean": 3, "vib_mean": 3, "hours_to_failure": 1}).to_string(index=False))
print("  주의: 고장 '이후' 구간까지 학습에 넣으면 정답을 훔쳐보는 셈이 된다 (누수, leakage)")

# ── (5) rolling — 표 위에서 만드는 시계열 특징 ────────────────────────────
m = m.sort_values(["equip", "time"])
for col in ["temp_mean", "vib_mean"]:
    grp = m.groupby("equip")[col]
    m[f"{col}_roll6"] = grp.transform(lambda s: s.rolling(6, min_periods=2).mean())
    m[f"{col}_slope6"] = grp.transform(lambda s: s.diff().rolling(6, min_periods=2).mean())

print("\n== (5) rolling 특징 ==")
print("  6시간 이동평균과 '시간당 변화율의 6시간 평균'(기울기). 전조는 절댓값보다 기울기에서 먼저 보인다")
cmp = (m.groupby("label")[["vib_mean", "vib_mean_slope6", "temp_mean_slope6"]]
         .mean().round(4))
print(cmp.to_string())
print("  label=1 구간에서 기울기가 눈에 띄게 크면, 이 특징은 쓸모가 있다는 뜻이다")

# ── (6) 모델에 넣을 최종 표 ───────────────────────────────────────────────
feat_cols = ["temp_mean", "temp_max", "vib_mean", "vib_std", "current_mean",
             "temp_mean_roll6", "vib_mean_roll6", "temp_mean_slope6", "vib_mean_slope6"]
dataset = m.dropna(subset=feat_cols + ["label"])[["equip", "time"] + feat_cols + ["label"]]
ds_path = os.path.join(OUT, "dataset_hourly.csv")
dataset.to_csv(ds_path, index=False)
print("\n== (6) 완성된 표 ==")
print(f"  {dataset.shape[0]}행 × 특징 {len(feat_cols)}개 + 라벨 1개 → {os.path.relpath(ds_path, HERE)}")
print("  여기까지가 'X와 y를 만든다'의 실제 내용이다. 모델 학습은 실습 3~5에서 이 표 모양을 그대로 받는다")

print("\n== 정리 ==")
print("  read_csv → describe 로 이상값 찾기 → 결측 종류 나누기 → resample → merge 로 라벨 → rolling 특징.")
print("  이 6단계가 2단계 제조 응용 과제에서 매번 반복된다. 순서를 외워 두면 강의를 따라가기 쉽다.")

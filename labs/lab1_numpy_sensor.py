"""실습 1 — NumPy를 센서 신호로 익힌다 (진동·온도·전류 3채널)

왜 이 순서인가
  가스미터·초음파 열량계 펌웨어에서 하던 일(링버퍼에 샘플을 쌓고, 영점을 빼고,
  이동평균으로 노이즈를 깎고, 임계값을 넘는 구간을 찾는다)이 NumPy에서는
  전부 '한 줄'이 된다. C의 for 루프를 배열 연산으로 바꿔 보는 것이 1단계 내내
  가장 많이 쓰게 될 사고방식이다.

무엇을 느끼는가
  (1) 배열의 shape 은 곧 (시간, 채널) 이다 — 모든 실습에서 shape 부터 본다
  (2) 벡터화는 편해서가 아니라 빨라서 쓴다 (직접 재 본다)
  (3) 브로드캐스팅 = 채널별 보정값을 한 번에 빼기
  (4) 슬라이딩 윈도우 = 이동평균·이동표준편차 = 2단계 시계열 특징의 원형
  (5) 불리언 마스킹 = 이상 구간 탐지의 가장 단순한 형태

실행: python3 lab1_numpy_sensor.py
"""
import os
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(42)

# ── (1) 배열 = (시간, 채널) ────────────────────────────────────────────────
# 1초에 10번 샘플링하는 설비 센서 3채널을 10분치(6000샘플) 만든다.
FS = 10                      # 샘플링 주파수 [Hz]
N = 60 * 10 * FS             # 10분
t = np.arange(N) / FS        # 초 단위 시간축
CH = ["진동", "온도", "전류"]

vib = 0.4 * np.sin(2 * np.pi * 1.7 * t) + rng.normal(0, 0.15, N)   # 회전체 기본 진동
tmp = 60 + 3 * np.sin(2 * np.pi * t / 300) + rng.normal(0, 0.3, N)  # 5분 주기 온도 변동
cur = 12 + rng.normal(0, 0.4, N)                                    # 전류

# 7분(=420초) 지점부터 30초간 베어링 이상: 진동이 커지고 전류가 따라 오른다
fault = (t >= 420) & (t < 450)
vib[fault] += np.linspace(0, 1.2, fault.sum()) + rng.normal(0, 0.2, fault.sum())
cur[fault] += 1.5

X = np.stack([vib, tmp, cur], axis=1)   # (시간, 채널)

print("== (1) shape 부터 본다 ==")
print(f"  X.shape = {X.shape}  ← (샘플 {X.shape[0]}개, 채널 {X.shape[1]}개)")
print(f"  X.dtype = {X.dtype}   메모리 = {X.nbytes / 1024:.0f} KB")
print(f"  X[0]    = {np.round(X[0], 3)}   ← 한 시점의 3채널 값 (행 = 시점)")
print(f"  X[:, 0] = 진동 채널 전체 {X[:, 0].shape}   ← 열 = 채널")

# ── (2) 벡터화는 빨라서 쓴다 ──────────────────────────────────────────────
def rms_loop(sig):
    """C로 짜던 방식 그대로 — 파이썬 for 루프"""
    total = 0.0
    for v in sig:
        total += v * v
    return (total / len(sig)) ** 0.5


def rms_vec(sig):
    """같은 계산, NumPy 한 줄"""
    return np.sqrt(np.mean(sig ** 2))


sig = X[:, 0]
t0 = time.perf_counter(); a = rms_loop(sig); t1 = time.perf_counter()
b = rms_vec(sig);          t2 = time.perf_counter()
print("\n== (2) 같은 RMS, 두 가지 방법 ==")
print(f"  for 루프 : {a:.6f}   {(t1 - t0) * 1e3:8.2f} ms")
print(f"  NumPy    : {b:.6f}   {(t2 - t1) * 1e3:8.2f} ms")
print(f"  → {(t1 - t0) / max(t2 - t1, 1e-9):.0f}배. 데이터가 100배 커지면 이 차이가 그대로 대기 시간이 된다")

# ── (3) 브로드캐스팅 = 채널별 보정 ────────────────────────────────────────
# 센서마다 영점(offset)과 스케일이 다르다. 모델에 넣기 전에 반드시 맞춘다.
mu = X.mean(axis=0)    # (3,) 채널별 평균
sd = X.std(axis=0)     # (3,) 채널별 표준편차
Z = (X - mu) / sd      # (6000,3) - (3,) → NumPy가 (3,)를 모든 행에 자동 정렬

print("\n== (3) 표준화 (z-score) ==")
print(f"  채널별 평균  {np.round(mu, 2)}   표준편차 {np.round(sd, 2)}")
print(f"  (X - mu).shape = {(X - mu).shape}  ← (6000,3) 와 (3,) 가 계산된다. 이것이 브로드캐스팅")
print(f"  표준화 후 평균 {np.round(Z.mean(axis=0), 6)}  표준편차 {np.round(Z.std(axis=0), 3)}")
print("  왜 필요한가: 온도 60 과 진동 0.4 를 그대로 넣으면 모델이 온도만 본다 (실습 3에서 실측한다)")

# ── (4) 슬라이딩 윈도우 = 이동평균·이동표준편차 ───────────────────────────
W = FS * 5                                     # 5초 창
win = sliding_window_view(X[:, 0], W)          # (N-W+1, W) — 복사 없이 '보기'만 바꾼다
ma = win.mean(axis=1)
ms = win.std(axis=1)
t_w = t[W - 1:]

print("\n== (4) 5초 이동창 ==")
print(f"  sliding_window_view → {win.shape}   원본 메모리 재사용 여부: {win.base is not None}")
print(f"  이동평균 {ma.shape}, 이동표준편차 {ms.shape}")
print("  이동평균은 노이즈를 깎고, 이동표준편차는 '흔들림이 커졌는가'를 잡는다.")
print("  → 2단계 시계열 실습에서 만드는 특징(feature)이 대부분 이 두 개의 변형이다")

# ── (5) 불리언 마스킹 = 가장 단순한 이상탐지 ─────────────────────────────
# 정상 구간(처음 5분)의 통계로 임계값을 잡고, 그 뒤를 판정한다.
base = ms[t_w < 300]
thr = base.mean() + 3 * base.std()             # 3-시그마 규칙
alarm = ms > thr

print("\n== (5) 3-시그마 임계값으로 이상 구간 찾기 ==")
print(f"  정상 구간 이동표준편차: 평균 {base.mean():.4f}, 표준편차 {base.std():.4f}")
print(f"  임계값 = 평균 + 3*표준편차 = {thr:.4f}")
print(f"  경보 샘플 수: {alarm.sum()} / {alarm.size}")

# 연속한 경보를 하나의 '구간'으로 묶는다 (True 가 끊기는 지점에서 자른다)
edges = np.diff(alarm.astype(int))
starts = np.flatnonzero(edges == 1) + 1
ends = np.flatnonzero(edges == -1) + 1
if alarm[0]:
    starts = np.r_[0, starts]
if alarm[-1]:
    ends = np.r_[ends, alarm.size]
print(f"  경보 구간 {len(starts)}개 (앞 5개만 표시):")
for s_i, e_i in list(zip(starts, ends))[:5]:
    dur = t_w[e_i - 1] - t_w[s_i]
    hit = "  ← 고장 구간 안" if t_w[s_i] < 450 and t_w[e_i - 1] > 420 else "  ← 오경보"
    print(f"    {t_w[s_i]:7.1f} ~ {t_w[e_i - 1]:7.1f} 초 ({dur:5.1f}s){hit}")
inside = sum(1 for a, b in zip(starts, ends) if t_w[a] < 450 and t_w[b - 1] > 420)
print(f"  요약: 고장 구간 안 {inside}개 / 바깥(오경보) {len(starts) - inside}개")
print("  읽는 법 두 가지:")
print("   - 검출이 늦다: 5초 창의 통계라 고장이 창을 채울 때까지 기다린다. 창 길이 = 지연 vs 안정성 trade-off")
print("   - 오경보가 섞인다: 3-시그마는 '정상도 가끔 넘는' 임계값이다. 짧은 구간을 버리는 규칙(예: 3초 이상 지속)이 현장에서 늘 따라붙는다")

# ── (6) axis 를 손에 익힌다 ───────────────────────────────────────────────
print("\n== (6) axis 는 '없어지는 축' 이다 ==")
print(f"  X.shape            = {X.shape}")
print(f"  X.mean(axis=0).shape = {X.mean(axis=0).shape}  ← 시간축이 사라짐 = 채널별 평균")
print(f"  X.mean(axis=1).shape = {X.mean(axis=1).shape}  ← 채널축이 사라짐 = 시점별 평균(의미 없음)")
print("  실무 사고: 지우고 싶은 축 번호를 axis 에 넣는다")

# ── (7) 그림으로 확인 ─────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(t, X[:, 0], lw=0.5, color="0.6", label="raw")
    ax[0].plot(t_w, ma, lw=1.2, color="C0", label="5s moving average")
    ax[0].set_ylabel("vibration"); ax[0].legend(loc="upper left", fontsize=8)
    ax[1].plot(t_w, ms, lw=1.0, color="C1", label="5s moving std")
    ax[1].axhline(thr, color="r", ls="--", lw=1, label="3-sigma threshold")
    ax[1].set_ylabel("moving std"); ax[1].legend(loc="upper left", fontsize=8)
    ax[2].fill_between(t_w, 0, alarm.astype(float), step="mid", color="r", alpha=0.5)
    ax[2].axvspan(420, 450, color="k", alpha=0.15)
    ax[2].set_ylabel("alarm"); ax[2].set_xlabel("time [s]"); ax[2].set_ylim(-0.1, 1.1)
    fig.suptitle("lab1 - moving statistics detect the injected bearing fault (420-450s, gray band)")
    fig.tight_layout()
    path = os.path.join(OUT, "lab1_sensor.png")
    fig.savefig(path, dpi=110)
    print(f"\n  그림 저장: {os.path.relpath(path, HERE)}")
except ImportError:
    print("\n  (matplotlib 없음 — 그림은 건너뛴다)")

print("\n== 정리 ==")
print("  shape·브로드캐스팅·axis·마스킹 네 가지가 NumPy의 전부라고 봐도 된다.")
print("  강의에서 코드가 빨리 넘어가도 이 네 가지로 되짚으면 따라갈 수 있다.")

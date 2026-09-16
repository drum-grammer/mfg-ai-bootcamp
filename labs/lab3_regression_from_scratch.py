"""실습 3 — 회귀를 손으로 만든다 (사출성형 공정 → 제품 치수)

문제 설정
  사출압력·금형온도·냉각시간·원료 수분율로 완성품의 치수 편차(mm)를 예측한다.
  전형적인 제조 회귀 문제이고, 강의 1단계의 '회귀' 파트가 이 형태로 나온다.

무엇을 느끼는가
  (1) 학습 = 손실을 줄이는 것. 손실(MSE)이 뭔지 숫자로 본다
  (2) 정답이 수식으로 한 번에 나오는 경우(정규방정식)가 있다 — 그런데 왜 경사하강법을 쓰는가
  (3) 학습률(learning rate)은 '적당히'가 없다 — 작으면 안 가고 크면 터진다. 직접 터뜨려 본다
  (4) 특징 스케일링을 빼먹으면 같은 학습률에서 수렴이 몇 배 느려진다 — 실측한다
  (5) train/test 를 나누지 않으면 아무것도 모른 채 좋아 보인다 — 다항 차수를 올려 과적합을 만든다

실행: python3 lab3_regression_from_scratch.py
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(1)

# ── (1) 데이터 — 단위가 제각각인 것이 핵심 ───────────────────────────────
N = 600
pressure = rng.normal(850, 60, N)      # 사출압력 [bar]      ~850
mold_temp = rng.normal(45, 4, N)       # 금형온도 [℃]        ~45
cool_time = rng.normal(12, 2.0, N)     # 냉각시간 [s]        ~12
moisture = rng.normal(0.04, 0.01, N)   # 원료 수분율 [%]     ~0.04  ← 크기가 4자리 차이

X = np.column_stack([pressure, mold_temp, cool_time, moisture])
FEATS = ["사출압력", "금형온도", "냉각시간", "수분율"]

# 실제 물리 관계(우리가 맞혀야 할 정답)
TRUE_W = np.array([0.0020, -0.0150, -0.0400, 3.0000])
TRUE_B = 0.35
y = X @ TRUE_W + TRUE_B + rng.normal(0, 0.02, N)   # 치수 편차 [mm]

print("== (1) 데이터 ==")
print(f"  X.shape = {X.shape}, y.shape = {y.shape}")
for name, col, w in zip(FEATS, X.T, TRUE_W):
    print(f"  {name:6s} 평균 {col.mean():8.3f}  범위 {col.min():8.3f}~{col.max():8.3f}   정답 계수 {w:+.4f}")
print("  주목: 사출압력은 850 근처, 수분율은 0.04 근처. 크기 차이가 2만 배다")

# ── (2) 손실(MSE)과 정규방정식 ────────────────────────────────────────────
def mse(w, b, X, y):
    return float(np.mean((X @ w + b - y) ** 2))


Xb = np.column_stack([X, np.ones(N)])                 # 절편을 열 하나로 붙인다
theta = np.linalg.lstsq(Xb, y, rcond=None)[0]         # 정규방정식의 수치적으로 안전한 버전
w_exact, b_exact = theta[:-1], theta[-1]
print("\n== (2) 정규방정식 — 한 번에 푸는 답 ==")
print(f"  계수 {np.round(w_exact, 4)}  절편 {b_exact:.4f}")
print(f"  정답 {np.round(TRUE_W, 4)}  절편 {TRUE_B:.4f}")
print(f"  MSE = {mse(w_exact, b_exact, X, y):.6f}   (잡음 분산 {0.02**2:.6f} 근처면 더 줄일 수 없다는 뜻)")
print("  → 그러면 왜 경사하강법을 배우는가: 이 공식은 선형 모델에만 있다.")
print("    신경망에는 이런 닫힌 해가 없어서 (3)의 방법밖에 쓸 수 없다")

# ── (3) 경사하강법을 직접 쓴다 ────────────────────────────────────────────
def grad_descent(X, y, lr, steps=400):
    """MSE의 기울기를 따라 w, b를 조금씩 옮긴다. 손실 기록을 함께 돌려준다."""
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    hist = []
    # 학습률이 크면 값이 폭주해 오버플로가 난다 — 그 자체가 이 실습의 관찰 대상이라
    # 경고를 끄고 결과(inf/nan)만 본다
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(steps):
            err = X @ w + b - y                  # (n,)
            gw = 2.0 / n * (X.T @ err)           # dL/dw
            gb = 2.0 / n * err.sum()             # dL/db
            w -= lr * gw
            b -= lr * gb
            loss = float(np.mean(err ** 2))
            hist.append(loss if np.isfinite(loss) else np.nan)
    return w, b, np.array(hist)


print("\n== (3) 원본 스케일 그대로, 학습률만 바꿔 본다 ==")
for lr in [1e-9, 1e-7, 1e-5]:
    w, b, h = grad_descent(X, y, lr)
    end = h[-1]
    verdict = "발산" if not np.isfinite(end) or end > h[0] else ("느림" if end > 0.01 else "수렴")
    shown = f"{end:.6f}" if np.isfinite(end) else "inf/nan"
    print(f"  lr={lr:<8.0e} 400스텝 후 MSE = {shown:>12s}  → {verdict}")
print("  스케일이 제각각이면 '되는 학습률'의 창이 바늘구멍처럼 좁아진다")

# ── (4) 스케일링 하나로 달라진다 ──────────────────────────────────────────
mu, sd = X.mean(axis=0), X.std(axis=0)
Xs = (X - mu) / sd
print("\n== (4) 표준화한 뒤 같은 실험 ==")
for lr in [1e-3, 1e-2, 1e-1]:
    w, b, h = grad_descent(Xs, y, lr)
    print(f"  lr={lr:<8.0e} 400스텝 후 MSE = {h[-1]:.6f}")
w_s, b_s, hist_s = grad_descent(Xs, y, 0.1, steps=400)
print(f"  → lr=0.1 로 400스텝이면 정규방정식(MSE {mse(w_exact, b_exact, X, y):.6f})에 사실상 도달한다")

# 표준화 공간의 계수를 원래 단위로 되돌린다 — 해석할 때 반드시 필요한 환산
w_back = w_s / sd
b_back = b_s - float((mu / sd) @ w_s)
print(f"  되돌린 계수 {np.round(w_back, 4)}  절편 {b_back:.4f}   (정답 {np.round(TRUE_W, 4)} / {TRUE_B})")
print("  스케일링은 '모델을 위한 전처리'이고, 해석은 원래 단위로 되돌려 놓고 한다")

# ── (5) scikit-learn 과 대조 ──────────────────────────────────────────────
try:
    from sklearn.linear_model import LinearRegression
    sk = LinearRegression().fit(X, y)
    print("\n== (5) sklearn 과 대조 ==")
    print(f"  sklearn  계수 {np.round(sk.coef_, 4)}  절편 {sk.intercept_:.4f}")
    print(f"  직접구현 계수 {np.round(w_back, 4)}  절편 {b_back:.4f}")
    print(f"  최대 차이 {np.abs(sk.coef_ - w_back).max():.2e}  → 같은 답. 라이브러리가 하는 일을 이제 안다")
except ImportError:
    print("\n== (5) sklearn 없음 — 건너뜀 ==")

# ── (6) train/test 와 과적합 ──────────────────────────────────────────────
perm = rng.permutation(N)
tr, te = perm[:int(N * 0.7)], perm[int(N * 0.7):]


def fit_poly(deg):
    """원래 특징에 거듭제곱 항을 덧붙여 모델을 복잡하게 만든다."""
    cols = [X]
    for p in range(2, deg + 1):
        cols.append(((X - mu) / sd) ** p)
    Xp = np.column_stack(cols)
    Xp = np.column_stack([Xp, np.ones(N)])
    th = np.linalg.lstsq(Xp[tr], y[tr], rcond=None)[0]
    pred = Xp @ th
    rmse = lambda i: float(np.sqrt(np.mean((pred[i] - y[i]) ** 2)))
    return rmse(tr), rmse(te), Xp.shape[1] - 1


print("\n== (6) 모델을 복잡하게 만들수록 어떻게 되는가 ==")
print("  차수  파라미터   train RMSE   test RMSE")
rows = []
for deg in [1, 2, 3, 5, 8, 11]:
    a, b_, k = fit_poly(deg)
    rows.append((deg, k, a, b_))
    print(f"  {deg:>3d}   {k:>7d}     {a:.5f}      {b_:.5f}")
best = min(rows, key=lambda r: r[3])
print(f"  → test 기준 최적은 {best[0]}차. train RMSE는 계속 줄지만 test는 어느 지점에서 돌아선다")
print("  이 갈라짐이 과적합이다. train 점수만 보고 좋아하는 실수가 1단계 퀴즈의 단골이다")

# ── (7) 그림 ──────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].plot(hist_s)
    ax[0].set_yscale("log"); ax[0].set_xlabel("step"); ax[0].set_ylabel("MSE")
    ax[0].set_title("(4) loss curve, scaled, lr=0.1")

    pred = X @ w_back + b_back
    ax[1].scatter(y, pred, s=6, alpha=0.5)
    lo, hi = y.min(), y.max()
    ax[1].plot([lo, hi], [lo, hi], "r--", lw=1)
    ax[1].set_xlabel("actual [mm]"); ax[1].set_ylabel("predicted [mm]")
    ax[1].set_title("(5) predicted vs actual")

    d = [r[0] for r in rows]
    ax[2].plot(d, [r[2] for r in rows], "o-", label="train")
    ax[2].plot(d, [r[3] for r in rows], "s-", label="test")
    ax[2].set_xlabel("polynomial degree"); ax[2].set_ylabel("RMSE"); ax[2].legend()
    ax[2].set_title("(6) overfitting")
    fig.tight_layout()
    path = os.path.join(OUT, "lab3_regression.png")
    fig.savefig(path, dpi=110)
    print(f"\n  그림 저장: {os.path.relpath(path, HERE)}")
except ImportError:
    print("\n  (matplotlib 없음 — 그림은 건너뛴다)")

print("\n== 정리 ==")
print("  손실 → 기울기 → 파라미터 갱신. 신경망도 정확히 이 세 줄이고, 달라지는 건 모델의 모양뿐이다.")
print("  전처리(스케일링)와 평가(train/test 분리)는 모델 선택보다 먼저 정해야 하는 것들이다.")

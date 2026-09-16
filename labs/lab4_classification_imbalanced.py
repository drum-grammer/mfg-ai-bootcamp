"""실습 4 — 불량 판정은 '정확도'로 평가하지 않는다 (불균형 분류)

문제 설정
  공정 파라미터 5개로 그 로트가 불량인지 판정한다. 불량률은 2%.
  품질관리·이상탐지·예지보전은 전부 이 모양이다 — 찾아야 할 것이 드물다.

무엇을 느끼는가
  (1) 로지스틱 회귀 = 선형회귀 + 시그모이드 + 로그손실. 직접 만들어 본다
  (2) 정확도 98%짜리 바보 모델을 만들 수 있다 — 그래서 정확도를 쓰지 않는다
  (3) 혼동행렬의 네 칸에 현장 용어(미검·과검)를 붙이면 지표가 이해된다
  (4) 임계값 0.5는 관습일 뿐이다. 미검 비용과 과검 비용을 넣고 직접 고른다
  (5) 불균형에서는 ROC-AUC가 후하게 나온다 — PR 곡선을 같이 본다

실행: python3 lab4_classification_imbalanced.py
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(3)

# ── (1) 데이터 — 불량률 2% ────────────────────────────────────────────────
N = 6000
FEATS = ["사출압력", "금형온도", "냉각시간", "수분율", "사이클편차"]
X = np.column_stack([
    rng.normal(850, 60, N),
    rng.normal(45, 4, N),
    rng.normal(12, 2.0, N),
    rng.normal(0.04, 0.01, N),
    rng.normal(0.0, 1.0, N),
])
mu, sd = X.mean(axis=0), X.std(axis=0)
Xs = (X - mu) / sd                       # 실습 3에서 배운 대로 먼저 표준화

# 불량 위험도: 수분율과 사이클편차가 높고 금형온도가 낮을 때 올라간다
logit = -5.6 + 1.5 * Xs[:, 3] + 1.2 * Xs[:, 4] - 0.9 * Xs[:, 1] + 0.4 * Xs[:, 2]
p = 1 / (1 + np.exp(-logit))
y = (rng.random(N) < p).astype(int)

perm = rng.permutation(N)
tr, te = perm[:4200], perm[4200:]
print("== (1) 데이터 ==")
print(f"  전체 {N}건 중 불량 {y.sum()}건 ({y.mean() * 100:.2f}%)")
print(f"  학습 {len(tr)}건(불량 {y[tr].sum()}) / 평가 {len(te)}건(불량 {y[te].sum()})")

# ── (2) 로지스틱 회귀를 직접 만든다 ───────────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(X, y, lr=0.3, steps=3000, weight_pos=1.0):
    """이진 교차엔트로피를 경사하강법으로 최소화한다.
    weight_pos > 1 이면 불량(양성) 한 건을 여러 건처럼 세어 준다 = class_weight."""
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    sw = np.where(y == 1, weight_pos, 1.0)
    sw = sw / sw.mean()
    for _ in range(steps):
        pred = sigmoid(X @ w + b)
        err = (pred - y) * sw
        w -= lr * (X.T @ err) / n
        b -= lr * err.sum() / n
    return w, b


w, b = fit_logistic(Xs[tr], y[tr])
score = sigmoid(Xs[te] @ w + b)             # 평가셋의 불량 확률
print("\n== (2) 직접 구현한 로지스틱 회귀 ==")
for name, coef in zip(FEATS, w):
    print(f"  {name:6s} 계수 {coef:+.3f}")
print("  (표준화한 값에 대한 계수라 크기 비교가 그대로 '영향력' 비교가 된다)")

try:
    from sklearn.linear_model import LogisticRegression
    sk = LogisticRegression(max_iter=2000).fit(Xs[tr], y[tr])
    print(f"  sklearn 계수 {np.round(sk.coef_.ravel(), 3)}")
    print(f"  직접구현 계수 {np.round(w, 3)}   → 방향과 크기가 맞으면 구현이 옳다")
except ImportError:
    sk = None
    print("  (sklearn 없음 — 대조 생략)")

# ── (3) 정확도 98%짜리 바보 모델 ──────────────────────────────────────────
dumb = np.zeros(len(te), dtype=int)          # 무조건 '양품'
acc_dumb = (dumb == y[te]).mean()
pred05 = (score >= 0.5).astype(int)
acc_model = (pred05 == y[te]).mean()
print("\n== (3) 정확도의 함정 ==")
print(f"  전부 '양품'이라고 찍는 모델 : 정확도 {acc_dumb * 100:.2f}%  불량 검출 0건")
print(f"  학습한 모델 (임계값 0.5)     : 정확도 {acc_model * 100:.2f}%  불량 검출 {int(((pred05 == 1) & (y[te] == 1)).sum())}건")
print("  → 정확도로는 두 모델을 구분할 수 없다. 이 수치를 보고서에 쓰면 안 된다")

# ── (4) 혼동행렬 = 현장 용어 ──────────────────────────────────────────────
def confusion(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return tp, fp, fn, tn


def prf(y_true, y_pred):
    tp, fp, fn, tn = confusion(y_true, y_pred)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


tp, fp, fn, tn = confusion(y[te], pred05)
prec, rec, f1 = prf(y[te], pred05)
print("\n== (4) 혼동행렬 (임계값 0.5) ==")
print("                예측 불량   예측 양품")
print(f"  실제 불량 |  TP {tp:5d}   FN {fn:5d}   ← FN = 미검(불량을 흘려보냄)")
print(f"  실제 양품 |  FP {fp:5d}   TN {tn:5d}   ← FP = 과검(멀쩡한 걸 잡아 세움)")
print(f"  정밀도(precision) {prec:.3f} = 잡은 것 중 진짜 불량 비율 → 과검이 적을수록 높다")
print(f"  재현율(recall)    {rec:.3f} = 진짜 불량 중 잡아낸 비율   → 미검이 적을수록 높다")
print(f"  F1                {f1:.3f} = 둘의 조화평균")

# ── (5) 임계값은 비용으로 고른다 ──────────────────────────────────────────
# 가정: 미검 1건이 고객 클레임으로 이어져 50만원, 과검 1건은 재검사 인건비 2만원
COST_FN, COST_FP = 500_000, 20_000
ths = np.linspace(0.01, 0.9, 90)
costs, recs, precs = [], [], []
for t in ths:
    pr = (score >= t).astype(int)
    _tp, _fp, _fn, _tn = confusion(y[te], pr)
    costs.append(_fn * COST_FN + _fp * COST_FP)
    p_, r_, _ = prf(y[te], pr)
    precs.append(p_); recs.append(r_)
costs = np.array(costs)
best_i = int(costs.argmin())
print("\n== (5) 미검 50만원 · 과검 2만원 이라면 임계값은? ==")
print(f"  임계값 0.50 : 비용 {costs[np.argmin(np.abs(ths - 0.5))]:>12,.0f}원")
print(f"  최적 {ths[best_i]:.2f} : 비용 {costs[best_i]:>12,.0f}원  (재현율 {recs[best_i]:.3f}, 정밀도 {precs[best_i]:.3f})")
print("  → 미검이 비쌀수록 임계값을 낮춰 많이 잡는다. '0.5'는 어디에도 근거가 없다.")
print("    현업과 합의할 것은 모델이 아니라 이 비용표다")

# ── (6) class_weight 로 불균형에 대응 ─────────────────────────────────────
print("\n== (6) 불량 한 건을 몇 건처럼 셀 것인가 (class_weight) ==")
print("  가중치   정밀도   재현율     F1")
for wp in [1, 5, 20, 50]:
    w2, b2 = fit_logistic(Xs[tr], y[tr], weight_pos=wp)
    s2 = sigmoid(Xs[te] @ w2 + b2)
    p_, r_, f_ = prf(y[te], (s2 >= 0.5).astype(int))
    print(f"  {wp:>5d}    {p_:.3f}    {r_:.3f}   {f_:.3f}")
print("  → 가중치를 올리면 재현율이 오르고 정밀도가 내려간다. 임계값 조정과 결국 같은 손잡이다")

# ── (7) ROC-AUC 대신 PR 을 본다 ───────────────────────────────────────────
try:
    from sklearn.metrics import average_precision_score, roc_auc_score
    print("\n== (7) 요약 지표 ==")
    print(f"  ROC-AUC            {roc_auc_score(y[te], score):.3f}   ← 불균형에서도 후하게 나온다")
    print(f"  PR-AUC (평균정밀도) {average_precision_score(y[te], score):.3f}   ← 기준선이 불량률 {y[te].mean():.3f}")
    base = y[te].mean()
    ap = average_precision_score(y[te], score)
    print(f"  PR-AUC {ap:.3f} 는 무작위 기준선({base:.3f})의 {ap / base:.0f}배다. ROC만 보고 판단하지 말 것")
except ImportError:
    print("\n== (7) sklearn 없음 — 지표 생략 ==")

# ── (8) 그림 ──────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].plot(recs, precs, "-")
    ax[0].axhline(y[te].mean(), color="r", ls="--", lw=1, label="random baseline")
    ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision"); ax[0].legend()
    ax[0].set_title("(7) precision-recall")

    ax[1].plot(ths, costs / 1e4)
    ax[1].axvline(ths[best_i], color="r", ls="--", lw=1)
    ax[1].set_xlabel("threshold"); ax[1].set_ylabel("expected cost [10k KRW]")
    ax[1].set_title("(5) cost vs threshold")

    ax[2].imshow([[tp, fn], [fp, tn]], cmap="Blues")
    for (i, j), v in zip([(0, 0), (0, 1), (1, 0), (1, 1)], [tp, fn, fp, tn]):
        ax[2].text(j, i, str(v), ha="center", va="center", fontsize=13)
    ax[2].set_xticks([0, 1], ["pred NG", "pred OK"])
    ax[2].set_yticks([0, 1], ["true NG", "true OK"])
    ax[2].set_title("(4) confusion matrix @0.5")
    fig.tight_layout()
    path = os.path.join(OUT, "lab4_classification.png")
    fig.savefig(path, dpi=110)
    print(f"\n  그림 저장: {os.path.relpath(path, HERE)}")
except ImportError:
    print("\n  (matplotlib 없음 — 그림은 건너뛴다)")

print("\n== 정리 ==")
print("  불균형 분류에서 보고할 것: 혼동행렬 네 칸 + 정밀도/재현율 + 임계값의 근거.")
print("  2단계 품질관리·이상탐지 과제의 평가는 전부 이 틀 위에서 이뤄진다.")

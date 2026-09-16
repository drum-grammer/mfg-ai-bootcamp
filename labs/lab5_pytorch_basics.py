"""실습 5 — PyTorch: 실습 3·4에서 손으로 하던 것을 그대로 옮긴다

왜 PyTorch인가
  실습 3에서 미분을 손으로 적었다. 모델이 조금만 복잡해져도 그 미분을 손으로 못 적는다.
  PyTorch는 '계산을 기록해 두었다가 미분을 대신 해 주는' 도구다. 그게 전부다.

무엇을 느끼는가
  (1) 텐서는 그냥 ndarray + 미분 기록 + GPU
  (2) backward() 가 실습 3에서 직접 쓴 gw, gb 와 같은 값을 만든다 (숫자로 대조한다)
  (3) 학습 루프는 다섯 줄 관용구다 — 부트캠프 내내 이 다섯 줄만 반복된다
  (4) 미니배치(DataLoader)는 데이터가 클 때를 위한 것이지 정확도를 위한 게 아니다
  (5) 선형 모델로 안 되는 경계가 있고, 은닉층 하나면 넘어간다 — 직접 확인한다

실행: python3 lab5_pytorch_basics.py
"""
import os

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
rng = np.random.default_rng(11)

DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")

# ── (1) 텐서 = ndarray + α ────────────────────────────────────────────────
a = np.arange(6, dtype=np.float32).reshape(2, 3)
t = torch.from_numpy(a)                      # 복사 없이 같은 메모리를 공유한다
print("== (1) 텐서 ==")
print(f"  ndarray {a.shape} {a.dtype}  →  tensor {tuple(t.shape)} {t.dtype}")
print(f"  연산 이름이 거의 같다: a.mean(axis=0)={a.mean(axis=0)}  t.mean(dim=0)={t.mean(dim=0).numpy()}")
print(f"  이 노트북이 쓸 장치: {DEVICE}")
print("  주의 두 가지: NumPy는 axis, torch는 dim / 기본 dtype이 NumPy는 float64, torch는 float32")

# ── (2) autograd 가 실습 3의 손계산과 같은 값을 준다 ──────────────────────
Xn = rng.normal(size=(200, 3)).astype(np.float32)
wn_true = np.array([1.5, -2.0, 0.7], dtype=np.float32)
yn = (Xn @ wn_true + 0.3 + rng.normal(0, 0.05, 200)).astype(np.float32)

w_np = np.zeros(3, dtype=np.float32)
b_np = np.float32(0.0)
err = Xn @ w_np + b_np - yn
gw_hand = 2.0 / len(Xn) * (Xn.T @ err)       # 실습 3에서 적었던 그 식
gb_hand = 2.0 / len(Xn) * err.sum()

Xt = torch.tensor(Xn)
yt = torch.tensor(yn)
w_t = torch.zeros(3, requires_grad=True)     # requires_grad=True → 이 값에 대한 미분을 추적한다
b_t = torch.zeros(1, requires_grad=True)
loss = ((Xt @ w_t + b_t - yt) ** 2).mean()
loss.backward()                               # 여기서 미분이 계산돼 .grad 에 쌓인다

print("\n== (2) 손으로 쓴 기울기 vs backward() ==")
print(f"  손계산  dL/dw = {np.round(gw_hand, 5)}   dL/db = {gb_hand:.5f}")
print(f"  autograd dL/dw = {np.round(w_t.grad.numpy(), 5)}   dL/db = {b_t.grad.item():.5f}")
print(f"  최대 차이 {np.abs(gw_hand - w_t.grad.numpy()).max():.2e}  → 같은 것을 자동으로 해 줄 뿐이다")

# ── (3) 다섯 줄 관용구 ────────────────────────────────────────────────────
model = nn.Linear(3, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
lossf = nn.MSELoss()
yt2 = yt.unsqueeze(1)                         # (200,) → (200,1) : 모델 출력 모양에 맞춘다

print("\n== (3) 학습 루프 — 이 다섯 줄이 전부다 ==")
print("    opt.zero_grad()          # 지난 기울기를 지운다 (빼먹으면 누적된다)")
print("    pred = model(x)          # 순전파")
print("    loss = lossf(pred, y)    # 손실")
print("    loss.backward()          # 역전파 = 기울기 계산")
print("    opt.step()               # 파라미터 갱신")
for step in range(500):
    opt.zero_grad()
    out = model(Xt)
    l = lossf(out, yt2)
    l.backward()
    opt.step()
    if step in (0, 99, 499):
        print(f"  step {step:>3d}  loss {l.item():.6f}")
print(f"  학습된 계수 {np.round(model.weight.detach().numpy().ravel(), 3)}  절편 {model.bias.item():.3f}")
print(f"  정답        {wn_true}  절편 0.300")

# ── (4) DataLoader — 데이터가 클 때 ──────────────────────────────────────
from torch.utils.data import DataLoader, TensorDataset

ds = TensorDataset(Xt, yt2)
dl = DataLoader(ds, batch_size=32, shuffle=True)
model2 = nn.Linear(3, 1)
opt2 = torch.optim.SGD(model2.parameters(), lr=0.1)
for epoch in range(30):
    for xb, yb in dl:                         # 한 에폭 = 데이터를 한 바퀴
        opt2.zero_grad()
        lossf(model2(xb), yb).backward()
        opt2.step()
print("\n== (4) 미니배치 ==")
print(f"  배치 32개로 30에폭 → 계수 {np.round(model2.weight.detach().numpy().ravel(), 3)}")
print(f"  전체를 한 번에 쓰던 (3)과 같은 답. 미니배치는 '메모리에 다 안 들어갈 때'를 위한 것이다")
print(f"  한 에폭 = {len(dl)}번의 갱신. 에폭 수와 갱신 횟수를 헷갈리지 말 것")

# ── (5) 선형으로 안 되는 경계 — 은닉층이 필요한 이유 ─────────────────────
# 금형온도가 너무 낮아도, 너무 높아도 불량이 난다 (공정 창, process window).
# 이런 U자 관계는 직선 하나로 가를 수 없다.
M = 4000
temp = rng.uniform(30, 60, M).astype(np.float32)
moist = rng.uniform(0.01, 0.07, M).astype(np.float32)
ok_window = (temp > 40) & (temp < 52) & (moist < 0.05)
yb_ = (~ok_window).astype(np.float32)         # 창 밖이면 불량
# 현실에는 오라벨·측정오차가 섞인다. 3%를 뒤집어 둔다 (정확도 100%가 나오면 오히려 의심할 것)
flip = rng.random(M) < 0.03
yb_[flip] = 1 - yb_[flip]
Xb_ = np.column_stack([(temp - 45) / 9, (moist - 0.04) / 0.017]).astype(np.float32)

Xtr, ytr = torch.tensor(Xb_[:3000]), torch.tensor(yb_[:3000]).unsqueeze(1)
Xte, yte = torch.tensor(Xb_[3000:]), torch.tensor(yb_[3000:]).unsqueeze(1)


def train(net, steps=1500, lr=0.1):
    o = torch.optim.Adam(net.parameters(), lr=lr)
    f = nn.BCEWithLogitsLoss()                # 시그모이드 + 로그손실을 안정적으로 한 번에
    for _ in range(steps):
        o.zero_grad()
        f(net(Xtr), ytr).backward()
        o.step()
    with torch.no_grad():
        acc = (((net(Xte) > 0).float()) == yte).float().mean().item()
    return acc


linear = nn.Linear(2, 1)
mlp = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))
acc_lin = train(linear)
acc_mlp = train(mlp)
print("\n== (5) 공정 창(process window) 문제 ==")
print(f"  선형 모델 정확도    {acc_lin * 100:.1f}%")
print(f"  은닉층 2개 MLP 정확도 {acc_mlp * 100:.1f}%")
print("  → 직선 하나로 '너무 낮아도 불량, 너무 높아도 불량'을 가를 수 없다.")
print("    딥러닝이 제조에서 쓰이는 이유의 8할은 이 비선형 경계다 (나머지 2할은 이미지·시계열 같은 원시 데이터)")

# ── (6) 저장과 불러오기 ──────────────────────────────────────────────────
ckpt = os.path.join(OUT, "lab5_mlp.pt")
torch.save(mlp.state_dict(), ckpt)
reloaded = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))
reloaded.load_state_dict(torch.load(ckpt))
reloaded.eval()
with torch.no_grad():
    same = torch.allclose(mlp(Xte), reloaded(Xte))
print("\n== (6) 저장·불러오기 ==")
print(f"  state_dict 저장 → {os.path.relpath(ckpt, HERE)}  ({os.path.getsize(ckpt) / 1024:.1f} KB)")
print(f"  다시 불러와 같은 출력인가: {same}")
print("  저장되는 것은 '가중치'뿐이다. 모델 구조 코드는 따로 있어야 한다 — 미니 프로젝트 제출 때 자주 놓친다")

# ── (7) 그림 ──────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gx, gy = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-2, 2, 200))
    grid = torch.tensor(np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    for a_, net, name in [(ax[0], linear, f"linear  ({acc_lin*100:.1f}%)"),
                          (ax[1], mlp, f"MLP  ({acc_mlp*100:.1f}%)")]:
        with torch.no_grad():
            zz = (net(grid) > 0).float().numpy().reshape(gx.shape)
        a_.contourf(gx, gy, zz, levels=[-0.1, 0.5, 1.1], colors=["#cfe8ff", "#ffd5d5"])
        sub = slice(0, 800)
        a_.scatter(Xb_[sub, 0], Xb_[sub, 1], c=yb_[sub], s=4, cmap="coolwarm", alpha=0.6)
        a_.set_title(name); a_.set_xlabel("mold temp (scaled)")
    ax[0].set_ylabel("moisture (scaled)")
    fig.suptitle("lab5 - a straight line cannot enclose a process window")
    fig.tight_layout()
    path = os.path.join(OUT, "lab5_boundary.png")
    fig.savefig(path, dpi=110)
    print(f"\n  그림 저장: {os.path.relpath(path, HERE)}")
except ImportError:
    print("\n  (matplotlib 없음 — 그림은 건너뛴다)")

print("\n== 정리 ==")
print("  PyTorch에서 새로 외울 것은 사실상 다섯 줄 관용구와 shape 맞추기뿐이다.")
print("  개강 후 강의 코드가 빨라지면, 모르는 줄이 (a)데이터 준비 (b)모델 정의 (c)다섯 줄 중 어디인지부터 분류할 것.")

"""실습 0 — 노트북 환경 점검 (개강 전 반드시 통과시킬 것)

부트캠프 준비물은 "개인 노트북 지참"뿐이고, 어떤 환경을 깔아 오라는 안내는
2026-09-11 기준 공지되지 않았다. 그래서 이 스크립트는 '무엇이 없는지'를 알려주고
없는 것을 깔 명령까지 찍어 준다. 개강 당일 강의실에서 pip를 돌리는 상황을 피하는 것이 목적.

실행:  python3 setup_check.py

통과 기준: 마지막 줄이 "환경 준비 완료"로 끝나면 lab1~lab5를 전부 돌릴 수 있다.
"""
import platform
import sys

# 부트캠프 1단계(NumPy·Pandas·PyTorch)에 필요한 최소 묶음.
# (import 이름, pip 이름, 어디에 쓰는가)
NEEDED = [
    ("numpy", "numpy", "배열·선형대수 — 모든 실습의 바닥"),
    ("pandas", "pandas", "표 형태 설비 로그·센서 데이터"),
    ("matplotlib", "matplotlib", "그래프"),
    ("sklearn", "scikit-learn", "회귀·분류·앙상블·평가지표"),
    ("torch", "torch", "딥러닝 (2단계 시계열·이미지 모델)"),
]

# ── (1) 파이썬 ────────────────────────────────────────────────────────────
print("== 1) 파이썬 ==")
print(f"  버전   : {platform.python_version()}  ({sys.executable})")
print(f"  플랫폼 : {platform.platform()}")
major, minor = sys.version_info[:2]
py_ok = (3, 10) <= (major, minor) <= (3, 12)
if py_ok:
    print("  판정   : OK — 3.10~3.12는 torch 휠이 확실히 있는 구간")
else:
    print("  판정   : 주의 — 3.10~3.12를 권장한다. 너무 최신(3.13+)이면 torch 휠이 없을 수 있다")

# ── (2) 패키지 ────────────────────────────────────────────────────────────
print("\n== 2) 패키지 ==")
missing = []
for mod_name, pip_name, why in NEEDED:
    try:
        mod = __import__(mod_name)
        ver = getattr(mod, "__version__", "?")
        print(f"  [O] {mod_name:12s} {ver:10s} {why}")
    except ImportError:
        print(f"  [X] {mod_name:12s} {'':10s} {why}")
        missing.append(pip_name)

if missing:
    print("\n  없는 것 설치:")
    print(f"    {sys.executable} -m pip install " + " ".join(missing))
    print("\n  → 설치 후 이 스크립트를 다시 돌릴 것. 아래 점검은 건너뛴다.")
    raise SystemExit(1)

# ── (3) 연산 장치 ─────────────────────────────────────────────────────────
print("\n== 3) 연산 장치 ==")
import torch

if torch.backends.mps.is_available():
    device = "mps"   # 애플 실리콘 맥
elif torch.cuda.is_available():
    device = "cuda"  # 엔비디아 GPU
else:
    device = "cpu"
print(f"  torch가 쓸 장치: {device}")
print("  cpu여도 1단계 실습은 전부 가능하다. 2단계 이미지 모델부터 체감 차이가 난다.")

# ── (4) 스모크 테스트 ─────────────────────────────────────────────────────
print("\n== 4) 스모크 테스트 — 아주 작은 학습 루프가 도는가 ==")
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))
true_w = np.array([2.0, -1.0, 0.5])
y = X @ true_w + rng.normal(scale=0.1, size=200)

xt = torch.tensor(X, dtype=torch.float32, device=device)
yt = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)
model = torch.nn.Linear(3, 1).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.05)
for _ in range(300):
    opt.zero_grad()
    loss = torch.nn.functional.mse_loss(model(xt), yt)
    loss.backward()
    opt.step()

learned = model.weight.detach().cpu().numpy().ravel()
print(f"  정답 계수  : {np.round(true_w, 3)}")
print(f"  학습된 계수: {np.round(learned, 3)}")
print(f"  최종 손실  : {loss.item():.5f}")
gap = float(np.abs(learned - true_w).max())
assert gap < 0.15, f"계수가 너무 다르다 (최대 오차 {gap:.3f}) — 설치 상태를 다시 볼 것"

# ── (5) 주피터 ────────────────────────────────────────────────────────────
print("\n== 5) 주피터 ==")
try:
    __import__("jupyterlab")
    print("  [O] jupyterlab — `jupyter lab` 으로 notebooks/ 를 열 수 있다")
except ImportError:
    print("  [ ] jupyterlab 없음 (선택). 노트북으로 실습하려면:")
    print(f"    {sys.executable} -m pip install jupyterlab")

print("\n환경 준비 완료")

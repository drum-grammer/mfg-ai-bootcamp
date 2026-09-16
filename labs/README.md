# 실습 코드 — KAIST 제조AI 엔지니어 부트캠프 2기 스터디

https://drum-grammer.github.io/mfg-ai-bootcamp/ 의 1단계 스터디 페이지가 가리키는 코드다.
전부 **합성 데이터**를 그 자리에서 만들어 쓰므로 내려받을 것이 없고, 예제 상황은 제조 현장(설비 센서·사출성형·불량 판정)이다.

```bash
python3 -m venv ~/venv/mfgai && source ~/venv/mfgai/bin/activate
pip install -U pip numpy pandas matplotlib scikit-learn torch jupyterlab
git clone https://github.com/drum-grammer/mfg-ai-bootcamp
cd mfg-ai-bootcamp/labs
python3 setup_check.py            # 환경 점검 — 마지막 줄이 "환경 준비 완료"면 된다
python3 lab1_numpy_sensor.py
```

| 파일 | 배우는 것 |
|---|---|
| `setup_check.py` | 환경 점검, 없는 패키지 설치 명령 |
| `lab1_numpy_sensor.py` | NumPy — shape·브로드캐스팅·axis·마스킹, 이동통계로 베어링 고장 잡기 |
| `lab2_pandas_equipment_log.py` | Pandas — 결측·리샘플링·merge로 라벨 붙은 표 만들기 |
| `lab3_regression_from_scratch.py` | 회귀를 손으로 — 손실·경사하강법·학습률·스케일링·과적합 |
| `lab4_classification_imbalanced.py` | 불균형 분류 — 정확도의 함정, 혼동행렬, 비용으로 임계값 정하기 |
| `lab5_pytorch_basics.py` | PyTorch — autograd, 학습 루프 다섯 줄, 선형 vs MLP |
| `notebooks/*.ipynb` | 위 여섯 개의 주피터 노트북판 (`make_notebooks.py` 산출물) |

코랩: `https://colab.research.google.com/github/drum-grammer/mfg-ai-bootcamp/blob/main/labs/notebooks/<이름>.ipynb`

원본은 비공개 저장소에 있고 배포 때 여기로 복사된다. 여기서 직접 고치지 말 것.

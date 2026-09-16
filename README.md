# KAIST 제조AI 엔지니어 부트캠프 2기 — 스터디 사이트

https://drum-grammer.github.io/mfg-ai-bootcamp/

KAIST 제조AI 엔지니어 부트캠프 2기 「기초·응용 과정」(2026-09-15 ~ 11-26, 화·목 오후, 판교)을 따라가면서 쓰는
**메인 문서와 단계별 스터디 페이지**. 개념 해설과 합성 데이터 실습을 한 장에 모았고, 실습은 numpy·pandas·scikit-learn·torch만으로 돈다.

| 페이지 | 내용 |
|---|---|
| [index.html](https://drum-grammer.github.io/mfg-ai-bootcamp/) | 메인 문서 — 과정 개요·10주 20회차 일정·목표와 시간 예산·주간 루틴·질문 전략·노트 양식·자료 지도 |
| [stage-1-basics.html](https://drum-grammer.github.io/mfg-ai-bootcamp/stage-1-basics.html) | 1단계 기초 AI — 환경 설치·개념 13개·실습 5종(실제 출력값·그림)·자가 점검 10문항 |
| [stage-2-applied.html](https://drum-grammer.github.io/mfg-ai-bootcamp/stage-2-applied.html) | 2단계 제조 응용 AI — 태스크 3종·시계열·이미지·비지도 이상탐지·공개 데이터셋·미니 프로젝트 후보 |
| [`labs/`](labs/) | 실습 코드 6개 + 주피터 노트북. 합성 데이터라 다운로드 없음 |

```bash
python3 -m venv ~/venv/mfgai && source ~/venv/mfgai/bin/activate
pip install -U pip numpy pandas matplotlib scikit-learn torch jupyterlab
git clone https://github.com/drum-grammer/mfg-ai-bootcamp && cd mfg-ai-bootcamp/labs
python3 setup_check.py && python3 lab1_numpy_sensor.py
```

정적 사이트다 — 서버도 데이터베이스도 없다. 이 저장소는 **생성물만** 담는다. 원본은 비공개 저장소 `drum-grammer/kaist-mfg-ai-bootcamp-2026`에 있고, 그 저장소의 GitHub Actions(`deploy-pages`)가 `main` push마다 여기로 밀어 넣는다.
여기서 직접 고치지 말 것 — 다음 배포에 덮어써진다. 같은 틀의 다른 사이트: [VLA 논문 스터디](https://drum-grammer.github.io/vla-study/).

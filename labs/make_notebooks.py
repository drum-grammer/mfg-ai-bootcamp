"""make_notebooks.py — lab*.py 를 주피터 노트북으로 변환한다

왜 생성기인가: 코드가 두 벌이면 한쪽만 고치는 사고가 난다. `.py`가 정본이고
노트북은 여기서 만들어진다. 실습 내용을 고칠 때는 `.py`만 고치고 이 스크립트를 다시 돌린다.

    python3 make_notebooks.py          # notebooks/*.ipynb 생성
    python3 make_notebooks.py --check  # 노트북이 최신인지 확인만 (종료 코드로 알림)

셀을 어디서 끊는가: `# ── 제목 ───` 꼴의 구획 주석마다 새 셀이 시작되고, 그 제목은
마크다운 셀로 올라간다. 실습 파일을 쓸 때 이 주석만 지켜 주면 된다.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "notebooks")

LABS = [
    "setup_check.py",
    "lab1_numpy_sensor.py",
    "lab2_pandas_equipment_log.py",
    "lab3_regression_from_scratch.py",
    "lab4_classification_imbalanced.py",
    "lab5_pytorch_basics.py",
]

# 마지막에 결과 그림을 붙여 보여 줄 실습
FIGURES = {
    "lab1_numpy_sensor.py": ["out/lab1_sensor.png"],
    "lab3_regression_from_scratch.py": ["out/lab3_regression.png"],
    "lab4_classification_imbalanced.py": ["out/lab4_classification.png"],
    "lab5_pytorch_basics.py": ["out/lab5_boundary.png"],
}

HEADER_RE = re.compile(r"^# ──\s*(.*?)\s*─*\s*$")


def md(lines):
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code(lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def as_source(text):
    """노트북 source는 줄 끝 개행을 포함한 리스트, 마지막 줄만 개행 없음."""
    lines = text.rstrip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []


def docstring_to_md(doc):
    """모듈 독스트링을 마크다운으로.

    `.py` 독스트링은 들여쓰기로 구조를 잡지만 마크다운에서 4칸 이상 들여쓰기는
    코드 블록이 된다. 그래서 (a) 이어지는 줄은 앞 줄에 붙이고 (b) `(1)` 꼴 항목은
    목록으로 바꾸고 (c) 나머지 들여쓰기는 없앤다. `실행:` 줄은 노트북에 불필요해 뺀다.
    """
    body = doc.strip("\n")
    first, _, rest = body.partition("\n")
    out = ["# " + first.strip(), ""]
    for raw in rest.strip("\n").split("\n"):
        s = raw.rstrip()
        if not s.strip():
            out.append("")
            continue
        if s.lstrip().startswith("실행:"):
            continue
        indent = len(s) - len(s.lstrip())
        item = re.match(r"\((\d+)\)\s*(.*)", s.lstrip())
        if indent >= 4 and out and out[-1] and not out[-1].startswith("#"):
            out[-1] = out[-1] + " " + s.lstrip()      # 이어지는 줄은 앞 줄에 붙인다
        elif item:
            if out and out[-1] and not out[-1].startswith(("-", "#")):
                out.append("")                        # 목록 앞에는 빈 줄이 있어야 한다
            out.append(f"- **({item.group(1)})** {item.group(2)}")
        else:
            out.append(s.lstrip())
    return as_source("\n".join(out).strip("\n"))


def split_cells(src):
    """본문을 `# ── 제목 ──` 주석 기준으로 잘라 (마크다운 제목, 코드) 조각 목록으로."""
    lines = src.split("\n")
    starts = [0]
    for i, ln in enumerate(lines):
        if i and HEADER_RE.match(ln):
            starts.append(i)
    starts.append(len(lines))
    chunks = []
    for a, b in zip(starts[:-1], starts[1:]):
        block = lines[a:b]
        title = None
        m = HEADER_RE.match(block[0]) if block else None
        if m and m.group(1):
            title, block = m.group(1), block[1:]
        text = "\n".join(block).strip("\n")
        if text or title:
            chunks.append((title, text))
    return chunks


def build(path):
    name = os.path.basename(path)
    src = open(path, encoding="utf-8").read()
    doc = re.match(r'^"""(.*?)"""\n', src, re.S)
    body = src[doc.end():] if doc else src
    cells = [md(docstring_to_md(doc.group(1)) if doc else ["# " + name])]

    # 노트북은 labs/ 에서 실행되도록 경로를 맞춘다 (out/ 저장·읽기)
    cells.append(md(as_source("## 준비 — 실습 폴더로 경로 맞추기\n\n"
                              "결과 파일을 `labs/out/` 에 쓰고 읽기 위해 작업 폴더를 옮긴다.")))
    cells.append(code(as_source(
        "import os, sys\n"
        "if os.path.basename(os.getcwd()) == 'notebooks':\n"
        "    os.chdir(os.path.abspath(os.path.join(os.getcwd(), '..')))\n"
        "sys.path.insert(0, os.getcwd())\n"
        "print('작업 폴더:', os.getcwd())")))

    # 노트북에서는 __file__ 이 없다. 실습 코드가 쓰는 HERE 를 작업 폴더로 대체한다.
    body = body.replace('HERE = os.path.dirname(os.path.abspath(__file__))',
                        'HERE = os.getcwd()   # 노트북에는 __file__ 이 없다')

    for title, text in split_cells(body):
        if title:
            cells.append(md(["## " + title]))
        if text:
            cells.append(code(as_source(text)))

    for fig in FIGURES.get(name, []):
        cells.append(md(as_source(f"## 결과 그림\n\n`{fig}` 를 노트북 안에서 본다.")))
        cells.append(code(as_source(
            "from IPython.display import Image, display\n"
            f"display(Image('{fig}'))")))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    check = "--check" in sys.argv
    os.makedirs(OUT, exist_ok=True)
    stale = []
    for name in LABS:
        nb = build(os.path.join(HERE, name))
        text = json.dumps(nb, ensure_ascii=False, indent=1) + "\n"
        dst = os.path.join(OUT, name.replace(".py", ".ipynb"))
        if check:
            cur = open(dst, encoding="utf-8").read() if os.path.exists(dst) else ""
            if cur != text:
                stale.append(os.path.basename(dst))
            continue
        open(dst, "w", encoding="utf-8").write(text)
        print(f"  생성: notebooks/{os.path.basename(dst)}  (셀 {len(nb['cells'])}개)")
    if check:
        if stale:
            print("낡은 노트북:", ", ".join(stale), "\n  python3 make_notebooks.py 를 실행하고 함께 커밋할 것")
            return 1
        print("노트북 최신")
    return 0


if __name__ == "__main__":
    sys.exit(main())

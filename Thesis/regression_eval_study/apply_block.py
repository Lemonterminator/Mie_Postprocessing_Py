"""apply_block.py <sec-number> <block-file> [--dry]

以 <h2 id="secN"> 為錨點，把該節整段換成 block-file 的內容。
只有在「數學式／tag／id／href／class／HTML 元素」七項計數與內容完全一致時才寫入。
"""
import sys, re, shutil, collections
from pathlib import Path

TARGET = Path(r"C:\Users\Jiang\Documents\Mie_Postprocessing_Py\Thesis\regression_eval_study\metrics_derivations_zh.html")

def _ws(s):
    """數學式內的換行／縮排對 MathJax 無意義，比對前正規化；其餘字元一個都不能變。"""
    return re.sub(r"\s+", " ", s).strip()

def invariants(t):
    return {
        "display_math": collections.Counter(_ws(m) for m in re.findall(r"\\\[.*?\\\]", t, flags=re.S)),
        "inline_math":  collections.Counter(_ws(m) for m in re.findall(r"\\\(.*?\\\)", t, flags=re.S)),
        "tag":          collections.Counter(re.findall(r"\\tag\{[^}]*\}", t)),
        "id":           collections.Counter(re.findall(r'id="([^"]+)"', t)),
        "href":         collections.Counter(re.findall(r'href="([^"]+)"', t)),
        "class":        collections.Counter(re.findall(r'class="([^"]+)"', t)),
        "element":      collections.Counter(re.findall(r"<(/?[a-zA-Z][a-zA-Z0-9]*)", t)),
    }

def locate(lines, sec):
    start = next(i for i, l in enumerate(lines) if l.strip().startswith(f'<h2 id="sec{sec}">'))
    end = len(lines)
    for i in range(start + 1, len(lines)):
        # 下一節的 h2，或（最後一節時）demo 腳本／頁尾
        if lines[i].strip().startswith('<h2 id="sec') or re.match(r"\s*<(script|footer)\b", lines[i]):
            end = i
            break
    while end > start and lines[end - 1].strip() == "":
        end -= 1
    return start, end

def main():
    sec = sys.argv[1]
    blockfile = Path(sys.argv[2])
    dry = "--dry" in sys.argv

    raw = TARGET.read_bytes().decode("utf-8")
    lines = raw.split("\n")
    start, end = locate(lines, sec)
    old = "\n".join(lines[start:end])

    new_lines = blockfile.read_bytes().decode("utf-8").rstrip("\n").split("\n")
    new = "\r\n".join(l.rstrip("\r") for l in new_lines)

    assert new.lstrip().startswith(f'<h2 id="sec{sec}">'), "新區塊必須以本節的 h2 開頭"
    assert '<h2 id="sec' not in new[40:], "新區塊不可包含其他章節的 h2"

    a, b = invariants(old), invariants(new)
    bad = False
    print(f"§{sec}  舊 {end-start} 行 → 新 {len(new_lines)} 行")
    for k in a:
        if a[k] != b[k]:
            bad = True
            print(f"  !! {k} 不一致")
            for it, n in (a[k] - b[k]).items(): print(f"     - 少了 x{n}: {str(it)[:120]}")
            for it, n in (b[k] - a[k]).items(): print(f"     + 多了 x{n}: {str(it)[:120]}")
        else:
            print(f"  ok {k}: {sum(a[k].values())}")

    HAN = re.compile(r"[\u4e00-\u9fff]")
    def prose(t):
        t = re.sub(r"\\\[.*?\\\]", " ", t, flags=re.S)
        t = re.sub(r"\\\(.*?\\\)", " ", t, flags=re.S)
        t = re.sub(r"<[^>]+>", " ", t)
        return t
    WORD = re.compile(r"[A-Za-z][A-Za-z\-']+")
    for lbl, t in (("舊", old), ("新", new)):
        p = prose(t)
        h = len(HAN.findall(p))
        w = len(WORD.findall(p))
        print(f"  {lbl}: 漢字 {h}, 英文詞 {w}, 密度 {100*w/max(h,1):.1f}/百字")

    if bad:
        print("\n*** 結構不一致，未寫入 ***")
        sys.exit(1)
    if dry:
        print("\n(dry run，未寫入)")
        return

    bak = TARGET.with_suffix(f".html.bak_pre_sec{sec}_zh")
    if not bak.exists():
        shutil.copy2(TARGET, bak)
    out = "\n".join(lines[:start] + [new] + lines[end:])
    TARGET.write_bytes(out.encode("utf-8"))
    print(f"\n已寫入 §{sec}。備份：{bak.name}")

main()

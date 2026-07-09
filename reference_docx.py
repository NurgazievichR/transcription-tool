"""Load reference transcripts from client .docx files."""

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


def parse_ts(s: str) -> float | None:
    m = re.match(r"^(\d+):(\d{2}):(\d{1,2}(?:\.\d+)?)$", s)
    if not m:
        return None
    h, m1, s1 = m.groups()
    return int(h) * 3600 + int(m1) * 60 + float(s1)


def load_reference_docx(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paras = []
    for p in root.findall(".//w:p", ns):
        texts = [t.text or "" for t in p.findall(".//w:t", ns)]
        line = "".join(texts).strip()
        if line:
            paras.append(line)

    start = 0
    for i, p in enumerate(paras):
        if p == "N/A" or parse_ts(p):
            start = i
            break

    segments = []
    i = start
    while i < len(paras):
        speaker = paras[i]
        if i + 1 >= len(paras):
            break
        ts = parse_ts(paras[i + 1])
        if ts is None:
            i += 1
            continue
        src = paras[i + 2] if i + 2 < len(paras) else ""
        eng = paras[i + 3] if i + 3 < len(paras) else ""
        segments.append({
            "speaker": speaker,
            "start": ts,
            "source": src,
            "english": eng,
        })
        i += 4
    return segments

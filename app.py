# -*- coding: utf-8 -*-
"""
ប្រព័ន្ធវាយតម្លៃដំណោះស្រាយគណិតវិទ្យាមូលដ្ឋានគ្រឹះ
Khmer foundational-arithmetic handwriting assessment — Streamlit demo.

Run locally:
    streamlit run app.py
"""

import os
import random

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

import core

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ប្រព័ន្ធវាយតម្លៃគណិតវិទ្យា",
    page_icon="✎",
    layout="wide",
)

DEVICE = "cpu"
CANVAS_W, CANVAS_H = 520, 380
DEFAULT_WEIGHTS = {
    "BiGRU": "models/bigru_best.pth",
    "BiLSTM": "models/bilstm_best.pth",
    "Attention_BiGRU": "models/attention_bigru_best.pth",
}

# ---------------------------------------------------------------------------
# Theme  (academic indigo + gold; Noto Serif/Sans Khmer)
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+Khmer:wght@500;700&family=Noto+Sans+Khmer:wght@300;400;500;700&display=swap');

:root{
  --ink:#16233B; --muted:#5C6B85; --paper:#F4F6FB; --surface:#FFFFFF;
  --primary:#2F4A9C; --primary-press:#243A7D; --accent:#E0A53B;
  --line:#DCE3F0;
  --poor:#C0473B; --fair:#D98A2B; --good:#2F6FB0; --excellent:#2E9E6B;
}
html, body, [class*="css"]{ font-family:'Noto Sans Khmer', sans-serif; color:var(--ink); }
.stApp{ background:
   radial-gradient(1200px 400px at 80% -10%, #EAF0FF 0%, rgba(234,240,255,0) 60%),
   var(--paper); }
#MainMenu, footer{ visibility:hidden; }
.block-container{ padding-top:1.4rem; max-width:1180px; }

/* ---- header ---- */
.kh-hero{ border-bottom:1px solid var(--line); padding-bottom:1.1rem; margin-bottom:1.4rem; }
.kh-eyebrow{ display:inline-flex; align-items:center; gap:.5rem; font-size:.82rem;
  font-weight:500; letter-spacing:.02em; color:var(--primary);
  background:#EAF0FF; border:1px solid #D5E0FA; padding:.28rem .7rem; border-radius:999px; }
.kh-title{ font-family:'Noto Serif Khmer', serif; font-weight:700;
  font-size:2.15rem; line-height:1.5; margin:.7rem 0 .2rem; color:var(--ink); }
.kh-sub{ color:var(--muted); font-size:1rem; font-weight:300; max-width:60ch; line-height:1.7;}

/* ---- cards ---- */
.kh-card{ background:var(--surface); border:1px solid var(--line); border-radius:16px;
  padding:1.2rem 1.3rem; box-shadow:0 1px 2px rgba(22,35,59,.04); }
.kh-card__label{ font-size:.8rem; font-weight:500; color:var(--muted);
  letter-spacing:.02em; margin-bottom:.55rem; }

/* ---- exercise (printed problem) ---- */
.kh-problem{ font-family:'Noto Serif Khmer', serif; font-weight:700;
  font-size:2.9rem; color:var(--ink); text-align:center; letter-spacing:.04em;
  padding:.7rem 0 .9rem; }
.kh-problem .op{ color:var(--primary); }
.kh-problem .q{ color:var(--accent); }

/* ---- result panel ---- */
.kh-empty{ text-align:center; color:var(--muted); padding:2.4rem 1rem; }
.kh-empty .mk{ font-size:2rem; color:var(--line); display:block; margin-bottom:.6rem; }

.kh-badge{ display:inline-block; font-family:'Noto Serif Khmer',serif; font-weight:700;
  font-size:1.5rem; padding:.45rem 1.1rem; border-radius:12px; color:#fff; }
.kh-badge--poor{ background:var(--poor);} .kh-badge--fair{ background:var(--fair);}
.kh-badge--good{ background:var(--good);} .kh-badge--excellent{ background:var(--excellent);}

.kh-score{ font-family:'Noto Serif Khmer',serif; font-weight:700; font-size:3.1rem;
  line-height:1; color:var(--ink); }
.kh-score small{ font-size:1.1rem; color:var(--muted); font-weight:400; }
.kh-feedback{ background:#FBF7EC; border:1px solid #F0E2C0; border-left:4px solid var(--accent);
  border-radius:10px; padding:.85rem 1rem; font-size:1.05rem; line-height:1.7; color:#5A4A28; }
.kh-meta{ font-size:.82rem; color:var(--muted); margin-top:.7rem; }

/* ---- buttons ---- */
.stButton>button{ width:100%; border-radius:11px; font-family:'Noto Sans Khmer',sans-serif;
  font-weight:500; padding:.6rem 1rem; border:1px solid var(--line); }
.stButton>button[kind="primary"]{ background:var(--primary); border-color:var(--primary);}
.stButton>button[kind="primary"]:hover{ background:var(--primary-press); border-color:var(--primary-press);}

/* ---- sidebar status dot ---- */
.kh-status{ display:flex; align-items:center; gap:.5rem; font-size:.9rem; margin:.3rem 0 .2rem;}
.kh-dot{ width:9px; height:9px; border-radius:50%; display:inline-block; }
.kh-dot--ok{ background:var(--excellent);} .kh-dot--demo{ background:var(--fair);}
.kh-divider{ border:0; border-top:1px solid var(--line); margin:1rem 0; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def grid_image(w, h, cell=26):
    img = Image.new("RGB", (w, h), "#FCFCFA")
    d = ImageDraw.Draw(img)
    for x in range(0, w, cell):
        d.line([(x, 0), (x, h)], fill="#ECEFF6", width=1)
    for y in range(0, h, cell):
        d.line([(0, y), (w, y)], fill="#ECEFF6", width=1)
    d.line([(0, h // 2), (w, h // 2)], fill="#DCE3F0", width=2)  # baseline
    return img


def extract_strokes(json_data):
    """Fabric.js freedraw paths -> list of strokes (list of [x,y])."""
    strokes = []
    if not json_data:
        return strokes
    for obj in json_data.get("objects", []):
        if obj.get("type") != "path":
            continue
        pts = []
        for cmd in obj.get("path", []):
            c = cmd[0]
            if c in ("M", "L") and len(cmd) >= 3:
                pts.append([cmd[1], cmd[2]])
            elif c == "Q" and len(cmd) >= 5:
                pts.append([cmd[3], cmd[4]])
            elif c == "C" and len(cmd) >= 7:
                pts.append([cmd[5], cmd[6]])
        if pts:
            strokes.append(pts)
    return strokes


@st.cache_resource(show_spinner=False)
def get_model(arch, path, mtime):
    return core.load_model(arch, path, DEVICE)


def fmt_problem(text):
    out = []
    for ch in text:
        if ch in "+-×៖":
            out.append(f'<span class="op">{ch}</span>')
        elif ch == "?":
            out.append('<span class="q">?</span>')
        else:
            out.append(ch)
    return "".join(out)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
ss = st.session_state
if "exercise" not in ss:
    ss.exercise, ss.meta = core.generate_exercise(random)
    ss.canvas_key = 0
    ss.result = None


# ---------------------------------------------------------------------------
# Sidebar — model controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ការកំណត់ម៉ូដែល")
    arch = st.radio(
        "ជ្រើសរើសម៉ូដែល",
        ["BiGRU", "BiLSTM", "Attention_BiGRU"],
        index=0,
        help="ប្តូររវាងស្ថាបត្យកម្មពីរ",
    )
    weights = st.text_input("ទីតាំងឯកសារទម្ងន់ (.pth)", DEFAULT_WEIGHTS[arch])

    mtime = os.path.getmtime(weights) if os.path.exists(weights) else 0
    model, status = get_model(arch, weights, mtime)

    if status == "loaded":
        st.markdown(
            '<div class="kh-status"><span class="kh-dot kh-dot--ok"></span>'
            f"បានផ្ទុកម៉ូដែល <b>{arch}</b></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="kh-status"><span class="kh-dot kh-dot--demo"></span>'
            "របៀបសាកល្បង (គ្មានទម្ងន់)</div>",
            unsafe_allow_html=True,
        )
        st.caption("ដាក់ឯកសារ .pth ដែលបានបណ្តុះបណ្តាល ដើម្បីទទួលលទ្ធផលពិត។")

    st.markdown('<hr class="kh-divider">', unsafe_allow_html=True)
    with st.expander("របៀបដំណើរការ"):
        st.markdown(
            "១. សរសេរដំណោះស្រាយលើក្រដាស\n\n"
            "២. ប្រព័ន្ធប្រមូលកូអរដោនេ រួចធ្វើ normalize និង center\n\n"
            "៣. ម៉ូដែលព្យាករណ៍ពិន្ទុ និងមតិយោបល់"
        )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="kh-hero">'
    '<span class="kh-eyebrow">✎ ការវាយតម្លៃដោយបញ្ញាសិប្បនិមិត្ត</span>'
    '<div class="kh-title">ប្រព័ន្ធវាយតម្លៃដំណោះស្រាយគណិតវិទ្យាមូលដ្ឋានគ្រឹះ</div>'
    '<div class="kh-sub">សរសេរចម្លើយលំហាត់នព្វន្ធដោយដៃ '
    "ហើយទទួលបានពិន្ទុ ចំណាត់ថ្នាក់ និងមតិយោបល់ភ្លាមៗ។</div>"
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Body
# ---------------------------------------------------------------------------
left, right = st.columns([1.15, 1], gap="large")

with left:
    st.markdown('<div class="kh-card__label">លំហាត់</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="kh-card"><div class="kh-problem">{fmt_problem(ss.exercise)}</div></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(
        '<div class="kh-card__label">សរសេរដំណោះស្រាយនៅទីនេះ</div>', unsafe_allow_html=True
    )

    canvas = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=5,
        stroke_color="#16233B",
        background_image=grid_image(CANVAS_W, CANVAS_H),
        update_streamlit=True,
        height=CANVAS_H,
        width=CANVAS_W,
        drawing_mode="freedraw",
        display_toolbar=True,
        key=f"canvas_{ss.canvas_key}",
    )

    b1, b2 = st.columns(2)
    new_clicked = b1.button("លំហាត់ថ្មី", use_container_width=True)
    submit_clicked = b2.button("ដាក់ស្នើ", type="primary", use_container_width=True)

    if new_clicked:
        ss.exercise, ss.meta = core.generate_exercise(random)
        ss.canvas_key += 1
        ss.result = None
        st.rerun()

    if submit_clicked:
        strokes = extract_strokes(canvas.json_data if canvas else None)
        if not strokes:
            st.warning("សូមសរសេរដំណោះស្រាយជាមុនសិន។")
        else:
            ss.result = core.predict(model, strokes, ss.exercise, DEVICE)
            ss.result["status"] = status

with right:
    st.markdown('<div class="kh-card__label">លទ្ធផល</div>', unsafe_allow_html=True)
    r = ss.result
    if not r:
        st.markdown(
            '<div class="kh-card kh-empty"><span class="mk">✎</span>'
            "សរសេរចម្លើយ រួចចុច <b>ដាក់ស្នើ</b> ដើម្បីមើលការវាយតម្លៃ។</div>",
            unsafe_allow_html=True,
        )
    else:
        if r.get("status") == "demo":
            st.info("របៀបសាកល្បង៖ លទ្ធផលនេះមិនមែនជាការព្យាករណ៍ពិតទេ (មិនទាន់ផ្ទុកទម្ងន់ម៉ូដែល)។")
        st.markdown(
            f'<div class="kh-card">'
            f'<div style="display:flex;align-items:center;justify-content:space-between;gap:1rem;">'
            f'  <div><div class="kh-card__label">ចំណាត់ថ្នាក់</div>'
            f'    <span class="kh-badge kh-badge--{r["grade_tone"]}">{r["grade"]}</span></div>'
            f'  <div style="text-align:right;"><div class="kh-card__label">ពិន្ទុ</div>'
            f'    <span class="kh-score">{core.to_khmer(r["score"])}<small>/១០</small></span></div>'
            f"</div>"
            f'<div style="margin-top:1.1rem;"><div class="kh-card__label">មតិយោបល់</div>'
            f'  <div class="kh-feedback">{r["feedback"]}</div></div>'
            f'<div class="kh-meta">ម៉ូដែល៖ {arch} · ទំនុកចិត្តពិន្ទុ៖ '
            f"{r['score_conf'] * 100:.0f}% · ចំណុចសរសេរ៖ {core.to_khmer(r['n_points'])}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
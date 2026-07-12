# -*- coding: utf-8 -*-
"""
core.py — model definitions, tokenizer, preprocessing, label maps,
exercise generation, and inference for the Khmer arithmetic demo.

The BiGRU / BiLSTM classes here are IDENTICAL to the training pipeline so the
saved state_dicts load cleanly.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence


# ===========================================================================
# Tokenizer  (character-level; recovered exactly from the training CSV)
# ===========================================================================
CHAR2ID = {
    " ": 1,
    "+": 2,
    "-": 3,
    "=": 4,
    "?": 5,
    "×": 6,
    "៖": 7,
    "០": 8,
    "១": 9,
    "២": 10,
    "៣": 11,
    "៤": 12,
    "៥": 13,
    "៦": 14,
    "៧": 15,
    "៨": 16,
    "៩": 17,
}
PAD_IDX = 0
MAX_LABEL_LEN = 13
KH_DIGITS = "០១២៣៤៥៦៧៨៩"


def to_khmer(n: int) -> str:
    return "".join(KH_DIGITS[int(d)] for d in str(n))


def encode_label(text: str, max_len: int = MAX_LABEL_LEN):
    """Char-level encode + pad to max_len. Returns (ids, true_len)."""
    ids = [CHAR2ID[c] for c in text if c in CHAR2ID]
    ids = ids[:max_len]
    true_len = max(1, len(ids))
    ids = ids + [PAD_IDX] * (max_len - len(ids))
    return ids, true_len


# ===========================================================================
# Human-readable output maps  (verified against the training CSV)
# ===========================================================================
SCORE_MAP = {0: 0, 1: 5, 2: 8, 3: 9, 4: 10}  # score_class -> real score
GRADE_MAP = {0: "ខ្សោយ", 1: "មធ្យម", 2: "ល្អ", 3: "ល្អ", 4: "ល្អឥតខ្ចោះ"}
FEEDBACK_MAP = {
    0: "ចម្លើយខុស សូមសរសេរប្រមាណវិធីពេញ",
    1: "ចម្លើយត្រូវ តែកាត់ពេក",
    2: "ចម្លើយត្រូវ តែបូកភ្លេចត្រាទុក",
    3: "ចម្លើយត្រូវ តែភ្លេចកាត់ក្បៀស",
    4: "ចម្លើយត្រូវ តែភ្លេចខ្ចី",
    5: "ចម្លើយត្រូវ តែភ្លេចត្រាទុក",
    6: "ចម្លើយត្រូវ តែសូមសរសេរប្រមាណវិធីពេញ",
    7: "ធ្វើបានល្អឥតខ្ចោះ មិនមានកំហុសទេ",
    8: "មិនត្រឹមត្រូវទេ",
    9: "សូមធ្វើប្រមាណវិធីពីស្តាំទៅឆ្វេង",
}
# grade_class -> semantic colour key used by the UI
GRADE_TONE = {0: "poor", 1: "fair", 2: "good", 3: "good", 4: "excellent"}


# ===========================================================================
# Exercise generation  ("A op B = ?", four operations, exact division)
# ===========================================================================
def generate_exercise(rng: random.Random = random):
    """Return (label_text, meta). label_text like '២៣ + ៣ = ?'."""
    op = rng.choice(["+", "-", "×", "៖"])
    if op == "+":
        a, b = rng.randint(1, 99), rng.randint(1, 99)
        answer = a + b
    elif op == "-":
        a = rng.randint(1, 99)
        b = rng.randint(1, a)  # keep result >= 0
        answer = a - b
    elif op == "×":
        a, b = rng.randint(1, 12), rng.randint(1, 9)
        answer = a * b
    else:  # exact division, remainder 0
        b = rng.randint(1, 9)
        q = rng.randint(1, 12)
        a = b * q
        answer = q
    text = f"{to_khmer(a)} {op} {to_khmer(b)} = ?"
    # guard: stay within the model's label length
    if len([c for c in text if c in CHAR2ID]) > MAX_LABEL_LEN:
        return generate_exercise(rng)
    return text, {"a": a, "b": b, "op": op, "answer": answer}


# ===========================================================================
# Coordinate preprocessing
#   1) center on centroid   2) scale by max axis-range (aspect preserving)
#   3) build per-step features [x, y, dx, dy, pen_up]  (matches training)
# ===========================================================================
def normalize_strokes(strokes):
    """strokes: list[ list[ [x,y] ] ] in raw canvas pixels -> normalized strokes."""
    allp = np.array([p for st in strokes for p in st], dtype=np.float64)
    if len(allp) == 0:
        return strokes
    c = allp.mean(axis=0)
    rng_x = allp[:, 0].max() - allp[:, 0].min()
    rng_y = allp[:, 1].max() - allp[:, 1].min()
    scale = max(rng_x, rng_y)
    if scale <= 1e-6:
        scale = 1.0
    out = []
    for st in strokes:
        out.append([[(p[0] - c[0]) / scale, (p[1] - c[1]) / scale] for p in st])
    return out


def strokes_to_features(strokes, max_points: int = 1000):
    """Normalized strokes -> (T,5) float32 features [x,y,dx,dy,pen_up]."""
    pts, pen = [], []
    for st in strokes:
        n = len(st)
        for i, p in enumerate(st):
            pts.append([float(p[0]), float(p[1])])
            pen.append(1.0 if i == n - 1 else 0.0)
    if len(pts) == 0:
        pts, pen = [[0.0, 0.0]], [1.0]
    pts = np.asarray(pts, dtype=np.float32)
    pen = np.asarray(pen, dtype=np.float32).reshape(-1, 1)
    d = np.zeros_like(pts)
    d[1:] = pts[1:] - pts[:-1]
    feat = np.concatenate([pts, d, pen], axis=1)
    if len(feat) > max_points:
        idx = np.linspace(0, len(feat) - 1, max_points).astype(int)
        feat = feat[idx]
    return feat


def preprocess_for_model(strokes, max_points: int = 1000):
    return strokes_to_features(normalize_strokes(strokes), max_points)


# ===========================================================================
# Model definitions  (must match the training pipeline exactly)
# ===========================================================================
DEFAULTS = dict(
    coord_feat_dim=5,
    coord_hidden=128,
    label_hidden=64,
    n_layers=2,
    dropout=0.3,
    embed_dim=64,
    vocab_size=18,
    pad_idx=0,
    n_score=5,
    n_feedback=10,
)


class _Cfg:
    def __init__(self, d):
        for k, v in {**DEFAULTS, **(d or {})}.items():
            setattr(self, k, v)


class AdditiveAttention(nn.Module):
    def __init__(self, in_dim, attn_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, x, mask):
        score = self.v(torch.tanh(self.proj(x))).squeeze(-1)  # (B,T)
        score = score.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(score, dim=1)
        ctx = torch.bmm(attn.unsqueeze(1), x).squeeze(1)  # (B,H)
        return ctx, attn


def lengths_to_mask(lengths, max_len, device):
    rng = torch.arange(max_len, device=device).unsqueeze(0)
    return rng < lengths.unsqueeze(1).to(device)


def _last_hidden(h_n, n_layers, n_dir, batch):
    h = h_n.view(n_layers, n_dir, batch, -1)[-1]
    return torch.cat([h[i] for i in range(n_dir)], dim=1)


class BiGRUClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.coord_gru = nn.GRU(
            cfg.coord_feat_dim,
            cfg.coord_hidden,
            cfg.n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
        )
        self.embed = nn.Embedding(
            cfg.vocab_size, cfg.embed_dim, padding_idx=cfg.pad_idx
        )
        self.label_gru = nn.GRU(
            cfg.embed_dim, cfg.label_hidden, 1, batch_first=True, bidirectional=True
        )
        fused = 2 * cfg.coord_hidden + 2 * cfg.label_hidden
        self.dropout = nn.Dropout(cfg.dropout)
        self.shared = nn.Sequential(
            nn.Linear(fused, fused // 2), nn.ReLU(), nn.Dropout(cfg.dropout)
        )
        self.score_head = nn.Linear(fused // 2, cfg.n_score)
        self.feedback_head = nn.Linear(fused // 2, cfg.n_feedback)

    def forward(self, coords, coord_lens, labels, label_lens):
        B = coords.size(0)
        packed = pack_padded_sequence(
            coords, coord_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, ch = self.coord_gru(packed)
        cvec = _last_hidden(ch, self.cfg.n_layers, 2, B)
        emb = self.embed(labels)
        lpacked = pack_padded_sequence(
            emb, label_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, lh = self.label_gru(lpacked)
        lvec = _last_hidden(lh, 1, 2, B)
        h = self.shared(self.dropout(torch.cat([cvec, lvec], dim=1)))
        return self.score_head(h), self.feedback_head(h)


class BiLSTMClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.coord_lstm = nn.LSTM(
            cfg.coord_feat_dim,
            cfg.coord_hidden,
            cfg.n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
        )
        self.embed = nn.Embedding(
            cfg.vocab_size, cfg.embed_dim, padding_idx=cfg.pad_idx
        )
        self.label_lstm = nn.LSTM(
            cfg.embed_dim, cfg.label_hidden, 1, batch_first=True, bidirectional=True
        )
        fused = 2 * cfg.coord_hidden + 2 * cfg.label_hidden
        self.dropout = nn.Dropout(cfg.dropout)
        self.shared = nn.Sequential(
            nn.Linear(fused, fused // 2), nn.ReLU(), nn.Dropout(cfg.dropout)
        )
        self.score_head = nn.Linear(fused // 2, cfg.n_score)
        self.feedback_head = nn.Linear(fused // 2, cfg.n_feedback)

    def forward(self, coords, coord_lens, labels, label_lens):
        B = coords.size(0)
        packed = pack_padded_sequence(
            coords, coord_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (ch, _) = self.coord_lstm(packed)
        cvec = _last_hidden(ch, self.cfg.n_layers, 2, B)
        emb = self.embed(labels)
        lpacked = pack_padded_sequence(
            emb, label_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (lh, _) = self.label_lstm(lpacked)
        lvec = _last_hidden(lh, 1, 2, B)
        h = self.shared(self.dropout(torch.cat([cvec, lvec], dim=1)))
        return self.score_head(h), self.feedback_head(h)


class AttentionBiGRUClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.coord_gru = nn.GRU(
            cfg.coord_feat_dim,
            cfg.coord_hidden,
            cfg.n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
        )
        self.coord_attn = AdditiveAttention(2 * cfg.coord_hidden, cfg.attn_dim)
        self.embed = nn.Embedding(
            cfg.vocab_size, cfg.embed_dim, padding_idx=cfg.pad_idx
        )
        self.label_gru = nn.GRU(
            cfg.embed_dim, cfg.label_hidden, 1, batch_first=True, bidirectional=True
        )
        self.label_attn = AdditiveAttention(2 * cfg.label_hidden, cfg.attn_dim)
        fused = 2 * cfg.coord_hidden + 2 * cfg.label_hidden
        self.shared = nn.Sequential(
            nn.Linear(fused, fused // 2), nn.ReLU(), nn.Dropout(cfg.dropout)
        )
        self.score_head = nn.Linear(fused // 2, cfg.n_score)
        self.feedback_head = nn.Linear(fused // 2, cfg.n_feedback)

    def forward(self, coords, coord_lens, labels, label_lens):
        device = coords.device
        packed = pack_padded_sequence(
            coords, coord_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        cout, _ = self.coord_gru(packed)
        cout, _ = pad_packed_sequence(cout, batch_first=True)
        cctx, _ = self.coord_attn(
            cout, lengths_to_mask(coord_lens, cout.size(1), device)
        )
        emb = self.embed(labels)
        lpacked = pack_padded_sequence(
            emb, label_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        lout, _ = self.label_gru(lpacked)
        lout, _ = pad_packed_sequence(lout, batch_first=True)
        lctx, _ = self.label_attn(
            lout, lengths_to_mask(label_lens, lout.size(1), device)
        )
        h = self.shared(torch.cat([cctx, lctx], dim=1))
        return self.score_head(h), self.feedback_head(h)


ARCHITECTURES = {
    "BiGRU": BiGRUClassifier,
    "BiLSTM": BiLSTMClassifier,
    "Attention_BiGRU": AttentionBiGRUClassifier,
}


# ===========================================================================
# Loading + inference
# ===========================================================================
def load_model(arch_name: str, weights_path: str, device="cpu"):
    """
    Returns (model, status) where status is 'loaded' or 'demo'.
    'demo' = weights not found, model uses random init (UI still works).
    """
    cfg_dict, state, status = None, None, "demo"
    if weights_path and os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            cfg_dict = ckpt.get("config")
            state = ckpt["model"]
        else:
            state = ckpt  # raw state_dict
        status = "loaded"
    cfg = _Cfg(cfg_dict)
    model = ARCHITECTURES[arch_name](cfg).to(device)
    if state is not None:
        model.load_state_dict(state, strict=False)
    model.eval()
    return model, status


@torch.no_grad()
def predict(model, strokes, label_text, device="cpu", max_points=1000):
    """Run one sample. Returns dict with classes, scores, confidences."""
    feats = preprocess_for_model(strokes, max_points)  # (T,5)
    coords = torch.from_numpy(feats).unsqueeze(0).to(device)  # (1,T,5)
    coord_lens = torch.tensor([feats.shape[0]], dtype=torch.long)

    ids, true_len = encode_label(label_text)
    labels = torch.tensor([ids], dtype=torch.long, device=device)
    label_lens = torch.tensor([true_len], dtype=torch.long)

    s_logit, f_logit = model(coords, coord_lens, labels, label_lens)
    s_prob = torch.softmax(s_logit, dim=1)[0]
    f_prob = torch.softmax(f_logit, dim=1)[0]
    s_cls = int(s_prob.argmax())
    f_cls = int(f_prob.argmax())
    return {
        "score_class": s_cls,
        "feedback_class": f_cls,
        "score": SCORE_MAP.get(s_cls, 0),
        "grade": GRADE_MAP.get(s_cls, "-"),
        "grade_tone": GRADE_TONE.get(s_cls, "fair"),
        "feedback": FEEDBACK_MAP.get(f_cls, "-"),
        "score_conf": float(s_prob.max()),
        "feedback_conf": float(f_prob.max()),
        "n_points": int(feats.shape[0]),
    }

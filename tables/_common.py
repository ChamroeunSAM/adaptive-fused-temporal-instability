"""Shared helpers for the tables and figures: paths, loading results, labels, and a PDF + CSV writer."""
import sys, re, csv, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table
from settings import RESULTS, SUBSETS, PERIOD_LABELS, subset_file   # noqa: F401
from labels import DESC, CATS, ORDER, desc   # noqa: F401

OUT = RESULTS / 'tables'; OUT.mkdir(parents=True, exist_ok=True)
SUBS = ['S1', 'S2', 'S3', 'S4']
SUBLAB = {'S1': 'S1: single-motorcycle crashes (two-year periods)', 'S2': 'S2: motorcycle and another vehicle type',
          'S3': 'S3: single vehicle, no motorcycle', 'S4': 'S4: multi-vehicle, no motorcycle'}
REFTXT = {'S1': 'Central region, daytime, no median, one motorcycle', 'S2': 'Central region, daytime, no median',
          'S3': 'Central region, daytime, no median, one pickup truck', 'S4': 'Central region, daytime, no median'}
STAGE_OF = lambda tag: 'Fixed-parameter logit' if tag.endswith('_s1') else 'Random parameters model'
fm = lambda x, d=3: f'{x:.{d}f}'.replace('-', '−')
neg = lambda s: s.replace('-', '−')


def load_subset(sub):
    Z = np.load(subset_file(sub), allow_pickle=True)
    return Z['X'], Z['y'], Z['t'], [str(n) for n in Z['names']], SUBSETS[sub]['T']


def rp_res(sub, tag): return json.load(open(RESULTS / 'rp' / sub / f'{tag}.json', encoding='utf-8'))
def rp_fit(sub, tag): return json.load(open(RESULTS / 'rp' / sub / f'{tag}_fit.json', encoding='utf-8'))
def decision(sub): return json.load(open(RESULTS / 'rp' / sub / 'decision.json', encoding='utf-8'))
def stage1(sub): return json.load(open(RESULTS / 'stage1' / f'{sub}.json', encoding='utf-8'))


def styles(size=7.3):
    return dict(c=ParagraphStyle('c', fontName='Helvetica', fontSize=size, leading=size * 1.2),
                v=ParagraphStyle('v', fontName='Helvetica', fontSize=size, leading=size * 1.2, alignment=1),
                h=ParagraphStyle('h', fontName='Helvetica-Bold', fontSize=size, leading=size * 1.2, alignment=1),
                sd=ParagraphStyle('d', fontName='Helvetica-Oblique', fontSize=size - 0.3, leading=size * 1.15, leftIndent=8),
                sec=ParagraphStyle('s', fontName='Helvetica-Bold', fontSize=size + 0.2, leading=size * 1.25),
                sub=ParagraphStyle('u', fontName='Helvetica-Oblique', fontSize=size, leading=size * 1.2),
                t=ParagraphStyle('t', fontName='Helvetica-Bold', fontSize=9.2, leading=11.5),
                n=ParagraphStyle('n', fontName='Helvetica', fontSize=6.9, leading=8.4))


def _text(v):
    if isinstance(v, Paragraph): v = v.text
    v = '' if v is None else str(v)
    v = re.sub(r'<super>(.*?)</super>', r'^\1', v); v = re.sub(r'<br/>', ' ', v)
    return re.sub(r'<[^>]+>', '', v).replace('−', '-').replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')


def save(name, pagesize, flow, title='', margins=(12, 12, 10, 10)):
    """Builds results/tables/<name>.pdf and writes every table in it as <name>.csv (cell text, minus signs as '-')."""
    from reportlab.lib.units import mm
    tables = [fl for fl in flow if isinstance(fl, Table)]          # reportlab consumes the flow list while building
    with open(OUT / f'{name}.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        for tb in tables:
            for row in tb._cellvalues: w.writerow([_text(c) for c in row])
    l, r, t_, b = (m * mm for m in margins)
    SimpleDocTemplate(str(OUT / f'{name}.pdf'), pagesize=pagesize, leftMargin=l, rightMargin=r, topMargin=t_, bottomMargin=b,
                      title=title or name).build(list(flow))
    print('saved', OUT / f'{name}.pdf')

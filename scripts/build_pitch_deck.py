"""Build a 16:9 investor deck from frozen completed evidence and editable text.

Standard-library preparation; bounded ReportLab rendering only. No training,
Torch, NumPy, model execution or investor outreach. Financial inputs are scenarios.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
INV = ROOT / 'investment'
CONTENT = INV / 'PITCH_DECK.json'
EVIDENCE = INV / 'pitch_deck_evidence_20261003.json'
OUTPUT = INV / 'sleeping_machines_pitch_deck.pdf'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    frozen = json.loads((INV / 'evidence_20261003.json').read_text())
    for name, digest in frozen['completed_evidence_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Changed completed parent: ' + name)
    native = frozen['native_language_rows']
    rows = {r['label']: r for r in native}
    timing = rows['p32/d4']
    credit = rows['p32/d4 + route credit']
    pool = rows['p32/d4/pool4 + route credit']
    best = rows['p96/d4 + route credit']
    lstm, transformer = frozen['saved_one_pass_controls']
    online_path = 'experiments/results/online_language/local_integrated_online_backbone_D8192_20260930T200000Z.json'
    online = json.loads((ROOT / online_path).read_text())
    metrics = dict(
        best_bpc=best['test256'], lstm_bpc=lstm['test'], transformer_bpc=transformer['test'],
        best_lstm_gain=lstm['test'] - best['test256'],
        best_lstm_fit_ratio=best['whole'] / lstm['whole'],
        credit_bpc=credit['test256'], timing_bpc=timing['test256'], pool_bpc=pool['test256'],
        credit_gain=timing['test256'] - credit['test256'],
        credit_fit_increase_pct=100 * (credit['whole'] / timing['whole'] - 1),
        credit_perplexity_reduction_pct=100 * (1 - 2 ** (credit['test256'] - timing['test256'])),
        transformer_fit_ratio=transformer['whole'] / credit['whole'],
        transformer_parameter_ratio=transformer['parameters'] / credit['parameters'],
        transformer_credit_gain=transformer['test'] - credit['test256'],
        pool_gain=credit['test256'] - pool['test256'],
        pool_fit_increase_pct=100 * (pool['whole'] / credit['whole'] - 1),
        pool_sparse_increase_pct=100 * (pool['sparse'] / credit['sparse'] - 1),
        depth_credit_gain=rows['p32/d8, skip2']['test256'] - rows['p32/d8, skip2 + route credit']['test256'],
        online_frozen=online['progress']['bpc']['frozen'], online_adapted=online['progress']['bpc']['online'],
        online_work_ratio=online['work']['online']['unit_special_flops'] / online['work']['frozen']['unit_special_flops'],
        online_gain=online['progress']['bpc']['frozen'] - online['progress']['bpc']['online'],
        raise_eur=3_000_000, pre_money_eur=50_000_000, stretch_pre_money_eur=100_000_000,
    )
    finance = dict(exit_equity_eur=10_000_000_000, retention=0.30, years=10, discount_rate=0.15,
                   failure_recovery_eur=0,
                   scope='Reverse underwriting assumptions, not observed success odds, appraisal or offer')
    pv_success = finance['exit_equity_eur'] * finance['retention'] / (1 + finance['discount_rate']) ** finance['years']
    metrics['required_probability_50_pct'] = 100 * metrics['pre_money_eur'] / pv_success
    metrics['required_probability_100_pct'] = 100 * metrics['stretch_pre_money_eur'] / pv_success
    metrics['new_investor_pct'] = 100 * metrics['raise_eur'] / (metrics['pre_money_eur'] + metrics['raise_eur'])
    metrics['stretch_new_investor_pct'] = 100 * metrics['raise_eur'] / (metrics['stretch_pre_money_eur'] + metrics['raise_eur'])
    ownership = metrics['new_investor_pct'] / 100 * finance['retention']
    finance.update(initial_investor_ownership=metrics['new_investor_pct'] / 100,
                   exit_investor_ownership=ownership, conditional_exit_proceeds_eur=ownership * finance['exit_equity_eur'],
                   conditional_gross_moic=ownership * finance['exit_equity_eur'] / metrics['raise_eur'])
    metrics['conditional_moic'] = finance['conditional_gross_moic']
    metrics['conditional_proceeds_m_eur'] = finance['conditional_exit_proceeds_eur'] / 1e6
    budget = [('Team: research + systems', 1_350_000), ('Compute + independent replication', 900_000),
              ('Hardware feasibility + measurement', 250_000), ('IP, legal, operations', 200_000),
              ('Contingency reserve', 300_000)]
    assert sum(v for _, v in budget) == metrics['raise_eur']
    assert math.isclose(finance['initial_investor_ownership'] * 53_000_000, 3_000_000)
    assert math.isclose(metrics['required_probability_50_pct'] / 100 * pv_success, 50_000_000)
    assert metrics['credit_gain'] > 0 and metrics['pool_fit_increase_pct'] > 0
    sensitivity = []
    for exit_eur in [3e9, 10e9, 30e9]:
        for retention in [.1, .3, .5]:
            pv = exit_eur * retention / 1.15 ** 10
            sensitivity.append(dict(exit_equity_eur=exit_eur, retention=retention, years=10,
                                    discount_rate=.15, required_probability_50=50e6 / pv,
                                    required_probability_100=100e6 / pv))
    ledger = []
    for r in [*native, *frozen['saved_one_pass_controls']]:
        is_native = r in native
        ledger.append(dict(model=r['label'], parameters=r['parameters'],
                           test_bpc_T256=r.get('test256') if is_native else r['test'],
                           whole_fit_TFLOPs_est=r['whole'] / 1e12,
                           fitting_MFLOPs_per_training_position_est=r['fit'] / 1e6,
                           inference_emulator_MFLOPs_per_evaluated_position_est=r['infer'] / 1e6,
                           inference_winner_MFLOPs_per_evaluated_position_est=(r['sparse'] if is_native else r['infer']) / 1e6,
                           backend_scope='trained sparse parity/rescore pending' if is_native else 'saved shape estimate',
                           available_slots=r.get('available_slots'), selected_writes=r.get('selected_writes'),
                           scored_keys=r.get('scored_keys')))
    parents = dict(frozen['completed_evidence_sha256'])
    parents['investment/evidence_20261003.json'] = sha(INV / 'evidence_20261003.json')
    data = dict(created_utc=datetime.now(timezone.utc).isoformat(), completed_evidence_sha256=parents,
                metrics=metrics, financial_assumptions=finance, budget_eur=dict(budget),
                financial_sensitivity=sensitivity, benchmark_ledger=ledger,
                online_work={k: online['work'][k]['unit_special_flops'] for k in ['frozen', 'online']},
                work_boundary=frozen['work_boundary'],
                status='Completed small-scale research + explicit prospective financing; no customer interest reported',
                metric_formulas=dict(credit_gain='timing T256 bpc - credited T256 bpc',
                    fit_ratio='control whole-fit estimated FLOPs / native whole-fit estimated FLOPs',
                    required_probability='pre-money * (1+discount)^years / (exit equity * future retention)',
                    initial_ownership='raise / (pre-money + raise)',
                    online_work_ratio='online total unit-special operations / frozen total unit-special operations'))
    EVIDENCE.write_text(json.dumps(data, indent=2) + '\n')
    for name, values in [('pitch_deck_benchmarks.csv', ledger), ('pitch_deck_financial_sensitivity.csv', sensitivity)]:
        with (INV / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(values)
    content = json.loads(CONTENT.read_text())
    notes = ['# Sleeping Machines — full pitch deck and diligence notes',
             '\nProposed raise: €3M. Bullish negotiating case: €50M priced pre-money; €100M stretch scenario.',
             '\nThis supersedes the pricing proposal in the older $10M discussion memo; it does not add new benchmark evidence.',
             '\nAll financial outcomes, budgets and milestone timelines are assumptions. No customer interest has been reported. Repository is private by founder instruction on 3 October 2026. This deck is a private review artifact; distribution and any future publication require a considered disclosure decision.',
             '\nResearch cut-off: completed records available on 3 October 2026. No pending training scores enter the deck.']
    for i, slide in enumerate(content['slides'], 1):
        notes.extend([f"\n## {i}. {slide['title'].format_map(metrics)}", slide.get('subtitle', '').format_map(metrics),
                      slide.get('notes', '').format_map(metrics)])
        for source in slide.get('sources', []):
            s = content['sources'][source]
            notes.append(f"- [{source}: {s['label']}]({s['url']}) — {s['scope']}")
    notes.extend(['\n## Reproduce and inspect',
                  'The editable slide narrative is PITCH_DECK.json. The frozen parent hashes, exact derived metrics and assumptions are in pitch_deck_evidence_20261003.json. CSV exports use identical column units for every model. Run the preparation only when deliberately updating the evidence cut-off; use a fresh publication tag to render. This build imports no numerical model runtime.',
                  '\nOpen diligence: company/jurisdiction, cap table and option pool; founder availability; contribution and IP chain of title (including credited co-author Karoliina Salminen); employer invention assignments; intended software/model licenses; budget quotes; three-seed modern controls; trained sparse-backend parity; measured system energy/traffic; design-partner willingness to pay. No contacts have been made by this work.'])
    (INV / 'PITCH_DECK_NOTES.md').write_text('\n\n'.join(notes) + '\n')
    print(json.dumps(dict(metrics=metrics, models=len(ledger), slides=len(content['slides']))))


def available():
    return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))


def verify():
    data = json.loads(EVIDENCE.read_text())
    for name, digest in data['completed_evidence_sha256'].items():
        if sha(ROOT / name) != digest: raise ValueError('Changed frozen parent: ' + name)
    return data


def render(stage):
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.colors import HexColor
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph
    font_dir = Path(importlib.util.find_spec('matplotlib').origin).parent / 'mpl-data/fonts/ttf'
    for name, filename in [('Deck', 'DejaVuSans.ttf'), ('DeckB', 'DejaVuSans-Bold.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    pdfmetrics.registerFontFamily('Deck', normal='Deck', bold='DeckB', italic='Deck', boldItalic='DeckB')
    content = json.loads(CONTENT.read_text()); data = verify(); metrics = data['metrics']
    W, H = 1280, 720
    bg, card, text, muted = '#0B1423', '#142237', '#F2F6FC', '#A9B9CC'
    cyan, green, amber, pink, violet = '#57CBE7', '#7DE3BE', '#F5C377', '#F29CAC', '#BCA8F2'
    c = canvas.Canvas(str(stage / OUTPUT.name), pagesize=(W, H), pageCompression=1)
    c.setTitle('Sleeping Machines | €3M research-to-platform pitch'); c.setAuthor('Sleeping Machines — Tero Keski-Valkama')
    layout = []

    def rect(x, y, w, h, color=card, radius=14, stroke=None):
        c.setFillColor(HexColor(color)); c.setStrokeColor(HexColor(stroke or color))
        c.roundRect(x, H-y-h, w, h, radius, fill=1, stroke=bool(stroke))

    def line(x1, y1, x2, y2, color=muted, width=1, dashed=False):
        c.setStrokeColor(HexColor(color)); c.setLineWidth(width)
        c.setDash(4, 4) if dashed else c.setDash()
        c.line(x1, H-y1, x2, H-y2); c.setDash()

    def p(value, x, y, w, size=19, color=text, bold=False, maxh=None):
        value = value.format_map(metrics)
        value = html.escape(value).replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>').replace('\n', '<br/>')
        paragraph = Paragraph(value, ParagraphStyle('d', fontName='DeckB' if bold else 'Deck', fontSize=size,
                              leading=size*1.32, textColor=HexColor(color), splitLongWords=False))
        _, height = paragraph.wrap(w, 1000)
        if maxh is not None and height > maxh + .1:
            raise ValueError(f'Content exceeds box on slide {len(layout)}: {value[:90]} ({height}>{maxh})')
        bottom = H-8 if y >= 674 else H-46
        if min(x,y) < 0 or x+w > W or y+height > bottom:
            raise ValueError('Content outside safe slide area: ' + value[:90])
        paragraph.drawOn(c, x, H-y-height)
        layout[-1]['text_boxes'].append(dict(x=x,y=y,w=w,h=height,size=size))
        return height

    def card_text(x, y, w, h, label, body, metric=None, accent=cyan):
        rect(x,y,w,h)
        p(label,x+22,y+22,w-44,14,accent,True)
        compact=h<210
        offset=58
        if metric:
            metric_size=34 if compact else 42
            while pdfmetrics.stringWidth(metric.format_map(metrics),'DeckB',metric_size)>w-44:
                metric_size-=1
                if metric_size<22: raise ValueError('Metric is too wide: '+metric)
            height=p(metric,x+22,y+54,w-44,metric_size,accent,True)
            offset=max(102 if compact else 121,54+height+12)
        p(body,x+22,y+offset,w-44,16 if compact else 18,maxh=h-offset-18)

    def banner(value, y=586, color=green):
        rect(52,y,1176,70,'#102C32',radius=9)
        p(value,72,y+12,1136,18,color,True,maxh=51)

    def table(headers, rows, x=52, y=190, widths=None, rowh=48, size=14):
        widths = widths or [1176/len(headers)]*len(headers)
        xx=x
        for head,width in zip(headers,widths):
            rect(xx,y,width,42,'#25405C',radius=0); p(head,xx+10,y+9,width-20,size-1,cyan,True,maxh=33); xx+=width
        for i,row in enumerate(rows):
            xx=x; yy=y+42+i*rowh
            for item,width in zip(row,widths):
                rect(xx,yy,width,rowh,card if i%2==0 else '#101D30',radius=0)
                p(str(item),xx+10,yy+10,width-20,size,maxh=rowh-14); xx+=width

    native = {r['model']:r for r in data['benchmark_ledger']}
    for index,s in enumerate(content['slides'],1):
        layout.append(dict(slide=index,title=s['title'].format_map(metrics),text_boxes=[]))
        c.setFillColor(HexColor(bg)); c.rect(0,0,W,H,fill=1,stroke=0)
        p(s.get('section','INVESTOR PRESENTATION').upper(),52,28,1176,12,cyan,True)
        kind=s['kind']
        title_height=p(s['title'],52,63,1176,36 if kind!='cover' else 61,text,True,maxh=102 if kind=='cover' else 96)
        subtitle_y=max(122,63+title_height+9) if kind!='cover' else 195
        if s.get('subtitle'): p(s['subtitle'],52,subtitle_y,1176,17,muted,maxh=56)

        if kind=='cover':
            p(s['hero'],52,287,740,29,maxh=155)
            p('Tero Keski-Valkama | Sole founder',52,470,850,21,cyan,True)
            p('Proposed raise €3M · Bullish case €50M pre-money\nResearch-stage deep tech · 3 October 2026',52,521,790,20,muted)
            for layer in range(4):
                for slot in range(4):
                    x=930+layer*65; yy=257+slot*78
                    active=(layer+slot)%4==0
                    rect(x,yy,36,36,green if active else '#25405C',radius=9)
                    if layer<3: line(x+36,yy+18,x+65,257+((slot+1)%4)*78+18,cyan if active else '#23344A',2)
            p('Capacity beyond activity',889,590,337,15,green,True)

        elif kind=='cards':
            cards=s['cards']; n=len(cards); width=(1176-22*(n-1))/n
            for i,a in enumerate(cards): card_text(52+i*(width+22),200,width,342,a['label'],a['body'],a.get('metric'),[cyan,green,amber,violet][i])
            if s.get('banner'): banner(s['banner'])

        elif kind=='market':
            cards=s['cards']
            for i,a in enumerate(cards): card_text(52+i*400,200,376,330,a['label'],a['body'],a['metric'],[cyan,green,amber][i])
            banner(s['banner'])

        elif kind=='architecture':
            for i,a in enumerate(s['steps']):
                x=52+i*300
                card_text(x,214,276,258,a['label'],a['body'],a.get('metric'),[cyan,green,amber,violet][i])
                if i<3:
                    line(x+277,335,x+295,335,cyan,2)
                    line(x+289,330,x+295,335,cyan,2); line(x+289,340,x+295,335,cyan,2)
            line(650,492,196,492,amber,2,True)
            p('Counterfactual credit: learn from selected and unrealized routes',194,506,890,20,amber,True)
            banner(s['banner'])

        elif kind=='scatter':
            x0,y0,pw,ph=115,205,695,325
            def point(x,y): return x0+x/120*pw, y0+(2.55-y)/.45*ph
            for tick in [0,30,60,90,120]:
                xx,_=point(tick,2.1); line(xx,y0,xx,y0+ph,'#213349'); p(str(tick),xx-13,y0+ph+10,50,13,muted)
            for tick in [2.1,2.2,2.3,2.4,2.5]:
                _,yy=point(0,tick); line(x0,yy,x0+pw,yy,'#213349'); p(f'{tick:.1f}',60,yy-10,50,13,muted)
            p('Test bits per character ↓',115,174,590,15,muted)
            p('Whole fitting TFLOPs, estimated ↓',247,570,565,15,muted)
            positions=[('p32/d4 + route credit',128,250,cyan,'Ours p32 / D4'),
                ('p32/d4/pool4 + route credit',290,315,green,'Ours p32 / pool4'),
                ('p32/d8, skip2 + route credit',340,395,cyan,'Ours p32 / D8'),
                ('p96/d4 + route credit',500,445,green,'Ours p96 / D4'),
                ('LSTM-256',118,505,violet,'LSTM-256'),('Transformer-256x2',543,239,pink,'Transformer-256×2')]
            for label,lx,ly,color,short in positions:
                r=native[label]; xx,yy=point(r['whole_fit_TFLOPs_est'],r['test_bpc_T256'])
                c.setFillColor(HexColor(color)); c.circle(xx,H-yy,6,fill=1,stroke=0)
                line(xx,yy,lx+8,ly+44 if yy>ly+18 else ly-7,color,.8)
                p(short,lx,ly,260,14,color,True); p(f"{r['test_bpc_T256']:.3f} bpc · {r['whole_fit_TFLOPs_est']:.2f} TF",lx,ly+20,250,12,muted)
            card_text(870,205,358,190,'QUALITY FRONTIER','Ahead of saved LSTM; {best_lstm_fit_ratio:.2f}× fitting work.','{best_bpc:.3f}',green)
            card_text(870,409,358,190,'SMALL TRANSFORMER CONTROL','p32 / D4: less estimated fitting work and better quality.','{transformer_fit_ratio:.1f}×',cyan)
            p(s['caveat'],52,620,1176,13,muted,maxh=35)

        elif kind=='credit':
            for i,(metric,label,body) in enumerate(s['stats']):
                card_text(52+i*400,205,376,236,label,body,metric,[cyan,green,amber][i])
            p('Same p32 / D4 model; hard forward routes retained',52,473,1120,23,text,True)
            p(s['detail'],52,520,1176,19,muted,maxh=62)
            banner(s['banner'],600)

        elif kind=='capacity':
            for i,(label,slots,scalars,keys,bpc) in enumerate([('Pool 2',16,512,16,metrics['credit_bpc']),('Pool 4',32,1024,32,metrics['pool_bpc'])]):
                x=52+i*410; rect(x,202,386,342)
                p(label,x+22,224,340,23,cyan,True)
                for j in range(slots):
                    xx=x+24+(j%8)*42; yy=276+(j//8)*37
                    rect(xx,yy,28,25,green if j<8 else '#2B4059',radius=4)
                p(f'{slots} slots · {scalars:,} value scalars\n8 writes · {keys} keys scored\n{bpc:.3f} bpc',x+22,441,342,18)
            card_text(890,202,338,342,'WHAT SCALES','Winner-only arithmetic +{pool_sparse_increase_pct:.2f}%. Fitting work +{pool_fit_increase_pct:.1f}%. Keys and value storage double.','8 writes',green)
            banner(s['banner'])

        elif kind=='inference':
            series=[('Ours p32 / pool4','p32/d4/pool4 + route credit'),('Ours p64 / pool2','p64/d4 + route credit'),('LSTM-256','LSTM-256')]
            p('Estimated MFLOPs per evaluated input position',52,184,820,17,muted)
            for i,(short,label) in enumerate(series):
                y=237+i*106; r=native[label]; p(short,52,y+10,260,18,text,True)
                for j,(field,color) in enumerate([('inference_emulator_MFLOPs_per_evaluated_position_est',cyan),('inference_winner_MFLOPs_per_evaluated_position_est',green)]):
                    v=r[field]; yy=y+j*31
                    rect(330,yy,v/.95*495,21,color,radius=3); p(f'{v:.3f}',335+v/.95*495,yy-1,90,15,color,True)
            p('Cyan: emulator / control forward     Green: winner-only / same control',52,584,1176,15,muted)
            card_text(932,231,296,309,'STATUS','All keys remain scored. Actual trained FP32 parity and held-out rescore are pending. Traffic, setup, latency and joules must be measured.',None,amber)
            p(s['caveat'],52,621,1176,13,muted,maxh=35)

        elif kind=='online':
            card_text(52,212,566,303,'PREDICT BEFORE UPDATE','8,191 new development targets; same persistent state, frozen versus full-backbone adaptation.','{online_frozen:.3f} → {online_adapted:.3f}',green)
            card_text(640,212,588,303,'COSTED LEARNING','Total online processing work includes credit, backward and optimizer. Retention under drift and native asynchronous learning remain open.','{online_work_ratio:.1f}× work',amber)
            banner(s['banner'])

        elif kind=='table':
            table(s['headers'],s['rows'],y=s.get('y',200),widths=s['widths'],rowh=s.get('rowh',62),size=s.get('size',15))
            if s.get('banner'): banner(s['banner'],s.get('banner_y',600))
            if s.get('caveat'): p(s['caveat'],52,623,1176,13,muted,maxh=32)

        elif kind=='roadmap':
            for i,a in enumerate(s['stages']):
                x=52+i*300; card_text(x,200,276,346,a['label'],a['body'],a.get('metric'),[cyan,green,amber,violet][i])
            banner(s['banner'])

        elif kind=='founder':
            p('Tero\nKeski-Valkama',52,209,510,47,text,True)
            p('Sole founder',52,363,510,25,cyan,True)
            p(s['summary'],52,416,520,20,muted,maxh=160)
            for i,a in enumerate(s['roles']):
                y=201+i*94; rect(616,y,612,82)
                p(a['date'],636,y+17,146,15,cyan,True)
                p(a['role'],797,y+15,410,17,text,True)
                p(a['detail'],797,y+43,410,14,muted,maxh=36)
            p(s['caveat'],52,623,1176,13,muted,maxh=32)

        elif kind=='budget':
            x=52
            colors=[cyan,green,violet,amber,pink]
            for i,(label,amount) in enumerate(data['budget_eur'].items()):
                width=1176*amount/3e6; rect(x,200,width,58,colors[i],radius=0); x+=width
            for i,(label,amount) in enumerate(data['budget_eur'].items()):
                y=290+i*49; rect(52,y+3,12,12,colors[i],radius=3)
                p(label,79,y,728,19)
                p(f'€{amount/1e6:.2f}M  |  {amount/3e6:.1%}',870,y,357,19,colors[i],True)
            banner(s['banner'])
            p(s['caveat'],52,623,1176,13,muted,maxh=32)

        elif kind=='valuation':
            card_text(52,201,376,235,'BULLISH PRICED ROUND','€3M raise → €53M post-money; {new_investor_pct:.2f}% new-investor ownership.','€50M',green)
            card_text(452,201,376,235,'REVERSE UNDERWRITING','Required platform-success probability in the stated €10B exit scenario.','{required_probability_50_pct:.2f}%',cyan)
            card_text(852,201,376,235,'STRETCH SCENARIO','€100M pre-money requires {required_probability_100_pct:.2f}% in the same model; {stretch_new_investor_pct:.2f}% ownership.','€100M',amber)
            p('V = p × exit equity × future retention / (1 + discount rate)^years',52,475,1176,23,text,True)
            p('Assumptions: €10B exit; 30% stake retention; 10 years; 15% discount; failure value zero.',52,523,1176,18,muted)
            banner(s['banner'])

        elif kind=='upside':
            table(s['headers'],s['rows'],y=202,widths=s['widths'],rowh=79,size=16)
            p(s['detail'],52,449,1176,20,muted,maxh=112)
            banner(s['banner'])

        elif kind=='closing':
            p('€3 million',52,202,740,83,green,True)
            p(s['hero'],52,325,1020,28,maxh=139)
            for i,a in enumerate(s['items']):
                p(f'{i+1:02d}',52+i*400,500,82,32,cyan,True)
                p(a,145+i*400,504,294,18,maxh=105)
            p('Tero Keski-Valkama · Private investor diligence available by arrangement',52,627,1176,16,muted)

        elif kind=='benchmark':
            labels=s['models']; rows=[]
            for label in labels:
                r=native[label]
                rows.append([s.get('short_labels',{}).get(label,label),f"{r['test_bpc_T256']:.3f}",f"{r['parameters']/1000:.1f}",
                    f"{r['whole_fit_TFLOPs_est']:.2f}",f"{r['fitting_MFLOPs_per_training_position_est']:.3f}",
                    f"{r['inference_emulator_MFLOPs_per_evaluated_position_est']:.3f}",
                    f"{r['inference_winner_MFLOPs_per_evaluated_position_est']:.3f}"])
            table(['Model','Test bpc','Params K','Fit total TF','Fit MF/pos','Emul MF/pos','Winner MF/pos'],rows,
                  y=188,widths=[315,100,120,150,150,170,171],rowh=43,size=13)
            p(s['caveat'],52,626,1176,13,muted,maxh=35)

        elif kind=='sensitivity':
            rows=[]
            for r in data['financial_sensitivity']:
                rows.append([f"€{r['exit_equity_eur']/1e9:.0f}B",f"{r['retention']:.0%}",
                             f"{r['required_probability_50']:.2%}",f"{r['required_probability_100']:.2%}"])
            table(['Exit equity','Stake retention','Required p: €50M','Required p: €100M'],rows,
                  y=188,widths=[294]*4,rowh=39,size=14)
            p(s['caveat'],52,601,1176,14,muted,maxh=58)

        elif kind=='sources':
            ids=s['source_list']
            for i,source in enumerate(ids):
                src=content['sources'][source]; y=192+i*61
                p(source,52,y,60,15,cyan,True)
                p(src['label'],117,y,1090,16,text,True)
                p(src['scope'],117,y+26,1090,13,muted,maxh=36)
                c.linkURL(src['url'],(117,H-y-51,1200,H-y),relative=0,thickness=0)

        else: raise ValueError('Unknown slide kind: '+kind)

        line(52,674,1228,674,'#2A3D52',.8)
        p('SLEEPING MACHINES  /  PRIVATE REVIEW  /  03 OCT 2026',52,685,460,10,muted)
        xx=480
        for source in s.get('sources',[]):
            src=content['sources'][source]; label=source+' '+src['short']
            width=pdfmetrics.stringWidth(label,'Deck',9)+19
            if xx+width>1170: raise ValueError('Footer sources overflow')
            c.setFont('Deck',9); c.setFillColor(HexColor(muted)); c.drawString(xx,H-695,label)
            c.linkURL(src['url'],(xx,H-700,xx+width,H-683),relative=0,thickness=0); xx+=width
        c.setFont('DeckB',11); c.setFillColor(HexColor(cyan)); c.drawRightString(1228,25,f'{index:02d} / {len(content["slides"]):02d}')
        c.showPage()
    c.save()
    (stage / 'layout.json').write_text(json.dumps(layout,indent=2)+'\n')
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules


def limits():
    os.nice(19)
    resource.setrlimit(resource.RLIMIT_AS,(1_000_000*1024,1_000_000*1024))


def publish(tag):
    if not tag or Path(tag).name!=tag: raise ValueError('Unique plain tag required')
    stage=ROOT / '.git/pitch-deck-preview' / tag
    record=INV / ('publication_'+tag+'.json')
    if stage.exists() or record.exists(): raise ValueError('Fresh tag required; old artifacts preserved')
    if available()<8192*1024: raise ValueError('Less than 8GiB available')
    verify()
    inputs=[CONTENT,EVIDENCE,Path(__file__).resolve(),INV/'PITCH_DECK_NOTES.md',
            INV/'pitch_deck_benchmarks.csv',INV/'pitch_deck_financial_sensitivity.csv']
    hashes={str(x.relative_to(ROOT)):sha(x) for x in inputs}
    previous=sha(OUTPUT) if OUTPUT.exists() else None
    stage.mkdir(parents=True)
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    start=time.monotonic(); peak=0
    with (stage/'render.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--render-stage',str(stage)],
                               env=env,stdout=log,stderr=subprocess.STDOUT,preexec_fn=limits)
        try:
            while child.poll() is None:
                try:
                    rss=next((int(x.split()[1]) for x in Path(f'/proc/{child.pid}/status').read_text().splitlines() if x.startswith('VmRSS:')),0)
                except FileNotFoundError: rss=0
                peak=max(peak,rss)
                if rss>300000 or available()<8192*1024 or time.monotonic()-start>120:
                    raise RuntimeError('Pitch deck resource guard tripped')
                time.sleep(.1)
            if child.returncode: raise RuntimeError('Deck render failed; inspect '+str(stage/'render.log'))
        finally:
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
    import pymupdf
    pages=[]
    with pymupdf.open(stage/OUTPUT.name) as doc:
        for i,page in enumerate(doc,1):
            value=page.get_text()
            if len(value.strip())<160: raise ValueError('Sparse slide '+str(i))
            for x0,y0,x1,y1,*_ in page.get_text('blocks'):
                if min(x0,y0)<0 or x1>page.rect.width or y1>page.rect.height: raise ValueError('Out-of-bounds slide '+str(i))
            if f'{i:02d} / {len(doc):02d}' not in value: raise ValueError('Slide pagination mismatch')
            page.get_pixmap(matrix=pymupdf.Matrix(.75,.75)).save(str(stage/f'slide_{i:02d}.png'))
            pages.append(dict(slide=i,characters=len(value)))
        # A bounded contact sheet for visual review; no image-processing dependency.
        contact=pymupdf.open()
        for offset in range(0,len(doc),8):
            grid=contact.new_page(width=1280,height=720)
            for j in range(min(8,len(doc)-offset)):
                box=pymupdf.Rect((j%2)*640,(j//2)*180,(j%2+1)*640,(j//2+1)*180)
                grid.show_pdf_page(box,doc,offset+j)
            grid.get_pixmap().save(str(stage/f'contact_{offset//8+1}.png'))
        contact.close()
    verify()
    if hashes!={str(x.relative_to(ROOT)):sha(x) for x in inputs}: raise ValueError('Concurrent deck-source change')
    if previous!=(sha(OUTPUT) if OUTPUT.exists() else None): raise ValueError('Concurrent deck publication')
    if OUTPUT.exists():
        archive=INV/'archive'/(tag+'_previous_'+OUTPUT.name)
        archive.parent.mkdir(exist_ok=True)
        if archive.exists(): raise ValueError('Preserve existing archive')
        shutil.copy2(OUTPUT,archive)
    temporary=OUTPUT.with_suffix('.publishing'); shutil.copy2(stage/OUTPUT.name,temporary); temporary.replace(OUTPUT)
    record.write_text(json.dumps(dict(status='completed',published_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashes,previous_sha256=previous,output_sha256=sha(OUTPUT),slides=pages,
        guards=dict(threads=1,rss_kib=300000,address_space_kib=1000000,min_available_mib=8192,timeout_s=120),
        render_wall_s=time.monotonic()-start,peak_render_rss_kib=peak,
        scope='Saved evidence, public-source research and financial scenarios only; no numerical model runtime'),indent=2)+'\n')
    print(json.dumps(dict(pdf=str(OUTPUT),slides=len(pages),preview=str(stage),peak_rss_kib=peak)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true'); mode.add_argument('--tag'); mode.add_argument('--render-stage')
    args=parser.parse_args()
    if args.prepare: prepare()
    elif args.render_stage: render(Path(args.render_stage))
    else: publish(args.tag)

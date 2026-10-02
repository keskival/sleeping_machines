"""Clarify measured live-state range without changing frozen numerical sources."""
import runpy
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=runpy.run_path(str(ROOT/'report/route_calibration_evidence.py'))


def load(read):
    data=ORIGINAL['load'](read)
    sizes={case['state_tensor_bytes'] for row in data['temperature']['rows'] for case in row['cases']}
    if sizes!={576,648,720}:raise ValueError('Changed measured calibration state evidence')
    return data


def pages(data):
    blocks=ORIGINAL['pages'](data)
    for page in blocks:
        if 'Clock-preserving route calibration' not in page[0][1]:continue
        for i,(kind,text) in enumerate(page):
            if kind=='small':
                page[i]=(kind,text.replace('has8 available units/720state bytes.',
                    'has8 available units;576..720 live-state bytes.'))
        page.append(('small','Accounting clarification,2 October:720bytes is the observed '
            'maximum, not every prefix. Completed cases retain576,648 or720 persistent '
            'tensor bytes as6,7 or8 units become occupied. Repeated receiver writes and '
            'available capacity do not imply all available units have live memory. '
            'Parameters, Python metadata and graphs are separate. Original report retained.'))
    return blocks

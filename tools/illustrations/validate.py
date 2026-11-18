#!/usr/bin/env python3
"""Validate generated SVG bounds, glyphs, license embedding and page references."""
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from PIL import ImageFont
from fontTools.ttLib import TTFont
ROOT=Path(__file__).resolve().parents[2]
report=json.loads((ROOT/'tools/illustrations/manifest.json').read_text())
font_path=ROOT/'tools/assets/Inter.ttf';cmap=TTFont(font_path).getBestCmap()
issues=[];checked=[]
for page in report['pages']:
    markdown=(ROOT/page['page']).read_text()
    paths=[page['opener']]+[f['file'] for f in page['figures']]
    for relative in paths:
        path=ROOT/'docs/images'/relative;source=path.read_text();root=ET.fromstring(source)
        w,h=map(float,(root.attrib['width'],root.attrib['height']))
        if w!=760:issues.append(f'{relative}: wrong canvas width')
        if '@font-face' not in source or 'font/woff2;base64,' not in source:issues.append(f'{relative}: missing embedded font')
        if 'font-license' not in source or 'SIL OPEN FONT LICENSE' not in source:issues.append(f'{relative}: missing font license')
        for el in root.iter():
            tag=el.tag.rsplit('}',1)[-1]
            if tag=='text':
                value=''.join(el.itertext());size=float(el.attrib.get('font-size','20'))
                absent={c for c in value if not c.isspace() and ord(c) not in cmap}
                if absent:issues.append(f'{relative}: missing glyphs {absent}')
                font=ImageFont.truetype(str(font_path),round(size*100))
                width=font.getlength(value)/100
                x=float(el.attrib.get('x','0'));y=float(el.attrib.get('y','0'));anchor=el.attrib.get('text-anchor','start')
                left=x-(width/2 if anchor=='middle' else width if anchor=='end' else 0)
                if left < -1 or left+width > w+1 or y-size < -1 or y+size*.3 > h+1:
                    issues.append(f'{relative}: text out of bounds: {value!r} x={left:.1f}..{left+width:.1f}')
            elif tag=='rect':
                x=float(el.attrib.get('x','0'));y=float(el.attrib.get('y','0'))
                rw=float(el.attrib.get('width','0'));rh=float(el.attrib.get('height','0'))
                if x<0 or y<0 or x+rw>w+.5 or y+rh>h+.5:issues.append(f'{relative}: rect out of bounds')
        if relative not in markdown:issues.append(f'{relative}: missing page insertion')
        checked.append(dict(file=relative,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),height=h))
    if '<!-- mgpu:opener:start -->' not in markdown:issues.append(f'{page["page"]}: missing opener')
    for fig in page['figures']:
        if markdown.count(f'<!-- mgpu:{fig["key"]}:start -->')!=1:issues.append(f'{page["page"]}: duplicate/missing figure marker')
out=dict(status='PASS' if not issues else 'FAIL',pages=len(report['pages']),svgs=len(checked),issues=issues,files=checked)
(ROOT/'tools/illustrations/validation.json').write_text(json.dumps(out,indent=2)+'\n')
for issue in issues:print(issue)
print(f'{out["status"]}: {len(checked)} SVGs across {out["pages"]} pages; fonts, bounds, license and insertions.')
raise SystemExit(bool(issues))

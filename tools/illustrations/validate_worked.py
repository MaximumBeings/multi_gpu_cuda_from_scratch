#!/usr/bin/env python3
"""Validate chapter-by-chapter worked figures without executing GPU programs."""
from pathlib import Path
import base64
import hashlib
import io
import json
import re
import xml.etree.ElementTree as ET
import yaml
from PIL import ImageFont
from fontTools.ttLib import TTFont

ROOT=Path(__file__).resolve().parents[2]

def pages():
    result=[]
    def visit(value):
        if isinstance(value,list):
            for item in value:visit(item)
        elif isinstance(value,dict):
            for title,item in value.items():
                if isinstance(item,str) and re.match(r'(part\d+/\d\d-|appendix/[a-g]-)',item):
                    identifier=Path(item).name.split('-')[0]
                    key='appendix-'+identifier if item.startswith('appendix/') else 'ch'+identifier
                    result.append((title,item,key))
                elif not isinstance(item,str):visit(item)
    visit(yaml.safe_load((ROOT/'mkdocs.yml').read_text())['nav'])
    return result

def verify():
    errors=[];records=[];entries=pages();expected=set()
    if len(entries)!=47:errors.append('Expected 40 chapters and seven appendices')
    for title,page,key in entries:
        text=(ROOT/'docs'/page).read_text();relative='worked/'+key+'.svg';expected.add(relative)
        blocks=re.findall(r'<!-- mgpu:worked:start -->(.*?)<!-- mgpu:worked:end -->',text,re.S)
        if len(blocks)!=1:
            errors.append(page+': expected one worked section');continue
        body=blocks[0]
        if relative not in body:errors.append(page+': missing worked figure link')
        if not re.search(r'^1\. ',body,re.M):errors.append(page+': missing numbered derivation')
        if not re.search(r'exercise|self-check|check yourself',body,re.I):errors.append(page+': missing exercise/check')
        if not re.search(r'answer|solution',body,re.I):errors.append(page+': missing explained answer')
        path=ROOT/'docs/images'/relative
        if not path.is_file():errors.append(relative+': missing file');continue
        source=path.read_text();svg=ET.fromstring(source)
        width,height=float(svg.attrib['width']),float(svg.attrib['height'])
        if width!=760 or height<=0:errors.append(relative+': invalid canvas')
        if 'font-license' not in source or 'SIL OPEN FONT LICENSE' not in source:errors.append(relative+': missing font license')
        embedded=re.search(r'font/woff2;base64,([A-Za-z0-9+/=]+)',source)
        if not embedded:errors.append(relative+': missing embedded type');continue
        if not any('BookDiagram' in element.attrib.get('font-family','') for element in svg.iter()):
            errors.append(relative+': embedded font is not applied to drawing text')
        cmap=TTFont(io.BytesIO(base64.b64decode(embedded[1]))).getBestCmap()
        for element in svg.iter():
            tag=element.tag.rsplit('}',1)[-1]
            if tag=='text':
                value=''.join(element.itertext());size=float(element.attrib.get('font-size',20))
                if size<14:errors.append(relative+': font below 14px')
                missing={char for char in value if not char.isspace() and ord(char) not in cmap}
                if missing:errors.append(relative+': missing embedded glyphs '+repr(missing))
                font=ImageFont.truetype(str(ROOT/'tools/assets/Inter.ttf'),round(size*100))
                length=font.getlength(value)/100;x=float(element.attrib.get('x',0));y=float(element.attrib.get('y',0));anchor=element.attrib.get('text-anchor','start')
                left=x-(length/2 if anchor=='middle' else length if anchor=='end' else 0)
                if left < -1 or left+length > width+1 or y-size < -1 or y+size*.3 > height+1:
                    errors.append(relative+': text outside canvas: '+repr(value))
            elif tag=='rect':
                x=float(element.attrib.get('x',0));y=float(element.attrib.get('y',0));w=float(element.attrib.get('width',0));h=float(element.attrib.get('height',0))
                if min(x,y,w,h)<0 or x+w>width+.5 or y+h>height+.5:errors.append(relative+': rectangle outside canvas')
        records.append(dict(page=page,file=relative,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    actual={'worked/'+p.name for p in (ROOT/'docs/images/worked').glob('*.svg')}
    if actual!=expected:errors.append('Worked figure set differs from active chapter set')
    report=dict(status='FAIL' if errors else 'PASS',pages=len(entries),figures=len(records),scope='Derived teaching examples; no new hardware execution.',issues=errors,files=records)
    (ROOT/'tools/illustrations/worked-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    for error in errors:print(error)
    print(f"{report['status']}: {len(entries)} worked lessons; {len(records)} numerical SVGs; coverage, fonts, glyphs, bounds and explained checks.")
    return bool(errors)
if __name__=='__main__':raise SystemExit(verify())

#!/usr/bin/env python3
"""Build and insert original, responsive book artwork with safe rerunnable markers.

Usage: python tools/illustrations/build.py [--no-insert]
Requires Pillow and fonttools[woff]. Does not modify any source-code fence.
"""
from pathlib import Path
import argparse
import base64
import hashlib
import io
import json
import re
import yaml
from catalog import CHAPTERS, APPENDICES

ROOT=Path(__file__).resolve().parents[2]
IMAGES=ROOT/'docs/images'
NAV_TITLES={}
def collect_titles(tree):
    if isinstance(tree,list):
        for item in tree:collect_titles(item)
    elif isinstance(tree,dict):
        for title,value in tree.items():
            if isinstance(value,str):NAV_TITLES[value]=title
            else:collect_titles(value)
collect_titles(yaml.safe_load((ROOT/'mkdocs.yml').read_text()).get('nav',[]))
def outside_fences(text):
    return re.sub(r'```.*?```',lambda m:''.join('\n' if c=='\n' else ' ' for c in m.group(0)),text,flags=re.S)
def page_title(page,text):
    match=re.search(r'^# (.+)$',outside_fences(text),re.M)
    return match.group(1) if match else NAV_TITLES.get(str(page.relative_to(ROOT/'docs')),page.stem.replace('-',' ').title())

def font_style():
    from fontTools import subset
    from fontTools.ttLib import TTFont
    text=''.join(p.read_text() for p in (ROOT/'docs').rglob('*.md'))
    text+=''.join(str(x) for x in (CHAPTERS,APPENDICES))+'μ→≠×0123456789'
    font=TTFont(ROOT/'tools/assets/Inter.ttf')
    options=subset.Options();options.flavor='woff2';options.layout_features=['*']
    cut=subset.Subsetter(options=options);cut.populate(text=text);cut.subset(font)
    font.flavor='woff2';buffer=io.BytesIO();font.save(buffer)
    encoded=base64.b64encode(buffer.getvalue()).decode()
    (ROOT/'tools/illustrations/font-style.svg').write_text('<style type="text/css">@font-face{font-family:BookDiagram;src:url(data:font/woff2;base64,'+encoded+') format("woff2");font-weight:100 900;font-style:normal}text{font-synthesis:none;text-rendering:optimizeLegibility}</style>')

font_style()
import proofs
proofs.OUT=IMAGES
from proofs import Drawing, PLUM, CORAL, LINE
from renderers import Figure,draw


def opener(path,key,title,kind,labels):
    d=Drawing(800,True)
    d.text(32,40,'MULTI-GPU FROM SCRATCH / '+key,14,'#c3b6e1',650)
    d.line([(32,57),(728,57)],'#514666',1)
    yy=d.paragraph(32,103,title,696,34,'#f7f3ff',43)
    art=yy+24
    accent='#f0a08b';pale='#b8a1e1';fill='#33294e'
    def box(x,y,w,h,label):
        d.rect(x,y,w,h,fill,pale,12);d.text(x+w/2,y+h/2+7,label,20,'#f7f3ff',550,'middle')
    if kind in {'capacity','ownership','symmetric','hybrid','replicas'}:
        for i in range(3):
            x=45+i*239
            box(x,art,190,58,('PE ' if kind=='symmetric' else 'GPU ')+str(i))
            for j in range(3):
                d.rect(x+12,art+83+j*52,166,40,fill,accent if (j+i)%3==0 else pale,5)
        d.line([(140,art+256),(618,art+256)],accent,2,False,True)
    elif kind in {'balance','weighted'}:
        for i,width in enumerate([500,360,235]):
            d.text(38,art+45+i*78,f'GPU {i}',20,'#c3b6e1')
            d.rect(155,art+15+i*78,width,52,fill,accent if i==0 else pale,8)
    elif kind in {'topology','numa','switch','rdma','staging','transfer','ipc','put','layers'}:
        box(40,art+20,245,185,'NODE A' if kind not in {'ipc','layers'} else ('PROCESS A' if kind=='ipc' else 'PARTITION A'))
        box(475,art+20,245,185,'NODE B' if kind not in {'ipc','layers'} else ('PROCESS B' if kind=='ipc' else 'PARTITION B'))
        d.line([(285,art+106),(475,art+106)],accent,3)
        d.line([(285,art+116),(475,art+116)],accent,3)
        d.text(380,art+253,'EXPLICIT HANDOFF',20,'#c3b6e1',550,'middle')
    elif kind in {'tree','experts','broadcast','reduce','hub'}:
        box(260,art,240,67,'ROUTE / COMBINE' if kind=='experts' else ('ROOT' if kind in {'broadcast','reduce','hub'} else 'HIERARCHY'))
        for i in range(3):
            x=45+i*239;d.line([(380,art+67),(380,art+115),(x+95,art+115),(x+95,art+167)],accent,2.5)
            box(x,art+167,190,73,('EXPERT ' if kind=='experts' else 'RANK ')+str(i))
    elif kind in {'flow','mapping','profile','netting'}:
        for i in range(4):
            x=40+i*181;box(x,art+50,137,128,str(i+1))
            if i<3:d.line([(x+141,art+114),(x+177,art+114)],accent,3)
        d.text(380,art+245,'TRACE THE DEPENDENCIES',20,'#c3b6e1',550,'middle')
    elif kind in {'pipeline','buckets','events','handoff','threads'}:
        for lane in range(3):
            d.text(32,art+48+lane*75,f'{"Stage" if kind in {"pipeline","buckets"} else "Queue"} {lane}',20,'#c3b6e1')
            for step in range(4):box(158+(step+lane)*79,art+15+lane*75,71,48,str(step))
    elif kind in {'column','row','summa','matrixgrid','tpgrid','tiles','halo'}:
        for r in range(4):
            for c in range(8):
                color=pale if c<4 else accent
                d.rect(85+c*75,art+10+r*59,68,52,fill,color,5)
        d.line([(380,art),(380,art+252)],accent,2,False,True)
    elif kind in {'ring','failure','frontier','fraud','graph'}:
        pos=[(90,art+5),(510,art+5),(510,art+170),(90,art+170)]
        for a,b in [(0,1),(1,2),(2,3),(3,0)]:
            d.line([(pos[a][0]+80,pos[a][1]+35),(pos[b][0]+80,pos[b][1]+35)],accent,3)
        for i,(x,y) in enumerate(pos):box(x,y,160,70,f'Rank {i}')
    elif kind in {'gather','scatter','chunks','alltoall'}:
        for rank in range(4):
            d.text(38,art+38+rank*57,f'R{rank}',20,'#c3b6e1')
            for c in range(8):d.rect(105+c*77,art+10+rank*57,68,40,fill,pale if c%4==rank else '#655474',5)
    else:
        box(260,art,240,74,'COORDINATE')
        for i in range(3):
            x=45+i*239;d.line([(380,art+74),(380,art+120),(x+95,art+120),(x+95,art+160)],accent,2.5)
            box(x,art+160,190,80,f'GPU {i}')
    footer=art+304
    d.text(32,footer,'TOPOLOGY · OWNERSHIP · COORDINATION',18,accent,600)
    d.text(32,footer+43,'Original conceptual chapter artwork',13,'#a89bbd')
    d.height=int(footer+69);d.parts[0]=f'<rect width="760" height="{d.height}" fill="#18152b"/>'
    (IMAGES/path).parent.mkdir(parents=True,exist_ok=True)
    d.write(path,title,'Original conceptual artwork for '+key+'. No measured timing or physical product topology is implied.')


def emit_block(key,path,alt):
    return f'<!-- mgpu:{key}:start -->\n[![{alt}]({path})]({path})\n<!-- mgpu:{key}:end -->'


def update_page(page,items,opener_file,insert):
    before=page.read_text();text=before
    relative='../images/' if page.parent!=ROOT/'docs' else 'images/'
    title=page_title(page,text)
    block=emit_block('opener',relative+opener_file,title)
    pattern=r'<!-- mgpu:opener:start -->.*?<!-- mgpu:opener:end -->'
    if re.search(pattern,text,re.S):text=re.sub(pattern,lambda _:block,text,flags=re.S)
    elif (heading:=re.search(r'^# .+$',outside_fences(text),re.M)):
        text=text[:heading.end()]+'\n\n'+block+text[heading.end():]
    else:text=block+'\n\n'+text
    for key,file,alt,section in items:
        block=emit_block(key,relative+file,alt)
        pattern=rf'<!-- mgpu:{re.escape(key)}:start -->.*?<!-- mgpu:{re.escape(key)}:end -->'
        if re.search(pattern,text,re.S):text=re.sub(pattern,lambda _:block,text,flags=re.S)
        elif section is not None:
            anchor=rf'^(## {re.escape(section)}[ \t]+[^\n]*)$'
            matches=list(re.finditer(anchor,outside_fences(text),re.M))
            if len(matches)!=1:raise ValueError(f'Unique section anchor missing: {page} {section}')
            at=matches[0].end();text=text[:at]+'\n\n'+block+text[at:]
        else:
            # General pages: insert before the first second-level heading.
            if (heading:=re.search(r'^## [^\n]+$',outside_fences(text),re.M)):
                at=heading.end();text=text[:at]+'\n\n'+block+text[at:]
            else:text+='\n\n'+block+'\n'
    code=lambda value:re.findall(r'```[^\n]*\n.*?```',value,re.S)
    if code(before)!=code(text):raise ValueError('A source/output fence changed: '+str(page))
    if insert and text!=before:
        if page.read_text()!=before:raise ValueError('Concurrent edit detected: '+str(page))
        page.write_text(text)
    return dict(page=str(page.relative_to(ROOT)),opener=opener_file,
                figures=[dict(key=k,file=f,alt=a,section=sec) for k,f,a,sec in items])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--no-insert',action='store_true');args=parser.parse_args()
    manifest=[]
    for page in sorted((ROOT/'docs').glob('part*/*.md'))+sorted((ROOT/'docs/appendix').glob('*.md')):
        appendix=page.parent.name=='appendix'
        ident=page.name.split('-')[0];specs=APPENDICES[ident] if appendix else CHAPTERS[int(ident)]
        title=page_title(page,page.read_text())
        short=re.sub(r'^(Chapter \d+|Appendix [A-Z]|\d+\.):?\s*','',title)
        key=('APPENDIX '+ident.upper()) if appendix else ('CHAPTER '+ident)
        opath=f'chapter-openers/{"appendix-" if appendix else ""}{ident}.svg'
        opener(opath,key,short,specs[0][1],specs[0][3])
        items=[]
        for sec,kind,ftitle,labels,note in specs:
            prefix=ident.upper() if appendix else str(int(ident))
            section=prefix+'.'+sec
            filename=f'technical/{"appendix-" if appendix else "ch"}{ident}/{sec}-{kind}.svg'
            f=Figure(key+' / '+prefix+'.'+sec,ftitle)
            bottom=draw(f,kind,labels)
            (IMAGES/filename).parent.mkdir(parents=True,exist_ok=True)
            f.finish(filename,ftitle,note,bottom)
            items.append(('figure-'+sec,filename,ftitle,section))
        manifest.append(update_page(page,items,opath,not args.no_insert))
    for name,title,kind,labels,note in [
        ('index.md','From one device to a coordinated system','flow',['Own local state','Establish communication','Order dependencies','Verify distributed result'],'The book connects memory ownership, communication and correctness before performance.'),
        ('getting-started.md','Separate build checks from hardware execution','flow',['Check tools','Discover devices','Inspect topology','Run reference checks'],'A successful host simulation or compilation does not validate a real multi-GPU transport.')]:
        page=ROOT/'docs'/name;key='FIELD GUIDE' if name=='index.md' else 'GETTING STARTED'
        opath='chapter-openers/'+name.replace('.md','.svg');opener(opath,key,'Multi-GPU Programming in CUDA C++' if name=='index.md' else 'From installation to a checked result',kind,labels)
        if name=='index.md':
            proofs.cover()
            (IMAGES/'00-cover.svg').replace(IMAGES/opath)
        filename='technical/'+('book-overview.svg' if name=='index.md' else name.replace('.md','.svg'));f=Figure(key,title);bottom=draw(f,kind,labels);f.finish(filename,title,note,bottom)
        if name=='index.md':
            from book_overview import build as build_book_overview, TITLE
            build_book_overview()
            title=TITLE
        manifest.append(update_page(page,[('figure-overview',filename,title,None)],opath,not args.no_insert))
    for path in IMAGES.rglob('*.svg'):
        # Portable marker color: Cairo and non-Chromium viewers may not support context-stroke.
        value=path.read_text().replace('fill="context-stroke"','fill="#686478"');path.write_text(value)
    report=dict(version=1,canvas_width=760,minimum_body_type=20,font='Inter / SIL OFL',
                policy='Original conceptual diagrams; code and captured-output fences preserved.',pages=manifest)
    (ROOT/'tools/illustrations/manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Generated {len(manifest)} openers + {sum(len(x["figures"]) for x in manifest)} technical SVGs. Inserted: {not args.no_insert}')

if __name__=='__main__':main()

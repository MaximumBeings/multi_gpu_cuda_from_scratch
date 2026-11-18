#!/usr/bin/env python3
"""Original Multi-GPU CUDA book visual proofs. Does not edit chapter Markdown.

Proof phase: cover + topology. Next families: collective chunk flow, stream/event
swimlanes, ownership/halo layouts, partitioned matrices, pipeline schedules.
Keep captured output and code fences intact. All diagrams fit a 760px canvas;
body labels use >=20px. Graphics express relationships rather than prose cards.
"""
from pathlib import Path
from html import escape
import re
from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/images/proofs'
STYLE = (ROOT/'tools/illustrations/font-style.svg').read_text()
LICENSE = (ROOT/'tools/assets/OFL.txt').read_text()
INK='#27253f'; MUTED='#686478'; PLUM='#6643a4'; CORAL='#b54738'
LILAC='#f0ebf8'; PEACH='#fff0e9'; PAPER='#fffdfa'; LINE='#ddd8e7'
FONT=ROOT/'tools/assets/Inter.ttf'

class Drawing:
    def __init__(self,height,dark=False):
        self.height=height;self.parts=[]
        self.rect(0,0,760,height,'#18152b' if dark else PAPER,'none',0)
    def rect(self,x,y,w,h,fill=PAPER,stroke=LINE,r=12):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    def text(self,x,y,s,size=20,color=INK,weight=450,anchor='start'):
        s=s.replace('→','to')
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(s)}</text>')
    def paragraph(self,x,y,s,width,size=20,color=INK,leading=28):
        s=s.replace('→','to')
        font=ImageFont.truetype(str(FONT),size*10)
        line=''
        for word in s.split():
            proposal=(line+' '+word).strip()
            if line and font.getlength(proposal)/10>width:
                self.text(x,y,line,size,color);y+=leading;line=word
            else:line=proposal
        if line:self.text(x,y,line,size,color);y+=leading
        return y
    def line(self,points,color=PLUM,sw=2.5,arrow=False,dash=False):
        d='M'+' L'.join(f'{x},{y}' for x,y in points)
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round"'+(' marker-end="url(#arrow)"' if arrow else '')+(' stroke-dasharray="6 6"' if dash else '')+'/>')
    def node(self,x,y,w,h,title,sub=None,accent=PLUM):
        self.rect(x,y,w,h,LILAC if accent==PLUM else PEACH)
        self.text(x+w/2,y+(h/2+7 if not sub else 31),title,22,accent,650,'middle')
        if sub:self.text(x+w/2,y+60,sub,20,INK,450,'middle')
    def write(self,name,title,desc):
        OUT.mkdir(parents=True,exist_ok=True)
        text=f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="{self.height}" viewBox="0 0 760 {self.height}" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>
<metadata id="font-license">{escape(LICENSE)}</metadata>{STYLE}
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="context-stroke"/></marker></defs>
<g font-family="BookDiagram,Inter,Arial,sans-serif">{''.join(self.parts)}</g></svg>'''
        (OUT/name).write_text(text)


def cover():
    d=Drawing(860,True)
    d.text(40,48,'FIELD GUIDE / CUDA C++',16,'#c3b6e1',650)
    d.text(40,112,'MULTI-GPU',52,'#f7f3ff',750)
    d.text(40,174,'FROM SCRATCH',45,'#f7f3ff',750)
    d.paragraph(40,217,'From peer memory to collectives. From one node to a cluster.',650,23,'#d5cae8',32)
    # Architectural artwork: two planes, connected by an explicit bridge.
    for x,label in [(42,'NODE A'),(410,'NODE B')]:
        d.rect(x,324,308,294,'#221d38','#514666',20)
        d.text(x+22,358,label,17,'#c3b6e1',650)
        for dx,dy,num in [(26,66,0),(170,66,1),(26,188,2),(170,188,3)]:
            d.rect(x+dx,324+dy,112,76,'#33294e','#b8a1e1',12)
            d.text(x+dx+56,371+dy,f'GPU {num}',21,'#f7f3ff',600,'middle')
        d.line([(x+138,428),(x+170,428)],'#b8a1e1')
        d.line([(x+138,550),(x+170,550)],'#b8a1e1')
        d.line([(x+82,466),(x+82,512)],'#b8a1e1')
        d.line([(x+226,466),(x+226,512)],'#b8a1e1')
    d.line([(350,472),(410,472)],'#f0a08b',4)
    d.line([(350,484),(410,484)],'#f0a08b',4)
    d.text(380,680,'MOVE BYTES. PRESERVE OWNERSHIP.',21,'#f0a08b',650,'middle')
    d.paragraph(40,753,'Topology · Synchronization · Parallel algorithms',680,23,'#d5cae8',32)
    d.text(40,822,'ORIGINAL SCHEMATIC / MULTI-GPU CUDA FIELD GUIDE',13,'#a89bbd')
    d.write('00-cover.svg','Multi-GPU from scratch','Original cover artwork: two conceptual GPU nodes joined across a network. Decorative schematic, not a product topology or measured system.')


def topology():
    d=Drawing(1040)
    d.text(28,31,'MULTI-GPU FROM SCRATCH / CHAPTER 02 / FIGURE PROOF',13,MUTED,650)
    d.line([(28,46),(732,46)],LINE,1)
    d.text(28,88,'Connectivity is a graph',34,INK,700)
    d.paragraph(28,128,'Identify the route between a specific pair. A capability query does not name the physical link.',700,20,MUTED)
    d.text(28,223,'01 / SHARED PCIe SWITCH',18,PLUM,650)
    d.node(28,253,178,90,'GPU 0','Local memory')
    d.node(291,253,178,90,'PCIe switch',accent=CORAL)
    d.node(554,253,178,90,'GPU 1','Local memory')
    d.line([(208,298),(287,298)],CORAL)
    d.line([(473,298),(550,298)],CORAL)
    d.text(380,382,'Peer route depends on platform support and topology.',20,MUTED,450,'middle')
    d.line([(28,413),(732,413)],LINE,1)
    d.text(28,456,'02 / DIRECT LINK',18,PLUM,650)
    d.node(86,487,190,90,'GPU 0','Local memory')
    d.node(484,487,190,90,'GPU 1','Local memory')
    d.line([(280,527),(480,527)],PLUM,3)
    d.line([(280,537),(480,537)],PLUM,3)
    d.text(380,510,'NVLink',20,PLUM,550,'middle')
    d.text(380,615,'Direct links exist only where the system wires them.',20,MUTED,450,'middle')
    d.line([(28,646),(732,646)],LINE,1)
    d.text(28,690,'03 / SWITCHED FABRIC',18,PLUM,650)
    d.node(260,716,240,70,'NVSwitch fabric')
    centers=[113,291,469,647]
    for i,x in enumerate(centers):
        d.line([(x,830),(x,809),(380,809),(380,790)],PLUM,2.5)
        d.node(x-78,834,156,70,f'GPU {i}')
    d.text(380,947,'Shared fabric ≠ guaranteed application bandwidth.',20,MUTED,450,'middle')
    d.line([(28,980),(732,980)],LINE,1)
    d.text(28,1011,'Conceptual paths; no product bandwidth or measured throughput is implied.',15,MUTED)
    d.write('02-topology.svg','Connectivity is a graph','Three conceptual interconnect arrangements: a PCIe switch between GPUs, a direct NVLink pair, and four GPUs connected to an NVSwitch fabric. Capability and physical route are distinct.')

if __name__=='__main__':
    cover();topology()
    print('Wrote two original SVG proofs; no chapter Markdown changed.')

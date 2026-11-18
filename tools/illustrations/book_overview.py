#!/usr/bin/env python3
"""Reproduce the homepage reading map using the book's licensed SVG system."""
import math
from PIL import ImageFont
import proofs
from proofs import Drawing, ROOT, FONT, INK, MUTED, LINE, PLUM, CORAL, PAPER

TITLE = 'From one GPU to a distributed application'
DESCRIPTION = ('Chapters 1–3 establish hardware topology and device ownership. Chapters 4–7 move and synchronize data; chapters 8–11 build collectives. Chapters 12–16 choose a parallel decomposition. Chapters 17–19 handle synchronization, imbalance and failures; chapters 20–23 extend communication beyond one node. Chapters 24–31 develop workloads and chapters 32–40 examine industrial systems. Appendices A–G cover installation, libraries, profiling, framework mappings and troubleshooting. At every scale, verify ownership, ordering and results before measuring performance.')
COLORS = [(PLUM,'#f0ebf8'),('#197d76','#e6f4f0'),(CORAL,'#fff0e9'),('#946b18','#fff3d8')]


def lines(text,width,size):
    font=ImageFont.truetype(str(FONT),size*100)
    out=[];line=''
    for word in text.split():
        proposed=(line+' '+word).strip()
        if line and font.getlength(proposed)/100>width:
            out.append(line);line=word
        else:line=proposed
    if line:out.append(line)
    return out


def build():
    d=Drawing(1800)
    d.text(28,30,'MULTI-GPU FROM SCRATCH / THE WHOLE BOOK / READING MAP',13,MUTED,650)
    d.line([(28,46),(732,46)],LINE,1)
    y=84
    for line in lines(TITLE,704,29):d.text(28,y,line,29,INK,650);y+=36
    for line in lines('Forty chapters, seven appendices: understand the route, divide the work, coordinate the result.',704,22):d.text(28,y+10,line,22,MUTED);y+=29
    y+=46

    def height(title,body,w):
        return 42+len(lines(title,w-54,22))*29+len(lines(body,w-40,22))*29

    def card(x,yy,w,h,title,body,color):
        accent,tint=COLORS[color]
        d.rect(x,yy,w,h,tint,LINE,14)
        d.rect(x+18,yy+22,4,22,accent,'none',2)
        ty=yy+39
        for line in lines(title,w-54,22):d.text(x+34,ty,line,22,accent,650);ty+=29
        for line in lines(body,w-40,22):d.text(x+20,ty+9,line,22,INK);ty+=29

    def row(items):
        nonlocal y
        w=704 if len(items)==1 else 342
        h=max(height(t,b,w) for t,b,c in items)
        for x,(title,body,color) in zip((28,390),items):card(x,y,w,h,title,body,color)
        y+=h

    def arrow():
        nonlocal y
        d.line([(380,y+4),(380,y+27)],MUTED,2,True);y+=36

    row([('01–03  Know the machine and its owners',
          'Capacity, PCIe, NVLink, NVSwitch and the CUDA device model.',0)])
    arrow()
    row([('04–07  Move data safely','Peer access, copies, streams, events and inter-process handoffs.',1),
         ('08–11  Build collectives','Broadcast, reduce, rings, all-to-all and the NCCL library.',0)])
    arrow()
    row([('12–16  Divide the workload',
          'Data, model, tensor and pipeline parallelism; domain decomposition and halo exchange.',0)])
    arrow()
    row([('17–19  Coordinate progress','Synchronization, heterogeneous load balance, stragglers and failures.',2),
         ('20–23  Cross the node boundary','MPI, CUDA-aware transport, RDMA, NVSHMEM and CUDA graphs.',1)])
    arrow()
    row([('24–31  Build applications','GEMM, training, LLM inference, N-body, BFS, Jacobi, rendering and Monte Carlo.',1),
         ('32–40  Examine real systems','MoE, serving, recommendations, weather, finance, fraud, genomics and imaging.',0)])
    arrow()
    row([('A–G  Install, inspect and troubleshoot',
          'Setup, quizzes, libraries, graphs, profiling, framework mappings and failure diagnosis.',3)])
    y+=24
    row([('The invariant at every scale',
          'Track ownership and dependencies. Check distributed results against a reference. Measure the actual route and critical path.',0)])
    y+=30
    d.line([(28,y),(732,y)],LINE,1)
    y+=29;d.text(28,y,'Conceptual reading map · Communication diagrams are not measured topology.',14,MUTED)
    d.height=math.ceil(y+24)
    d.parts[0]=f'<rect width="760" height="{d.height}" fill="{PAPER}"/>'
    proofs.OUT=ROOT/'docs/images/technical'
    d.write('book-overview.svg',TITLE,DESCRIPTION)
    p=proofs.OUT/'book-overview.svg';p.write_text(p.read_text().replace('fill="context-stroke"','fill="#686478"'))
    print(f'Created {p.relative_to(ROOT)} (760 × {d.height}).')

if __name__=='__main__':build()

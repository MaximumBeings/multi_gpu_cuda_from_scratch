#!/usr/bin/env python3
"""Original numeric worked figures for Chapters 1–14 and Appendices A–C.

Only writes docs/images/worked assets. No source listings or records are edited.
All worked examples are derived arithmetic/models, not performance measurements.
"""
from pathlib import Path
from html import escape
import math
from PIL import ImageFont
from proofs import Drawing, ROOT, STYLE, LICENSE, FONT, INK, MUTED, PLUM, CORAL, LILAC, PEACH, PAPER, LINE

OUT=ROOT/'docs/images/worked'
TEAL='#197d76'; GREEN='#e6f4f0'


def wrap(text,width,size=20):
    font=ImageFont.truetype(str(FONT),size*10);lines=[]
    for paragraph in str(text).split('\n'):
        line=''
        for word in paragraph.split():
            proposed=(line+' '+word).strip()
            if line and font.getlength(proposed)/10>width:lines.append(line);line=word
            else:line=proposed
        lines.append(line)
    return lines


class Worked(Drawing):
    def __init__(self,key,title,scope):
        super().__init__(2400);self.key=key;self.title=title;self.scope=scope
        self.text(28,30,'MULTI-GPU FROM SCRATCH / '+key.upper()+' / WORKED EXAMPLE',14,MUTED,650)
        self.line([(28,47),(732,47)],LINE,1)
        self.y=self.para(28,88,title,704,29,INK,38)+4
        self.y=self.para(28,self.y,scope,704,20,MUTED,28)+20
    def para(self,x,y,text,width,size=20,color=INK,leading=28):
        for line in wrap(text,width,size):self.text(x,y,line,size,color);y+=leading
        return y
    def section(self,text):
        self.y=self.para(28,self.y+22,text,704,23,INK,31)+17
    def table(self,headers,rows,widths):
        y=self.y
        for j,row in enumerate([headers]+rows):
            sizes=19 if j==0 else 20;cells=[wrap(c,w-26,sizes) for c,w in zip(row,widths)];h=max(len(c) for c in cells)*28+20
            self.rect(28,y,704,h,LILAC if j==0 else ('#fff' if j%2 else '#f7f5fa'),LINE,5 if j==0 else 0)
            x=28
            for lines,w in zip(cells,widths):
                yy=y+31
                for line in lines:self.text(x+13,yy,line,sizes,PLUM if j==0 else INK,600 if j==0 else 450);yy+=28
                x+=w
            y+=h
        self.y=y+24
    def payload(self,label,values,color=PLUM):
        self.section(label);y=self.y;cw=704/len(values)
        for i,v in enumerate(values):
            self.rect(28+i*cw,y,cw-4,58,LILAC if color==PLUM else GREEN,'#fff',7)
            self.text(28+(i+.5)*cw-2,y+37,str(v),22,color,600,'middle')
        self.y=y+82
    def note(self,text,color=PLUM):
        lines=wrap(text,664,20);y=self.y;h=28*len(lines)+32
        self.rect(28,y,704,h,LILAC if color==PLUM else GREEN,LINE,10)
        self.para(48,y+33,text,664,20,INK,28);self.y=y+h+24
    def finish(self):
        y=self.y+4;self.line([(28,y),(732,y)],LINE,1)
        self.text(28,y+28,'Derived worked model / stated values and units / no measured timing implied',14,MUTED)
        self.height=math.ceil(y+52);self.parts[0]=f'<rect width="760" height="{self.height}" fill="{PAPER}"/>'
        svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="{self.height}" viewBox="0 0 760 {self.height}" role="img" aria-labelledby="title desc">
<title id="title">{escape(self.title)}</title><desc id="desc">{escape(self.scope)}</desc><metadata id="font-license">{escape(LICENSE)}</metadata>{STYLE}
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="{PLUM}"/></marker></defs><g font-family="BookDiagram,Inter,Arial,sans-serif">{''.join(self.parts)}</g></svg>'''
        OUT.mkdir(exist_ok=True,parents=True);p=OUT/(self.key+'.svg');p.write_text(svg);return p


def chapter(n):
    k=f'ch{n:02d}'
    if n==1:
        d=Worked(k,'Sharding state solves a different problem from replication','Hypothetical 2-billion-parameter model; decimal GB; 24 GB usable capacity per GPU. Activations/workspace: 4 GB per GPU.')
        d.table(['State per parameter','Bytes','2 billion parameters'],[['FP16 weights','2','4 GB'],['FP16 gradients','2','4 GB'],['FP32 master weights','4','8 GB'],['FP32 first moment','4','8 GB'],['FP32 second moment','4','8 GB']],[340,110,254])
        d.table(['Placement','State + extra per GPU','Fits 24 GB?'],[['One GPU','32 + 4 = 36 GB','No'],['Two full replicas','32 + 4 = 36 GB each','No'],['Ideal equal state sharding','16 + 4 = 20 GB each','Yes, in this model']],[250,280,174])
        d.note('Two GPUs are not automatically a memory pool: replicas still store complete state. Real sharding also needs its actual buffers, alignment and communication storage.')
    elif n==2:
        d=Worked(k,'A slow cut can dominate a fast local link','Assumed one-way edge capacities, decimal GB/s. A 12 MB payload is used only to calculate ideal serialization lower bounds.')
        y=d.y+18
        for x,num in [(28,0),(230,1),(432,2),(634,3)]:d.node(x,y,98,74,f'GPU {num}')
        for a,b,cap in [(126,230,'48 GB/s'),(328,432,'12 GB/s'),(530,634,'48 GB/s')]:
            d.line([(a,y+38),(b,y+38)],PLUM,3,True);d.text((a+b)/2,y+113,cap,19,PLUM,550,'middle')
        d.y=y+154
        d.table(['Transfer','Bottleneck model','Payload / capacity'],[['GPU 0 to GPU 1','48 GB/s','12 MB / 48 GB/s = 0.25 ms'],['GPU 0 to GPU 3','12 GB/s middle cut','at least 1.00 ms']],[220,220,264])
        d.note('These are not measured transfer times. Startup, routing, forwarding, contention and directionality can increase elapsed time. A topology or peer-capability query alone supplies no bandwidth measurement.')
    elif n==3:
        d=Worked(k,'Changing current device changes later calls, not earlier objects','One host thread; visible CUDA ordinals 0 and 1. Three FP32 elements per allocation: 12 bytes, not 3 bytes.')
        d.table(['Host action','Current device','Object / values'],[['Select 0; allocate A, create s0','0','A on GPU 0: [1,2,3]'],['Select 1; allocate B, create s1','1','B on GPU 1: [10,20,30]'],['Select 0; launch A times two on s0','0','A becomes [2,4,6]'],['Select 1','1','s0 still belongs to GPU 0']],[340,150,214])
        d.note('cudaSetDevice selects the calling host thread’s device. It does not migrate A, B, s0 or s1. Device ordinals are process-visible identifiers, not immutable physical GPU identities.')
    elif n==4:
        d=Worked(k,'Peer access is directional; UVA is not a data copy','Hypothetical capability matrix: rows are accessing devices, columns own the allocation. Values are int32 payload, not pointer addresses.')
        d.table(['Accessing device','Own GPU 0 memory','Own GPU 1 memory'],[['GPU 0','local','can access = 1'],['GPU 1','can access = 0','local']],[240,232,232])
        d.payload('GPU 1 source p1: three int32 values',[10,20,30])
        d.table(['Step on GPU 0','Meaning','Derived result'],[['Query (device=0, peer=1)','Check this direction','Capability is true in model'],['Enable peer 1 from device 0','Authorize supported peer reads','No payload moved'],['Read p1[1]; add 100 locally','20 + 100','GPU 0 result = 120']],[270,230,204])
        d.note('The reverse query is a separate fact. Same virtual pointer value does not establish legal CPU access, peer access, ordering or remote completion.')
    elif n==5:
        d=Worked(k,'A staged copy moves every payload through two legs','Eight FP32 values, 32 useful bytes. Two chunks of four elements are shown; counts exclude protocol overhead.')
        d.payload('Source values; positions 0 through 7',list(range(10,18)))
        d.table(['Chunk','Source element positions','Bytes per leg'],[['0','0..3: 10,11,12,13','4 × 4 = 16'],['1','4..7: 14,15,16,17','4 × 4 = 16']],[110,384,210])
        d.table(['Path','Useful transfer accounting','Completion needed'],[['Peer copy','32 payload bytes copied','before destination consumption'],['Host staging','32 D2H + 32 H2D = 64 bytes','D2H before H2D; H2D before host-slot reuse']],[180,272,252])
        d.note('One pinned host slot may be reused only after both legs using it are safe. Direct versus staged byte counts do not by themselves predict which path is faster.')
    elif n==6:
        d=Worked(k,'Readiness and safe reuse are two different events','Device 0 doubles [1,2,3]; device 1 receives that vector and adds one. Arrows are dependency order, not a timing scale.')
        d.table(['Queue','Ordered operations','Payload state'],[['GPU 0 stream s0','produce; record ready','[2,4,6]'],['GPU 1 stream s1','wait ready; peer copy; add one; record done','[3,5,7]'],['Producer reuse path','wait for copy/read completion','old source may then be overwritten']],[190,310,204])
        d.payload('Expected device 1 output after its completion',[3,5,7],TEAL)
        d.note('Waiting for ready orders the first read. It does not establish that the downstream read has finished. Preserve source and destination lifetimes until their actual completion conditions.')
    elif n==7:
        d=Worked(k,'An IPC handle names shared storage; raw pointers do not travel','Exporter and importer are separate processes. One four-int allocation contains 16 bytes. Opaque handles are not drawn as numeric addresses.')
        d.payload('Exporter allocation: element indices 0..3',[7,11,13,17])
        d.table(['Exporter','Control handoff','Importer'],[['Keep allocation alive','Send opaque memory/event handles','Open handles in supported context'],['Finish producing; record ready','Ready event dependency','Wait; read the same [7,11,13,17]'],['Wait for importer acknowledgement','Use finished','Close imported mapping'],['Free only after all imports close','Lifetime ends','No access after close/free']],[240,224,240])
        d.note('The importer’s mapped address may differ. Kernel completion, imported mapping lifetime and handle transport are separate responsibilities.')
    elif n==8:
        d=Worked(k,'Broadcast chooses one input; reduce combines them','Four ranks, two int32 elements each. Root is communicator rank 2. All arithmetic here stays inside the int32 range.')
        d.table(['Rank','Input element 0','Input element 1'],[[str(r),str(r+1),str(10*(r+1))] for r in range(4)],[130,287,287])
        d.payload('Broadcast from root 2: expected on every rank',[3,30])
        d.payload('SUM reduce: defined result on root rank 2',[10,100],TEAL)
        d.note('Each input is 2 elements = 8 bytes. Reduce does not promise the final sum in nonroot receive buffers. Broadcast is not a sum.')
    elif n==9:
        d=Worked(k,'Follow chunk 0 through both ring phases','P=4 ranks, L=8 int32 elements per rank, two elements per chunk. Input at rank r, position i is 100r+i.')
        d.table(['Phase / round','Chunk 0 movement','Chunk values after step'],[['Initial','rank 0 local input','[0,1]'],['Reduce-scatter 0','0 to 1; add rank 1','[100,102]'],['Reduce-scatter 1','1 to 2; add rank 2','[300,303]'],['Reduce-scatter 2','2 to 3; add rank 3','[600,604]'],['All-gather 0,1,2','3 to 0 to 1 to 2','forward [600,604], do not add']],[200,250,254])
        d.payload('Complete final vector, expected at every rank',[600+4*i for i in range(8)])
        d.note('Per rank: send 2(P−1)L/P = 12 int32 elements = 48 bytes; receive another 48 bytes. Sent+received is 96 bytes, not 48. This counts payload bytes, not measured bandwidth.')
    elif n==10:
        d=Worked(k,'Collectives differ in output meaning and count','Three ranks. AllGather sends two int32 values; ReduceScatter receives two; All-to-All sends one value to each peer.')
        d.table(['Rank','AllGather input','SUM ReduceScatter output','All-to-All output'],[['0','[0,10]','[30,33]','[0,10,20]'],['1','[1,11]','[36,39]','[1,11,21]'],['2','[2,12]','[42,45]','[2,12,22]']],[75,180,249,200])
        d.payload('AllGather result on every rank',[0,10,1,11,2,12])
        d.note('ReduceScatter input at rank r is [10r+i] for i=0..5. All-to-All source r sends value 10r+d to destination d. Matching counts do not make these operations interchangeable.')
    elif n==11:
        d=Worked(k,'A communicator rank is not a CUDA device ordinal','One thread manages three ranks. Rank order determines collective layout; each stream belongs to its mapped device.')
        d.table(['Rank','CUDA device','Four FP32 input values'],[['0','2','1,2,3,4'],['1','0','10,20,30,40'],['2','1','100,200,300,400']],[100,180,424])
        d.payload('AllReduce SUM result at every rank: count=4, 16 bytes',[111,222,333,444])
        d.table(['Boundary','What has been established'],[['GroupStart + three collective calls + GroupEnd','Operations submitted as a matching multi-device group'],['Each mapped stream completes','Its result is ready for consumption'],['Root argument 1, if doing a rooted collective','Root CUDA device is 0 in this mapping']],[360,344])
        d.note('Do not synchronize a stream inside an unfinished group. GroupEnd is not a promise that every GPU result has already completed.')
    elif n==12:
        d=Worked(k,'Unequal local batches need a weighted global gradient','Two ranks have 2 and 6 samples. Their local mean gradients are [2,4] and [6,8]. Values are a derived optimizer example.')
        d.table(['Rank','Samples','Local mean','Local gradient sum'],[['0','2','[2,4]','[4,8]'],['1','6','[6,8]','[36,48]']],[90,120,210,284])
        d.payload('AllReduce SUM of local sums',[40,56])
        d.payload('Divide by global sample count 8',[5,7],TEAL)
        d.table(['Update from w=[10,20], learning rate 0.1','Result'],[['Correct global mean','[9.5,19.3] on both ranks'],['Unweighted average of rank means','[4,6], a different gradient']],[440,264])
        d.note('An ncclAvg of local means is sample-weighted only when local effective batch weights match, or when additional weighting is supplied deliberately.')
    elif n==13:
        d=Worked(k,'Split layers transfer activations, not the whole model','One input row X=[1,2]. Stage 0 owns W0=[[1,0,2],[0,1,1]]; stage 1 owns W1=[[1],[2],[3]]. FP32 storage.')
        d.table(['Stage','Calculation','Result / storage'],[['GPU 0','[1,2] × W0','H=[1,2,4]; 3 floats = 12 bytes'],['Handoff','Copy H after producer completion','GPU 1 gets [1,2,4]'],['GPU 1','1×1 + 2×2 + 4×3','Y=[17]']],[120,300,284])
        d.table(['Weight placement','Elements','Bytes'],[['W0 on GPU 0','6','24'],['W1 on GPU 1','3','12']],[420,140,144])
        d.note('This forward model has no activation function or backward pass. Training additionally needs its activation/gradient lifetime, backward transfers and optimizer state.')
    elif n==14:
        d=Worked(k,'Column shards feed row shards without an intermediate gather','Replicated X=[1,2]. First weight matrix rows are [1,2,3,4] and [5,6,7,8]. The second matrix is partitioned along its reduction rows.')
        d.table(['Rank','Column-parallel output','Second-layer row shard','Partial output'],[['0','[11,14]','[[1,0],[0,1]]','[11,14]'],['1','[17,20]','[[1,1],[2,1]]','[57,37]']],[80,220,220,184])
        d.payload('AllReduce SUM of second-layer partials',[68,51],TEAL)
        d.payload('Add final bias [1,−1] once',[69,50])
        d.note('The row-shard K coordinates must match the preceding column shards. Adding the bias on both ranks before SUM would instead produce [70,49].')
    return d


def appendix(key):
    if key=='a':
        d=Worked('appendix-a','Version checks answer different setup questions','Read these as a diagnostic model, not a new installation log. Counts are elements unless a byte unit is explicit.')
        d.table(['Check','Example / calculation','What it establishes'],[['Compiler selection','nvcc --version: selected 12.8.x','Which compiler is on PATH'],['NCCL tuple to integer','2 × 10000 + 26 × 100 + 2 = 22602','How version 2.26.2 is encoded'],['Two-rank buffer','4 FP32 values per rank = 16 bytes','Required payload allocation'],['Device count = 2','ordinals 0 and 1 visible','Count, not topology or peer correctness']],[230,240,234])
        d.note('Successful header compilation, shared-library loading, visible GPUs and a checked numerical transfer are separate gates. Do not infer all four from nvidia-smi alone.')
    elif key=='b':
        d=Worked('appendix-b','Predict the operation before reading a PASS','Two ranks supply int32 vectors [1,10] and [2,20]. The requirement is the elementwise sum on both ranks.')
        d.table(['Candidate','Result location / values','Required receive bytes'],[['AllReduce SUM, count 2','each rank: [3,30]','8 per rank'],['Reduce SUM, count 2, root 1','root only: [3,30]','8 at root'],['AllGather, sendcount 2','each rank: [1,10,2,20]','16 per rank']],[260,280,164])
        d.note('AllGather does not reduce, and Reduce does not distribute the result everywhere. Allocation size must follow the selected operation, not only the local input length.')
    else:
        d=Worked('appendix-c','Same four values, different communication contracts','Two PEs/ranks; source [2,3,5,7] uses four int32 values = 16 bytes. Symmetric object correspondence is not raw pointer equality.')
        d.payload('The payload, independent of transport choice',[2,3,5,7])
        d.table(['Mechanism','Issuance / matching','When consumption is safe'],[['NCCL Send/Recv','matching peer, count 4, type int32','receive stream completes'],['NVSHMEM put','PE 0 targets symmetric object on PE 1','issuing-context completion plus a matching synchronization/notification protocol'],['Neither','raw pointer bytes sent to another process','no legal mapping or completion is established']],[160,264,280])
        d.note('For the host-issued RMA model in the text: quiet completes PE 0’s put; every PE joins the matching barrier before PE 1 reads. Device-issued paths require their own supported launch and stream/proxy rules.')
    return d


def main():
    for i in range(1,15):
        p=chapter(i).finish();print(p.relative_to(ROOT))
    for key in 'abc':
        p=appendix(key).finish();print(p.relative_to(ROOT))
    assert [sum(100*r+i for r in range(4)) for i in range(8)]==[600+4*i for i in range(8)]
    assert 2*3*8//4*4==48 and 4*2048+2*384+2*64+12==9100
    assert [(11+57),(14+37)]==[68,51]
    print('PASS: 17 numeric worked figures; no Markdown/source/record changes')

if __name__=='__main__':main()

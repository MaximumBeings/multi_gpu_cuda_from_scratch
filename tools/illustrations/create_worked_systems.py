#!/usr/bin/env python3
"""Numeric worked systems figures; source programs and execution records stay unchanged."""
from pathlib import Path
from html import escape
import math
import xml.etree.ElementTree as ET
from PIL import ImageFont
from proofs import Drawing, ROOT, STYLE, LICENSE, FONT, INK, MUTED, PLUM, CORAL, LILAC, PEACH, LINE

TEAL = '#14786f'
OUT = ROOT/'docs/images/worked'
SPECS = {
'15': dict(title='Four microbatches through three stages', scope='Forward-only equal-stage schedule. One slot is an abstract work unit, not a measured time.',
 headers=['Stage / slot','0','1','2','3','4','5'], widths=[152,92,92,92,92,92,92],
 rows=[['Stage 0','b0','b1','b2','b3','—','—'],['Stage 1','—','b0','b1','b2','b3','—'],['Stage 2','—','—','b0','b1','b2','b3']],
 second=(['Batch input','After +1','After ×2','After −3'],[[str(x),str(x+1),str(2*(x+1)),str(2*(x+1)-3)] for x in range(1,5)]),
 result='12 useful stage-slots / 18 available = 2/3 utilization. Makespan is 4+3−1=6 slots; outputs retain order [1,3,5,7].',
 pitfall='A forward-only schedule is not a 1F1B training schedule. Transfer costs and unequal stages require a different timeline.'),
'16': dict(title='Two shards, one old-generation halo', scope='One-dimensional three-point averaging, six cells, two ranks. External ghost values are fixed at zero.',
 headers=['Rank','Owned old cells','Required old ghosts'],widths=[100,302,302],
 rows=[['0','indices 0–2: [0,1,2]','left 0; right 3 from rank 1'],['1','indices 3–5: [3,4,5]','left 2 from rank 0; right 0']],
 second=(['Rank','Three local new values','Interface check'],[['0','[1/3,1,2]','(1+2+3)/3=2'],['1','[3,4,3]','(2+3+4)/3=3']]),
 result='Combined new grid: [1/3,1,2,3,4,3]. Every update reads old cells; halo bytes must arrive before boundary updates.',
 pitfall='A missing interface halo changes the answer even though every local address can remain valid.'),
'17': dict(title='A local wait is not a phase boundary', scope='Two-rank stream-ordered readiness reduction. Work lengths 2 and 5 are hypothetical scheduling units.',
 headers=['Rank','Before collective','Ready contribution'],widths=[100,302,302],
 rows=[['0','local preceding work ends at slot 2','1'],['1','local preceding work ends at slot 5','1']],
 second=(['Phase','Rank 0','Rank 1'],[['Local wait only','own stream complete','may still be working'],['Matching sum reduction','1+1=2','1+1=2'],['Completion wait','wait on collective stream','wait on collective stream']]),
 result='Both receive 2 after the matching reduction completes. This orders the named streams, not unrelated work on every stream.',
 pitfall='ncclGroupEnd returning does not mean device completion. Missing participation cannot be repaired with a local cudaDeviceSynchronize.'),
'18': dict(title='Partition work by measured task speed', scope='Planning model: 12 work units, rank speeds [1,2,1] work units per second. Times are calculated, not measured.',
 headers=['Rank','Equal count / time','Weighted count / time'],widths=[100,302,302],
 rows=[['0','4 / 4 s','3 / 3 s'],['1','4 / 2 s','6 / 3 s'],['2','4 / 4 s','3 / 3 s']],
 second=(['Total work','Cumulative boundaries','Counts'],[['12','[0,3,9,12]','[3,6,3]'],['11','[0,2,8,11]','[2,6,3]']]),
 result='At 12 units, modeled makespan improves from 4 s to 3 s. At 11, integer rounding still covers all work once.',
 pitfall='SM counts are not task throughput. Include memory capacity and communication constraints before accepting a weighted partition.'),
'19': dict(title='Count a completed task exactly once', scope='Speculative task model, not a fault-injected NCCL experiment. Durations use a hypothetical common clock.',
 headers=['Attempt','Start / duration','Completion'],widths=[180,262,262],
 rows=[['Task C primary','0 / 9 units','9'],['Task C backup','5 / 3 units','8'],['Accepted result','first valid attempt','C contributes 5 once']],
 second=(['Task A','Task B','Task C','Combined value'],[['2','3','5','10'],['2','3','5+5 (bug)','15 (wrong)']]),
 result='A timeout at 6 does not prove the primary failed. Accepting the backup at 8 requires deduplication before combining results.',
 pitfall='Aborting a failed communicator is not restoration of application state, membership or optimizer state.'),
'20': dict(title='Trace all three MPI ring messages', scope='Host-buffer arithmetic trace: three ranks, two integers per message, distinct send and receive buffers.',
 headers=['Rank','Send to successor','Receive from predecessor'],widths=[100,302,302],
 rows=[['0','[10,11] to rank 1','[30,31] from rank 2'],['1','[20,21] to rank 2','[10,11] from rank 0'],['2','[30,31] to rank 0','[20,21] from rank 1']],
 second=(['Check','Required value'],[['Element count','2 MPI_INT on each side'],['Tag','7 in both matched directions'],['After receive','validate both integers, not just exit status']]),
 result='For rank r: next=(r+1)%3, previous=(r+2)%3. MPI_Sendrecv matches each edge without relying on eager send buffering.',
 pitfall='MPI world rank is not a CUDA device ordinal. Per-process visibility masks renumber the accessible GPU devices.'),
'21': dict(title='Control metadata does not move the payload', scope='A four-integer payload [2,4,6,8], assuming 32-bit integers: 16 bytes. Counts below are logical copy traffic.',
 headers=['Route','Local host-staging bytes','Network payload'],widths=[190,272,242],
 rows=[['Staged GPU to GPU','16 D2H + 16 H2D = 32','16 bytes'],['Supported RDMA path','0 payload staging bytes','16 bytes']],
 second=(['Element index / value','0 / 2','1 / 4','2 / 6','3 / 8'],[['Receiver check','2','4','6','8']]),
 result='Capability=1 is not a received-data check. Validate every element after the transport and consumer-ordering contract completes.',
 pitfall='A CUDA-aware pointer API, NCCL bootstrap success or a device attribute does not prove that this transfer used GPUDirect RDMA.'),
'22': dict(title='A symmetric object receives a ring put', scope='Three-PE paper trace of the checked ring-put protocol. Local virtual addresses need not be numerically equal.',
 headers=['Issuing PE','Target PE','Value written'],widths=[190,242,272],
 rows=[['0','1','777'],['1','2','778'],['2','0','779']],
 second=(['Receiving PE','Final target','Expected predecessor'],[['0','779','2'],['1','777','0'],['2','778','1']]),
 result='Initialize, rendezvous, put and quiet, complete issuing kernels, rendezvous again, then read targets. Every target has one writer.',
 pitfall='Symmetric allocation identifies a corresponding object; it is not permission to send an arbitrary numeric pointer from another PE.'),
'23': dict(title='Replay preserves the fork and the join', scope='Scalar model of the two-device graph: A starts at zero; device 0 increments A, device 1 writes B=A+1.',
 headers=['Replay','A after device 0','B after device 1'],widths=[130,287,287],
 rows=[['1','1','2'],['2','2','3'],['3','3','4']],
 second=(['Dependency','Required relation'],[['Fork event','A write precedes device 1 peer read'],['Join event','device 1 work rejoins the origin stream'],['After replay wait','A=3 and B=4; allocations remain live']]),
 result='A replay executes dependencies again. It does not reset A unless the reset is part of the represented graph.',
 pitfall='Capture does not replace peer access, matching collective participants or valid buffer lifetimes.'),
'24': dict(title='Four SUMMA owners accumulate two panels', scope='Paper model: A is 4×2, B is 2×4, C=AB. A 2×2 process grid gives each rank a 2×2 output tile.',
 headers=['Owner','After k=0','After adding k=1'],widths=[116,280,308],
 rows=[['(0,0)','1 2 / 3 6','21 42 / 43 86'],['(0,1)','3 4 / 9 12','63 84 / 129 172'],['(1,0)','5 10 / 7 14','65 130 / 87 174'],['(1,1)','15 20 / 21 28','195 260 / 261 348']],
 second=(['Input','Rows'],[['A','[1,2]; [3,4]; [5,6]; [7,8]'],['B','[1,2,3,4]; [10,20,30,40]']]),
 result='C[0,0]=1×1+2×10=21; C[3,3]=7×4+8×40=348. Row/column panel broadcasts feed the matching output owners.',
 pitfall='A correct tile shape does not determine its input panels. Resetting C between panels loses the earlier contribution.'),
'25': dict(title='Reduce gradients before updating replicas', scope='Scalar least-squares model: loss=(wx−y)^2/2 per sample, y=x, w=2, equal two-sample batches.',
 headers=['Rank','Local x values','Mean gradient'],widths=[100,302,302],
 rows=[['0','[1,2]','(1+4)/2 = 2.5'],['1','[3,4]','(9+16)/2 = 12.5']],
 second=(['Reduction','Gradient used','New w; learning rate 0.1'],[['Sum then divide by 2','(2.5+12.5)/2=7.5','2−0.1×7.5=1.25'],['Sum without division','15 (wrong convention)','0.5 (different update)']]),
 result='The global four-sample derivative is (1+4+9+16)/4=7.5. Both replicas must apply the same completed reduction.',
 pitfall='Equal-rank averaging is not generally equal-sample averaging when local batch counts differ.'),
'26': dict(title='TP groups stay inside their pipeline stage', scope='Four-rank two-stage linear model: TP=2, PP=2. Input [1,2] is replicated in the first stage.',
 headers=['Rank / stage','Owned computation','Result'],widths=[160,344,200],
 rows=[['0 / stage 0','W rows [1,0]; [0,1]','h0=[1,2]'],['1 / stage 0','W rows [1,1]; [2,1]','h1=[3,4]'],['2 / stage 1','[1,10] dot h0','21'],['3 / stage 1','[100,1000] dot h1','4300']],
 second=(['Pipeline handoff','Within-stage TP reduction'],[['rank 0 to 2: [1,2]','rank 2 + rank 3: 21+4300=4321'],['rank 1 to 3: [3,4]','both stage-1 ranks receive 4321']]),
 result='Rank = stage×2+TP lane. The second-stage sum belongs to communicator {2,3}, not the four-rank world.',
 pitfall='Pipeline membership and tensor-parallel membership are different. A world all-reduce can mix unrelated layer values.'),
'27': dict(title='Each target consumes every source body', scope='Three unit masses at x=[0,1,3], G=1, distinct positions, no softening. This is a one-dimensional force model.',
 headers=['Target / owner','Nonself contributions','Acceleration'],widths=[170,310,224],
 rows=[['x=0 / rank 0','+1 and +1/9','10/9'],['x=1 / rank 1','−1 and +1/4','−3/4'],['x=3 / rank 2','−1/9 and −1/4','−13/36']],
 second=(['Conservation check','Common denominator'],[['Sum of accelerations','40/36−27/36−13/36=0'],['Per-rank source records','3 total: one local, two remote']]),
 result='An all-pairs owner needs the complete source set. A nearest-neighbor halo would incorrectly omit the distant mass.',
 pitfall='Self interaction is excluded; softened or Barnes–Hut forces have a different numerical contract.'),
'28': dict(title='Frontier ownership changes at every level', scope='Directed graph: 0→{1,2}, 1→3, 2→{3,4}, 3→5, 4→5. Owner(v)=floor(v/2), three ranks.',
 headers=['BFS level','Rank 0: {0,1}','Rank 1: {2,3}','Rank 2: {4,5}'],widths=[110,198,198,198],
 rows=[['0','{0}','{}','{}'],['1','{1}','{2}','{}'],['2','{}','{3}','{4}'],['3','{}','{}','{5}'],['next','{}','{}','{}']],
 second=(['Duplicate discovery','Owner action'],[['Vertex 3 from 1 and 2','rank 1 inserts once'],['Vertex 5 from 3 and 4','rank 2 inserts once']]),
 result='Distances are [0,1,1,2,2,3]. Terminate only when the combined next-frontier count is zero across every rank.',
 pitfall='An empty local frontier is not global termination; that rank must still participate in the level protocol.'),
'appendix-d': dict(title='Change a parameter without changing the graph', scope='Two-node scalar graph model: node A adds a constant, node B multiplies by two. Input x=3.',
 headers=['Version','Node A result','Node B result'],widths=[210,247,247],
 rows=[['A adds 1','3+1=4','4×2=8'],['A updated to add 2','3+2=5','5×2=10']],
 second=(['Update choice','Contract'],[['Known kernel-node parameter','use supported executable-node update API'],['Replacement graph description','check cudaGraphExecUpdate and result info'],['Rejected update','use a valid rebuild path; do not ignore failure']]),
 result='The edge A-before-B remains. Compatibility and buffer lifetime are still required; an update is not a new dependency proof.',
 pitfall='Per-device cooperative synchronization is not a cross-device barrier, and deprecated multi-device launch support must not be assumed.'),
'appendix-e': dict(title='Local durations are not the whole makespan', scope='Illustrative common-clock timeline, not measured CUDA event timestamps. Event pairs stay on their owning devices.',
 headers=['Interval','Start / end','Duration'],widths=[220,242,242],
 rows=[['Device 0 local interval','0 to 6 ms','6 ms'],['Device 1 local interval','4 to 7 ms','3 ms'],['Complete host phase','0 to 7 ms','7 ms']],
 second=(['Four-rank 12 MB buffer','Count; decimal MB'],[['Ring sent bytes per rank','2×(3/4)×12=18 MB'],['Effective algorithm rate at 7 ms','12/7=1.714 GB/s'],['Sent-volume rate over same interval','18/7=2.571 GB/s']]),
 result='max(local durations)=6 ms, while modeled phase makespan is 7 ms. Report the chosen scope before deriving a rate.',
 pitfall='Do not subtract raw event timestamps from different GPUs or label sent-plus-received bytes as a one-direction bus rate.'),
}

class Worked(Drawing):
    def __init__(self,key,spec):
        super().__init__(1000)
        self.key=key;self.spec=spec
        self.parts[0]='<rect width="760" height="1000" fill="#fff"/>'
        self.text(28,30,'MULTI-GPU FROM SCRATCH / WORKED '+key.upper(),14,MUTED,650)
        self.line([(28,46),(732,46)],LINE,1)
        self.y=self.paragraph(28,86,spec['title'],704,29,INK,36)+8
        self.y=self.paragraph(28,self.y,spec['scope'],704,20,MUTED,28)+20
    def wrapped(self,s,w,size=20):
        font=ImageFont.truetype(str(FONT),size*10);lines=[];line=''
        for word in str(s).split():
            trial=(line+' '+word).strip()
            if line and font.getlength(trial)/10>w:lines.append(line);line=word
            else:line=trial
        if line:lines.append(line)
        return lines or ['']
    def table(self,headers,rows,widths=None):
        widths=widths or [704/len(headers)]*len(headers)
        heights=[len(self.wrapped(t,w-24,19)) for t,w in zip(headers,widths)]
        h=max(heights)*26+24
        self.rect(28,self.y,704,h,LILAC,LINE,8);x=28
        for t,w in zip(headers,widths):
            self.paragraph(x+12,self.y+30,str(t),w-24,19,PLUM,26);x+=w
        self.y+=h
        for i,row in enumerate(rows):
            h=max(len(self.wrapped(t,w-24)) for t,w in zip(row,widths))*28+24
            self.rect(28,self.y,704,h,'#fff' if i%2 else '#f8f6fb',LINE,0);x=28
            for t,w in zip(row,widths):
                self.paragraph(x+12,self.y+32,str(t),w-24,20,INK,28);x+=w
            self.y+=h
        self.y+=28
    def note(self,title,text,color):
        lines=self.wrapped(text,664);h=54+len(lines)*28
        self.rect(28,self.y,704,h,LILAC if color==PLUM else '#e7f4f0',LINE,12)
        self.text(48,self.y+32,title,22,color,650)
        self.paragraph(48,self.y+65,text,664,20,INK,28);self.y+=h+22
    def save(self):
        self.table(self.spec['headers'],self.spec['rows'],self.spec.get('widths'))
        self.table(*self.spec['second'])
        self.note('Expected result — arithmetic, not a measurement',self.spec['result'],TEAL)
        self.note('Preserve the contract',self.spec['pitfall'],PLUM)
        self.line([(28,self.y),(732,self.y)],LINE,1)
        self.text(28,self.y+29,'Derived worked example · Units, participants and assumptions are part of the result.',14,MUTED)
        self.height=math.ceil(self.y+54);self.parts[0]=f'<rect width="760" height="{self.height}" fill="#fff"/>'
        svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="{self.height}" viewBox="0 0 760 {self.height}" role="img" aria-labelledby="title desc"><title id="title">{escape(self.spec['title'])}</title><desc id="desc">{escape(self.spec['scope']+' '+self.spec['result'])}</desc><metadata id="font-license">{escape(LICENSE)}</metadata>{STYLE}<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="context-stroke"/></marker></defs><g font-family="BookDiagram,Inter,Arial,sans-serif">{''.join(self.parts)}</g></svg>'''
        OUT.mkdir(exist_ok=True,parents=True)
        filename='ch'+self.key+'.svg' if self.key.isdigit() else self.key+'.svg'
        target=OUT/filename;target.write_text(svg)
        root=ET.parse(target).getroot();texts=[e for e in root.iter() if e.tag.endswith('}text')]
        assert root.get('width')=='760' and min(float(e.get('font-size')) for e in texts)>=14
        assert all(0<=float(e.get('x'))<=760 and 0<=float(e.get('y'))<=self.height for e in texts)
        assert 'data:font/woff2;base64,' in svg
        return target

if __name__=='__main__':
    for key,spec in SPECS.items():print(Worked(key,spec).save().relative_to(ROOT))

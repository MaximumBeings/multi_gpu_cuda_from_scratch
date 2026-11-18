#!/usr/bin/env python3
"""Generate numerical ownership diagrams for chapters29–40 and appendicesF/G.

All numbers are paper models, not new execution or industrial performance data.
"""
from pathlib import Path
from html import escape
import xml.etree.ElementTree as ET
from proofs import Drawing, ROOT, STYLE, LICENSE, INK, MUTED, PLUM, CORAL, LINE

OUT=ROOT/'docs/images/worked'
TEAL='#126e66'; MINT='#e6f4ef'
CASES={
'ch29':('One Jacobi sweep, two different norms',
 'Dimensionless 1D stencil model; each rank owns two interior unknowns. Exchange the right generation before reading neighbors.',
 ['Physical endpoint 1','old interior [0,0 | 0,0]','Physical endpoint 0'],
 ['Owner / statistic','Rank 0: unknowns1–2','Rank 1: unknowns3–4'],
 [('Old interface sent','x2=0','x3=0'),('New local iterate','[1,0.5]','[0.5,0.5]'),('True residual entries','[0.5,1.5]','[1,0.5]'),('Local residual sum²','2.5','1.25')],
 'Global residual² = 2.5+1.25 = 3.75; residual norm = sqrt(3.75). Update norm = sqrt(1.75): a different statistic.',
 'Reduction combines squared residuals. The residual needs the new interface; the Jacobi sweep uses old values.'),
'ch30':('Four rendering tiles, two schedules',
 'Hypothetical independent tile costs in time units. The complete immutable scene already fits each worker.',
 ['Tile A:8','Tile B:2','Tile C:6','Tile D:4'],
 ['Schedule','GPU 0 work','GPU 1 work'],
 [('Contiguous','A+B = 10','C+D = 10'),('Alternating','A+C = 14','B+D = 6'),('Compute makespan','max(10,10)=10','max(14,6)=14')],
 'Each tile:1024 RGBA FP32 pixels =16384 bytes. Four output tiles =65536 payload bytes to assemble.',
 'Disjoint complete pixels need assembly, not a sum. Equal tile counts need not mean equal compute cost.'),
'ch31':('Merge means without losing between-group variation',
 'Toy losses in monetary units. M2 is the sum of squared deviations from each group’s mean, not its variance.',
 ['Rank 0 losses [1,3]','Rank 1 losses [5,7]'],
 ['Quantity','Rank 0','Rank 1'],
 [('Count n','2','2'),('Mean mu','2','6'),('Centered sum M2','2','2'),('Between-mean delta','6−2 =4','Correction:16')],
 'Merged n=4; mean=4; M2=2+2+16=20. Population variance=5. Sample variance=20/3.',
 'Nearest-rank75th percentile is5. Mean/M2 do not retain enough information to recover arbitrary quantiles.'),
'ch32':('Route token IDs, then restore their original order',
 'Top-1 paper MoE: H=4 two-byte elements,8 activation bytes per token. No capacity overflow in the initial case.',
 ['E0 receives token IDs0,3','E1 receives token IDs1,2'],
 ['Original token','Expert / toy function','Restored first value'],
 [('0: input1','E0: multiply by2','2'),('1: input2','E1: multiply by3','6'),('2: input3','E1: multiply by3','9'),('3: input4','E0: multiply by2','8')],
 'Source-to-expert token counts:[[1,1],[1,1]]. Four assignments =32 dispatch payload bytes;16 are remote here.',
 'Combine uses recorded token identity. Equal bytes or expert-local order do not establish correct token order.'),
'ch33':('Reshard a request’s KV cache by head',
 'Paper model:S=8 tokens,L=2 layers,2 KV heads,d=4,2-byte elements. Count both K and V.',
 ['Prefill rank0: head0,256B','Prefill rank1: head1,256B'],
 ['Object','Calculation','Logical bytes'],
 [('Full prompt KV','2×8×2×2×4×2','512'),('One head shard','512/2','256'),('One generated token','2×2×2×4×2','64'),('After4 generated tokens','512+4×64','768')],
 'One decode worker needs both shards:256+256=512B, placed by layer/token/head coordinates before decode reads.',
 'A transfer enqueue is not readiness. Metadata, page padding, weights and replicated-head policies are excluded.'),
'ch34':('Fetch the embedding rows each sample requested',
 'Row-sharded toy table:8 rows,D=4 FP32 elements. Each vector is16 bytes; each owner stores4 rows.',
 ['Owner0: IDs0–3','Owner1: IDs4–7'],
 ['Sample owner','Requested IDs','Remote result'],
 [('Rank0','[1,6]','ID6:16 bytes'),('Rank1','[2,7]','ID2:16 bytes'),('Rank0 assembled slot0','ID1','[10,11,12,13]'),('Rank0 assembled slot1','ID6','[60,61,62,63]')],
 'Full weights:128B;64B per shard. Remote lookup return payload:32B before request IDs and metadata.',
 'Read deduplication can fill repeated slots once. Backward must still accumulate every gradient contribution.'),
'ch35':('Transpose a4×4 grid from rows to columns',
 'Paper distributed FFT layout. Each full complex value is8 bytes; each rank initially owns two complete rows.',
 ['Rank0: rows0–1','Rank1: rows2–3'],
 ['Block movement','Values','Payload'],
 [('Rank0 to rank1','Rows0–1,columns2–3:4','32 bytes'),('Rank1 to rank0','Rows2–3,columns0–1:4','32 bytes'),('Rank0 keeps locally','Rows0–1,columns0–1:4','32 bytes'),('Rank1 keeps locally','Rows2–3,columns2–3:4','32 bytes')],
 'After redistribution, rank0 has columns0–1 and rank1 columns2–3. Remote sender payload totals64B.',
 'A fixed narrow halo does not provide every column. Local values do not traverse the inter-rank link.'),
'ch36':('Net within paths, floor, then average each date',
 'Toy trade values in monetary units. Each path has equal weight; both trades share one modeled netting set.',
 ['Rank0: path A','Rank1: path B'],
 ['Date / operation','Path A','Path B'],
 [('Date1 trades','[10,−8]','[−4,6]'),('Date1 exposure','max(2,0)=2','max(2,0)=2'),('Date2 trades','[3,−5]','[7,−1]'),('Date2 exposure','max(−2,0)=0','max(6,0)=6')],
 'EE(date1)=(2+2)/2=2; EE(date2)=(0+6)/2=3. Interval weights1,3 give EPE approximation2.75.',
 'Flooring individual trades first gives a different result. EE/EPE alone are not a CVA model or a portfolio recommendation.'),
'ch37':('Compare complete market-data paths at one boundary',
 'Hypothetical serial batch:8 messages×256 bytes=2048 payload bytes. Times are microseconds, not published measurements.',
 ['Input:8 sequence-tagged messages','Output:GPU consumption complete'],
 ['Staged path component','Time','Cumulative time'],
 [('NIC/host arrival','3 us','3 us'),('H2D setup','2 us','5 us'),('H2D payload','1 us','6 us'),('GPU processing','4 us','10 us')],
 'A separate direct route taking4us to GPU-visible input plus4us processing finishes in8us under this toy model.',
 'Kernel bypass, NIC-to-GPU DMA and CPU GDRCopy mappings are different mechanisms. Count setup and readiness.'),
'ch38':('Deduplicate feature traffic, preserve sampled edges',
 'Paper feature fetch:GPU0 owns IDs0–3;GPU1 owns4–7. D=8 FP32 values gives32 bytes per feature row.',
 ['Sample slots:[1,4,4,7]','Remote unique IDs:{4,7}'],
 ['Logical slot','Fetched owner','Component0'],
 [('Slot0:ID1','GPU0 local','1'),('Slot1:ID4','GPU1,shared fetch','4'),('Slot2:ID4','GPU1,same fetched row','4'),('Slot3:ID7','GPU1','7')],
 'Remote payload:2×32=64B, instead of96B without read deduplication. Sampled sum=1+4+4+7=16; mean=4.',
 'Summing unique IDs gives12, the wrong sampled aggregation. Stable feature versions and duplicate slots remain necessary.'),
'ch39':('Separate core result ownership from context overlap',
 'Workflow model only, not a genomics execution or clinical-quality result. Region coordinates are half-open intervals.',
 ['Worker0 core:[0,50)','Worker1 core:[50,100)'],
 ['Worker','May read context','Emits core-owned results'],
 [('0','[0,52)','[0,50)'),('1','[48,100)','[50,100)'),('Position50','Can be context on0','Emitted by worker1'),('Read-alignment IDs','Worker0:0–4','Worker1:5–9')],
 'Ten read IDs have unique alignment owners. The region context overlaps at positions48–51, but emitted cores do not.',
 'Reference build, record/read-pair boundaries and algorithm-specific context matter. Stage independence is not zero workflow cost.'),
'ch40':('Sum detector contributions; keep voxel ownership',
 'Toy linear operator, dimensionless values. Volume x=[1,2|3,4]; detector rows are[1,1,1,1] and[1,0,1,0].',
 ['Slab0 voxels:[1,2]','Slab1 voxels:[3,4]'],
 ['Operation','Slab0','Slab1'],
 [('Forward partial bins','[3,1]','[7,3]'),('Forward sum by bin','3+7=10','1+3=4'),('Residual input','[2,5] available','[2,5] available'),('Backproject local voxels','[7,2]','[7,2]')],
 'Adjoint check:<Ax,r>=10×2+4×5=40=<x,A^T r>. A4×3 FP32 boundary plane is48B;both directions carry96B.',
 'Projection reduction and later stencil halo exchange are distinct. This model does not certify geometry or image quality.'),
'appendix-f':('A sum collective is not the whole gradient contract',
 'Two equally sized local minibatches; each local gradient is already a mean. SGD learning rate0.1;parameters[10,20].',
 ['Rank0 gradient:[2,4]','Rank1 gradient:[6,8]'],
 ['Step','Calculation','Result'],
 [('Raw SUM collective','[2,4]+[6,8]','[8,12]'),('Global sample mean','[8,12]/2','[4,6]'),('SGD update','[10,20]−0.1×[4,6]','[9.6,19.4]'),('Wrong:divide twice','Mean gradient/2','[2,3]')],
 'Unequal local sample counts1 and3 require(1×[2,4]+3×[6,8])/4=[5,7] for this objective.',
 'DDP defaults, custom hooks and loss scaling are separate from raw all_reduce(SUM). Match the serial objective before translating APIs.'),
'appendix-g':('A wait cycle and a read from the wrong generation',
 'Protocol model:rank0 owns3,rank1 owns7. Correct current exchange gives sum10 on each rank.',
 ['Rank0 waits inRecv(from1)','Rank1 waits inRecv(from0)'],
 ['State','Rank0 sees','Consequence'],
 [('Both receive before send','No matching send posted','Wait cycle'),('Consume stale incoming2','3+2','Wrong result5'),('Then incoming becomes7','A later wait completes','Old result stays5'),('Consume after completion','3+7','Correct result10')],
 'MPI_Sendrecv or correctly matched nonblocking exchange can remove this ordering cycle. Complete dependencies before consumers read.',
 'No message-size trick repairs unmatched receives. Matching counts/tags/order and buffer lifetime remain required.')}


class CaseDrawing(Drawing):
    def __init__(self):
        super().__init__(1800)
        self.y=0
    def heading(self,title,lede):
        self.text(30,32,'MULTI-GPU / WORKED OWNERSHIP AND COST',14,MUTED,650)
        self.line([(30,48),(730,48)],LINE,1)
        self.y=self.paragraph(30,90,title,700,29,INK,38)+10
        self.y=self.paragraph(30,self.y,lede,700,20,MUTED,28)+22
    def boxes(self,items):
        n=len(items);gap=14;w=(700-gap*(n-1))/n
        h=112 if n<4 else 95
        for i,s in enumerate(items):
            x=30+i*(w+gap);self.rect(x,self.y,w,h,'#f0ebf8' if i%2==0 else MINT)
            self.paragraph(x+14,self.y+32,s,w-28,20,PLUM if i%2==0 else TEAL,27)
        self.y+=h+32
    def table(self,headers,rows):
        widths=[220,240,240];x=30;y=self.y
        self.rect(30,y,700,56,'#eae6f2',LINE,8)
        for h,w in zip(headers,widths):self.paragraph(x+12,y+23,h,w-24,19,INK,24);x+=w
        y+=56
        for i,row in enumerate(rows):
            heights=[]
            # Wrap identically without emitting: use font metrics.
            from PIL import ImageFont
            from proofs import FONT
            f=ImageFont.truetype(str(FONT),190)
            for cell,w in zip(row,widths):
                lines=1;line=''
                for word in str(cell).split():
                    candidate=(line+' '+word).strip()
                    if line and f.getlength(candidate)/10>w-24:lines+=1;line=word
                    else:line=candidate
                heights.append(26*lines+22)
            h=max(heights);self.rect(30,y,700,h,'#fff' if i%2 else '#f8f6fc',LINE,0);x=30
            for cell,w in zip(row,widths):self.paragraph(x+12,y+28,str(cell),w-24,19,INK,26);x+=w
            y+=h
        self.y=y+28
    def result(self,text):
        self.rect(30,self.y,700,130,MINT,LINE,10)
        self.text(48,self.y+28,'EXPECTED NUMERIC CHECK',15,TEAL,650)
        self.paragraph(48,self.y+58,text,664,21,INK,29)
        self.y+=156
    def finish(self,name,title,note):
        self.y=self.paragraph(30,self.y,note,700,19,CORAL,27)+20
        self.line([(30,self.y),(730,self.y)],LINE,1)
        self.text(30,self.y+28,'Derived paper model; separate from captured code and industrial measurements.',14,MUTED)
        self.height=round(self.y+50);self.parts[0]=f'<rect width="760" height="{self.height}" fill="#fff"/>'
        OUT.mkdir(parents=True,exist_ok=True)
        svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="{self.height}" viewBox="0 0 760 {self.height}" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(note)}</desc><metadata id="font-license">{escape(LICENSE)}</metadata>{STYLE}<g font-family="BookDiagram,Inter,Arial,sans-serif">{''.join(self.parts)}</g></svg>'''
        path=OUT/(name+'.svg');path.write_text(svg);return path


def main():
    for name,(title,lede,boxes,headers,rows,result,note) in CASES.items():
        d=CaseDrawing();d.heading(title,lede);d.boxes(boxes);d.table(headers,rows);d.result(result);p=d.finish(name,title,note)
        tree=ET.parse(p).getroot();assert tree.get('width')=='760'
        for e in tree.iter():
            if e.tag.endswith('}text'):assert float(e.get('font-size'))>=14
        print(p.relative_to(ROOT))

if __name__=='__main__':main()

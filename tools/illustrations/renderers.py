"""Semantic SVG layouts for the original Multi-GPU CUDA illustration system."""
from proofs import Drawing, PLUM, CORAL, LILAC, PEACH, INK, MUTED, LINE, PAPER
import math

COLORS=[PLUM,CORAL,'#197d76','#946b18']
TINTS=[LILAC,PEACH,'#e6f4f0','#fff3d8']

class Figure(Drawing):
    def __init__(self,book_id,title):
        super().__init__(1000)
        self.title=title
        self.text(28,30,'MULTI-GPU FROM SCRATCH / '+book_id,13,MUTED,650)
        self.line([(28,46),(732,46)],LINE,1)
        self.top=self.paragraph(28,86,title,704,29,INK,36)+22
    def box(self,x,y,w,h,label,color=0):
        self.rect(x,y,w,h,TINTS[color%4],LINE,12)
        # Center wrapped labels, keeping at least 20px type.
        from PIL import ImageFont
        from proofs import FONT
        font=ImageFont.truetype(str(FONT),200)
        lines=[];line=''
        for word in label.split():
            proposal=(line+' '+word).strip()
            if line and font.getlength(proposal)/10>w-24:lines.append(line);line=word
            else:line=proposal
        if line:lines.append(line)
        yy=y+h/2-(len(lines)-1)*13+7
        for line in lines:self.text(x+w/2,yy,line,20,COLORS[color%4],600,'middle');yy+=26
    def arrow(self,points,color=0,dash=False):
        self.line(points,COLORS[color%4],2.5,True,dash)
    def finish(self,path,title,note,bottom):
        y=bottom+32
        self.rect(28,y,704,24+len(self._wrap_note(note))*28,LILAC,LINE,12)
        y=self.paragraph(48,y+32,note,664,20,INK,28)+22
        self.line([(28,y),(732,y)],LINE,1)
        self.text(28,y+29,'Conceptual schematic · Read the section for assumptions and evidence.',14,MUTED)
        self.height=math.ceil(y+53)
        self.parts[0]=f'<rect width="760" height="{self.height}" fill="{PAPER}"/>'
        self.write(path,title,note)
    def _wrap_note(self,s):
        from PIL import ImageFont
        from proofs import FONT
        font=ImageFont.truetype(str(FONT),200);out=[];line=''
        for word in s.split():
            p=(line+' '+word).strip()
            if line and font.getlength(p)/10>664:out.append(line);line=word
            else:line=p
        out.append(line);return out


def flow(d,labels):
    y=d.top;w=310;h=100
    for i,label in enumerate(labels):
        row=i//2;col=i%2 if row%2==0 else 1-i%2
        x=28+col*394;yy=y+row*148
        d.box(x,yy,w,h,label,i)
        if i:
            prow=(i-1)//2;pcol=(i-1)%2 if prow%2==0 else 1-(i-1)%2
            px=28+pcol*394;py=y+prow*148
            if row==prow:d.arrow([(px+(w if col>pcol else 0),py+50),(x+(0 if col>pcol else w),yy+50)],i-1)
            else:d.arrow([(px+w/2,py+h),(x+w/2,yy)],i-1)
    return y+((len(labels)-1)//2)*148+h


def transfer(d,labels,staging=False,rdma=False):
    y=d.top
    if len(labels)==2:labels=[labels[0],'Selected data route',labels[1]]
    for i,label in enumerate(labels[:3]):
        d.box(152,y+i*150,456,92,label,i)
        if i:d.arrow([(380,y+(i-1)*150+92),(380,y+i*150)],i-1)
    if staging:d.text(634,y+196,'Host',20,MUTED,450,'middle')
    if rdma:
        d.line([(110,y+40),(110,y+350)],CORAL,2,False,True)
        d.text(105,y+403,'Host staging omitted only when the path supports it.',20,MUTED)
        return y+420
    return y+392


def topology(d,labels):
    y=d.top
    d.box(28,y,204,95,'GPU 0 / memory',0);d.box(528,y,204,95,'GPU 1 / memory',1)
    d.box(278,y+160,204,95,'PCIe switch',3)
    d.arrow([(130,y+95),(130,y+208),(274,y+208)],3)
    d.arrow([(486,y+208),(630,y+208),(630,y+95)],3)
    d.line([(236,y+40),(524,y+40)],PLUM,3)
    d.line([(236,y+49),(524,y+49)],PLUM,3)
    d.text(380,y+26,'Optional direct NVLink',20,PLUM,450,'middle')
    d.text(380,y+300,'Query capability; inspect the actual physical route.',20,MUTED,450,'middle')
    return y+322


def numa(d,labels):
    y=d.top
    for i in range(2):
        x=28+i*390;d.rect(x,y,314,305,'#f8f5fb',LINE,14)
        d.box(x+45,y+20,224,70,'CPU '+str(i),i)
        d.box(x+45,y+128,224,70,'PCIe switch',i)
        d.box(x+45,y+236,224,55,'GPU '+str(i),i)
        d.arrow([(x+157,y+90),(x+157,y+128)],i)
        d.arrow([(x+157,y+198),(x+157,y+236)],i)
    d.line([(342,y+55),(418,y+55)],CORAL,3)
    d.text(380,y+360,'An inter-socket route adds shared resources.',20,MUTED,450,'middle')
    return y+380


def switch(d,labels):
    y=d.top;d.box(242,y,276,95,'Shared switch / fabric',0)
    for i,label in enumerate(labels[:4]):
        x=28+i*181;d.box(x,y+220,161,92,label,i)
        d.line([(380,y+95),(380,y+155),(x+80,y+155),(x+80,y+220)],COLORS[i],2.5)
    d.text(380,y+363,'Each endpoint retains its own local memory.',20,MUTED,450,'middle')
    return y+386


def ring(d,labels,failure=False):
    y=d.top;nodes=[(60,y),(480,y),(480,y+245),(60,y+245)]
    for i,(x,yy) in enumerate(nodes):
        d.box(x,yy,220,90,labels[i] if i<len(labels) else f'Rank {i}',i)
        for j in range(4):d.rect(x+j*49+12,yy+104,42,25,TINTS[j],COLORS[j],4)
    edges=[[(284,y+45),(476,y+45)],[(590,y+137),(590,y+241)],[(476,y+290),(284,y+290)],[(170,y+241),(170,y+137)]]
    for i,pts in enumerate(edges):d.arrow(pts,i,dash=failure and i==1)
    if failure:
        if 'cycle of waits' in d.title.lower():
            d.text(380,y+176,'WAIT CYCLE',22,CORAL,650,'middle')
            d.text(380,y+414,'Every participant waits on another participant.',20,MUTED,450,'middle')
        else:
            d.text(590,y+196,'×',40,CORAL,650,'middle')
            d.text(380,y+414,'Missing progress breaks the dependency chain.',20,MUTED,450,'middle')
    else:
        is_reduce='all-reduce' in d.title.lower()
        d.text(380,y+167,'REDUCE-SCATTER' if is_reduce else 'MATCHED SENDS',20,PLUM,650,'middle')
        d.text(380,y+198,'then ALL-GATHER' if is_reduce else 'AND RECEIVES',20,CORAL,650,'middle')
        d.text(380,y+414,'Chunks circulate between immediate ring neighbors.',20,MUTED,450,'middle')
    return y+435


def collective(d,labels,kind):
    y=d.top;colors=range(4)
    if kind in {'gather','scatter','alltoall','chunks'}:
        d.text(188,y+15,'BEFORE',17,MUTED,650,'middle');d.text(570,y+15,'AFTER',17,MUTED,650,'middle')
        for rank in range(4):
            yy=y+50+rank*85;d.text(28,yy+30,f'R{rank}',20,MUTED,550)
            for col in range(4):
                before=kind in {'alltoall','scatter'} or col==rank
                after=kind in {'gather','alltoall','chunks'} or col==rank
                if kind=='chunks':before=col==rank
                aidx=rank if kind=='alltoall' else col
                bidx=col if kind=='alltoall' else col
                for base,active,cidx in [(70,before,aidx),(435,after,bidx)]:
                    d.rect(base+col*65,yy,57,55,TINTS[cidx] if active else '#f0eef2',LINE,6)
                    if active:
                        text=f'{rank}→{col}' if kind=='alltoall' and base==70 else (f'{col}→{rank}' if kind=='alltoall' else chr(65+col))
                        d.text(base+col*65+28,yy+34,text,20,COLORS[cidx],600,'middle')
            d.arrow([(343,yy+28),(422,yy+28)],rank)
        d.text(380,y+430,{'gather':'Concatenate contributions; retain their rank order.','scatter':'Reduce matching coordinates; distribute output ownership.','alltoall':'Source/destination identities determine each piece.','chunks':'Reduced chunk shards become a replicated result.'}[kind],20,MUTED,450,'middle')
        return y+450
    # Root fan-in/fan-out.
    root=(labels[0] if kind=='broadcast' else labels[-1]) if labels else 'Root'
    d.box(243,y,274,90,root,0)
    names=(labels[1:] if kind=='broadcast' else labels[:-1]) or ['Rank 0','Rank 1','Rank 2']
    for i,name in enumerate(names):
        x=28+i*241;d.box(x,y+230,222,90,name,i+1)
        pts=[(380,y+90),(380,y+155),(x+111,y+155),(x+111,y+226)]
        if kind=='reduce':pts=list(reversed(pts))
        d.arrow(pts,i+1)
    d.text(380,y+377,'COPY' if kind=='broadcast' else 'COMBINE MATCHING ELEMENTS',20,PLUM,650,'middle')
    return y+400


def events(d,labels,kind):
    y=d.top;left=185;cw=82
    for i in range(7):
        d.line([(left+i*cw,y+30),(left+i*cw,y+295)],LINE,1)
        d.text(left+i*cw,y+15,str(i),15,MUTED,450,'middle')
    d.text(730,y-14,'logical time →',16,MUTED,450,'end')
    for i,name in enumerate(labels[:2]):d.paragraph(28,y+90+i*140,name,145,20,MUTED,26)
    d.box(left,y+55,cw*2-6,70,'Produce',0)
    d.box(left+cw*2,y+55,cw-6,70,'Event',3)
    if kind=='events':
        d.box(left,y+195,cw*2-6,70,'Independent',1)
        d.box(left+cw*3,y+195,cw*2-6,70,'Later work',1)
        d.line([(left+cw*2.5,y+130),(left+cw*2.5,y+275)],CORAL,2,False,True)
    else:
        d.box(left+cw*2,y+195,cw-6,70,'Wait',3)
        d.box(left+cw*3,y+195,cw*2-6,70,'Consume',1)
        d.arrow([(left+cw*2.5,y+130),(left+cw*2.5,y+187)],3)
    return y+320


def pipeline(d,labels,buckets=False):
    y=d.top;left=164;cw=78;rows=3
    d.text(730,y,'logical time →',16,MUTED,450,'end')
    for t in range(8):
        d.line([(left+t*cw,y+25),(left+t*cw,y+333)],LINE,1)
        d.text(left+t*cw,y+21,str(t),15,MUTED,450,'middle')
    for rank in range(rows):
        yy=y+55+rank*100;d.paragraph(28,yy+30,labels[rank] if rank<len(labels) else f'Stage {rank}',126,20,MUTED,26)
        for batch in range(4):
            d.box(left+(batch+rank)*cw,yy,cw-5,65,f'{"B" if buckets else "μ"}{batch}',batch)
    d.text(380,y+390,'Fill → steady progress → drain',20,MUTED,450,'middle')
    return y+413


def ownership(d,labels,kind):
    y=d.top
    for i,label in enumerate(labels[:3]):
        x=28+i*(704/len(labels));w=704/len(labels)-16
        d.rect(x,y,w,285,'#f8f5fb',LINE,14)
        d.paragraph(x+16,y+35,label,w-32,20,MUTED,26)
        for j in range(3):d.box(x+20,y+80+j*60,w-40,45,('Object '+str(j)) if kind=='symmetric' else ('Owned region '+str(j)),i)
    d.text(380,y+340,'Logical identity ≠ a universal numeric address' if kind=='symmetric' else 'Access permission and lifetime remain explicit.',20,MUTED,450,'middle')
    return y+360


def threads(d,labels):
    y=d.top
    for i,name in enumerate(labels[:2]):
        x=28+i*390;d.rect(x,y,314,365,'#f8f5fb',LINE,14)
        d.text(x+157,y+35,name,21,MUTED,600,'middle')
        for j,t in enumerate([f'Select device {i}',f'Create stream on {i}',f'Launch work on {i}']):
            d.box(x+28,y+65+j*100,258,70,t,i)
            if j<2:d.arrow([(x+157,y+135+j*100),(x+157,y+165+j*100)],i)
    return y+365


def ipc(d,labels):
    y=d.top
    for i,name in enumerate(labels):
        x=28+i*390;d.rect(x,y,314,300,'#f8f5fb',LINE,14);d.text(x+157,y+36,name,22,MUTED,650,'middle')
        d.box(x+27,y+100,260,100,'Export allocation' if i==0 else 'Import mapped access',i)
    d.arrow([(342,y+150),(418,y+150)],1)
    d.text(380,y+348,'Exchange an opaque IPC handle',20,PLUM,550,'middle')
    return y+370


def put(d,labels):
    y=d.top;d.box(28,y,294,110,labels[0],0);d.box(438,y,294,110,labels[1],1)
    d.arrow([(326,y+55),(434,y+55)],0)
    d.text(380,y+150,'issue remote operation',20,MUTED,450,'middle')
    d.arrow([(585,y+115),(585,y+260),(175,y+260),(175,y+115)],1,True)
    d.text(380,y+308,'Completion / notification protocol',20,CORAL,550,'middle')
    return y+333


def halo(d,labels):
    y=d.top;cw=53
    for rank,name in enumerate(labels[:3]):
        yy=y+rank*150;d.text(28,yy+28,name,21,MUTED,600)
        for cell in range(9):
            ghost=cell in (0,8);x=205+cell*cw
            d.rect(x,yy,47,54,'#f0eef2' if ghost else TINTS[rank],LINE,5)
            d.text(x+23,yy+34,'G' if ghost else str(cell-1),19,MUTED if ghost else COLORS[rank],550,'middle')
        if rank<2:
            d.arrow([(205+7*cw+23,yy+58),(205+7*cw+23,yy+95),(205+23,yy+95),(205+23,yy+146)],rank)
            d.arrow([(205+cw+23,yy+146),(205+cw+23,yy+111),(205+8*cw+23,yy+111),(205+8*cw+23,yy+58)],rank+1)
    d.text(380,y+468,'G = ghost storage; colored cells = owned interior',20,MUTED,450,'middle')
    return y+490


def matrix(d,labels,kind):
    y=d.top
    if kind=='row':return row_products(d,labels)
    if kind in {'tpgrid','matrixgrid','summa'}:
        for row in range(2):
            for col in range(2):
                x=140+col*300;yy=y+90+row*160
                label=f'C{row}{col}' if kind=='summa' else (f'PP {col} / TP {row}' if kind=='tpgrid' else f'Rank ({row},{col})')
                d.box(x,yy,180,110,label,row*2+col)
        for row in range(2):d.arrow([(324,y+145+row*160),(436,y+145+row*160)],row)
        for col in range(2):d.line([(230+col*300,y+204),(230+col*300,y+246)],CORAL,2.5)
        if kind=='summa':
            d.text(380,y+35,'A panels along rows · B panels along columns',20,PLUM,550,'middle')
        elif kind=='tpgrid':d.text(380,y+35,'Pipeline direction →',20,PLUM,550,'middle')
        else:d.text(380,y+35,'Row groups and column groups',20,PLUM,550,'middle')
        return y+395
    # Explicit matrix cells: full X, partitioned W, resulting Y ownership.
    for i,name in enumerate(labels[:3]):
        x=30+i*242;d.text(x+107,y+30,name,20,MUTED,600,'middle')
        for r in range(4 if i==1 else 6):
            for c in range(4):
                idx=(c//2 if kind=='column' else r//3) if i==1 else (c//2 if kind=='column' and i==2 else 0)
                d.rect(x+c*51,y+65+r*43,47,39,TINTS[idx],LINE,3)
    d.text(258,y+202,'×',28,INK,600,'middle');d.text(499,y+202,'=',28,INK,600,'middle')
    if kind=='row':
        d.text(380,y+386,'Local products share output coordinates → reduce partial sums',20,CORAL,550,'middle')
    else:d.text(380,y+386,'Output features retain the column-shard ownership.',20,PLUM,550,'middle')
    return y+410


def capacity(d,labels):
    y=d.top
    d.text(380,y+18,'ONE LOGICAL WORKING SET',18,MUTED,650,'middle')
    for i,label in enumerate(labels):d.box(28+i*238,y+48,220,110,label,i)
    d.line([(138,y+162),(138,y+210),(622,y+210),(622,y+162)],PLUM,2)
    d.text(380,y+253,'Partition / place according to the algorithm',20,PLUM,600,'middle')
    for i in range(3):
        d.box(28+i*238,y+310,220,110,f'GPU {i} local memory',i)
        d.arrow([(138+i*238,y+278),(138+i*238,y+306)],i)
    return y+425


def balance(d,labels,weighted=False):
    y=d.top;left=205
    d.text(730,y+15,'relative work / completion',17,MUTED,450,'end')
    for i,label in enumerate(labels[:3]):
        yy=y+70+i*110;d.paragraph(28,yy+28,label,155,20,MUTED,26)
        width=(420 if weighted else [220,320,420][i]);d.rect(left,yy,width,58,TINTS[i],COLORS[i],8)
        d.text(left+width/2,yy+36,(['more work','medium work','less work'][i] if weighted else ['earlier finish','middle finish','later finish'][i]),20,COLORS[i],550,'middle')
    d.line([(left+420,y+45),(left+420,y+365)],CORAL,2,False,True)
    d.text(380,y+414,'Illustrative relative schedule; no hardware timings.',20,MUTED,450,'middle')
    return y+438


def layers(d,labels):
    y=d.top
    if labels and ('Prefill' in labels[0] or 'CUDA device' in labels[0]):return pools_or_stack(d,labels)
    for i,label in enumerate(labels[:2]):
        x=28+i*390;d.rect(x,y,314,275,'#f8f5fb',LINE,14)
        d.paragraph(x+20,y+35,label,274,20,MUTED,26)
        for j in range(3):d.box(x+35,y+72+j*59,244,46,f'Layer {i*3+j}',i)
    d.arrow([(346,y+143),(414,y+143)],1)
    d.text(380,y+330,'Activation / state handoff',20,PLUM,550,'middle')
    return y+355


def replicas(d,labels,hybrid=False):
    y=d.top
    for i,label in enumerate(labels[:3]):
        x=28+i*238;d.text(x+110,y+25,label,20,MUTED,550,'middle')
        d.box(x,y+55,220,100,'Model replica' if not hybrid else ('Embedding shard '+str(i)),i)
        d.box(x,y+205,220,80,'Data shard '+str(i) if not hybrid else 'Dense replica',i)
        d.arrow([(x+110,y+159),(x+110,y+201)],i)
    d.line([(138,y+330),(614,y+330)],PLUM,2.5)
    for i in range(3):d.line([(138+i*238,y+289),(138+i*238,y+330)],COLORS[i],2.5)
    d.text(380,y+377,'Corresponding state must follow the same update contract.',20,MUTED,450,'middle')
    return y+400


def tiles(d,labels):
    y=d.top
    for r in range(6):
        for c in range(9):
            owner=c//3;d.rect(92+c*64,y+r*54,59,49,TINTS[owner],LINE,4)
    for i,label in enumerate(labels[:3]):d.text(185+i*192,y+369,label,20,COLORS[i],550,'middle')
    return y+392


def graph(d,labels,kind):
    y=d.top
    if kind=='experts':
        d.box(250,y,260,85,'Token identities + routes',0)
        for i in range(3):
            x=28+i*241;d.box(x,y+205,222,85,f'Expert {i}',i+1)
            d.arrow([(380,y+89),(380,y+140),(x+111,y+140),(x+111,y+201)],i+1)
            d.arrow([(x+111,y+294),(x+111,y+345),(380,y+345),(380,y+388)],i+1)
        d.box(250,y+392,260,85,'Combine by token',0);return y+480
    positions=[(140,y+30),(500,y+30),(500,y+270),(140,y+270)]
    edges=[(0,1),(1,2),(2,3)] if kind=='graph' else [(0,1),(1,2),(2,3),(3,0),(0,2)]
    for a,b in edges:
        ax,ay=positions[a][0]+60,positions[a][1]+45;bx,by=positions[b][0]+60,positions[b][1]+45
        dx,dy=bx-ax,by-ay;scale=max(abs(dx)/60,abs(dy)/45)
        ux,uy=dx/scale,dy/scale
        d.arrow([(ax+ux,ay+uy),(bx-ux,by-uy)],a,dash=kind=='frontier' and a==2)
    for i,(x,yy) in enumerate(positions):d.box(x,yy,120,90,labels[i] if i<len(labels) else f'Node {i}',i)
    return y+395


def tree(d,labels):
    y=d.top
    if labels and labels[0]=='Local histograms':return histogram(d)
    d.box(245,y,270,85,'Spatial hierarchy',0)
    for i,label in enumerate(labels[:2]):
        x=45+i*390;d.box(x,y+210,280,95,label,i+1)
        d.arrow([(x+140,y+206),(x+140,y+150),(380,y+150),(380,y+89)],i+1)
    return y+330


def mapping(d,labels):
    y=d.top
    for i,label in enumerate(labels):
        yy=y+i*93
        pair=label.split('→');left=pair[0].strip();right=pair[1].strip() if len(pair)>1 else 'Choose by semantics'
        d.box(28,yy,300,70,left,i);d.box(432,yy,300,70,right,i)
        d.arrow([(332,yy+35),(428,yy+35)],i)
    return y+(len(labels)-1)*93+80


def profile(d,labels):
    y=d.top
    for i,label in enumerate(labels):
        x=53+i*181;d.text(x+50,y+30,label,20,MUTED,550,'middle')
        d.rect(x,y+80,100,250,'#f0eef2',LINE,5)
        for path in range(3):
            yy=y+110+path*65;d.rect(x+12,yy,76,45,TINTS[path],LINE,5)
    d.text(380,y+385,'Combine path contributions at each observation date.',20,PLUM,550,'middle')
    return y+410


def draw(d,kind,labels):
    if kind=='flow':return flow(d,labels)
    if kind in {'transfer','staging','rdma'}:return transfer(d,labels,kind=='staging',kind=='rdma')
    if kind=='topology':return topology(d,labels)
    if kind=='numa':return numa(d,labels)
    if kind=='switch':return switch(d,labels)
    if kind in {'ring','failure'}:return ring(d,labels,kind=='failure')
    if kind in {'broadcast','reduce','gather','scatter','alltoall','chunks'}:return collective(d,labels,kind)
    if kind in {'events','handoff'}:return events(d,labels,kind)
    if kind in {'pipeline','buckets'}:return pipeline(d,labels,kind=='buckets')
    if kind in {'ownership','symmetric'}:return ownership(d,labels,kind)
    if kind=='threads':return threads(d,labels)
    if kind=='ipc':return ipc(d,labels)
    if kind=='put':return put(d,labels)
    if kind=='halo':return halo(d,labels)
    if kind in {'column','row','summa','tpgrid','matrixgrid'}:return matrix(d,labels,kind)
    if kind=='capacity':return capacity(d,labels)
    if kind in {'balance','weighted'}:return balance(d,labels,kind=='weighted')
    if kind=='layers':return layers(d,labels)
    if kind in {'replicas','hybrid'}:return replicas(d,labels,kind=='hybrid')
    if kind=='tiles':return tiles(d,labels)
    if kind in {'graph','frontier','fraud','experts'}:return graph(d,labels,kind)
    if kind=='tree':return tree(d,labels)
    if kind=='mapping':return mapping(d,labels)
    if kind=='profile':return profile(d,labels)
    if kind=='timing':return timing(d,labels)
    if kind=='hub':return collective(d,['Rank 0','Rank 1','Rank 2','Host / root'], 'reduce')
    if kind=='netting':return netting(d,labels)
    raise ValueError('Unknown renderer '+kind)


def netting(d,labels):
    y=d.top
    d.box(28,y,300,90,labels[0],0);d.box(432,y,300,90,labels[1],1)
    for x,color in [(178,0),(582,1)]:d.arrow([(x,y+94),(x,y+145),(380,y+145),(380,y+196)],color)
    d.box(230,y+200,300,90,labels[2],2)
    d.arrow([(380,y+294),(380,y+356)],2)
    d.box(230,y+360,300,90,labels[3],3)
    return y+450


def row_products(d,labels):
    y=d.top
    for rank in range(2):
        yy=y+rank*155
        d.text(28,yy+35,f'Rank {rank}',20,COLORS[rank],600)
        for group,label in enumerate([f'X{rank}',f'W{rank}',f'P{rank}']):
            x=160+group*186
            d.text(x+64,yy+21,label,20,COLORS[rank],600,'middle')
            for r in range(2):
                for c in range(2 if group==0 else 3):d.rect(x+c*42,yy+39+r*36,38,32,TINTS[rank],LINE,3)
        d.text(305,yy+85,'×',26,INK,550);d.text(491,yy+85,'=',26,INK,550)
    d.arrow([(658,y+75),(700,y+75),(700,y+326),(380,y+326),(380,y+361)],0)
    d.arrow([(595,y+268),(595,y+326),(380,y+326),(380,y+361)],1)
    d.box(230,y+365,300,95,'Y = P0 + P1',2)
    d.text(380,y+507,'P0 and P1 share the same output coordinates.',20,MUTED,450,'middle')
    return y+530


def pools_or_stack(d,labels):
    y=d.top
    if 'CUDA device' in labels[0]:
        for i,label in enumerate(['Application / algorithms','MPI · NCCL · NVSHMEM','CUDA runtime / driver','Physical devices / links']):
            d.box(150,y+i*100,460,75,label,i)
            if i:d.line([(380,y+(i-1)*100+79),(380,y+i*100-4)],PLUM,2.5)
        return y+390
    for i,label in enumerate(labels):
        x=28+i*390;d.rect(x,y,314,300,'#f8f5fb',LINE,14)
        d.text(x+157,y+37,label,22,MUTED,650,'middle')
        steps=['Prompt tokens','Build KV state','Completed prefill'] if i==0 else ['Read KV state','Decode step','Append token']
        for j,step in enumerate(steps):d.box(x+28,y+65+j*72,258,55,step,i)
    d.arrow([(346,y+165),(414,y+165)],1)
    d.text(380,y+355,'KV state and request metadata cross the pool boundary.',20,PLUM,550,'middle')
    return y+380

def histogram(d):
    y=d.top
    for i in range(3):
        x=28+i*241;d.box(x,y,222,80,f'Local histogram {i}',i)
        d.arrow([(x+111,y+84),(x+111,y+135),(380,y+135),(380,y+186)],i)
    d.box(230,y+190,300,85,'Reduce statistics',2)
    d.arrow([(380,y+279),(380,y+331)],2)
    d.box(230,y+335,300,85,'Choose next split',0)
    return y+425

def timing(d,labels):
    y=d.top
    rows=[('GPU 0 events',90,3,'local interval 0'),('GPU 1 events',225,4,'local interval 1')]
    for label,offset,duration,caption in rows:
        yy=y+offset;d.text(28,yy+30,label,20,MUTED,550)
        d.box(240,yy,430,70,caption,0 if offset==90 else 1)
        d.text(455,yy+107,'elapsed on this device only',20,MUTED,450,'middle')
    d.text(28,y+395,'Host clock',20,MUTED,550)
    d.line([(240,y+389),(700,y+389)],CORAL,3)
    d.line([(240,y+374),(240,y+404)],CORAL,2)
    d.line([(700,y+374),(700,y+404)],CORAL,2)
    d.paragraph(240,y+442,'Before submission through completion on every participant',450,20,INK,28)
    d.text(380,y+548,'Local event clocks do not form a shared device time axis.',20,MUTED,450,'middle')
    return y+575

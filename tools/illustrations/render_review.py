#!/usr/bin/env python3
"""Render every manifested figure and contact sheets for a visual review."""
from pathlib import Path
import argparse
import json
import cairosvg
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,default=ROOT/'build/illustration-preview')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
manifest=json.loads((ROOT/'tools/illustrations/manifest.json').read_text())
paths=[]
for page in manifest['pages']:
 paths.append(page['opener']);paths.extend(x['file'] for x in page['figures'])
for path in paths:
 target=a.output/(path.replace('/','-')+'.png')
 cairosvg.svg2png(url=str(ROOT/'docs/images'/path),write_to=str(target),output_width=760)
for family,cols,tileh in [('technical',9,240),('chapter-openers',7,230)]:
 files=[a.output/(x.replace('/','-')+'.png') for x in paths if x.startswith(family+'/')]
 sheet=Image.new('RGB',(cols*190,((len(files)+cols-1)//cols)*tileh),'#e5e2eb');draw=ImageDraw.Draw(sheet)
 for i,path in enumerate(files):
  im=Image.open(path).convert('RGB');im.thumbnail((180,tileh-28))
  x=(i%cols)*190+(190-im.width)//2;y=(i//cols)*tileh+20
  sheet.paste(im,(x,y));draw.text(((i%cols)*190+5,(i//cols)*tileh+3),path.name.replace('.svg.png','').replace(family+'-','')[:28],fill='#27253f')
 sheet.save(a.output/(family+'-contact.png'))
print(f'Rendered {len(paths)} SVGs and two contact sheets to {a.output}')

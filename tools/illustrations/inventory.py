#!/usr/bin/env python3
"""Inventory prose graphics separately from code/output; never edit book pages."""
from pathlib import Path
import json
import re
ROOT=Path(__file__).resolve().parents[2]
records=[]
for path in sorted((ROOT/'docs').rglob('*.md')):
    source=path.read_text()
    fences=[]
    for match in re.finditer(r'```([^\n]*)\n(.*?)```',source,re.S):
        language,body=match.groups()
        if language.strip() not in {'','text'}:
            continue
        graphic=any(token in body for token in ['-->', '<--', '+---', '│', '──', 'GPU0 ---'])
        fences.append(dict(line=source.count('\n',0,match.start())+1,
                           language=language.strip() or 'unlabeled',
                           classification='review-graphic' if graphic else 'preserve-output-or-reference',
                           preview=body[:140].replace('\n',' / ')))
    records.append(dict(page=str(path.relative_to(ROOT)),fences=fences))
report=dict(identity='Multi-GPU from scratch / plum, coral, warm paper',
            page_count=len(records),chapter_count=40,appendix_count=7,
            policy='Manual semantic review before replacement; preserve all code and captured evidence.',
            proof_files=['docs/images/proofs/00-cover.svg','docs/images/proofs/02-topology.svg'],
            rollout=[
                'Review two visual proofs and palette in the reading column.',
                'Define one chapter opener per chapter plus index/getting-started; preserve heading hierarchy.',
                'Map each genuine prose diagram to topology, collective flow, timeline, ownership, matrix, or cost-model layout.',
                'Review figure claims with the technical audit; link to existing chapter primary sources.',
                'Replace reviewed ASCII figures individually; retain captured outputs and numerical reference blocks.',
                'Render every SVG with embedded licensed font and inspect representative figures plus automated text bounds.',
                'Apply max-width:100%, height:auto and min-width:0 in the site reading column; verify mobile and desktop.'],
            pages=records)
out=ROOT/'tools/illustrations/inventory.json';out.write_text(json.dumps(report,indent=2)+'\n')
print(f'Inventoried {len(records)} pages; no chapter content changed: {out}')

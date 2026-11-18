# Original book illustration system

The book uses 49 chapter/appendix/landing-page openers and 99 technical SVGs.
The palette is plum, coral, warm paper, teal, and gold. Technical body labels
use 20 px on a 760 px canvas; headings wrap and the canvas height follows content.
Every production SVG embeds the licensed Inter subset and its OFL metadata.

`catalog.py` is the manually curated section-to-figure map. `renderers.py` draws
actual topology, chunk ownership, timelines, matrix partitions, halos, and other
relationships. These are conceptual schematics, not new measurement evidence.
The original code, captured outputs, and ASCII reference material remain intact.

Install `requirements.txt` from this directory, then run from the repository root:

```bash
python tools/illustrations/build.py
python tools/illustrations/validate.py
python tools/illustrations/render_review.py
```

`build.py --no-insert` generates assets without changing Markdown. Normal builds
use unique generated markers and section anchors, ignore headings inside fences,
check that every fenced block is unchanged, and reject concurrent page changes.
Regeneration replaces those markers rather than adding duplicate illustrations.
Do not use the generator to rewrite technical prose or historical evidence.

`manifest.json` maps each page to its artwork. `validation.json` records SVG hashes,
bounds, embedded-font/license checks, glyph coverage, and page references.
`render_review.py` writes individual PNGs and two contact sheets for inspection.
CairoSVG may need a local fontconfig configuration that maps BookDiagram to Inter;
web browsers use the embedded WOFF2 font. SVG arrows are paths, not font glyphs.

Responsive CSS keeps the images in the reading column; each image also links to
its standalone SVG for zooming. Original ASCII explanations remain available as
text. Check primary sources and the surrounding section when changing a figure's
meaning or adding a numerical claim.

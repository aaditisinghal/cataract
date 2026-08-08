# Paper build — Review Round 5 (surgical merge)

**Why this exists:** the original docx-js builder source was lost when a scratchpad
cleared mid-session. To avoid rebuilding the intricate IEEE A4 two-column layout
(6 alternating 1-col/2-col continuous section breaks for full-width tables/figures)
from scratch — and risking layout regressions with no local DOCX renderer to catch
them — R5 edits the **committed** `PersistenceOfVision.docx` in place, surgically.

## Pipeline
1. Unzip the committed DOCX → `word/document.xml` is pristine (docx-js writes one
   run per paragraph, so paragraph text is cleanly addressable).
2. `sections/sec{0..8}.json` + `abstract.json` — section bodies extracted from the
   DOCX, then edited by the R5 prose workflow (contribution-3 restructure, K/m
   symbol split, metaphor dedup, B17 circularity, B19 margin, ~10% trim). Captions
   and table cells are carried in these bodies and filtered at merge time.
3. `tables.json` — the 5 native DOCX tables, extracted verbatim (numbers preserved).
4. `merge.py` — order-preserving paragraph alignment (difflib) between original
   prose blocks and edited bodies. Replaces/deletes/inserts **prose only**; matches
   edited captions to existing caption blocks by Fig/Table id and replaces their
   text. Figures, tables, equations, and all column-break choreography are untouched.
   Run: `python3 merge.py` (dry run) then `python3 merge.py --apply`.
5. Fig 5 (rId16) regenerated as a combined recovery+margin panel (matches the new
   B19 caption); aspect 1.354 matches the original slot (no distortion).
6. References [10]/[16] given first-author *et al.* (verified via arXiv abstract
   pages); [3] given an access date.
7. The 9 data figures scaled 0.82 (equations left full size) for page economy.
8. Rezip `word/` → `PersistenceOfVision.docx`. `document.xml.pre-r5.bak` is the
   committed-version body for diffing/rollback.

## Invariants verified post-merge
- XML well-formed; 15 images + 5 tables intact; every rId resolves in rels.
- Contributions 1)–5) once each; "threefold" removed; ghost-vectors demoted to prose.
- K = lineup size (200/999/20) and k = projection rank (96) preserved; retained
  patch count renamed m (m=8/m=64). No stray patch-count K.
- Eq. (1)–(6) sequential; Fig 1–9 and Table I–V all referenced.
- SOTA title retained.

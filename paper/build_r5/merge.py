import re, json, difflib, sys
DRY = "--apply" not in sys.argv
SRC = "u/word/document.xml"
xml = open(SRC, encoding="utf-8").read()
head, body = xml.split("<w:body>", 1)
body_inner, tail = body.split("</w:body>", 1) if "</w:body>" in body else (body, "")

# tokenize into ordered blocks, but keep the exact raw substrings + gaps
# We re-find blocks and rebuild by walking with regex, preserving anything between blocks.
BLOCK = re.compile(r'<w:tbl>.*?</w:tbl>|<w:p\b.*?</w:p>', re.S)
def ptext(p): return "".join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p))
def unesc(t):
    return (t.replace("&amp;","&").replace("&apos;","'").replace("&quot;",'"')
             .replace("&lt;","<").replace("&gt;",">"))
def esc(t):
    return unesc(t).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
def kind(t):
    if t.startswith("<w:tbl"): return "TBL"
    if "<w:drawing>" in t: return "IMG"
    tx = ptext(t).strip()
    if re.match(r'^[IVX]+\.\s', tx): return "HEAD"
    if re.match(r'^(Fig\.|TABLE|Table\b)', tx): return "CAP"
    return "P"

matches = list(BLOCK.finditer(body_inner))
toks = [m.group(0) for m in matches]
spans = [(m.start(), m.end()) for m in matches]

heads = {}
for i, t in enumerate(toks):
    m = re.match(r'^([IVX]+)\.\s', ptext(t).strip())
    if m: heads[m.group(1)] = i
hidx = sorted(heads.values()) + [len(toks)]

def set_text(block, newtext):
    # replace text of first <w:t...>...</w:t>; blank any additional <w:t> in block
    done = [False]
    def repl(mm):
        if not done[0]:
            done[0] = True
            return mm.group(1) + esc(newtext) + "</w:t>"
        return mm.group(1) + "</w:t>"
    return re.sub(r'(<w:t[^>]*>)(?:[^<]*)</w:t>', repl, block)

def cap_id(s):
    m = re.match(r'^(Fig\.\s*\d+|TABLE\s+[IVX]+|Table\s+[IVX]+)', s.strip())
    return re.sub(r'\s+', ' ', m.group(1)).upper().replace("FIG.", "FIG") if m else None

# plan: dict block_idx -> ('REPLACE', text) or ('DELETE',) ; inserts: list of (after_block_idx, text)
plan = {}
inserts = []
SECMAP = {0:'I', 2:'III', 3:'IV', 4:'V', 7:'VIII'}
def ratio(a,b): return difflib.SequenceMatcher(None, a, b).ratio()

for si, rn in SECMAP.items():
    start = heads[rn]; end = min(x for x in hidx if x > start)
    seg = list(range(start+1, end))
    O = [(j, ptext(toks[j])) for j in seg if kind(toks[j]) == "P" and ptext(toks[j]).strip()]
    capblocks = [(j, ptext(toks[j])) for j in seg if kind(toks[j]) == "CAP"]
    Eall = [p.strip() for p in json.load(open(f"sections/sec{si}.json"))["body"].split("\n\n") if p.strip()]
    Ecap = [p for p in Eall if re.match(r'^(Fig\.|TABLE|Table\b)', p)]
    E = [p for p in Eall if not re.match(r'^(Fig\.|TABLE|Table\b)', p)]
    print(f"\n===== sec{si} (§{rn})  O_prose={len(O)}  E_prose={len(E)}  caps: doc={len(capblocks)} edited={len(Ecap)} =====")

    # ---- captions: match by id, replace ----
    capmap = {cap_id(t): j for j, t in capblocks}
    for ec in Ecap:
        cid = cap_id(ec)
        if cid in capmap:
            j = capmap[cid]
            if ptext(toks[j]).strip() != unesc(ec).strip():
                plan[j] = ('REPLACE', ec)
                print(f"   CAP replace [{j}] {cid}: {ec[:70]}")
        else:
            print(f"   !! CAP no doc-match for edited caption: {ec[:60]}")

    # ---- prose: order-preserving greedy 2-pointer align ----
    i = j = 0
    while i < len(O) and j < len(E):
        r = ratio(O[i][1], E[j])
        if r >= 0.45:
            if r < 0.999:
                plan[O[i][0]] = ('REPLACE', E[j])
            i += 1; j += 1
        else:
            look_ins = ratio(O[i][1], E[j+1]) if j+1 < len(E) else -1   # O[i] matches a later E -> E[j] is insert
            look_del = ratio(O[i+1][1], E[j]) if i+1 < len(O) else -1   # E[j] matches a later O -> O[i] is delete
            if look_ins >= look_del:
                inserts.append((O[i-1][0] if i>0 else start, E[j])); print(f"   +INSERT after blk {O[i-1][0] if i>0 else start}: {E[j][:70]}"); j += 1
            else:
                plan[O[i][0]] = ('DELETE',); print(f"   -DELETE [{O[i][0]}]: {O[i][1][:70]}"); i += 1
    while i < len(O):
        plan[O[i][0]] = ('DELETE',); print(f"   -DELETE(tail) [{O[i][0]}]: {O[i][1][:70]}"); i += 1
    while j < len(E):
        anchor = O[-1][0] if O else start
        inserts.append((anchor, E[j])); print(f"   +INSERT(tail) after {anchor}: {E[j][:70]}"); j += 1

nrep = sum(1 for v in plan.values() if v[0]=='REPLACE')
ndel = sum(1 for v in plan.values() if v[0]=='DELETE')
print(f"\nPLAN TOTALS: replace={nrep} delete={ndel} insert={len(inserts)}")

if DRY:
    print("\n(DRY RUN — no file written; pass --apply to write)")
    sys.exit(0)

# ---- apply: rebuild body_inner by walking spans ----
# choose a prose template block (for inserts) = first prose block of sec0
tmpl_idx = next(j for j,_ in [(x,0) for x in range(len(toks))] if kind(toks[j])=="P")
tmpl = toks[tmpl_idx]
def make_para(text):
    return set_text(tmpl, text)

ins_by_anchor = {}
for a, t in inserts: ins_by_anchor.setdefault(a, []).append(t)

out = []
prev_end = 0
for idx, (s, e) in enumerate(spans):
    out.append(body_inner[prev_end:s])   # keep inter-block content (sectPr paras etc. are their own blocks though)
    act = plan.get(idx)
    if act and act[0] == 'DELETE':
        pass  # drop the block
    elif act and act[0] == 'REPLACE':
        out.append(set_text(toks[idx], act[1]))
    else:
        out.append(toks[idx])
    for t in ins_by_anchor.get(idx, []):
        out.append(make_para(t))
    prev_end = e
out.append(body_inner[prev_end:])
new_body = "".join(out)
newxml = head + "<w:body>" + new_body + "</w:body>" + tail
open("u/word/document.xml", "w", encoding="utf-8").write(newxml)
print("APPLIED -> u/word/document.xml  (len %d -> %d)" % (len(xml), len(newxml)))

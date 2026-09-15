#!/usr/bin/env python3
"""Build the article docx: pandoc + post-processing (tables, captions, footer page numbers, media cleanup)."""
import subprocess, zipfile, shutil, re, sys
md, out = sys.argv[1], sys.argv[2]
subprocess.run(['pandoc', md, '-f', 'markdown+autolink_bare_uris', '-o', out, '--reference-doc=orig.docx'], check=True)
tmp = out + '.tmp'
zin = zipfile.ZipFile(out)
rels = zin.read('word/_rels/document.xml.rels').decode('utf8')
referenced = set(re.findall(r'Target="media/([^"]+)"', rels))
doc = zin.read('word/document.xml').decode('utf8')

def add_ppr(p, extra):
    """Insert extra pPr children after pStyle (schema order)."""
    if '<w:pPr>' not in p:
        return p.replace('<w:p>', '<w:p><w:pPr>' + extra + '</w:pPr>', 1)
    head = p.split('<w:pPr>', 1)
    ppr_rest = head[1]
    m = re.match(r'(<w:pStyle[^>]*/>)', ppr_rest)
    if m:
        return head[0] + '<w:pPr>' + m.group(1) + extra + ppr_rest[m.end():]
    return head[0] + '<w:pPr>' + extra + ppr_rest

def fix_table(m):
    t = m.group(0); rows = re.split(r'(?=<w:tr[ >])', t)
    trs = [i for i, r in enumerate(rows) if r.startswith('<w:tr')]
    for n, i in enumerate(trs):
        r = rows[i]; extra = '<w:cantSplit/>' + ('<w:tblHeader/>' if n == 0 else '')
        if '<w:trPr>' in r.split('<w:tc>')[0]:
            r = r.replace('<w:trPr>', '<w:trPr>' + extra, 1)
        else:
            r = re.sub(r'^(<w:tr(?: [^>]*)?>)', r'\1<w:trPr>' + extra + '</w:trPr>', r, count=1)
        if len(trs) <= 12 and n < len(trs) - 1:
            r = re.sub(r'<w:p>.*?</w:p>', lambda pm: add_ppr(pm.group(0), '<w:keepNext/>') if '<w:keepNext/>' not in pm.group(0) else pm.group(0), r, flags=re.S)
        rows[i] = r
    return ''.join(rows)
doc = re.sub(r'<w:tbl>.*?</w:tbl>', fix_table, doc, flags=re.S)

def fix_cap(m):
    p = m.group(0); txt = ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p))
    if txt.strip().startswith(('Таблица','Table')) and '<w:keepNext/>' not in p:
        return add_ppr(p, '<w:keepNext/>')
    return p
doc = re.sub(r'<w:p>.*?</w:p>', fix_cap, doc, flags=re.S)

sect = ('<w:sectPr><w:footerReference w:type="default" r:id="rIdFooter1"/>'
        '<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1418" w:right="1418" w:bottom="1418" w:left="1418" w:header="709" w:footer="709" w:gutter="0"/></w:sectPr>')
doc = re.sub(r'<w:sectPr\s*/>', sect, doc, count=1)
footer = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
          '<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:fldChar w:fldCharType="begin"/></w:r>'
          '<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r>'
          '<w:r><w:t>1</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')
rels = rels.replace('</Relationships>', '<Relationship Id="rIdFooter1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/></Relationships>')

zout = zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    if item.filename.startswith('word/media/') and item.filename.split('/')[-1] not in referenced:
        continue
    data = zin.read(item.filename)
    if item.filename == 'word/document.xml': data = doc.encode('utf8')
    elif item.filename == 'word/_rels/document.xml.rels': data = rels.encode('utf8')
    elif item.filename == '[Content_Types].xml':
        x = data.decode('utf8')
        if 'Extension="png"' not in x:
            x = x.replace('<Default Extension="xml"', '<Default Extension="png" ContentType="image/png"/><Default Extension="xml"', 1)
        x = x.replace('</Types>', '<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>')
        data = x.encode('utf8')
    zout.writestr(item, data)
zout.writestr('word/footer1.xml', footer)
zout.close(); zin.close(); shutil.move(tmp, out)
print('built', out, 'hyperlinks:', doc.count('<w:hyperlink'))

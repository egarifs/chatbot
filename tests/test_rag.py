"""Pipeline tests. Fixtures are generated with stdlib so no binary files live in the repo."""

import zipfile

import pytest

from rag import build_index, chunk, load_document, retrieve


def _pdf(path, pages):
    """Write a minimal valid PDF with one text line per page."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None,
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 3 0 R >> >> /Contents {len(objs)} 0 R >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = "%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n" + "".join(f"{o:010} 00000 n \n" for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    path.write_bytes(out.encode("latin-1"))


def _docx(path, text):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
        z.writestr("_rels/.rels",
                   '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="r1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
        z.writestr("word/document.xml",
                   '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                   f'<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>')


MD = """Intro paragraph.

# Produk
Ringkasan produk.

## Harga
Paket dasar seharga 50 ribu rupiah per bulan.

```
# bukan heading
```
"""


def test_pdf_has_pages(tmp_path):
    f = tmp_path / "a.pdf"
    _pdf(f, ["Halaman satu tentang kucing", "Halaman dua tentang roket"])
    docs = load_document(f)
    assert [d["page"] for d in docs] == [1, 2]
    assert "roket" in docs[1]["text"]


def test_docx(tmp_path):
    f = tmp_path / "a.docx"
    _docx(f, "Isi dokumen word")
    assert load_document(f)[0]["text"].strip() == "Isi dokumen word"


def test_md_headings_ignore_code_block(tmp_path):
    f = tmp_path / "a.md"
    f.write_text(MD, encoding="utf-8")
    headings = [d["heading"] for d in load_document(f)]
    assert headings == ["(Pembuka)", "Produk", "Harga"]


def test_txt_and_chunk_ids(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("kata " * 600, encoding="utf-8")
    chunks = chunk(load_document(f))
    assert len(chunks) > 1
    assert [c["chunk_id"] for c in chunks] == list(range(len(chunks)))
    assert all(c["source"] == "a.txt" and len(c["text"]) <= 1000 for c in chunks)


@pytest.mark.parametrize("name,content", [
    ("a.pptx", b"x"),                  # unsupported
    ("a.txt", b"   \n"),               # empty
    ("a.txt", "é".encode("latin-1")),  # non-UTF-8
    ("a.pdf", b"not a pdf"),           # corrupt
    ("a.docx", b"not a zip"),          # corrupt
])
def test_bad_input_raises_valueerror(tmp_path, name, content):
    f = tmp_path / name
    f.write_bytes(content)
    with pytest.raises(ValueError):
        load_document(f)


def test_retrieve_finds_relevant_chunk_cross_language(tmp_path):
    f = tmp_path / "a.md"
    f.write_text(MD, encoding="utf-8")
    hits = retrieve("How much does the basic plan cost?", build_index(chunk(load_document(f))), k=2)
    assert len(hits) == 2
    assert hits[0]["heading"] == "Harga"
    assert all(0 <= h["score"] <= 1 for h in hits)

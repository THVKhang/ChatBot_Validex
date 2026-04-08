# Metadata Schema (Week 1)

## Muc tieu
Dinh nghia schema metadata co ban cho du lieu retrieval trong prototype.

## Required fields
- id: dinh danh duy nhat cho document/chunk
- title: tieu de ngan cua tai lieu/chunk
- source: nguon du lieu (faq_page, internal_blog, ...)
- topic: nhan chu de chinh
- document_type: loai tai lieu (faq, guide, checklist, ...)
- approved: trang thai su dung duoc cho prototype (true/false)
- content: noi dung text se duoc truy xuat

## JSON example
```json
{
  "id": "doc_001_chunk_01",
  "title": "Police Check FAQ",
  "source": "faq_page",
  "topic": "police check",
  "document_type": "faq",
  "approved": true,
  "content": "Police check la quy trinh xac minh ly lich..."
}
```

## Current sample file
- data/metadata/documents.json

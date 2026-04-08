# Demo Script (5 Minutes)

## Muc tieu demo
Chung minh prototype chatbot da chay on dinh voi 2 tinh huong:
1. Create blog tu prompt tu do
2. Follow-up shorten tu context luot truoc

## Chuan bi truoc demo
- Mo terminal tai root project
- Dam bao da cai dependencies
- Chay test nhanh

Lenh goi y:
```bash
python -m pytest -q
```
Ky vong: test pass (hien tai 29 passed)

## Demo Case 1 (Create Blog)
### Input
Write a blog about what a police check is for first-time job applicants, in a clear and professional tone.

### Cach chay
```bash
python -m app.main
```
Nhap prompt o tren.

### Ky vong output
- Co block Parsed: intent/topic/audience/tone/length
- Co block Retrieved Top Docs
- Co Generated Title, Outline, Draft
- Co Sources Used

### Diem can noi voi mentor
- Parser da clean topic de title gon
- Retriever da rerank theo metadata
- Generator output schema da chuan hoa

## Demo Case 2 (Shorten Follow-up)
### Input tiep theo trong cung session
Make it shorter

### Ky vong output
- Parsed intent = shorten
- Co dong: context: using previous draft
- Draft ngan hon va co tham chieu draft truoc

### Diem can noi voi mentor
- Prototype da co pseudo multi-turn
- Co kha nang sua bai theo luot chat tiep theo

## Script thuyet trinh 60s
- Em da hoan thien flow parser -> retriever -> generator o muc prototype dung duoc.
- Retrieval benchmark da cai thien tu 6/10 len 10/10 tren cung bo query.
- He thong da ho tro follow-up nhu "Make it shorter" bang context luot truoc.

## Q&A du kien
### Neu mentor hoi: "Da san sang production chua?"
Tra loi: chua. Day la prototype, chua co authentication, moderation, observability, va deployment pipeline.

### Neu mentor hoi: "Diem manh hien tai la gi?"
Tra loi: flow on dinh, output schema ro rang, co benchmark retrieval, co test suite pass.

### Neu mentor hoi: "Buoc tiep theo la gi?"
Tra loi: nang semantic retrieval, toi uu quality generation, va bo sung UI chat multi-turn day du.

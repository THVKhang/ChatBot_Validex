# Week 2 Demo Case

## Demo objective
Chung minh flow end-to-end: parse -> retrieve -> generate hoat dong on dinh hon Week 1.

## Input prompt
Write a blog about what a police check is for first-time job applicants, in a clear and professional tone.

## Parsed result
```json
{
  "intent": "create_blog",
  "topic": "police check",
  "audience": "first-time job applicants",
  "tone": "clear_professional",
  "length": "medium"
}
```

## Top retrieved documents
1. doc_08_who_needs_police_check (score=14)
2. doc_07_required_documents (score=8)
3. doc_04_candidate_experience (score=5)

## Generated title
What A Police Check Is For First-Time Job Applicants, In A Clear And Professional Tone

## Generated outline
1. Introduction
2. What a police check means
3. Who may need one
4. How long it takes and what to prepare
5. Conclusion

## Generated short draft
What A Police Check Is For First-Time Job Applicants, In A Clear And Professional Tone la mot chu de quan trong trong quy trinh tuyen dung va compliance. Bai viet nay huong den doi tuong first-time job applicants, su dung giong van professional, nham giai thich de hieu va ap dung duoc ngay.

Truoc het, police check giup doanh nghiep xac minh thong tin co ban, giam rui ro va tang do tin cay. Voi nhung vai tro nhay cam hoac moi truong can tinh an toan cao, buoc nay thuong duoc xem la tieu chuan toi thieu. Neu truyen thong khong ro rang, ung vien de bo quy trinh hoac hieu sai muc dich kiem tra.

Tiep theo, doanh nghiep nen thong bao ky ve pham vi kiem tra, thoi gian xu ly va cac giay to can thiet. Cach trinh bay minh bach se cai thien candidate experience va giam tre han onboarding. Doi voi nhan vien backoffice, mot checklist ngan gon giup thong nhat cach huong dan giua cac team.

Cuoi cung, noi dung blog nen ket hop dinh nghia, use case thuc te va huong dan hanh dong cu the. Ban co the ket bai bang CTA nhu: lien he team ho tro de nhan checklist police check phu hop tung vi tri.

## Notes
- Diem manh: flow parse -> retrieve -> generate chay on dinh, co output schema ro rang.
- Van de gap phai: title/topic con dai vi parser lay nguyen menh de prompt.
- Huong cai thien tuan tiep theo: tach title cleaner va bo sung post-processing cho topic.

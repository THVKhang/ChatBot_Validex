# Retrieval Testcases

## Cach dung
- Dien ket qua thuc te sau moi lan chay test retriever
- Danh dau pass/fail theo muc do dung voi expected topic
- Ghi ghi chu de dieu chinh top-k, token match, hoac data chunking

## Bang test
| Query | Expected topic | Top 3 results (thuc te) | Danh gia | Nhan xet |
|---|---|---|---|---|
| What is a police check? | police check definition | doc_08_who_needs_police_check(8); doc_01_police_check(7); doc_04_candidate_experience(4) | [x] Pass [ ] Fail | Top 3 deu lien quan police check definition/use case. |
| How long does a police check take? | processing time | doc_06_faq_processing_time(14); doc_08_who_needs_police_check(14); doc_01_police_check(9) | [x] Pass [ ] Fail | Da uu tien nhom tai lieu FAQ/time manh hon. |
| Who needs a police check? | target audience / use case | doc_08_who_needs_police_check(18); doc_01_police_check(9); doc_04_candidate_experience(6) | [x] Pass [ ] Fail | Ket qua dung trong tam audience/use case. |
| What documents are required? | requirements / documents | doc_07_required_documents(17); doc_01_police_check(1); doc_02_employer_guide(1) | [x] Pass [ ] Fail | Da uu tien tai lieu requirements ro rang. |
| Police check for employment | employment use case | doc_08_who_needs_police_check(14); doc_01_police_check(13); doc_07_required_documents(10) | [x] Pass [ ] Fail | Top ket qua sat voi employment va compliance. |

## Bo sung query (them 5 query)
| Query | Expected topic | Top 3 results (thuc te) | Danh gia | Nhan xet |
|---|---|---|---|---|
| Police check for first-time applicants | first-time applicant use case | doc_08_who_needs_police_check(20); doc_01_police_check(9); doc_06_faq_processing_time(9) | [x] Pass [ ] Fail | Uu tien dung tai lieu first-time applicants. |
| Recruitment compliance checklist | compliance | doc_05_compliance_note(8); doc_07_required_documents(7); doc_02_employer_guide(5) | [x] Pass [ ] Fail | Da fix query fail truoc do: compliance doc len top 1. |
| How to explain police check to candidates? | communication guide | doc_08_who_needs_police_check(15); doc_01_police_check(7); doc_04_candidate_experience(4) | [x] Pass [ ] Fail | Ket qua on dinh cho candidate communication. |
| Employer onboarding with police check | onboarding + timing | doc_01_police_check(9); doc_08_who_needs_police_check(9); doc_04_candidate_experience(5) | [x] Pass [ ] Fail | Top ket qua lien quan onboarding flow. |
| Background screening process | screening overview | doc_01_police_check(4); doc_08_who_needs_police_check(4); doc_04_candidate_experience(3) | [x] Pass [ ] Fail | Co 3 ket qua sat nghia screening/police check. |

## Tong ket retrieval (cuoi Week 2)
- So query pass: 10/10
- So query fail: 0/10
- Van de chinh gap phai: can tiep tuc giam score bang nhau giua cac doc lien quan cao.
- Hanh dong da dieu chinh: rerank theo metadata topic/document_type/approved + bo sung map file_stem.
- Muc cai thien so voi Week 1: tang tu 6/10 len 10/10 tren cung bo 10 query benchmark.

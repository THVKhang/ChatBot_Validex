# Week 1 Completion Checklist

## Muc tieu tuan 1
- Chot scope de tai
- Dinh nghia architecture tong quat
- Tao project skeleton
- Chuan bi du lieu mau cho retrieval
- Demo duoc flow nho parser -> retriever -> generator

## Bang danh gia Done / Not done / Evidence
| Hang muc | Dieu kien dat | Trang thai | Evidence |
|---|---|---|---|
| 1. Chot scope | Tra loi ro 5 cau hoi: ai dung, input, output, lam gi, khong lam gi | Done | docs/problem_statement.md |
| 2. Kien truc tong quat | Co flow User prompt -> Parser -> Retriever -> Generator -> Output | Done | docs/architecture.md |
| 3. Project skeleton | Co app/, data/, docs/, tests/, README.md | Done | app/, data/, docs/, tests/, README.md |
| 4. Du lieu mau | Co 5-10 tai lieu mau trong raw va processed | Done (6 tai lieu) | data/raw/, data/processed/ |
| 5. Metadata co ban | Co schema va JSON mau gom id, title, source, topic, content | Done | docs/metadata_schema.md, data/metadata/documents.json |
| 6. Parser prototype | Co ham parse_user_input(text) va output format co dinh | Done | app/parser.py |
| 7. Retriever prototype | Query vao tra ve top ket qua lien quan | Done | app/retriever.py, notebooks/exploration.ipynb |
| 8. Demo mini flow | Chay thu 1 prompt va in parser/retrieval/draft | Done | app/main.py, notebooks/exploration.ipynb |
| 9. README | Mo ta project, cach chay, cau truc module | Done | README.md |
| 10. Bao cao tuan 1 | Co the trinh bay ngan gon 3 y voi file minh chung | Done | Muc "Tom tat bao cao" trong file nay |

## 5 dau hieu manh nhat de ket luan "Week 1 xong"
1. Co mo ta scope ro rang
2. Co architecture parser -> retriever -> generator
3. Co project skeleton day du
4. Co du lieu mau da xu ly
5. Co retrieval prototype chay duoc

## Moc danh gia
### Muc dat
- Chot scope
- Co architecture
- Co project skeleton
- Co du lieu mau
- Co retrieval prototype

### Muc dat tot
- Co parser basic
- Co metadata schema
- Co demo mini
- Co README day du
- Co tong ket tuan 1

## Tom tat bao cao (3 y)
1. Da chot scope va kien truc he thong.
2. Da chuan bi du lieu mau va tao cau truc project.
3. Da lam retrieval prototype dau tien va noi thu flow co ban.

## Ke hoach theo ngay (tham chieu)
- Ngay 1: Chot scope, viet problem statement
- Ngay 2: Ve architecture, chia module
- Ngay 3: Thu thap du lieu mau
- Ngay 4: Lam sach du lieu, tao metadata
- Ngay 5: Khoi tao cau truc project
- Ngay 6: Lam retriever prototype don gian
- Ngay 7: Noi flow tu prompt den ket qua demo

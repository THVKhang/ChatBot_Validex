from dataclasses import asdict, dataclass
import re

from app.parser import ParsedPrompt
from app.retriever import RetrievedDoc


@dataclass
class GeneratedBlog:
    title: str
    outline: list[str]
    draft: str
    sources_used: list[str]


def format_title(topic: str) -> str:
    compact = re.sub(r"\s+", " ", topic).strip(" .,!?:;\n\t")
    if not compact or compact.lower() == "current draft":
        return "Current Draft Update"

    words = compact.split()
    if len(words) > 14:
        words = words[:14]
    compact = " ".join(words)

    stop_words = {"a", "an", "the", "and", "or", "for", "to", "of", "in", "on", "with"}
    titled: list[str] = []
    for index, word in enumerate(compact.split()):
        if index > 0 and index < len(compact.split()) - 1 and word.lower() in stop_words:
            titled.append(word.lower())
        else:
            titled.append(word.capitalize())
    return " ".join(titled)


def generate_outline(parsed: ParsedPrompt) -> list[str]:
    return [
        f"Mo dau: {parsed.topic}",
        "Noi dung chinh: dinh nghia, boi canh, gia tri",
        "Huong dan thuc te va luu y",
        "Ket luan va call-to-action",
    ]


def generate_draft(parsed: ParsedPrompt, docs: list[RetrievedDoc]) -> str:
    references = "\n".join([f"- {doc.doc_id} (score={doc.score})" for doc in docs])
    snippets = "\n".join([f"- {doc.content[:200]}" for doc in docs[:2]])

    if not references:
        references = "- Khong tim thay tai lieu phu hop"
    if not snippets:
        snippets = "- Chua co snippet tham chieu"

    return (
        f"Tieu de goi y: {parsed.topic.title()}\n\n"
        f"Giong van: {parsed.tone}\n"
        f"Doi tuong doc gia: {parsed.audience}\n"
        f"Do dai: {parsed.length}\n\n"
        "Noi dung nhap:\n"
        f"Bai viet nay giai thich ve {parsed.topic}. "
        "Ban co the bo sung so lieu va case study noi bo de tang do tin cay.\n\n"
        "Snippet tham chieu:\n"
        f"{snippets}\n\n"
        "Nguon da truy xuat:\n"
        f"{references}\n"
    )


def generate_blog_output(
    parsed: ParsedPrompt,
    docs: list[RetrievedDoc],
    previous_draft: str | None = None,
) -> GeneratedBlog:
    outline = generate_outline(parsed)
    sources_used = [doc.doc_id for doc in docs]
    clean_title = format_title(parsed.topic)

    source_snippets = "\n".join([f"- {doc.content[:220]}" for doc in docs[:3]])
    if not source_snippets:
        source_snippets = "- No retrieval context available"

    grounded_points = "\n".join(
        [f"- [{doc.doc_id}] {doc.content[:140]}" for doc in docs[:3]]
    )
    if not grounded_points:
        grounded_points = "- Chua co diem du lieu de grounding"

    tone_line = parsed.tone.replace("_", " ")
    previous_excerpt = ""
    if previous_draft:
        previous_excerpt = previous_draft[:350].strip()

    if parsed.intent == "shorten":
        draft = (
            f"{clean_title}: police check giup doanh nghiep xac minh thong tin va giam rui ro trong tuyen dung. "
            f"Cho doi tuong {parsed.audience}, phien ban nay duoc rut gon voi giong van {tone_line}. "
            "Can lam ro 3 diem: muc dich kiem tra, doi tuong can kiem tra, va bo giay to can chuan bi. "
            "CTA goi y: lien he team ho tro de nhan checklist trien khai nhanh.\n\n"
            + (f"Ngu canh tu draft truoc:\n- {previous_excerpt}\n\n" if previous_excerpt else "")
            + "Ngu canh truy xuat:\n"
            + f"{source_snippets}"
        )
    elif parsed.intent == "rewrite":
        draft = (
            f"Phien ban rewrite cho chu de {clean_title}. Noi dung duoc dieu chinh theo tone {tone_line} "
            f"va huong den {parsed.audience}.\n\n"
            "Ban rewrite can giu cau truc ro rang: mo dau dinh nghia, phan giua la use case, ket la hanh dong tiep theo. "
            "Ngan ngu can nhat quan, tranh lap y, va uu tien thong tin co the ap dung ngay trong onboarding.\n\n"
            + (f"Ngu canh tu draft truoc:\n- {previous_excerpt}\n\n" if previous_excerpt else "")
            + "Ngu canh truy xuat:\n"
            + f"{source_snippets}"
        )
    elif parsed.length == "short":
        draft = (
            f"{clean_title} la buoc xac minh quan trong trong tuyen dung. "
            f"Bai viet huong den {parsed.audience} voi giong van {tone_line}, tap trung vao gia tri thuc te va huong dan ngan gon. "
            "Ban nen neu ro muc dich, thoi gian xu ly, va checklist giay to co ban de nguoi doc hanh dong ngay.\n\n"
            "Grounded points:\n"
            f"{grounded_points}\n\n"
            "Ngu canh truy xuat:\n"
            f"{source_snippets}"
        )
    elif parsed.length == "long":
        draft = (
            f"{clean_title} la mot chu de then chot trong quan tri rui ro va quality cua quy trinh tuyen dung. "
            f"Bai viet nay huong den {parsed.audience}, su dung giong van {tone_line}, va trinh bay theo huong de hieu nhung day du boi canh.\n\n"
            "Truoc het, police check dong vai tro nhu mot co che xac minh co ban de doanh nghiep giam thieu sai sot trong quyet dinh nhan su. "
            "Voi cac vi tri nhay cam, day khong chi la buoc ky thuat ma con lien quan den compliance va uy tin to chuc. "
            "Khi truyen thong dung cach, ung vien se hieu ro day la buoc bao ve ca hai ben thay vi mot thu tuc can tro.\n\n"
            "Tiep theo, mot bai blog chat luong nen phan tach ro doi tuong can kiem tra, bo giay to can nop, va timeline xu ly du kien. "
            "Dieu nay giup team backoffice thong nhat thong diep, giam so cau hoi lap lai, va tang toc do onboarding. "
            "Ngoai ra, bo sung vi du tinh huong thuc te se giup nguoi doc nhanh chong lien he voi cong viec hang ngay.\n\n"
            "Sau cung, nen ket bai bang mot checklist hanh dong: xac dinh yeu cau theo vai tro, thong bao minh bach cho ung vien, "
            "theo doi trang thai xu ly, va co dau moi ho tro khi phat sinh vuong mac. "
            "Voi cach viet nay, noi dung vua dat muc thong tin can thiet vua giu duoc tinh ung dung.\n\n"
            "Grounded points:\n"
            f"{grounded_points}\n\n"
            "Ngu canh truy xuat:\n"
            f"{source_snippets}"
        )
    else:
        draft = (
            f"{clean_title} la mot chu de quan trong trong quy trinh tuyen dung va compliance. "
            f"Bai viet nay huong den doi tuong {parsed.audience}, su dung giong van {tone_line}, "
            "nham giai thich de hieu va ap dung duoc ngay.\n\n"
            "Truoc het, police check giup doanh nghiep xac minh thong tin co ban, giam rui ro va tang do tin cay. "
            "Voi nhung vai tro nhay cam hoac moi truong can tinh an toan cao, buoc nay thuong duoc xem la tieu chuan toi thieu. "
            "Neu truyen thong khong ro rang, ung vien de bo quy trinh hoac hieu sai muc dich kiem tra.\n\n"
            "Tiep theo, doanh nghiep nen thong bao ky ve pham vi kiem tra, thoi gian xu ly va cac giay to can thiet. "
            "Cach trinh bay minh bach se cai thien candidate experience va giam tre han onboarding. "
            "Doi voi nhan vien backoffice, mot checklist ngan gon giup thong nhat cach huong dan giua cac team.\n\n"
            "Grounded points:\n"
            f"{grounded_points}\n\n"
            "Cuoi cung, noi dung blog nen ket hop dinh nghia, use case thuc te va huong dan hanh dong cu the. "
            "Ban co the ket bai bang CTA nhu: lien he team ho tro de nhan checklist police check phu hop tung vi tri.\n\n"
            "Ngu canh truy xuat:\n"
            f"{source_snippets}"
        )

    return GeneratedBlog(
        title=clean_title,
        outline=outline,
        draft=draft,
        sources_used=sources_used,
    )


def generated_blog_to_dict(output: GeneratedBlog) -> dict:
    return asdict(output)

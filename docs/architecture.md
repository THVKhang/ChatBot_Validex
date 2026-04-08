# Architecture

## Tong quan luong xu ly
User Prompt -> Parser/Intent Understanding -> Retrieval tu Knowledge Base -> Generation bang LLM -> Tra ve Blog Draft

## Module chinh

### Parser
- Input: prompt text
- Output: intent, topic, tone, audience, length
- Vai tro: chuan hoa yeu cau nguoi dung

### Retriever
- Input: query da chuan hoa
- Output: top tai lieu/chunks lien quan
- Vai tro: lay ngu canh phuc vu generation

### Generator
- Input: prompt da chuan hoa + tai lieu truy xuat
- Output: outline/blog draft
- Vai tro: sinh noi dung theo muc tieu

### Session Manager
- Input: lich su chat, ban nhap truoc do
- Output: context cho luot tiep theo
- Vai tro: ho tro hoi thoai nhieu luot

### UI
- Input: prompt nguoi dung
- Output: hien thi ket qua
- Vai tro: diem tuong tac cho demo prototype

## Huong tiep can
- Kien truc: LLM + RAG
- Giao dien: chatbot demo don gian (Streamlit)
- Du lieu: FAQ, blog cu, tai lieu mo ta dich vu, knowledge base mau

import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings

def load_and_split_pdfs(pdf_paths):
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for path in pdf_paths:
        if not os.path.exists(path):
            print(f"❌ Файл не найден: {path}")
            continue

        loader = PyMuPDFLoader(path)
        documents = loader.load()
        chunks = splitter.split_documents(documents)
        print(f"✅ {len(chunks)} чанков из {path}")
        all_chunks.extend(chunks)

    return all_chunks

def build_vectorstore(chunks, persist_path):
    embeddings = ChatOllama(model="mistral", base_url="http://host.docker.internal:8080")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(persist_path)
    print(f"✅ Векторная база сохранена в: {persist_path}")

if __name__ == "__main__":
    pdf_files = ["data/file1.pdf"]
    persist_dir = "vectorstore-chekhov/"  # 👈 новая директория

    chunks = load_and_split_pdfs(pdf_files)
    if chunks:
        build_vectorstore(chunks, persist_dir)
    else:
        print("❌ Чанки не найдены — база не будет создана.")

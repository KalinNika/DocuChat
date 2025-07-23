from langchain_community.vectorstores import FAISS
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import time
import requests

def wait_for_ollama(timeout=60):
    print("⏳ Ждём запуска Ollama...")
    url = os.getenv("OLLAMA_BASE_URL", "http://localhost:8080")
    start = time.time()
    while True:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                print("✅ Ollama доступен.")
                return
        except Exception:
            pass
        if time.time() - start > timeout:
            raise TimeoutError("❌ Ollama не ответил за 60 секунд.")
        time.sleep(1)

# Вызов перед созданием OllamaEmbeddings
wait_for_ollama()

# --- Параметры ---
persist_dir = "vectorstore-chekhov"
model_name = "mistral"
show_chunks = True

# --- ЯВНЫЙ URL без os.getenv ---
ollama_url = "http://192.168.1.5:11434"

# --- Векторка ---
embeddings = OllamaEmbeddings(model=model_name, base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:8080")
)
vectorstore = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# --- Модель ---
llm = ChatOllama(model=model_name, base_url=ollama_url)

# --- Промпт ---
template = """Ты — ассистент. Используй только контекст:

Контекст:
{context}

Вопрос:
{question}

Ответ:"""

prompt = PromptTemplate(input_variables=["context", "question"], template=template)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# --- Вопрос ---
query = input("Вопрос: ")
result = qa_chain.invoke({"query": query})
print(result["result"])


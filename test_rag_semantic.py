import time
import requests

try:
    requests.get("http://host.docker.internal:11434").raise_for_status()
    print("✅ Ollama доступен.")
except:
    raise Exception("❌ Ollama не доступен.")


wait_for_ollama()
from langchain_community.vectorstores import FAISS
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from bert_score import score

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
import re

# === Параметры ===
persist_dir = "vectorstore-chekhov"
model_name = "mistral"
threshold = 0.80

# === Векторная база ===
embeddings = OllamaEmbeddings(model=model_name, base_url="http://host.docker.internal:8080")
vectorstore = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
llm = ChatOllama(model=model_name, base_url="http://host.docker.internal:8080")


# === Промпт ===
template = """
Ты — интеллектуальный помощник. Отвечай исключительно на **русском языке**, используя **только предоставленный контекст**. Не придумывай факты.

📎 Контекст:
{context}

❓ Вопрос:
{question}

🧾 Ответ (на русском языке, кратко и по сути):
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# === Тесты ===
tests = [
    {
        "question": "Кто был женат на Каролине Карловне?",
        "expected": "Майор Щелколобов был женат на Каролине Карловне."
    },
    {
        "question": "Что сделала майорша в лодке, из-за чего перевернулась лодка?",
        "expected": "Лодка перевернулась, когда майорша вырвала плеть из рук майора."
    },
    {
        "question": "Кем работал Иван Павлович в начале рассказа?",
        "expected": "Иван Павлович работал у майора в должности дворника."
    },
    {
        "question": "Что пообещала майорша Ивану Павловичу, если он её спасёт?",
        "expected": "Майорша пообещала Ивану Павловичу выйти за него замуж, если он её спасёт."
    },
    {
        "question": "Чем закончилась история Ивана Павловича после спасения обоих?",
        "expected": "После спасения Иван Павлович продолжил службу в волостном правлении."
    }
]

# === Метрика: BERTScore ===
def calc_similarity(answer: str, expected: str) -> float:
    P, R, F1 = score([answer], [expected], lang="ru", model_type="xlm-roberta-large")
    return F1.item()

# === Очистка текста ===
def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()

# === Запуск тестов ===
def run_tests():
    success = 0
    print("🚀 Запуск автотестов...")
    for i, test in enumerate(tests, 1):
        question = test["question"]
        expected = test["expected"]
        result = qa_chain.invoke({"query": question})
        answer = result["result"]
        similarity = calc_similarity(normalize(answer), normalize(expected))

        print(f"🧪 Тест {i}: {question}")
        print("📄 Ответ: ", answer)
        print("📊 Сходство:", round(similarity, 2))

        if similarity >= threshold:
            print("✅ УСПЕХ")
            success += 1
        else:
            print("❌ НЕСООТВЕТСТВИЕ")
        print("-" * 80)

    print(f"📊 Результат: {success} из {len(tests)} тестов пройдено.")

if __name__ == "__main__":
    run_tests()

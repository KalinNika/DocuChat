# 1. Базовый образ Python
FROM python:3.11-slim

# 2. Рабочая директория
WORKDIR /app

# 3. Устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Копируем весь проект
COPY . .

# 5. Указываем переменную окружения для Ollama (локальный Ollama хост)
ENV OLLAMA_BASE_URL=http://host.docker.internal:8080

# 6. Запуск автотестов по умолчанию
CMD ["python", "tests/test_rag_semantic.py"]


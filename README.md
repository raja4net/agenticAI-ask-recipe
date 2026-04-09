# 🤖 Local Agentic AI Recipe + Web Assistant using Ollama, PhiData, pgvector, Docker, and DuckDuckGo fallback search
![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-green)
![Docker](https://img.shields.io/badge/Docker-Required-blue)
![Postgres](https://img.shields.io/badge/PostgreSQL-pgvector-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

A fully local **Agentic AI system** that:
- 📚 Reads recipes from PDFs
- 🧠 Stores embeddings in PostgreSQL (pgvector)
- 🤖 Uses Ollama for local LLM + embeddings
- 🌐 Falls back to web search when needed
- 💬 Supports conversational memory (follow-up questions)

---

## 🚀 Demo

### Example interaction:
You: give me a recipe of butter chicken

RecipeGenie:
[Full recipe with ingredients, steps, calories]

You: is 540 calories per serving?

RecipeGenie:
Yes, according to the recipe, calories per serving are 540.

---

## 🧠 Architecture
User Input -> Knowledge Base (PDF + pgvector) -> (if not found) Web Search (DuckDuckGo) -> Ollama LLM (llama3.1) -> Final Answer

---

## 📦 Tech Stack

- **LLM**: Ollama (`llama3.1`)
- **Embeddings**: Ollama (`nomic-embed-text`)
- **Vector DB**: PostgreSQL + pgvector (Docker)
- **Agent Framework**: PhiData
- **Web Search**: DuckDuckGo
---

## 🛠️ Installation Guide

### 1️⃣ Clone Repo

```bash
git clone https://github.com/raja4net/agenticAI-ask-recipe.git
cd agenticAI-ask-recipe
```

2️⃣ Setup Python Environment
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
```

# Windows
venv\Scripts\activate
___
3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
___

4️⃣ Install Ollama

👉 https://ollama.com/download

Verify:
```bash
ollama --version
```
___

5️⃣ Download Models
```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```
___

6️⃣ Start PostgreSQL + pgvector (Docker)
```bash
docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  phidata/pgvector:16
```
___

7️⃣ Add PDFs
Create folder:
```bash
mkdir pdfs
```

Add your recipe PDFs inside.
___

8️⃣ Run the App
```bash
python AskLocalOrWeb.py
```
___
📂 Project Structure
```bash
.
├── AskLocalOrWeb.py
├── requirements.txt
├── README.md
└── pdfs/
```

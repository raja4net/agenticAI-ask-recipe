from pathlib import Path
from phi.agent import Agent
from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader
from phi.vectordb.pgvector import PgVector
from phi.model.ollama import Ollama
from phi.embedder.ollama import OllamaEmbedder
from phi.tools.duckduckgo import DuckDuckGo

# -----------------------------
# Config
# -----------------------------
DB_URL = "postgresql+psycopg://ai:ai@localhost:5532/ai"
PDF_FOLDER = "pdfs"
TABLE_NAME = "recipes"

KB_FALLBACK = "Could not find that answer from its knowledge base."
WEB_FALLBACK = "Could not find that answer on the web."
EXIT_WORDS = {"exit", "quit"}

# -----------------------------
# Build knowledge base
# -----------------------------
reader = PDFReader(chunk=True)

embedder = OllamaEmbedder(
    model="nomic-embed-text",
    dimensions=768,
)

knowledge_base = PDFKnowledgeBase(
    path=PDF_FOLDER,
    reader=reader,
    vector_db=PgVector(
        table_name=TABLE_NAME,
        db_url=DB_URL,
        embedder=embedder,
    ),
)

all_documents = []
for pdf_file in Path(PDF_FOLDER).glob("*.pdf"):
    print(f"Reading {pdf_file.name}...")
    docs = knowledge_base.reader.read(pdf=pdf_file)
    docs = [doc for doc in docs if getattr(doc, "content", None) and doc.content.strip()]
    all_documents.extend(docs)

if not all_documents:
    raise ValueError("No non-empty documents were found in the pdfs folder.")

knowledge_base.vector_db.drop()
knowledge_base.vector_db.create()
knowledge_base.load_documents(all_documents)

print(f"\nLoaded {len(all_documents)} documents into the knowledge base.\n")

# -----------------------------
# Agent 1: Knowledge-base agent
# -----------------------------
kb_agent = Agent(
    name="RecipeGenieKB",
    model=Ollama(id="llama3.1"),
    knowledge=knowledge_base,
    add_context=True,
    search_knowledge=False,
    prevent_hallucinations=True,
    num_documents=5,
    markdown=True,
    instructions=[
        "You are a recipe assistant.",
        "You MUST answer only from the provided knowledge base context.",
        f"If the answer is not found in the knowledge base, reply exactly: {KB_FALLBACK}",
        "Do not make up recipes, ingredients, prep time, calories, allergens, or facts.",
        "Do not describe your training data or your general knowledge.",
        "If the user asks what recipes are available, only answer from retrieved context.",
    ],
)

# -----------------------------
# Agent 2: Web-search fallback agent
# -----------------------------
web_agent = Agent(
    name="RecipeGenieWeb",
    model=Ollama(id="llama3.1"),
    tools=[DuckDuckGo()],
    show_tool_calls=False,
    prevent_hallucinations=True,
    markdown=True,
    instructions=[
        "You are a web research assistant.",
        "Answer using web results only.",
        f"If you cannot find a reliable answer, reply exactly: {WEB_FALLBACK}",
        "If the question depends on a previously discussed recipe, use the provided recipe context.",
        "Keep the answer concise and practical.",
        "Include source links when useful.",
    ],
)

# -----------------------------
# Helpers
# -----------------------------
def extract_text(response) -> str:
    text = getattr(response, "content", str(response))
    return text.strip() if text else ""


def normalize_kb_answer(text: str) -> str:
    if not text:
        return KB_FALLBACK

    lower = text.lower().strip()

    fallback_signals = [
        "i don't know",
        "i dont know",
        "i do not know",
        "could not find",
        "not found in the knowledge base",
        "not in the knowledge base",
        "no relevant context",
        "no recipes available",
    ]

    if any(signal in lower for signal in fallback_signals):
        return KB_FALLBACK

    return text


def normalize_web_answer(text: str) -> str:
    if not text:
        return WEB_FALLBACK

    lower = text.lower().strip()

    fallback_signals = [
        "i don't know",
        "i dont know",
        "i do not know",
        "could not find",
        "no reliable answer",
        "no relevant results",
    ]

    if any(signal in lower for signal in fallback_signals):
        return WEB_FALLBACK

    return text


def should_short_circuit_listing_query(user_question: str) -> bool:
    q = user_question.lower()
    list_queries = [
        "which recipes have you got",
        "which recipes do you have",
        "what recipes do you have",
        "what recipes have you got",
        "list recipes",
        "show all recipes",
        "what do you have in your knowledge base",
        "what is in your knowledge base",
    ]
    return any(item in q for item in list_queries)


def is_followup_question(user_question: str) -> bool:
    q = user_question.lower().strip()

    followup_signals = [
        "calories",
        "how many calories",
        "prep time",
        "preparation time",
        "cook time",
        "cooking time",
        "ingredients",
        "allergens",
        "servings",
        "per serving",
        "is it",
        "is this",
        "is that",
        "does it",
        "what about",
        "how much",
        "how long",
    ]

    return any(signal in q for signal in followup_signals)


def enrich_with_memory(user_question: str, last_question: str | None, last_context: str | None) -> str:
    if not is_followup_question(user_question):
        return user_question

    if not last_context:
        return user_question

    return (
        f"Previous recipe/question: {last_question}\n\n"
        f"Previous answer/context:\n{last_context}\n\n"
        f"Now answer this follow-up question strictly based on the above context if possible:\n"
        f"{user_question}"
    )


# -----------------------------
# Interactive loop
# -----------------------------
print("RecipeGenie is ready.")
print("Ask a recipe question.")
print("It will check the PDF knowledge base first, then the web if needed.")
print("Type 'exit' to quit.\n")

last_question = None
last_context = None

while True:
    user_question = input("You: ").strip()

    if not user_question:
        continue

    if user_question.lower() in EXIT_WORDS:
        print("Goodbye!")
        break

    if should_short_circuit_listing_query(user_question):
        print(f"\nRecipeGenie: {KB_FALLBACK}\n")
        continue

    # Add state memory for follow-up questions
    effective_question = enrich_with_memory(user_question, last_question, last_context)

    try:
        kb_response = kb_agent.run(effective_question)
        kb_answer = normalize_kb_answer(extract_text(kb_response))

        if kb_answer != KB_FALLBACK:
            print(f"\nRecipeGenie (knowledge base): {kb_answer}\n")
            last_question = user_question
            last_context = kb_answer
            continue

        print("\nRecipeGenie: Not found in knowledge base. Searching the web...\n")

        web_prompt = (
            f"User question: {user_question}\n\n"
            f"Previous recipe/question: {last_question or 'None'}\n\n"
            f"Previous answer/context:\n{last_context or 'None'}\n\n"
            "If the current question is a follow-up, use the previous context to understand it. "
            "Prefer recipe pages or reliable cooking sources. "
            "If the answer is not reliably available, reply exactly: "
            f"{WEB_FALLBACK}"
        )

        web_response = web_agent.run(web_prompt)
        web_answer = normalize_web_answer(extract_text(web_response))

        print(f"RecipeGenie (web): {web_answer}\n")

        last_question = user_question
        if web_answer != WEB_FALLBACK:
            last_context = web_answer

    except Exception as e:
        print(f"\nRecipeGenie: Error: {e}\n")
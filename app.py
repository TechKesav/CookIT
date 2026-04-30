import os

import markdown
import requests
import torch
from flask import Flask, render_template, request
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

app = Flask(__name__)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
DEBUG_MODE = os.getenv("FLASK_DEBUG", "0") == "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device},
)
vector_db = Chroma(persist_directory="./chroma_db_nccn1", embedding_function=embedding_function)


def get_relevant_context(query):
    context = ""
    results = vector_db.similarity_search(query, k=3)
    for result in results:
        context += result.page_content + "\n"
    return context[:4000]


def build_prompt(query, context):
    return f"""
You are a recipe assistant.
Use only the provided CONTEXT.
If a detail is missing in context, write "Not available in source".

User ingredients or question: {query}

CONTEXT:
{context}

Output format:
1. Recipe name
2. Ingredients
3. Steps
4. Estimated cooking time
5. Nutrition notes (if present in context)
""".strip()


def generate_with_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    except requests.RequestException as exc:
        raise RuntimeError(
            "Unable to connect to local Ollama. Start Ollama and run: ollama run llama3.2"
        ) from exc

    if response.status_code != 200:
        raise RuntimeError(f"Ollama error ({response.status_code}): {response.text}")

    data = response.json()
    text = data.get("response", "").strip()
    if not text:
        raise RuntimeError("Ollama returned empty output.")
    return text


def format_user_error(exc):
    error_text = str(exc)
    if "ollama" in error_text.lower():
        return "Local Ollama is not reachable. Start Ollama and run: ollama run llama3.2"
    return f"Error: {error_text}"


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form.get("prompt", "").strip()
        if not query:
            return render_template("index.html", response=markdown.markdown("Please enter ingredients."))

        try:
            context = get_relevant_context(query)
            prompt = build_prompt(query, context)
            answer = generate_with_ollama(prompt)
            html_response = markdown.markdown(answer)
        except Exception as exc:
            html_response = markdown.markdown(format_user_error(exc))

        return render_template("index.html", response=html_response)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=DEBUG_MODE, use_reloader=False)

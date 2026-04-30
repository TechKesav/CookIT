import os.path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as genai
from flask import Flask, render_template, request, url_for
import markdown
import torch

app = Flask(__name__, static_folder='static')

# Use GPU acceleration if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
languages = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'ta': 'Tamil',
    'zh-cn': 'Chinese',
    'ja': 'Japanese',
    'hi': 'Hindi'
}

def get_relevant_context(query):
    # Get relevant context from vector store
    context = ""
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                               model_kwargs={'device': device})
    vector_db = Chroma(persist_directory="./chroma_db_nccn1", embedding_function=embedding_function)
    search_result = vector_db.similarity_search(query, k=6)
    for result in search_result:
        context += result.page_content + "\n"
    return context


def rag_prompt(query, context,language_code):
    escaped = context.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = (f"""
    Provide a detailed and technical answer to the question: '{query}'.
    Please include all relevant information and specifications with time.
    Provide a nutritional details.
    Provide the recipe name.
    CONTEXT: '{context}'
    Translate the answer to {language_code} only.
    provide the suitable emoji for the sentences.

     Answer:
    """).format(query=query, context=context,language_code=language_code)
    return prompt


def generate_ans(prompt):
    # Generate answer using generative model
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(model_name='gemini-pro')
    answer = model.generate_content(prompt)
    return answer.text


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        query = request.form["prompt"]
        if query=="" or query==" ":
            return render_template("index3.html", response="Provide the valid Input...... Please🥹", languages=languages)
        selected_language = request.form["language"]
        print(f"Selected Language: {selected_language}")
        print(query)

        context = get_relevant_context(query)
        prompt = rag_prompt(query, context,selected_language)
        answer = generate_ans(prompt)
        html_response = markdown.markdown(answer)

        return render_template("index3.html", response=html_response, languages=languages)
    return render_template("index3.html",languages=languages)


if __name__ == "__main__":
    app.run()

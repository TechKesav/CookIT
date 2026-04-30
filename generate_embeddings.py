from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

pdf_files = [
    "./Document.pdf",
    "./Recipes Book.pdf",
    "./Food_Recipes_From_AYUSH.pdf",
    "./DFW_Oct2013_RecipesAndCuisine.pdf",
    "./Step_by_Step_Guide_to_Indian_cooking_Khalid_Aziz.pdf",
    "./pdfcoffee.com_cultinst-of-kerala-pdf-free.pdf",
    "./Indian-Recipes (1).pdf",
]

loaders = [PyPDFLoader(path) for path in pdf_files]
docs = []
for loader in loaders:
    docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(docs)

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

vector_store = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db_nccn1")
print(vector_store._collection.count())

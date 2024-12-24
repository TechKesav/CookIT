import collections

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# loaders=[PyPDFLoader("./Document.pdf")]
# docs=[]
#
# for file in loaders:
#     docs.extend(file.load())
#
# print(len(docs))
# text_spliter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
# docs=text_spliter.split_documents(docs)
# embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={'device':'cpu'})
# vector_store=Chroma.from_documents(docs,embedding_function,persist_directory="./chroma_db_nccn1")
# print(vector_store._collection.count())

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load PDF document
loaders = [PyPDFLoader("./Document.pdf"),PyPDFLoader("./Recipe-Book.pdf"),PyPDFLoader("./recipe-book (1).pdf"),PyPDFLoader("./SHARAN-Recipe-Book.pdf"),PyPDFLoader("./english-recipe-book.pdf"),PyPDFLoader("./Food_Recipes_From_AYUSH.pdf"),PyPDFLoader("./Recipe-Book.pdf"),PyPDFLoader("./Recipes Book.pdf"),PyPDFLoader("./DFW_Oct2013_RecipesAndCuisine.pdf"),PyPDFLoader("./Step_by_Step_Guide_to_Indian_cooking_Khalid_Aziz.pdf"),PyPDFLoader("./Kerala Cuisine Contest 2020-Veg.pdf"),PyPDFLoader("./Indian-Recipes (1).pdf"),PyPDFLoader("./chasingtheflavoursofmalabar.pdf")]
docs = []
for file in loaders:
    docs.extend(file.load())

# Split documents into chunks
text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_spliter.split_documents(docs)

# Create vector store
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                           model_kwargs={'device': 'cpu'})
vector_store = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db_nccn1")
print(vector_store._collection.count())

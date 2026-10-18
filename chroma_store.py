from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


def create_chroma_vectorstore(document=[], embeddings_model="text-embedding-3-small"):

    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory="./chroma_db", embedding_function=embeddings_model
    )
    vectorstore.delete_collection()
    vectorstore = Chroma(
        persist_directory="./chroma_db", embedding_function=embeddings_model
    )
    vectorstore.add_documents(document)
    return vectorstore

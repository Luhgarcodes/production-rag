from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from ollama import embeddings
import numpy as np
load_dotenv()


embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


def basic_embeddings():
    text="what is machine Learning?"
    single_embedding = embeddings_model.embed_query(text)
    print(f"Vector dimensions:{len(single_embedding)}")
    print(f"First 5 Values:{single_embedding[:5]}")
    print(f"Vector norm: {np.linalg.norm(single_embedding):.4f}")
    
def batch_embeddings():
    text=[
        "what is machine learning","Explain the concept of overfitting in ML","How does the neural network work?"
    ]
    batch_embedding = embeddings_model.embed_documents(text)
    for i, emb in enumerate(batch_embedding):
        print(f"Text {i+1} - Vector dimensions:{len(emb)}")
        print(f"Text {i+1} - First 5 Values: {emb[:5]}")
        print(f"Text {i+1} - Vector norm: {np.linalg.norm(emb):.4f}")


def similarity_search():
    docs = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Machine learning enables AI applications",
        "Deep learning uses neural networks",
        "Cats are popular pets",
    ]
    query = "What Programing language exists?"
    doc_vector = embeddings_model.embed_documents(docs)
    query_vector = embeddings_model.embed_query(query)

    def cosine_similarity(vec1,vec2):
        return np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

    similarities = [cosine_similarity(query_vector, doc_vec) for doc_vec in doc_vector]

    # rank documents by similarity
    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)
    print(f"Query:{query}")    
    print(f"Ranked by similarity")    
    for doc,sim in ranked_docs:
        print(f"Score:{sim:.4f}, Document:{doc}")

    

if __name__ == "__main__":
    # basic_embeddings()
    # batch_embeddings()
    similarity_search()
    
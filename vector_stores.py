from chroma_store import create_chroma_vectorstore
import tempfile
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_chroma import Chroma

load_dotenv()

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
# Sample documents
SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "vector_guide", "topic": "database"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for more accurate LLM responses.",
        metadata={"source": "rag_guide", "topic": "architecture"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "embeddings_guide", "topic": "fundamentals"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "chroma_docs", "topic": "database"},
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook.",
        metadata={"source": "faiss_docs", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed vector database service for production workloads.",
        metadata={"source": "pinecone_docs", "topic": "database"},
    ),
]

vectorstore = create_chroma_vectorstore(SAMPLE_DOCS)


def chroma_basics():

    print(
        f"Vector store created {vectorstore._collection.count()} documents and persisted"
    )

    query = "what is langchain"
    results = vectorstore.similarity_search(query, k=2)

    print(f"Top 2 results for query: '{query}'")
    for i, doc in enumerate(results, 1):
        print(f"Result {i+1}:{doc.page_content}(Source:{doc.metadata['source']})")


def similarity_search_with_scores():

    # perform similarity search with scores
    query = "Explain vector stores."
    results_with_scores = vectorstore.similarity_search_with_score(query, k=3)

    print(f"Top 3 results with scores for query '{query}':")
    for i, (doc, score) in enumerate(results_with_scores):
        final_score = 1 / (1 + score)  # Convert distance to similarity
        print(
            f"Result {i+1}: {doc.page_content} (Score: {final_score:.4f}, Source: {doc.metadata['source']})"
        )


def metadata_filtering():
    query = "What database are available?"
    results = vectorstore.similarity_search(query, k=5)
    print(f"Results without filtering metadata for query {query}")
    for i, doc in enumerate(results):
        print(f"Results {i+1}:{doc.page_content}(source:{doc.metadata['source']})")

    filter_criteria = {"topic": "database"}
    results_filtered = vectorstore.similarity_search(query, k=5, filter=filter_criteria)
    print("----------------------------------------------------------------")
    print(f"Results with filtering metadata for query {query}")
    for i, doc in enumerate(results_filtered):
        print(f"Results {i+1}:{doc.page_content}(Source:{doc.metadata['source']})")


if __name__ == "__main__":
    # vectorstore.delete()
    # chroma_basics()
    # similarity_search_with_scores()
    metadata_filtering()

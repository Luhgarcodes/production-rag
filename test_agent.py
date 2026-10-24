from app.agent import ProductionAgent

agent = ProductionAgent()

queries = [
    "What is LangGraph in one sentence?",
    "What is 2 + 2?",
    "Explain the difference between RAG and fine-tuning in 2 sentences.",
]

for query in queries:
    print(f"Query: {query}")
    response = agent.invoke(query)
    print(f"Response: {response}\n")

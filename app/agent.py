from typing import Optional
from typing_extensions import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langsmith import traceable

from app.config import get_settings
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    error: Optional[str]
    retry_count: int
    model_used: str


class ProductionAgent:

    def __init__(self):
        settings = get_settings()

        self.primary_model_name = settings.primary_model
        self.fallback_model_name = settings.fallback_model

        self.primary_llm = ChatOpenAI(
            model=self.primary_model_name,
            temperature=0,
            timeout=30,
            max_retries=0,
        )

        self.fallback_llm = ChatOpenAI(
            model=self.fallback_model_name,
            temperature=0,
            timeout=30,
            max_retries=0,
        )

        self.max_retries = settings.max_retries
        self.graph = self._build_graph()

    def _build_graph(self):

        def process_message(state: AgentState) -> dict:
            try:
                response = self.primary_llm.invoke(state["messages"])

                return {
                    "messages": [response],
                    "model_used": self.primary_model_name,
                    "error": None,
                }

            except Exception as e:
                return {
                    "error": str(e),
                    "retry_count": state["retry_count"] + 1,
                    "model_used": "",
                }

        def try_fallback(state: AgentState) -> dict:
            try:
                response = self.fallback_llm.invoke(state["messages"])

                return {
                    "messages": [response],
                    "error": None,
                    "model_used": self.fallback_model_name,
                }

            except Exception as e:
                return {
                    "error": str(e),
                    "model_used": "",
                }

        def handle_error(state: AgentState) -> dict:
            error_message = (
                f"Both primary and fallback models failed. "
                f"Last error: {state.get('error')}. "
                f"Primary retries attempted: {state['retry_count']}. "
                f"Please try again."
            )

            return {
                "messages": [AIMessage(content=error_message)],
                "model_used": "error_handler",
            }

        def route_after_process(state: AgentState) -> str:
            if state.get("error") is None:
                return "done"

            if state["retry_count"] < self.max_retries:
                return "retry"

            return "fallback"

        def route_after_fallback(state: AgentState) -> str:
            if state.get("error") is None:
                return "done"

            return "error"

        graph = StateGraph(AgentState)

        graph.add_node("process", process_message)
        graph.add_node("fallback", try_fallback)
        graph.add_node("error", handle_error)

        graph.add_edge(START, "process")

        graph.add_conditional_edges(
            "process",
            route_after_process,
            {
                "done": END,
                "retry": "process",
                "fallback": "fallback",
            },
        )

        graph.add_conditional_edges(
            "fallback",
            route_after_fallback,
            {
                "done": END,
                "error": "error",
            },
        )

        graph.add_edge("error", END)

        return graph.compile()

    @traceable(name="production_agent_invoke")
    def invoke(self, query: str) -> dict:
        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=query)],
                "error": None,
                "retry_count": 0,
                "model_used": "",
            }
        )

        return {
            "response": result["messages"][-1].content,
            "model_used": result.get("model_used", "unknown"),
            "error": result.get("error"),
        }


def demo_agent():
    agent = ProductionAgent()

    queries = [
        "What is LangGraph in one sentence?",
        "What is 2 + 2?",
        "Explain the difference between RAG and fine-tuning in 2 sentences.",
    ]

    for query in queries:
        print(f"\nQuery: {query}")

        result = agent.invoke(query)

        print(f"Response: {result['response']}")
        print(f"Model used: {result['model_used']}")

        if result["error"]:
            print(f"Error: {result['error']}")


if __name__ == "__main__":
    demo_agent()

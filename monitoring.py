"""
Monitoring and Logging for Production
Structured logging, metrics, and alerts
"""

from datetime import timezone
from datetime import datetime
import logging
import json
from langchain_openai import ChatOpenAI


class JSONFormatter:
    """format logs as json for log aggregation"""

    def format(self, record):
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }

        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)

        return json.dumps(log_obj)


class MetricsCollector:
    """Collect and aggregate metrics."""

    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "errors_total": 0,
            "latency_sum": 0,
            "latency_count": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def record_request(
        self,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        error: bool = False,
        cache_hit: bool = False,
    ):
        self.metrics["requests_total"] += 1
        self.metrics["latency_sum"] += latency_ms
        self.metrics["latency_count"] += 1
        self.metrics["tokens_input"] += input_tokens
        self.metrics["tokens_output"] += output_tokens

        if error:
            self.metrics["errors_total"] += 1

        if cache_hit:
            self.metrics["cache_hits"] += 1
        else:
            self.metrics["cache_misses"] += 1

    def get_summary(self) -> dict:
        avg_latency = (
            self.metrics["latency_sum"] / self.metrics["latency_count"]
            if self.metrics["latency_count"] > 0
            else 0
        )
        error_rate = (
            self.metrics["errors_total"] / self.metrics["requests_total"]
            if self.metrics["requests_total"] > 0
            else 0
        )
        cache_hit_rate = (
            self.metrics["cache_hits"]
            / (self.metrics["cache_hits"] + self.metrics["cache_misses"])
            if (self.metrics["cache_hits"] + self.metrics["cache_misses"]) > 0
            else 0
        )

        return {
            "total_requests": self.metrics["requests_total"],
            "total_errors": self.metrics["errors_total"],
            "error_rate": f"{error_rate:.2%}",
            "avg_latency_ms": round(avg_latency, 2),
            "total_input_tokens": self.metrics["tokens_input"],
            "total_output_tokens": self.metrics["tokens_output"],
            "cache_hit_rate": f"{cache_hit_rate:.2%}",
        }


def setup_logging():
    """Setup logging."""
    logger = logging.getLogger("langgraph_app")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    return logger


class InstrumentedLLM:
    """LLM with full instrumentation"""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.metrics = MetricsCollector()
        self.logger = setup_logging()


def demo_monitoring():
    """Demonstrate monitoring."""
    llm = InstrumentedLLM()

    print("Monitoring Demo:\n")

    queries = [
        "What is Python?",
        "Explain machine learning.",
        "What is 2 + 2?",
    ]

    for query in queries:
        result = llm.invoke(query)
        print(f"Query: {query[:30]}... -> {result[:30]}...")

    print("\nMetrics Summary:")
    summary = llm.metrics.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    # logger = setup_logging()
    # logger.info("Logging setup complete", extra={"extra_data": {"app": "langgraph"}})
    demo_monitoring()

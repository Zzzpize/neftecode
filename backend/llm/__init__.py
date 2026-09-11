"""LLM-слой: GigaChat-клиент, MCP-инструменты (read-only), форматтер и Q&A."""

from llm.formatter import Formatter, format_recommendation, format_recommendation_fallback
from llm.gigachat_client import (
    GigaChatClient,
    LLMResponse,
    LLMUnavailableError,
    ToolCall,
)
from llm.guardrails import (
    collect_numbers,
    extract_numbers,
    find_invented_numbers,
    validate_numbers,
)
from llm.mcp_server import MCPServer
from llm.qa_handler import QaHandler, answer

__all__ = [
    "GigaChatClient",
    "LLMResponse",
    "LLMUnavailableError",
    "ToolCall",
    "MCPServer",
    "Formatter",
    "format_recommendation",
    "format_recommendation_fallback",
    "collect_numbers",
    "extract_numbers",
    "find_invented_numbers",
    "validate_numbers",
    "QaHandler",
    "answer",
]


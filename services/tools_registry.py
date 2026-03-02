"""
Compatibility facade for tool definitions and execution.
"""

from services.tool_dispatch import execute_tool
from services.tool_handlers import TOOL_HANDLERS, list_tool_summaries
from services.tool_specs import TOOL_SPECS

__all__ = [
    "TOOL_HANDLERS",
    "TOOL_SPECS",
    "execute_tool",
    "list_tool_summaries",
]

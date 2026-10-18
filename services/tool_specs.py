TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the live web for current or missing information. Returns leads "
                "(title, URL, snippet); use summarize_url to open a promising result when "
                "the answer depends on the page, then synthesize the answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "Search query"},
                    "num": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                    "safe": {"type": "string", "enum": ["off", "active"], "default": "off"},
                    "gl": {"type": "string", "description": "Country code, e.g., 'us'"},
                    "lr": {"type": "string", "description": "Language restrict, e.g., 'lang_en'"},
                    "image": {
                        "type": "boolean",
                        "description": (
                            "Keyword-based image results for q. This is not a reverse-image lookup; "
                            "use reverse_image_search for an attached image."
                        ),
                        "default": False,
                    },
                },
                "required": ["q"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_agent_run_status",
            "description": (
                "Retrieve privacy-bounded orchestration status for this user's current "
                "conversation scope: provider/model, completion state, tools actually run, "
                "retries, approvals, duration, and public evidence URLs. Use this when the "
                "user asks whether a search/tool really ran or wants task progress/audit data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Optional exact run ID. Omit for the latest completed run.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 3,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reverse_image_search",
            "description": (
                "Perform a genuine content-based reverse-image lookup on an image attached "
                "to the current Discord request. Auto mode uses Google Cloud Vision first "
                "and escalates a no-match public URL to configured Google Lens. Returns the "
                "provider chain, exact/partial matches, source pages, visual candidates, and "
                "best-guess labels. Use this—not web_search or keyword image search—when the "
                "user asks where an image came from, to find its source, or to identify a "
                "matching panel."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_index": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                        "description": "Zero-based attached-image index.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["all", "exact", "visual"],
                        "default": "all",
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 10,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather or a short forecast for a place name or address.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Place or address, e.g. 'Raleigh NC'."},
                    "range": {
                        "type": "string",
                        "enum": ["current", "24h", "7d"],
                        "description": "current conditions, next 24 hours, or next 7 days.",
                        "default": "current",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_quote",
            "description": "Fetch latest stock price and change for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'AAPL'."},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_url",
            "description": (
                "Open and read an HTTP/HTTPS URL. Fetches the page, extracts its main "
                "content, and returns a condensed text block for answering questions, "
                "checking claims, or summarizing. Use whenever a relevant URL appears in "
                "the user's request or after web_search identifies a useful source."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "HTTP/HTTPS URL to summarize."},
                    "max_len": {
                        "type": "integer",
                        "description": "Max characters of condensed text.",
                        "minimum": 1000,
                        "maximum": 12000,
                        "default": 6000,
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_youtube_transcript",
            "description": "Return the raw transcript text for a YouTube URL if available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full YouTube URL (watch?v=… or youtu.be/…)"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_recent_commits",
            "description": "Get my recent git commits. Use this to answer questions about what I've changed recently.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "description": "Number of commits to fetch (max 50)", "default": 10},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit_diff",
            "description": "Get the diff/changes for a specific commit by SHA. Use after git_recent_commits to see details.",
            "parameters": {
                "type": "object",
                "properties": {"sha": {"type": "string", "description": "Commit SHA (short or full)"}},
                "required": ["sha"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_read_file",
            "description": "Read content of one of my source files. Use to explain my own code.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path relative to repo root, e.g. 'discord_bot.py'"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_search_code",
            "description": "Search my codebase for a literal substring (case-insensitive). Returns matching lines with file and line number. Search SHORT distinctive substrings, not full guessed expressions — e.g. 'hybrid_command' not '@commands.hybrid_command', 'command(name=' not '@bot.tree.command'. An empty result means that exact substring isn't present; broaden or drop a prefix and retry before concluding something doesn't exist.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Short literal substring to find (case-insensitive). Prefer distinctive fragments over full decorators/paths."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_search_history",
            "description": "Search git commit history for a pattern. Use this for leaked secrets, old code, or questions about past commits, not just the current tree.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Regex or text pattern to search through commit history."},
                    "max_results": {"type": "integer", "description": "Maximum commits to return", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_file_list",
            "description": "List all files in my repository.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_repo_info",
            "description": "Get basic info about my repository: branch, remote, last commit.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_find_api_calls",
            "description": "Find concrete API call sites in the current codebase and related commits in git history. Use this for requests like 'check git for API calls', 'find OpenAI/Gemini call sites', or blue-team reviews.",
            "parameters": {
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Optional provider filter such as 'openai', 'gemini', 'anthropic', or 'stability'."
                    },
                    "max_results": {"type": "integer", "description": "Maximum results to return", "default": 12},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search long-term memory (Elasticsearch) for past conversation context. Use this for recall questions, especially time-based prompts like '2 weeks ago', 'last month', or 'yesterday'. Supports scoped retrieval and optional target user lookup when enabled.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query (e.g. 'favorite pokemon', 'project ideas')"},
                    "limit": {"type": "integer", "description": "Max results to return", "default": 5},
                    "target_user_id": {"type": "string", "description": "Optional Discord user ID to search for. Only works when cross-user memory search is enabled."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_behavioral_instruction",
            "description": "Update your long-term behavioral instructions for the current user. Use this when the user asks you to change how you speak, behave, or interact with them from now on or permanently (e.g. 'always speak in uwu', 'be sassy', 'call me Captain', 'reset your personality', 'from now on talk like you are gargling rocks'). New instructions replace conflicting old ones.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "The full behavioral instruction to store. e.g. 'Always answer in 1920s slang.' Set to empty string to clear."
                    }
                },
                "required": ["instruction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_own_logs",
            "description": "Read recent runtime diagnostics. Non-owners receive only a small, recent, redacted error-category summary; the owner may retrieve broader sanitized diagnostics. Use this when asked about errors, crashes, restarts, or health.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lines": {"type": "integer", "description": "Requested diagnostic entries. Public requests are capped at 20; owner requests at 120.", "default": 20},
                    "level": {
                        "type": "string",
                        "enum": ["all", "warning", "error"],
                        "description": "Filter to warnings+errors or errors only.",
                        "default": "all",
                    },
                    "grep": {"type": "string", "description": "Optional case-insensitive substring filter."},
                    "since_minutes": {"type": "integer", "description": "Lookback window. Public requests are capped at 60 minutes; owner requests at 2880.", "default": 60},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": "Store a durable fact about the current user for future conversations (e.g. 'has a dog named Kevin', 'works night shifts', 'building a Discord bot called Multivac'). Use when the user shares personal info that will matter later. Do NOT store secrets, one-off trivia, or anything the user asks you to forget.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "One short sentence, third person, e.g. 'Prefers concise answers.'"},
                    "category": {
                        "type": "string",
                        "enum": ["identity", "preference", "project", "event", "relationship", "other"],
                        "description": "Rough category of the fact.",
                    },
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_fact",
            "description": "Delete stored facts about the current user that match a phrase. Use when the user says something is wrong, outdated, or asks you to forget it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "match": {"type": "string", "description": "Phrase to match against stored facts, e.g. 'dog named Kevin'."},
                },
                "required": ["match"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_tools",
            "description": "List all my available tools and what they do. Call this to see what capabilities I have.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

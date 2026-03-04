TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for fresh information. Returns top results (title, URL, snippet).",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "Search query"},
                    "num": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                    "safe": {"type": "string", "enum": ["off", "active"], "default": "off"},
                    "gl": {"type": "string", "description": "Country code, e.g., 'us'"},
                    "lr": {"type": "string", "description": "Language restrict, e.g., 'lang_en'"},
                    "image": {"type": "boolean", "description": "Image search", "default": False},
                },
                "required": ["q"],
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
            "description": "Fetch a URL, extract the main article content, and return a condensed text block.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "HTTP/HTTPS URL to summarize."},
                    "max_len": {
                        "type": "integer",
                        "description": "Max characters of condensed text.",
                        "default": 3000,
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
            "description": "Search my codebase for a pattern. Returns matching lines with file and line number.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query (case-insensitive)"}},
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
            "name": "search_memory",
            "description": "Search my long-term memory (Elasticsearch) for past conversations or context. Use this to remember things the user told you previously.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query (e.g. 'favorite pokemon', 'project ideas')"},
                    "limit": {"type": "integer", "description": "Max results to return", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_behavioral_instruction",
            "description": "Update your long-term behavioral instructions for the current user. Use this when the user asks you to change how you speak, behave, or interact with them permanently (e.g. 'always speak in uwu', 'be sassy', 'call me Captain').",
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
            "name": "list_available_tools",
            "description": "List all my available tools and what they do. Call this to see what capabilities I have.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_sora_video",
            "description": "Generate a video using OpenAI Sora. STRICT LIMIT: 2 videos per user per hour.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed description of the video to generate."
                    }
                },
                "required": ["prompt"],
            },
        },
    },
]

![Multivac Header](https://i.imgur.com/Ruo7oC1.png)

# Multivac: Modular AI Agent

**Multivac** is a state-of-the-art, modular Discord bot designed to act as a unified AI assistant. By orchestrating multiple Large Language Models (LLMs) and a suite of live tools, it provides a persistent, intelligent, and multimodal experience that goes far beyond standard chatbots.

## 🚀 Key Features

### 🧠 Multi-Model Intelligence
Multivac is not tied to a single provider. It dynamically leverages the best model for the task:
- **Google Gemini (1.5 Pro / Flash)**: Primary engine for reasoning, multimodal understanding (vision), and large context windows.
- **OpenAI (GPT-4o)**: Fallback and specialized tasks.
- **Anthropic Claude (3.5 Sonnet)**: High-quality creative writing and complex instruction following.
- **Stability AI**: Professional-grade image generation.

### 📚 Deep Memory (RAG)
Powered by an **Elasticsearch** backend, Multivac processes and indexes conversation history.
- **Proactive Memory Injection**: Deep retrieval mechanisms automatically find relevant context from past conversations—whether they happened yesterday or months ago.
- **Persistent Persona**: Remembers user preferences and instructions across sessions.

### 🛠️ Live Tool Ecosystem
Multivac isn't just a chatbot; it's an agent that can *do* things:
- **Web Search**: Real-time access to Google Search for the latest news, weather, and data.
- **Code Execution**: A sandboxed Python environment for complex math, data analysis, plotting, and file generation.
- **Vision & Imaging**:
  - **Generation**: Create stunning images using Imagen 3 or Stable Diffusion (`imagine ...`).
  - **Editing**: Edit images using natural language prompts (`edit ...`).
  - **Analysis**: "See" and analyze image attachments in chat.
- **Utilities**:
  - **Weather**: live forecasts via OpenWeatherMap.
  - **Stocks**: Financial data via AlphaVantage/Polygon.
  - **Summarization**: Unfurls and summarizes linked articles automatically.

### ✨ Rich User Interface
Built for a clean and premium Discord experience:
- **Smart Expansion**: Long responses are automatically collapsed with a "preview" and can be expanded interactively to prevent chat clutter.
- **Live Progress Bars**: Visual feedback for long-running generations (Images, Research).
- **Interactive Views**: Custom UI components for model selection and moderation handling.

## ⚙️ Architecture

The project is structured as a modular Python application:
- **`discord_bot.py`**: The core event loop and interaction handler.
- **`bot/`**: Chat, image, video, UI, and context helper modules for the Discord runtime.
- **`providers/`**: External AI/media integrations (OpenAI, Gemini, Claude, Stability, Sora).
- **`services/`**: Internal persistence, memory, search, weather, stock, URL, and utility services.
- **`scripts/`** and **`dev/`**: Diagnostics, probes, verification helpers, and one-off maintenance scripts.
- **`docs/`** and **`assets/`**: Supporting documentation and repository assets.

## 🔧 Setup & Configuration

### Prerequisites
- Python 3.10+
- poetry (recommended) or pip

### Environment Variables
Configure the following keys in your environment (or `.env`):
```bash
# Core
DISCORD_TOKEN=...

# AI Providers
OPENAI_API_KEY=...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
STABILITY_KEY=...

# Search & Tools
GOOGLE_API_KEY=...
GOOGLE_CSE_ID=...
OPENWEATHER_API_KEY=...
```

### Installation
```bash
# Install dependencies
poetry install

# Run the bot
poetry run python main.py
```

## 🎮 Usage

- **Chat**: Just mention `@Multivac` or reply to its messages.
- **Image Generation**: "imagine a cyberpunk city" (defaults to best model) or "stable imagine ..."
- **Context Search**: "What did we discuss about the project last week?" (Triggers RAG search)
- **Commands**:
  - `/ping`: Check latency.
  - `/memory_fetch_more`: Backfill history into the searchable database.

---
*Built with ❤️ by Sardistic*

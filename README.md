# LangChain Search Agent Demo

A simple LangChain agent that uses Ollama for the LLM and Tavily for web search capabilities.

## Features
- Local LLM execution via [Ollama](https://ollama.com/)
- Web search integration using [Tavily](https://tavily.com/)
- Environment variable management with `python-dotenv`

## Prerequisites
- Python 3.13+
- [Ollama](https://ollama.com/) installed and running
- A Tavily API Key

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd langchain-hello-world
    ```

2.  **Install dependencies:**
    This project uses `pyproject.toml`. You can install dependencies using `pip`:
    ```bash
    pip install .
    ```

3.  **Environment Variables:**
    Copy the example environment file and fill in your API keys:
    ```bash
    cp .env.example .env
    ```
    Edit `.env` and add your `TAVILY_API_KEY`.

4.  **Run the agent:**
    ```bash
    python main.py
    ```

## Usage
The agent is currently configured to answer questions about current events (e.g., "What is currently happening in Poland?"). You can modify the query in `main.py`.

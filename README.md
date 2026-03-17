# LangGraph ReAct Agent

A [LangGraph](https://langchain-ai.github.io/langgraph/) implementation of the [ReAct](https://react-lm.github.io/) (Reasoning + Acting) pattern, where an LLM iteratively reasons about a query and calls tools until it has a final answer.

## Graph

![Agent graph](graph.png)

The agent follows a simple loop:

1. **`agent_reason`** — the LLM receives the message history and decides whether to call a tool or return a final answer
2. **`act`** — if a tool call was requested, the tool is executed and its result is appended to the message history
3. The loop repeats until the LLM produces a response with no tool calls

## Tools

| Tool | Description |
|------|-------------|
| `TavilySearch` | Web search returning up to 3 results |
| `quadruple` | Multiplies a number by 4 |

## Project structure

```
main.py      # Builds and runs the StateGraph
nodes.py     # Defines agent_reason and tool_node graph nodes
react.py     # Configures the LLM (via OpenRouter) and binds tools
```

## Setup

**1. Install dependencies** (requires [uv](https://docs.astral.sh/uv/))

```bash
uv sync
```

**2. Configure environment variables**

```bash
cp .env.example .env
```

Fill in the following values in `.env`:

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | Your [OpenRouter](https://openrouter.ai/) API key |
| `OPENROUTER_GENERATIVE_MODEL` | Model name, e.g. `openai/gpt-4o` |
| `OPENROUTER_GENERATIVE_BASE_URL` | OpenRouter base URL |
| `TAVILY_API_KEY` | Your [Tavily](https://tavily.com/) search API key |
| `LANGSMITH_API_KEY` | (Optional) [LangSmith](https://smith.langchain.com/) tracing key |
| `LANGSMITH_TRACING` | Set to `true` to enable LangSmith tracing |
| `LANGSMITH_PROJECT` | LangSmith project name |

**3. Run the agent**

```bash
uv run main.py
```

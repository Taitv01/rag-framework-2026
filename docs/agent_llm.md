# Agent LLM — let the calling AI model answer

`src/core/agent_llm.py` lets RAG run with **the model of the AI agent that runs it**
(Claude, Codex, Gemini, …) instead of an API key or a local model server. Whichever
model calls RAG answers RAG's prompts, and the report credits that model.

## How it works

The agent cannot be called back in the middle of a Python run, so a check takes two passes:

| Pass | What happens |
|---|---|
| 1 | `AgentLLM.generate()` records each prompt it has no answer for and returns a neutral placeholder. The caller sees `complete == False`, writes the requests file (`write_requests()`), and must not publish results. |
| — | The agent reads the requests file, answers every prompt itself, and writes the answers file with its own model id. |
| 2 | The same command runs again; `generate()` replays the answers, `complete` is True, and `answered_by` names the model for the report. |

Prompts are keyed by SHA-256 (of the system prompt plus the prompt), so an edited story
asks only the changed prompts again, and answers to prompts that no longer occur are ignored.

## Files

Requests (written by RAG):

```json
{
  "answers_file": ".../rag_answers.json",
  "instructions": "Answer every request below yourself ...",
  "system_prompt": "...",
  "requests": [{"id": "<sha256>", "prompt": "..."}]
}
```

Answers (written by the agent; keep earlier answers, add new ids):

```json
{"model": "claude-opus-5-5", "answers": {"<sha256>": "OK"}}
```

## Usage

```python
from pathlib import Path
import importlib.util

# Load by path: importing the `src` package would load the whole embedding stack.
spec = importlib.util.spec_from_file_location("agent_llm", Path("src/core/agent_llm.py"))
agent_llm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_llm)

llm = agent_llm.AgentLLM("rag_answers.json", "rag_requests.json", system_prompt="...")
issues = ConsistencyChecker(llm, character_manager=...).check_chapter(text, 1)
if not llm.complete:
    llm.write_requests()  # hand over to the agent, then run again
```

A configured OpenAI-compatible endpoint (local Ollama / LM Studio or a hosted API) remains
the alternative when no agent is in the loop; callers choose which `llm` to pass.

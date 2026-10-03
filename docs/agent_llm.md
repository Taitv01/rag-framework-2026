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

### Pipelines with dependent prompts

A RAG query asks prompts that depend on earlier answers: rewrite the question, grade
what the rewrite retrieved, then answer from the documents that passed. A placeholder
early in that chain would make every later prompt worthless, so:

- `llm.start_chain()` begins one unit of work (one question). After the chain's first
  unanswered prompt, later prompts are not requested (`llm.deferred` counts them);
  `complete` stays False until none are deferred.
- `with llm.independent():` marks prompts that do not depend on each other (grading
  the retrieved documents), so they are all requested in the same run.

Each run therefore asks one more layer. `scripts/eval.py answer` needs four runs for the
default AdvancedRAG pipeline: rewrite, grades, answer, then the faithfulness judge.

### Inside LLMManager

`AgentChatModel` stands in for the LangChain chat model of an `LLMManager`, so whole
pipelines (AdvancedRAG, graders sharing its LLM) run on the calling model unchanged:

```python
rag = AdvancedRAG(...)
rag.llm._llm = AgentChatModel(AgentLLM("answers.json", "requests.json"))
```

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

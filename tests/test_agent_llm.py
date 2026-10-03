"""
Tests for AgentLLM: the calling agent answers RAG prompts in two passes.

Both modules are loaded by file path, as story pipelines do, so the tests
never import the ``src`` package and its embedding stack.
"""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def load(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent_llm = load("src/core/agent_llm.py", "agent_llm_under_test")
checker_module = load("src/story/consistency_checker.py", "consistency_checker_under_test")


class Characters:
    def get_all_characters_context(self):
        return "An: cậu bé tốt bụng, mặc áo chàm"


def write_answers(path: Path, answers: dict, model="claude-opus-5-5"):
    path.write_text(json.dumps({"model": model, "answers": answers}, ensure_ascii=False), encoding="utf-8")


def test_first_pass_records_prompts_and_second_pass_replays(tmp_path):
    answers, requests = tmp_path / "answers.json", tmp_path / "requests.json"
    llm = agent_llm.AgentLLM(answers, requests, system_prompt="Chỉ báo lỗi có bằng chứng.")
    checker = checker_module.ConsistencyChecker(llm, character_manager=Characters())

    assert checker.check_chapter("An mặc áo đỏ.", 1) == []
    assert not llm.complete
    saved = json.loads(llm.write_requests().read_text(encoding="utf-8"))
    ids = [item["id"] for item in saved["requests"]]
    assert len(ids) == 2  # character + timeline; no plot or world manager given
    assert saved["system_prompt"] == "Chỉ báo lỗi có bằng chứng."

    write_answers(answers, {ids[0]: "Chương 1: An mặc áo đỏ, hồ sơ ghi áo chàm", ids[1]: "OK"})
    llm = agent_llm.AgentLLM(answers, requests, system_prompt="Chỉ báo lỗi có bằng chứng.")
    issues = checker_module.ConsistencyChecker(llm, character_manager=Characters()).check_chapter("An mặc áo đỏ.", 1)

    assert llm.complete and llm.answered_by == "claude-opus-5-5" and llm.answers_used == 2
    assert [issue.category for issue in issues] == ["character"]


def test_changed_prompt_is_asked_again(tmp_path):
    answers = tmp_path / "answers.json"
    old_id = agent_llm.prompt_id("câu hỏi cũ")
    write_answers(answers, {old_id: "OK"})
    llm = agent_llm.AgentLLM(answers, tmp_path / "requests.json")

    assert llm.generate("câu hỏi mới") == "OK"  # placeholder, not the stale answer
    assert list(llm.pending) == [agent_llm.prompt_id("câu hỏi mới")]
    assert not llm.complete


def test_answers_without_model_name_are_not_complete(tmp_path):
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps({"answers": {agent_llm.prompt_id("p"): "OK"}}), encoding="utf-8")
    llm = agent_llm.AgentLLM(answers, tmp_path / "requests.json")

    assert llm.generate("p") == "OK" and llm.pending == {}
    assert not llm.complete
    assert "no \"model\" value" in json.loads(llm.write_requests().read_text(encoding="utf-8"))["instructions"]


def test_system_prompt_is_part_of_the_id():
    assert agent_llm.prompt_id("p", "a") != agent_llm.prompt_id("p", "b") != agent_llm.prompt_id("p")


def test_malformed_answers_file_is_rejected(tmp_path):
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps({"answers": ["OK"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="answers"):
        agent_llm.AgentLLM(answers, tmp_path / "requests.json")


def test_chain_requests_only_prompts_built_on_real_answers(tmp_path):
    answers, requests = tmp_path / "answers.json", tmp_path / "requests.json"

    def pipeline(llm, question):
        llm.start_chain()
        query = llm.generate(f"rewrite: {question}")
        return [llm.generate(f"grade {query} doc{i}") for i in range(2)]

    llm = agent_llm.AgentLLM(answers, requests)
    pipeline(llm, "q1")
    pipeline(llm, "q2")
    assert sorted(llm.pending.values()) == ["rewrite: q1", "rewrite: q2"]
    assert llm.deferred == 4 and not llm.complete

    write_answers(answers, {agent_llm.prompt_id("rewrite: q1"): "Q1", agent_llm.prompt_id("rewrite: q2"): "Q2"})
    llm = agent_llm.AgentLLM(answers, requests)
    pipeline(llm, "q1")
    pipeline(llm, "q2")
    # One new layer per run: the first grade of each chain, built on the real rewrite.
    assert sorted(llm.pending.values()) == ["grade Q1 doc0", "grade Q2 doc0"]
    assert llm.deferred == 2


def test_prompts_outside_a_chain_are_all_requested(tmp_path):
    llm = agent_llm.AgentLLM(tmp_path / "answers.json", tmp_path / "requests.json")
    llm.generate("a")
    llm.generate("b")
    assert len(llm.pending) == 2 and llm.deferred == 0


def test_agent_chat_model_flattens_messages(tmp_path):
    from types import SimpleNamespace

    answers = tmp_path / "answers.json"
    system = SimpleNamespace(type="system", content="Be brief.")
    user = SimpleNamespace(type="human", content="Hỏi?")
    write_answers(answers, {
        agent_llm.prompt_id("Hỏi?"): "Đáp.",
        agent_llm.prompt_id("[system]\nBe brief.\n\n[human]\nHỏi?"): "Ngắn.",
    })
    chat = agent_llm.AgentChatModel(agent_llm.AgentLLM(answers, tmp_path / "requests.json"))

    assert chat.invoke([user]).content == "Đáp."
    assert chat.invoke([system, user]).content == "Ngắn."
    assert [chunk.content for chunk in chat.stream([user])] == ["Đáp."]


def test_independent_prompts_are_asked_together(tmp_path):
    answers, requests = tmp_path / "answers.json", tmp_path / "requests.json"
    write_answers(answers, {agent_llm.prompt_id("rewrite"): "Q"})

    llm = agent_llm.AgentLLM(answers, requests)
    llm.start_chain()
    query = llm.generate("rewrite")
    with llm.independent():
        grades = [llm.generate(f"grade {query} doc{i}") for i in range(3)]
    llm.generate(f"answer from {grades}")

    assert sorted(llm.pending.values()) == ["grade Q doc0", "grade Q doc1", "grade Q doc2"]
    assert llm.deferred == 1  # the answer waits for all three grades


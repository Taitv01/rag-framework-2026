"""Tests for example utility helpers that do not call external services."""

from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]


def load_example(filename: str):
    path = ROOT / "examples" / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_video_script_markdown_cleaner():
    module = load_example("10_video_script_generator.py")

    assert module.clean_markdown_fence("```markdown\n# Title\n```") == "# Title"
    assert module.clean_markdown_fence("plain text") == "plain text"


def test_video_script_timestamp_helpers():
    module = load_example("10_video_script_generator.py")

    assert module.seconds_to_timestamp(0) == "00:00"
    assert module.seconds_to_timestamp(75) == "01:15"
    assert module.transition_time_range(3, seconds_per_transition=8) == "00:16 - 00:24"


def test_video_script_output_builder():
    module = load_example("10_video_script_generator.py")

    output = module.build_output("chars", "scenes", "transitions")

    assert "Character Bible" in output
    assert "chars" in output
    assert "scenes" in output
    assert "transitions" in output


def test_proxy_check_build_url():
    module = load_example("11_llm_proxy_check.py")

    assert module.build_url("https://example.com/v1") == "https://example.com/v1/chat/completions"
    assert module.build_url("https://example.com/v1/chat/completions") == "https://example.com/v1/chat/completions"

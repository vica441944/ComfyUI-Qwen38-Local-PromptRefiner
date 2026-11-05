"""Local Prompt Refiner must not return the raw brief as a successful prompt.

Qwen3.8's chat template opens ``<think>`` unless ``enable_thinking`` is false.
The completion then starts inside that block and often restates the brief.
These tests lock the visible-answer split, the echo rejection, and the node path.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if "folder_paths" not in sys.modules:
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(ROOT / ".test-models")
    folder_paths.add_model_folder_path = lambda *_args, **_kwargs: None
    sys.modules["folder_paths"] = folder_paths

import nodes  # noqa: E402


BRIEF = (
    "分镜2 使用参考 仓库门口的小程\n"
    "雨夜，小程站在旧仓库门口。他低声说「门后有人」。"
    "镜头从湿漉漉的鞋面慢慢推到他的眼睛，铁门在风里响。"
)
NEAR_COPY = BRIEF.replace("眼睛", "眼神")
H3 = (
    "subject_definitions:\n"
    "<Picture 1> 小程, fully_preserved.\n"
    "\n"
    "summary:\n"
    "A rain-soaked warehouse door opens the second group.\n"
    "\n"
    "retention_analysis:\n"
    "<Picture 1>: fully_preserved identity, coat, and doorway.\n"
    "\n"
    "detailed_description:\n"
    "[Shot 1] At 0.00, a low camera rises from the wet shoes to 小程's eyes. "
    "He whispers <d>[Chinese] 门后有人</d> while the iron door shivers.\n"
    "[Shot 2] Hold on his face as rain crosses the key light.\n"
    "\n"
    "overall_soundscape:\n"
    "Rain, a metal door, and a close whisper.\n"
    "\n"
    "non_diegetic_music:\n"
    "N/A"
)
H3_WITH_BRIEF_QUOTE = H3.replace(
    "[Shot 1] At 0.00, a low camera rises from the wet shoes to 小程's eyes. ",
    "[Shot 1] At 0.00, a low camera rises from the wet shoes to 小程's eyes. "
    + BRIEF
    + " ",
)


def _completion(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


class ReasoningSplitTests(unittest.TestCase):
    def test_closed_think_block_keeps_the_answer(self) -> None:
        raw = f"<think>draft the prompt</think>\n\n{H3}"
        self.assertEqual(nodes._strip_reasoning(raw), H3)

    def test_preopened_think_drops_the_brief_restatement(self) -> None:
        raw = f"I will restate the task.\n{BRIEF}\n</think>\n\n{H3}"
        self.assertEqual(nodes._strip_reasoning(raw), H3)
        self.assertNotIn(BRIEF, nodes._strip_reasoning(raw))

    def test_unclosed_think_body_is_not_a_successful_refinement(self) -> None:
        raw = f"<think>\n{BRIEF}"
        self.assertEqual(nodes._strip_reasoning(raw), "")
        with self.assertRaises(nodes.PromptRefinerError) as caught:
            nodes._accept_local_refined_prompt(
                raw,
                brief=BRIEF,
                engine=nodes.FULL_REFERENCE_ENGINE,
                lora_triggers=(),
            )
        self.assertIn("echoed the brief", str(caught.exception))


class EchoRejectionTests(unittest.TestCase):
    def test_exact_brief_is_rejected(self) -> None:
        with self.assertRaises(nodes.PromptRefinerError) as caught:
            nodes._accept_local_refined_prompt(
                BRIEF,
                brief=BRIEF,
                engine=nodes.FULL_REFERENCE_ENGINE,
                lora_triggers=(),
            )
        self.assertIn("echoed the brief", str(caught.exception))

    def test_wrapper_around_the_brief_is_rejected(self) -> None:
        wrapped = f"Here is your refined prompt:\n{BRIEF}"
        with self.assertRaises(nodes.PromptRefinerError) as caught:
            nodes._accept_local_refined_prompt(
                wrapped,
                brief=BRIEF,
                engine=nodes.FULL_REFERENCE_ENGINE,
                lora_triggers=(),
            )
        self.assertIn("echoed the brief", str(caught.exception))

    def test_near_copy_of_the_brief_is_rejected(self) -> None:
        self.assertTrue(nodes._is_unrefined_echo(NEAR_COPY, BRIEF))
        with self.assertRaises(nodes.PromptRefinerError):
            nodes._accept_local_refined_prompt(
                NEAR_COPY,
                brief=BRIEF,
                engine="Auto (standard image)",
                lora_triggers=(),
            )

    def test_h3_fields_filled_with_the_brief_are_rejected(self) -> None:
        pasted = "\n\n".join(f"{field}:\n{BRIEF}" for field in nodes.H3_FULL_REFERENCE_FIELDS)
        with self.assertRaises(nodes.PromptRefinerError) as caught:
            nodes._accept_local_refined_prompt(
                pasted,
                brief=BRIEF,
                engine=nodes.FULL_REFERENCE_ENGINE,
                lora_triggers=(),
            )
        self.assertIn("echoed the brief", str(caught.exception))

    def test_preopened_think_uses_the_engine_prompt(self) -> None:
        raw = f"{BRIEF}\n</think>\n\n{H3}"
        prompt = nodes._accept_local_refined_prompt(
            raw,
            brief=BRIEF,
            engine=nodes.FULL_REFERENCE_ENGINE,
            lora_triggers=(),
        )
        self.assertEqual(prompt, H3)
        self.assertNotEqual(prompt, BRIEF)
        self.assertNotIn("分镜2 使用参考", prompt)

    def test_engine_prompt_inside_an_unclosed_think_is_kept(self) -> None:
        prompt = nodes._accept_local_refined_prompt(
            f"<think>\n{H3}",
            brief=BRIEF,
            engine=nodes.FULL_REFERENCE_ENGINE,
            lora_triggers=(),
        )
        self.assertEqual(prompt, H3)

    def test_brief_after_think_does_not_hide_an_inner_engine_prompt(self) -> None:
        raw = f"<think>\n{H3}\n</think>\n\n{BRIEF}"
        prompt = nodes._accept_local_refined_prompt(
            raw,
            brief=BRIEF,
            engine=nodes.FULL_REFERENCE_ENGINE,
            lora_triggers=(),
        )
        self.assertEqual(prompt, H3)

    def test_engine_prompt_may_quote_the_brief(self) -> None:
        prompt = nodes._accept_local_refined_prompt(
            H3_WITH_BRIEF_QUOTE,
            brief=BRIEF,
            engine=nodes.FULL_REFERENCE_ENGINE,
            lora_triggers=(),
        )
        self.assertIn("subject_definitions:", prompt)
        self.assertIn("[Shot 1]", prompt)
        self.assertIn("分镜2 使用参考", prompt)
        self.assertNotEqual(nodes._compact_prompt_text(prompt), nodes._compact_prompt_text(BRIEF))

    def test_user_turn_echo_is_rejected(self) -> None:
        user_text = f"/no_think\n@minimax-fullref\n{BRIEF}"
        with self.assertRaises(nodes.PromptRefinerError):
            nodes._accept_local_refined_prompt(
                user_text,
                brief=BRIEF,
                engine=nodes.FULL_REFERENCE_ENGINE,
                lora_triggers=(),
                user_text=user_text,
            )

    def test_non_h3_rewrite_is_kept(self) -> None:
        prompt = nodes._accept_local_refined_prompt(
            "A rain-soaked warehouse doorway, 35mm, sodium spill on wet concrete, the courier frozen mid-step.",
            brief=BRIEF,
            engine="Auto (standard image)",
            lora_triggers=(),
        )
        self.assertIn("warehouse", prompt)

    def test_h3_missing_fields_raise_instead_of_returning_text(self) -> None:
        with self.assertRaises(nodes.PromptRefinerError) as caught:
            nodes._accept_local_refined_prompt(
                "A loose paragraph about the warehouse door.",
                brief=BRIEF,
                engine=nodes.FULL_REFERENCE_ENGINE,
                lora_triggers=(),
            )
        self.assertIn("did not return", str(caught.exception))
        self.assertNotIn(BRIEF, str(caught.exception))


class ThinkingDisabledTests(unittest.TestCase):
    def test_handler_receives_enable_thinking_false(self) -> None:
        seen: dict[str, object] = {}

        def original(**kwargs: object) -> dict:
            seen["enable_thinking"] = kwargs.get("enable_thinking")
            return _completion("ok")

        llm = _FakeLLM(original)
        nodes._create_prompt_completion(
            llm,
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.2,
            top_p=0.9,
            max_tokens=32,
            seed=1,
        )
        self.assertIs(seen["enable_thinking"], False)
        self.assertNotIn("enable_thinking", llm.calls[0])
        first = llm.chat_handler
        nodes._create_prompt_completion(
            llm,
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.2,
            top_p=0.9,
            max_tokens=32,
            seed=1,
        )
        self.assertIs(llm.chat_handler, first)

    def test_handler_without_thinking_kwarg_still_runs(self) -> None:
        def original(**kwargs: object) -> dict:
            if "enable_thinking" in kwargs:
                raise TypeError("unexpected keyword argument 'enable_thinking'")
            return _completion("plain")

        llm = _FakeLLM(original)
        response = nodes._create_prompt_completion(
            llm,
            messages=[],
            temperature=0.2,
            top_p=0.9,
            max_tokens=8,
            seed=1,
        )
        self.assertEqual(response["choices"][0]["message"]["content"], "plain")

    def test_unrelated_type_error_is_not_swallowed(self) -> None:
        def original(**_kwargs: object) -> dict:
            raise TypeError("model file missing")

        llm = _FakeLLM(original)
        with self.assertRaises(TypeError) as caught:
            nodes._create_prompt_completion(llm, messages=[], temperature=0.2, top_p=0.9, max_tokens=8, seed=1)
        self.assertIn("model file missing", str(caught.exception))

    def test_request_marks_the_turn_no_think(self) -> None:
        messages = nodes._build_request(
            nodes.FULL_REFERENCE_ENGINE,
            BRIEF,
            "",
            (),
            (),
            (),
            (),
            (),
            None,
        )
        text = nodes._chat_message_text(messages[-1])
        self.assertTrue(text.startswith("/no_think\n"))
        self.assertIn("@minimax-fullref", text)
        self.assertIn("分镜2 使用参考", text)


class LocalRefinerNodeTests(unittest.TestCase):
    def test_refine_returns_engine_prompt_hidden_after_the_brief(self) -> None:
        raw = f"Task restatement:\n{BRIEF}\n</think>\n\n{H3}"
        prompt, debug = _refine(BRIEF, [raw])
        self.assertEqual(prompt, H3)
        self.assertNotEqual(prompt, BRIEF)
        self.assertNotIn("分镜2 使用参考", prompt)
        self.assertIn("thinking=disabled", debug)
        self.assertIn("refine_attempts=1", debug)

    def test_refine_rejects_brief_echo_after_retry(self) -> None:
        with self.assertRaises(nodes.PromptRefinerError) as caught:
            _refine(BRIEF, [BRIEF, f"Here is the prompt:\n{BRIEF}"])
        self.assertIn("echoed the brief", str(caught.exception))
        self.assertIn("failed after 2 attempts", str(caught.exception))

    def test_refine_retries_once_then_keeps_the_engine_prompt(self) -> None:
        prompt, debug = _refine(BRIEF, [BRIEF, H3])
        self.assertEqual(prompt, H3)
        self.assertIn("refine_attempts=2", debug)

    def test_external_brief_is_the_text_that_must_not_be_echoed(self) -> None:
        widget = "widget brief that should stay unused"
        with self.assertRaises(nodes.PromptRefinerError):
            _refine(widget, [BRIEF, BRIEF], external_brief=BRIEF)
        prompt, debug = _refine(widget, [H3], external_brief=BRIEF)
        self.assertEqual(prompt, H3)
        self.assertIn("brief_source=external_brief", debug)
        self.assertNotEqual(prompt, BRIEF)


class _FakeLLM:
    def __init__(self, handler) -> None:
        self.chat_handler = None
        self.chat_format = "chat_template.default"
        self._chat_handlers = {"chat_template.default": handler}
        self.calls: list[dict] = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(dict(kwargs))
        handler = self.chat_handler or self._chat_handlers[self.chat_format]
        return handler(**kwargs)


def _refine(
    brief: str,
    contents: list[str],
    external_brief: str = "",
    output_language: str = "English",
) -> tuple[str, str]:
    pending = list(contents)

    def complete(_llm, **_kwargs):
        if not pending:
            raise AssertionError("refine called the model more times than expected")
        return _completion(pending.pop(0))

    runtime = types.SimpleNamespace(llm=object())
    with (
        mock.patch.object(nodes, "_resolve_model_file", return_value="model.gguf"),
        mock.patch.object(nodes._RuntimeCache, "get", return_value=runtime),
        mock.patch.object(nodes._RuntimeCache, "unload"),
        mock.patch.object(nodes, "_create_prompt_completion", side_effect=complete),
    ):
        return nodes.Qwen38LocalPromptRefiner().refine(
            brief=brief,
            engine=nodes.FULL_REFERENCE_ENGINE,
            model_name="model.gguf",
            mmproj_name="none",
            vision_handler="auto (Qwen3-VL)",
            context_length=2048,
            gpu_layers=0,
            temperature=0.2,
            top_p=0.9,
            max_tokens=256,
            seed=1,
            keep_model_loaded=False,
            external_brief=external_brief,
            output_language=output_language,
        )


if __name__ == "__main__":
    unittest.main()

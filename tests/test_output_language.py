"""Output prompt bodies default to Chinese. Structural tokens stay put.

These tests do not call a live model. They check that the language rule is in
the request, and that a finished English cinematic body is rejected when the
node is set to 中文.
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


R2V = nodes.DIRECTOR_LONG_R2V_MODE
BALANCED = nodes.DIRECTOR_LONG_QUALITY_PROFILES[nodes.DIRECTOR_LONG_QUALITY_BALANCED]
ZH = nodes.OUTPUT_LANGUAGE_CHINESE
EN = nodes.OUTPUT_LANGUAGE_ENGLISH

BRIEF = """分镜5使用参考:<Picture 1> 小程
分镜5: 小程站在门口。
分镜6使用参考:<Picture 1> 小程 <Picture 2> 柳二姐
分镜6: 中近景。柳二姐站在画面中央，正对镜头。特写鞋面。
"""


def h3(detail: str, summary: str, sound: str, music: str = "N/A") -> str:
    return (
        "subject_definitions:\n"
        "<Picture 1> 小程, fully_preserved.\n"
        "\n"
        "summary:\n"
        f"{summary}\n"
        "\n"
        "retention_analysis:\n"
        "<Picture 1>: fully_preserved.\n"
        "\n"
        "detailed_description:\n"
        f"{detail}\n"
        "\n"
        "overall_soundscape:\n"
        f"{sound}\n"
        "\n"
        "non_diegetic_music:\n"
        f"{music}"
    )


ENGLISH_H3 = h3(
    "[Shot 1] At 00:00.000, a low camera rises from the wet shoes to his eyes. "
    "He whispers <d>[Chinese] 门后有人</d> while the iron door shivers.",
    "A rain-soaked warehouse door opens the second group.",
    "Rain, a metal door, and a close whisper.",
)
CHINESE_H3 = h3(
    "[Brief Shot 5] 在 00:00.000，低机位从湿鞋面抬到小程的眼睛，铁门在风里轻轻震动。"
    "他低声说 <d>[Chinese] 门后有人</d>。\n"
    "[Brief Shot 6] 在 00:03.000，中近景里柳二姐站在画面中央，身体微微前倾，"
    "镜头从肩线缓缓推到鞋面特写，门把停在她右手下方。",
    "雨夜的仓库门口，小程从湿鞋面抬眼，铁门在风里响。",
    "雨声、金属门响和贴近的低语。",
)
CHINESE_WITH_ENGLISH_DIALOGUE = h3(
    "[Shot 1] 在 00:00.000，低机位从湿鞋面抬到小程的眼睛，铁门轻轻震动。"
    "他低声说 <d>[English] someone is behind the door</d>，雨水穿过主光。",
    "雨夜仓库门口，小程从鞋面抬眼，风把铁门推开一条缝。",
    "雨声、金属门响和贴近的人声。",
)


def _completion(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def _message_text(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


class OutputLanguageWidgetTests(unittest.TestCase):
    def test_three_nodes_default_to_chinese(self) -> None:
        nodes_under_test = (
            nodes.Qwen38LocalPromptRefiner,
            nodes.Qwen38LongVideoDirectorPlanner,
            nodes.Qwen38DirectorGroupReRefiner,
        )
        for node_cls in nodes_under_test:
            widget = node_cls.INPUT_TYPES()["required"]["output_language"]
            self.assertEqual(widget[0], [ZH, EN])
            self.assertEqual(widget[1]["default"], ZH)

    def test_blank_and_aliases_normalize_and_unknown_raises(self) -> None:
        self.assertEqual(nodes._normalize_output_language(None), ZH)
        self.assertEqual(nodes._normalize_output_language(""), ZH)
        self.assertEqual(nodes._normalize_output_language("zh-CN"), ZH)
        self.assertEqual(nodes._normalize_output_language("English"), EN)
        with self.assertRaises(nodes.PromptRefinerError) as caught:
            nodes._normalize_output_language("français")
        self.assertIn("Unsupported output language", str(caught.exception))

    def test_cache_key_changes_with_language_and_defaults_to_chinese(self) -> None:
        shared = dict(
            brief="story",
            director_mode=R2V,
            total_duration_seconds=60,
            segment_duration_seconds=10,
            model_name="model.gguf",
            mmproj_name="none",
            vision_handler="auto (Qwen3-VL)",
            context_length=32768,
            gpu_layers=-1,
            temperature=0.4,
            top_p=0.9,
            max_tokens=4096,
            planning_quality=BALANCED.name,
            prompt_refinement=nodes.DIRECTOR_PROMPT_REFINEMENT_PER_SEGMENT,
            seed=1,
            example_preset="None",
            lora_presets=(),
            video_frame_samples=3,
            reference_text="",
            manual_lora_trigger_words="",
            assets={},
        )
        omitted = nodes._director_draft_cache_key(**shared)
        chinese = nodes._director_draft_cache_key(**shared, output_language=ZH)
        english = nodes._director_draft_cache_key(**shared, output_language=EN)
        self.assertEqual(omitted, chinese)
        self.assertNotEqual(chinese, english)
        self.assertIn(ZH, chinese)
        self.assertIn(EN, english)


class OutputLanguageRequestTests(unittest.TestCase):
    def test_historical_callers_keep_english_shot_wording(self) -> None:
        system = nodes._long_director_structure_system(R2V, (10.0, 10.0))
        self.assertIn("Write it in English.", system)
        self.assertIn("50–90 English words.", system)
        self.assertNotIn("OUTPUT LANGUAGE: 中文", system)
        segment = nodes._DirectorLongSegment(
            index=1,
            duration_seconds=12.0,
            prompt="",
            picture_slots=(1,),
            video_slots=(),
            video_audio_slots=(),
            audio_slots=(),
            continuity_from_prev=False,
            beat=BRIEF,
            dialogue="门后有人",
            handoff="Her hand stays on the latch.",
        )
        messages = nodes._build_long_director_segment_request(
            R2V,
            BRIEF,
            "",
            (segment,),
            segment,
            ((1, "pic"),),
            (),
            (),
            None,
            BALANCED,
        )
        user = _message_text(messages[1])
        self.assertIn("timing in English", user)
        self.assertIn("Do not paste the brief", user)
        self.assertNotIn("OUTPUT LANGUAGE", user)

    def test_chinese_option_reaches_structure_and_segment_requests(self) -> None:
        system = nodes._long_director_structure_system(R2V, (10.0, 10.0), output_language=ZH)
        self.assertIn("OUTPUT LANGUAGE: 中文", system)
        self.assertIn("Simplified Chinese", system)
        self.assertIn("HANDOFF:", system)
        self.assertIn("Keep the HANDOFF: label in English.", system)
        self.assertIn("只写这一组能看见和听见的内容", system)
        self.assertNotIn("focused story slice for this group only", system)
        self.assertNotIn("Write it in English.", system)
        self.assertIn("original language and wording", system)
        segment_system = nodes._long_director_segment_system(R2V, BALANCED, ZH)
        self.assertIn("Chinese sentences", segment_system)
        self.assertIn("Do not translate assigned or quoted DIALOGUE", segment_system)
        self.assertIn("[Brief Shot N]", segment_system)
        segment = nodes._DirectorLongSegment(
            index=1,
            duration_seconds=12.0,
            prompt="",
            picture_slots=(1,),
            video_slots=(),
            video_audio_slots=(),
            audio_slots=(),
            continuity_from_prev=False,
            beat=BRIEF,
            dialogue="门后有人",
            handoff="手停在门把上。",
        )
        messages = nodes._build_long_director_segment_request(
            R2V,
            BRIEF,
            "",
            (segment,),
            segment,
            ((1, "pic"),),
            (),
            (),
            None,
            BALANCED,
            ZH,
        )
        blob = _message_text(messages[0]) + "\n" + _message_text(messages[1])
        self.assertIn("prose in Simplified Chinese", blob)
        self.assertIn("OUTPUT LANGUAGE: 中文", blob)
        self.assertIn("[Brief Shot 6]", blob)
        self.assertIn("<d>[Language] exact dialogue</d>", blob)
        self.assertNotIn("timing in English", blob)
        self.assertIn("Do not paste the brief", blob)

    def test_english_option_is_explicit(self) -> None:
        messages = nodes._build_request(
            nodes.FULL_REFERENCE_ENGINE,
            "A night office.",
            "",
            (),
            (),
            (),
            (),
            (),
            None,
            EN,
        )
        blob = _message_text(messages[0]) + "\n" + _message_text(messages[1])
        self.assertIn("OUTPUT LANGUAGE: English", blob)
        self.assertNotIn("OUTPUT LANGUAGE: 中文", blob)
        self.assertIn("subject_definitions:", blob)

    def test_local_request_defaults_are_unchanged_until_the_node_passes_chinese(self) -> None:
        historical = nodes._build_request(
            nodes.FULL_REFERENCE_ENGINE,
            "分镜",
            "",
            (),
            (),
            (),
            (),
            (),
            None,
        )
        self.assertNotIn("OUTPUT LANGUAGE", _message_text(historical[0]))
        chinese = nodes._build_request(
            nodes.FULL_REFERENCE_ENGINE,
            "分镜",
            "",
            (),
            (),
            (),
            (),
            (),
            None,
            ZH,
        )
        system = _message_text(chinese[0])
        self.assertIn("OUTPUT LANGUAGE: 中文", system)
        self.assertIn("fully_preserved", system)
        self.assertIn("Do not translate assigned or quoted DIALOGUE", system)


class ChineseBodyCheckTests(unittest.TestCase):
    def test_chinese_bodies_pass_and_english_cinematic_prose_fails(self) -> None:
        kept = nodes._finalize_director_segment_prompt(
            CHINESE_H3,
            engine=nodes.FULL_REFERENCE_ENGINE,
            picture_slots=(),
            lora_triggers=(),
            index=1,
            output_language=ZH,
        )
        self.assertIn("中近景", kept)
        self.assertIn("<d>[Chinese] 门后有人</d>", kept)
        self.assertIn("[Brief Shot 6]", kept)
        with self.assertRaises(nodes.PromptRefinerError) as caught:
            nodes._finalize_director_segment_prompt(
                ENGLISH_H3,
                engine=nodes.FULL_REFERENCE_ENGINE,
                picture_slots=(),
                lora_triggers=(),
                index=1,
                output_language=ZH,
            )
        self.assertIn("not Chinese", str(caught.exception))
        self.assertIn("detailed_description", str(caught.exception))

    def test_english_dialogue_inside_a_chinese_shot_is_kept(self) -> None:
        kept = nodes._require_chinese_prompt_bodies(
            CHINESE_WITH_ENGLISH_DIALOGUE,
            nodes.FULL_REFERENCE_ENGINE,
        )
        self.assertIsNone(kept)
        narrative = nodes._h3_field_body(CHINESE_WITH_ENGLISH_DIALOGUE, "detailed_description:")
        self.assertIn("someone is behind the door", narrative)

    def test_chinese_dialogue_does_not_make_an_english_shot_pass(self) -> None:
        with self.assertRaises(nodes.PromptRefinerError):
            nodes._require_chinese_prompt_bodies(ENGLISH_H3, nodes.FULL_REFERENCE_ENGINE)

    def test_na_music_is_allowed(self) -> None:
        nodes._require_chinese_prompt_bodies(CHINESE_H3, nodes.FULL_REFERENCE_ENGINE)

    def test_short_camera_cue_stays_a_dump_and_a_chinese_shot_does_not(self) -> None:
        cue = "中近景。柳二姐站在画面中央，正对镜头。特写鞋面。"
        self.assertTrue(nodes._shot_body_is_raw_brief_dump(cue, ""))
        rewritten = (
            "[Brief Shot 6] 在 00:03.000，中近景里柳二姐站在画面中央，身体微微前倾，"
            "镜头从肩线缓缓推到鞋面特写，门把停在她右手下方，呼吸放轻。"
        )
        self.assertFalse(nodes._shot_body_is_raw_brief_dump(rewritten, BRIEF))
        self.assertGreater(len(rewritten), nodes._RAW_CUE_COMPACT_LIMIT)

    def test_chinese_mode_still_rejects_a_raw_brief_dump(self) -> None:
        detail = (
            "[Brief Shot 6] 中近景。柳二姐站在画面中央，正对镜头。特写鞋面。\n"
            "<Picture 2> 柳二姐 <Picture 1> 小程 <Picture 2> 柳二姐的卧室 小程"
        )
        with self.assertRaises(nodes.PromptRefinerError) as caught:
            nodes._finalize_director_segment_prompt(
                h3(detail, "雨夜门口。", "雨声。"),
                engine=nodes.FULL_REFERENCE_ENGINE,
                picture_slots=(1, 2),
                lora_triggers=(),
                index=2,
                required_shot_ids=(6,),
                shot_contract_text=BRIEF,
                output_language=ZH,
            )
        self.assertIn("raw brief", str(caught.exception))
        self.assertNotIn("not Chinese", str(caught.exception))


class OutputLanguageNodePathTests(unittest.TestCase):
    def test_local_refiner_default_asks_for_chinese_and_retries_english(self) -> None:
        pending = [ENGLISH_H3, CHINESE_H3]
        seen: list[list[dict]] = []

        def complete(_llm, **kwargs):
            seen.append(kwargs["messages"])
            return _completion(pending.pop(0))

        runtime = types.SimpleNamespace(llm=object())
        with (
            mock.patch.object(nodes, "_resolve_model_file", return_value="model.gguf"),
            mock.patch.object(nodes._RuntimeCache, "get", return_value=runtime),
            mock.patch.object(nodes._RuntimeCache, "unload"),
            mock.patch.object(nodes, "_create_prompt_completion", side_effect=complete),
        ):
            prompt, debug = nodes.Qwen38LocalPromptRefiner().refine(
                brief="雨夜，小程站在仓库门口。",
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
            )
        self.assertIn("中近景", prompt)
        self.assertIn("<d>[Chinese] 门后有人</d>", prompt)
        self.assertIn("output_language=中文", debug)
        self.assertIn("refine_attempts=2", debug)
        first = _message_text(seen[0][0]) + "\n" + _message_text(seen[0][1])
        self.assertIn("OUTPUT LANGUAGE: 中文", first)
        retry = _message_text(seen[1][-1])
        self.assertIn("not Chinese", retry)
        self.assertIn("中文", retry)

    def test_segment_refine_retries_until_the_body_is_chinese(self) -> None:
        calls: list[str] = []

        def complete(messages, _max_tokens, _seed):
            user = _message_text(messages[-1])
            calls.append(user)
            if len(calls) == 1:
                self.assertIn("OUTPUT LANGUAGE: 中文", _message_text(messages[0]))
                return ENGLISH_H3
            self.assertIn("not Chinese", user)
            return CHINESE_H3

        _title, segments, _rebalanced, _debug, record = nodes._plan_director_with_independent_segment_refine(
            complete,
            mode=R2V,
            brief="雨夜，小程站在仓库门口，低声说门后有人。",
            reference_text="",
            schedule=(10.0,),
            image_references=(),
            director_videos=(),
            audio_slots=(),
            video_audio_slots=(),
            lora_triggers=(),
            prompt_example=None,
            quality=BALANCED,
            requested_max_tokens=1024,
            seed=1,
            output_language=ZH,
        )
        self.assertEqual(len(calls), 2)
        self.assertIn("中近景", segments[0].prompt)
        self.assertEqual(record["passes"][0]["attempts"], 2)

    def test_re_refiner_request_carries_the_chinese_contract(self) -> None:
        pending = [ENGLISH_H3, CHINESE_H3]
        seen: list[list[dict]] = []

        def complete(_llm, **kwargs):
            seen.append(kwargs["messages"])
            return _completion(pending.pop(0))

        runtime = types.SimpleNamespace(llm=object())
        with (
            mock.patch.object(nodes, "_resolve_model_file", return_value="model.gguf"),
            mock.patch.object(nodes._RuntimeCache, "get", return_value=runtime),
            mock.patch.object(nodes._RuntimeCache, "unload"),
            mock.patch.object(nodes, "_create_prompt_completion", side_effect=complete),
        ):
            result = nodes.Qwen38DirectorGroupReRefiner().refine_group(
                current_prompt=ENGLISH_H3,
                revision_request="把镜头改成中文叙述，台词不要改。",
                target_group_id="12",
                duration_seconds=8.0,
                previous_group_context="",
                next_group_context="",
                model_name="model.gguf",
                context_length=2048,
                gpu_layers=0,
                temperature=0.2,
                top_p=0.9,
                max_tokens=512,
                seed=1,
                keep_model_loaded=False,
            )
        prompt = result["result"][0]
        self.assertIn("中近景", prompt)
        blob = _message_text(seen[0][0]) + "\n" + _message_text(seen[0][1])
        self.assertIn("OUTPUT LANGUAGE: 中文", blob)
        self.assertIn("Do not translate", blob)
        self.assertIn("<Picture N>", blob)


if __name__ == "__main__":
    unittest.main()

"""Bare 主体/主角 is an unbound person, not a guess for Subject 1.

Input briefs fail before the model runs. A refined prompt that still says the
bare role word fails and is retried. A name, <Subject N>, or entity_id written
on that same mention is a binding. The host line 所绘主体 is not a person.
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


def h3(detail: str, summary: str = "雨夜仓库门口，小程从湿鞋面抬眼。") -> str:
    return (
        "subject_definitions:\n"
        "<Subject 1> 小程, fully_preserved.\n"
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
        "雨声、金属门响和贴近的低语。\n"
        "\n"
        "non_diegetic_music:\n"
        "N/A"
    )


CLEAN = h3(
    "[Brief Shot 1] 在 00:00.000，低机位从湿鞋面抬到小程的眼睛。"
    "他低声说 <d>[Chinese] 门后有人</d>。"
)
BARE = h3(
    "[Brief Shot 1] 在 00:00.000，低机位从湿鞋面抬到主体的眼睛。"
    "他低声说 <d>[Chinese] 门后有人</d>。"
)
NAMED_BRIEF = "<Picture 1> 小程\n小程站在仓库门口。"


def _completion(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def _message_text(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


class UnboundBriefTests(unittest.TestCase):
    def test_bare_role_words_and_close_variants_are_rejected(self) -> None:
        samples = (
            "主体走进房间。",
            "主角转身看向镜头。",
            "该主体站在门口。",
            "此主体抬起头。",
            "主人公推开门。",
            "主体人物向镜头走来。",
            "主角人物停下。",
            "主体们站成一排。",
            "主体1走向门口。",
        )
        for sample in samples:
            with self.subTest(sample=sample):
                with self.assertRaises(nodes.PromptRefinerError) as caught:
                    nodes._reject_unbound_brief_roles(sample)
                message = str(caught.exception)
                self.assertIn("unbound 主体/主角", message)
                self.assertIn("Subject 1", message)
                self.assertIn("<Subject N>", message)

    def test_a_known_name_elsewhere_does_not_bind_a_bare_role(self) -> None:
        brief = "<Picture 1> 小程\n<Subject 1> 小程\n主体走进房间。只有小程在场。"
        with self.assertRaises(nodes.PromptRefinerError) as caught:
            nodes._reject_unbound_brief_roles(brief)
        self.assertIn("unbound 主体/主角", str(caught.exception))
        self.assertIn("小程", str(caught.exception))
        self.assertIn("Subject 1", str(caught.exception))

    def test_local_bindings_and_fixture_names_pass(self) -> None:
        samples = (
            "<Picture 1> 主角甲\n主角甲走进客厅。",
            "主体A停在门口。",
            "主角B转身。",
            "主体（小程）推开门。",
            "<Picture 1> 小程\n主体：小程推开门。",
            "主体是小程。小程推开门。",
            "<Subject 1>主体推开门。",
            "主体<Subject 2>推开门。",
            "<Subject 1>（主体）推开门。",
            "entity_id=hero_01的主体推开门。",
            "主体 entity_id=hero_01 推开门。",
            "小程（主体）推开门。",
            "人物停下，没有对白。",
            "<Picture 1> 小程\n小程站在门口。",
        )
        for sample in samples:
            with self.subTest(sample=sample):
                nodes._reject_unbound_brief_roles(sample)

    def test_verb_after_a_colon_is_not_a_name(self) -> None:
        with self.assertRaises(nodes.PromptRefinerError):
            nodes._reject_unbound_brief_roles("主体：小程走进房间。")
        with self.assertRaises(nodes.PromptRefinerError):
            nodes._reject_unbound_brief_roles("主体是走进房间。")
        with self.assertRaises(nodes.PromptRefinerError):
            nodes._reject_unbound_brief_roles("主体：然后，镜头推近。")

    def test_dialogue_and_host_sentence_are_not_person_references(self) -> None:
        nodes._require_no_unbound_role_words(
            "小程低声说 <d>[Chinese] 你才是主体</d>，然后推开门。",
            context_text="<Picture 1> 小程",
            source="prompt",
        )
        nodes._require_no_unbound_role_words(
            "<Picture 1> 是所绘主体或环境的权威视觉参考，fully_preserved。\n小程推开门。",
            context_text="<Picture 1> 小程",
            source="prompt",
        )

    def test_requests_forbid_inventing_subject_1(self) -> None:
        messages = nodes._build_request(
            nodes.FULL_REFERENCE_ENGINE,
            NAMED_BRIEF,
            "",
            (),
            (),
            (),
            (),
            (),
            None,
            ZH,
        )
        blob = _message_text(messages[0]) + "\n" + _message_text(messages[1])
        self.assertIn("SUBJECT BINDING", blob)
        self.assertIn("Do not guess", blob)
        self.assertIn("Subject 1", blob)
        self.assertIn("OUTPUT LANGUAGE: 中文", blob)
        segment = nodes._DirectorLongSegment(
            index=1,
            duration_seconds=8.0,
            prompt="",
            picture_slots=(1,),
            video_slots=(),
            video_audio_slots=(),
            audio_slots=(),
            continuity_from_prev=False,
            beat=NAMED_BRIEF,
            dialogue="",
            handoff="手停在门把上。",
        )
        segment_messages = nodes._build_long_director_segment_request(
            R2V,
            NAMED_BRIEF,
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
        segment_blob = _message_text(segment_messages[0]) + "\n" + _message_text(segment_messages[1])
        self.assertIn("SUBJECT BINDING", segment_blob)
        self.assertIn("HANDOFF", segment_blob)
        self.assertIn("CONTINUITY_FROM_PREV", segment_blob)
        structure = nodes._long_director_structure_system(R2V, (8.0,), output_language=ZH)
        self.assertIn("SUBJECT BINDING", structure)
        self.assertIn("HANDOFF:", structure)
        self.assertIn("Write that one sentence in Simplified Chinese.", structure)


class UnboundOutputTests(unittest.TestCase):
    def test_refined_prompt_with_a_bare_role_is_rejected(self) -> None:
        with self.assertRaises(nodes.PromptRefinerError) as caught:
            nodes._accept_local_refined_prompt(
                BARE,
                brief=NAMED_BRIEF,
                engine=nodes.FULL_REFERENCE_ENGINE,
                lora_triggers=(),
                output_language=ZH,
            )
        self.assertIn("The refined prompt uses unbound 主体/主角", str(caught.exception))
        self.assertIn("Subject 1", str(caught.exception))
        kept = nodes._accept_local_refined_prompt(
            CLEAN,
            brief=NAMED_BRIEF,
            engine=nodes.FULL_REFERENCE_ENGINE,
            lora_triggers=(),
            output_language=ZH,
        )
        self.assertIn("小程", kept)
        self.assertNotIn("主体的眼睛", kept)

    def test_host_picture_line_does_not_fail_a_named_chinese_shot(self) -> None:
        prompt = nodes._finalize_director_segment_prompt(
            CLEAN,
            engine=nodes.FULL_REFERENCE_ENGINE,
            picture_slots=(2,),
            lora_triggers=(),
            index=1,
            output_language=ZH,
            context_text=NAMED_BRIEF,
        )
        self.assertIn("所绘主体或环境", prompt)
        self.assertIn("小程的眼睛", prompt)
        with self.assertRaises(nodes.PromptRefinerError) as caught:
            nodes._finalize_director_segment_prompt(
                BARE,
                engine=nodes.FULL_REFERENCE_ENGINE,
                picture_slots=(2,),
                lora_triggers=(),
                index=1,
                output_language=ZH,
                context_text=NAMED_BRIEF,
            )
        message = str(caught.exception)
        self.assertIn("主体的眼睛", message)
        self.assertNotIn("所绘主体", message)

    def test_single_shot_plan_rejects_a_bare_role_in_the_shot(self) -> None:
        def marked(detail: str) -> str:
            return (
                "TITLE: Door\n"
                "=== SEGMENT 1 ===\n"
                "DURATION_SECONDS: 8\n"
                "CONTINUITY_FROM_PREV: false\n"
                "PICTURES:\n"
                "VIDEOS:\n"
                "VIDEO_AUDIOS:\n"
                "AUDIOS:\n"
                "PROMPT:\n"
                f"{detail}\n"
                "=== END SEGMENT ==="
            )

        with self.assertRaises(nodes.PromptRefinerError) as caught:
            nodes._validate_long_director_plan(
                marked(BARE),
                R2V,
                (8.0,),
                (),
                (),
                (),
                (),
                (),
                output_language=ZH,
                context_text=NAMED_BRIEF,
            )
        self.assertIn("unbound 主体/主角", str(caught.exception))
        title, segments, _rebalanced = nodes._validate_long_director_plan(
            marked(CLEAN),
            R2V,
            (8.0,),
            (),
            (),
            (),
            (),
            (),
            output_language=ZH,
            context_text=NAMED_BRIEF,
        )
        self.assertEqual(title, "Door")
        self.assertIn("小程", segments[0].prompt)
        self.assertNotIn("主体的眼睛", segments[0].prompt)


class UnboundPathTests(unittest.TestCase):
    def test_local_refiner_rejects_external_brief_before_the_model(self) -> None:
        def complete(*_args, **_kwargs):
            raise AssertionError("model should not run")

        with mock.patch.object(nodes, "_create_prompt_completion", side_effect=complete):
            with self.assertRaises(nodes.PromptRefinerError) as caught:
                nodes.Qwen38LocalPromptRefiner().refine(
                    brief=NAMED_BRIEF,
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
                    external_brief="主体走进仓库。",
                )
        self.assertIn("The brief uses unbound 主体/主角", str(caught.exception))

    def test_planner_node_rejects_external_brief_before_scheduling(self) -> None:
        def schedule(*_args, **_kwargs):
            raise AssertionError("duration schedule should not run")

        with mock.patch.object(nodes, "_director_duration_schedule", side_effect=schedule):
            with self.assertRaises(nodes.PromptRefinerError) as caught:
                nodes.Qwen38LongVideoDirectorPlanner().plan(
                    brief=NAMED_BRIEF,
                    director_mode=R2V,
                    total_duration_seconds=8.0,
                    segment_duration_seconds=8.0,
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
                    external_brief="主角走向镜头。",
                )
        self.assertIn("The brief uses unbound 主体/主角", str(caught.exception))
        self.assertIn("主角", str(caught.exception))

    def test_clean_external_brief_wins_over_a_dirty_widget(self) -> None:
        pending = [CLEAN]

        def complete(_llm, **_kwargs):
            return _completion(pending.pop(0))

        runtime = types.SimpleNamespace(llm=object())
        with (
            mock.patch.object(nodes, "_resolve_model_file", return_value="model.gguf"),
            mock.patch.object(nodes._RuntimeCache, "get", return_value=runtime),
            mock.patch.object(nodes._RuntimeCache, "unload"),
            mock.patch.object(nodes, "_create_prompt_completion", side_effect=complete),
        ):
            prompt, debug = nodes.Qwen38LocalPromptRefiner().refine(
                brief="主体走进仓库。",
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
                output_language=EN,
                external_brief=NAMED_BRIEF,
            )
        self.assertIn("小程", prompt)
        self.assertIn("brief_source=external_brief", debug)
        self.assertNotIn("主体的眼睛", prompt)

    def test_local_refiner_retries_a_bare_role_in_the_prompt(self) -> None:
        pending = [BARE, CLEAN]
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
                brief=NAMED_BRIEF,
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
                output_language=ZH,
            )
        self.assertIn("小程的眼睛", prompt)
        self.assertNotIn("主体的眼睛", prompt)
        self.assertIn("refine_attempts=2", debug)
        retry = _message_text(seen[1][-1])
        self.assertIn("unbound 主体/主角", retry)
        self.assertIn("Subject 1", retry)
        self.assertIn("中文", retry)

    def test_planner_rejects_the_brief_before_completion(self) -> None:
        def complete(*_args):
            raise AssertionError("model should not run")

        with self.assertRaises(nodes.PromptRefinerError) as caught:
            nodes._plan_director_with_independent_segment_refine(
                complete,
                mode=R2V,
                brief="<Picture 1> 小程\n主体站在门口。",
                reference_text="",
                schedule=(8.0,),
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
        self.assertIn("The brief uses unbound 主体/主角", str(caught.exception))

    def test_segment_refine_retries_a_bare_role(self) -> None:
        calls: list[str] = []

        def complete(messages, _max_tokens, _seed):
            user = _message_text(messages[-1])
            calls.append(user)
            if len(calls) == 1:
                self.assertIn("SUBJECT BINDING", user)
                return BARE
            self.assertIn("unbound 主体/主角", user)
            self.assertIn("Subject 1", user)
            return CLEAN

        _title, segments, _rebalanced, _debug, record = nodes._plan_director_with_independent_segment_refine(
            complete,
            mode=R2V,
            brief=NAMED_BRIEF,
            reference_text="",
            schedule=(8.0,),
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
        self.assertIn("小程的眼睛", segments[0].prompt)
        self.assertNotIn("主体的眼睛", segments[0].prompt)
        self.assertEqual(record["passes"][0]["attempts"], 2)

    def test_re_refiner_retries_a_bare_role(self) -> None:
        pending = [BARE, CLEAN]
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
                current_prompt=CLEAN,
                revision_request="把抬眼的动作再写清楚一点。",
                target_group_id="4",
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
                output_language=ZH,
            )
        prompt = result["result"][0]
        self.assertIn("小程的眼睛", prompt)
        self.assertNotIn("主体的眼睛", prompt)
        first = _message_text(seen[0][0]) + "\n" + _message_text(seen[0][1])
        self.assertIn("SUBJECT BINDING", first)
        retry = _message_text(seen[1][-1])
        self.assertIn("unbound 主体/主角", retry)
        self.assertIn("Subject 1", retry)


if __name__ == "__main__":
    unittest.main()

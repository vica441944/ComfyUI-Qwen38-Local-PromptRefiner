"""Synthetic shot-first contracts. No user screenplay or model execution."""
import json
import re
import unittest
import types
from unittest.mock import patch
from test_long_director_refine import nodes, R2V, BALANCED, r2v_prompt


def shot(index, body="The courier opens the gate.", refs="", duration=None):
    return f"分镜{index}使用参考:{refs}\n" + (f"时长:{duration}秒\n" if duration is not None else "") + f"分镜{index}: {body}\n"


def cards(brief, total, overrun=False, images=(), audios=()):
    schedule = nodes._explicit_shot_schedule(brief, total, overrun)
    return nodes._explicit_shot_segments(brief, schedule, R2V, images, (), (), audios)


class ExplicitShotTests(unittest.TestCase):
    def test_public_plan_external_brief_short_group(self):
        runtime = types.SimpleNamespace(llm=object())
        output = r2v_prompt("A courier opens a gate.", picture=False).replace("[Shot 1]", "[Brief Shot 7]")
        with (
            patch.object(nodes, "_resolve_model_file", return_value=None),
            patch.object(nodes._RuntimeCache, "get", return_value=runtime),
            patch.object(nodes._RuntimeCache, "unload"),
            patch.object(nodes, "_current_director_group_packers", return_value=None),
            patch.object(nodes, "_create_prompt_completion", return_value={"choices": [{"message": {"content": output}}]}) as complete,
        ):
            result = nodes.Qwen38LongVideoDirectorPlanner().plan(
                brief="unused local brief", external_brief=shot(7),
                director_mode=R2V, total_duration_seconds=60,
                external_total_duration_seconds=2, segment_duration_seconds=15,
                model_name="model.gguf", mmproj_name="None", vision_handler="auto",
                context_length=4096, gpu_layers=0, temperature=0.4, top_p=0.9,
                max_tokens=1024, seed=1, keep_model_loaded=False,
                output_language=nodes.OUTPUT_LANGUAGE_ENGLISH,
                director_execution=nodes.DIRECTOR_EXECUTION_DIRECT,
                prompt_refinement=nodes.DIRECTOR_PROMPT_REFINEMENT_SINGLE_SHOT,
                unique_id="shot-contract-test",
            )
        self.assertEqual(complete.call_count, 1)
        self.assertEqual(len(result["result"][0]), 1)
        self.assertEqual(result["result"][0][0]["duration_sec"], 2)

    def test_order_and_no_time_driven_extra_groups(self):
        brief = shot(9, "FIRST_ACTION", "<Picture 2>") + shot(3, "SECOND_ACTION", "<Picture 1> <Audio 1>")
        result = cards(brief, 12, images=(1, 2, 3), audios=(1,))
        self.assertEqual([nodes._assigned_brief_shot_ids(s.beat) for s in result], [(9,), (3,)])
        self.assertEqual([s.picture_slots for s in result], [(2,), (1,)])
        self.assertEqual([s.audio_slots for s in result], [(), (1,)])
        self.assertNotIn("SECOND_ACTION", result[0].beat)
        self.assertNotIn("FIRST_ACTION", result[1].beat)

    def test_twelve_seconds_is_one_group(self):
        self.assertEqual([s.duration_seconds for s in cards(shot(1), 12)], [12])

    def test_two_seconds_and_empty_refs(self):
        result = cards(shot(1), 2, images=(1, 2))
        self.assertEqual(result[0].duration_seconds, 2)
        self.assertEqual(result[0].picture_slots, ())
        self.assertIn("courier", result[0].beat)

    def test_explicit_durations_and_residual(self):
        result = cards(shot(1, duration=2) + shot(2), 12)
        self.assertEqual([s.duration_seconds for s in result], [2, 10])

    def test_only_long_shot_splits(self):
        result = cards(shot(1, "The courier opens the gate. Then the courier closes it.", duration=17) + shot(2, "The train arrives.", duration=2), 19)
        self.assertEqual([nodes._assigned_brief_shot_ids(s.beat) for s in result], [(1,), (1,), (2,)])
        self.assertEqual([s.duration_seconds for s in result], [8.5, 8.5, 2])
        self.assertEqual([s.continuity_from_prev for s in result], [False, True, False])
        self.assertNotIn("train", result[0].beat + result[1].beat)

    def test_millisecond_conservation(self):
        result = cards(shot(1) + shot(2) + shot(3), 7.001)
        self.assertAlmostEqual(sum(s.duration_seconds for s in result), 7.001)
        self.assertTrue(all(0 < s.duration_seconds < 4 for s in result))

    def test_limits_and_conflicts(self):
        for total in (0, -1, 260, float("nan")):
            with self.assertRaises(nodes.PromptRefinerError):
                cards(shot(1), total)
        with self.assertRaises(nodes.PromptRefinerError):
            cards(shot(1), 601, True)
        with self.assertRaises(nodes.PromptRefinerError):
            cards(shot(1, duration=2), 5)
        self.assertEqual(cards(shot(1, duration=5), 2, True)[0].duration_seconds, 5)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(nodes.PromptRefinerError):
            cards(shot(1) + shot(1), 10)

    def test_dialogue_budget_error_not_silent_drop(self):
        brief = shot(1, 'The courier says “Please wait by the gate until the train arrives tomorrow morning.”')
        with self.assertRaises(nodes.PromptRefinerError):
            cards(brief, 2)
        result = cards(brief, 2, True)
        self.assertIn("Please wait", result[0].dialogue)
        self.assertGreater(sum(s.duration_seconds for s in result), 2)

    def test_short_group_packing_and_ui(self):
        result = cards(shot(1), 2)
        with patch.object(nodes, "_current_director_group_packers", return_value=None):
            groups, _ = nodes._pack_director_long_groups(R2V, result, {}, {}, {}, {})
        self.assertEqual(groups[0]["duration_sec"], 2)
        self.assertEqual(nodes._director_ui_group_specs(R2V, result)[0]["duration_sec"], 2)

    def test_no_structure_call_and_isolated_refine(self):
        brief = shot(5, "FIRST_GATE") + shot(8, "SECOND_TRAIN")
        calls = []
        def complete(messages, *_):
            text = messages[1]["content"][0]["text"]
            calls.append(text)
            self.assertNotIn("STRUCTURE mode", messages[0]["content"])
            sid = 5 if "FIRST_GATE" in text else 8
            self.assertNotIn("SECOND_TRAIN" if sid == 5 else "FIRST_GATE", text)
            return r2v_prompt("A courier walks along the platform.", picture=False).replace("[Shot 1]", f"[Brief Shot {sid}]")
        _, result, _, _, record = nodes._plan_director_with_independent_segment_refine(
            complete, mode=R2V, brief=brief, reference_text="", schedule=(4,),
            image_references=(), director_videos=(), audio_slots=(), video_audio_slots=(),
            lora_triggers=(), prompt_example=None, quality=BALANCED,
            requested_max_tokens=4096, seed=1, max_segment_seconds=4)
        self.assertEqual(len(calls), 2)
        self.assertEqual([s.duration_seconds for s in result], [2, 2])
        self.assertEqual(record["structure_attempts"], 0)
        self.assertEqual(record["grouping_contract"], "explicit_shots_v1")

    def test_legacy_minimum_unchanged(self):
        with self.assertRaises(nodes.PromptRefinerError):
            nodes._director_duration_schedule(2, 15)
        self.assertEqual(nodes._story_duration_schedule("A courier walks.", budget_seconds=24, max_segment_seconds=12)[0], (12, 12))

"""Planner tests that do not load a GGUF model.

Per-segment refinement is checked with a fake completion function.  Legacy
single-shot parsing, duration rebalance, and Director group packing stay covered
so the new path cannot silently change those contracts.
"""

from __future__ import annotations

import base64
import io
import json
import sys
import types
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if "folder_paths" not in sys.modules:
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(ROOT / ".test-models")
    folder_paths.add_model_folder_path = lambda *_args, **_kwargs: None
    sys.modules["folder_paths"] = folder_paths

import nodes  # noqa: E402


R2V = nodes.DIRECTOR_LONG_R2V_MODE
FL2V = nodes.DIRECTOR_LONG_FL2V_MODE
BALANCED = nodes.DIRECTOR_LONG_QUALITY_PROFILES[nodes.DIRECTOR_LONG_QUALITY_BALANCED]
HIGH = nodes.DIRECTOR_LONG_QUALITY_PROFILES[nodes.DIRECTOR_LONG_QUALITY_HIGH]


def r2v_prompt(marker: str, *, picture: bool = True) -> str:
    picture_line = "<Picture 1> hero, fully_preserved.\n" if picture else ""
    retention = "<Picture 1>: fully_preserved.\n" if picture else ""
    return (
        "subject_definitions:\n"
        f"{picture_line}"
        "summary:\n"
        f"{marker} changes the scene.\n"
        "retention_analysis:\n"
        f"{retention}"
        "detailed_description:\n"
        f"[Shot 1] {marker} crosses the room.\n"
        "overall_soundscape:\n"
        "Room tone and footsteps.\n"
        "non_diegetic_music:\n"
        "N/A"
    )


def fl2v_prompt(marker: str) -> str:
    return (
        "integrated_multimodal_description:\n"
        f"[Shot 1] {marker} walks through the doorway.\n"
        "overall_soundscape:\n"
        "Hall tone.\n"
        "non_diegetic_music:\n"
        "N/A"
    )


def structure_plan() -> str:
    return """
TITLE: Night office
=== SEGMENT 1 ===
DURATION_SECONDS: 8
CONTINUITY_FROM_PREV: true
PICTURES: 1
VIDEOS: 1
VIDEO_AUDIOS:
AUDIOS:
BEAT:
BEAT_ONE. (S1) enters and sets a cup on the desk, ending facing the window.
DIALOGUE:
DIALOGUE_ONE hello there
HANDOFF:
HANDOFF_ONE cup down, facing the window.
=== END SEGMENT ===
=== SEGMENT 2 ===
DURATION_SECONDS: 12
CONTINUITY_FROM_PREV: true
PICTURES: 1
VIDEOS: 2
VIDEO_AUDIOS: 2
AUDIOS:
BEAT:
BEAT_TWO. (S1) answers the phone and turns toward the door.
DIALOGUE:
DIALOGUE_TWO I can hear you
HANDOFF:
HANDOFF_TWO phone at the ear, body toward the door.
=== END SEGMENT ===
=== SEGMENT 3 ===
DURATION_SECONDS: 10
CONTINUITY_FROM_PREV: false
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS: 1
BEAT:
BEAT_THREE. Hard cut to the hallway as (S2) waits by the lift.
DIALOGUE:
NONE
HANDOFF:
HANDOFF_THREE standing still beside the lift doors.
=== END SEGMENT ===
""".strip()


class LongDirectorRefineTests(unittest.TestCase):
    def test_refine_budget_does_not_grow_with_group_count(self) -> None:
        short_plan = nodes._long_director_effective_max_tokens(4096, R2V, (10.0, 10.0), BALANCED)
        long_plan = nodes._long_director_effective_max_tokens(4096, R2V, (10.0,) * 12, BALANCED)
        per_segment = nodes._long_director_segment_refine_max_tokens(4096, R2V, BALANCED)
        self.assertGreater(long_plan, short_plan)
        self.assertEqual(per_segment, 4096)
        self.assertGreater(long_plan, per_segment * 2)
        self.assertEqual(
            nodes._long_director_segment_refine_max_tokens(1000, R2V, HIGH),
            HIGH.r2v_tokens_per_segment,
        )
        self.assertEqual(nodes._long_director_structure_max_tokens(12, 4096), 512 + 12 * 480)
        self.assertEqual(nodes._long_director_structure_max_tokens(30, 4096), 512 + 30 * 480)
        self.assertLess(nodes._long_director_structure_max_tokens(12, 4096), long_plan)

    def test_legacy_system_prompt_still_plans_every_group_at_once(self) -> None:
        system = nodes._long_director_system(R2V, (10.0, 10.0, 10.0), BALANCED)
        self.assertIn("Produce exactly 3 segments", system)
        self.assertIn("repeat the SEGMENT block exactly 3 times", system)
        segment_system = nodes._long_director_segment_system(R2V, BALANCED)
        self.assertIn("exactly one R2V group", segment_system)
        self.assertNotIn("repeat the SEGMENT block", segment_system)

    def test_independent_refine_keeps_each_beat_focused(self) -> None:
        calls: list[dict] = []

        def complete(messages, max_tokens, seed):
            system = messages[0]["content"]
            user = messages[1]["content"][0]["text"]
            calls.append(
                {
                    "system": system,
                    "user": user,
                    "parts": messages[1]["content"],
                    "max_tokens": max_tokens,
                    "seed": seed,
                }
            )
            if "STRUCTURE mode" in system:
                return structure_plan()
            if "Segment 1 of 3" in user:
                return r2v_prompt("UNIQUE_PROMPT_ONE")
            if "Segment 2 of 3" in user:
                return r2v_prompt("UNIQUE_PROMPT_TWO")
            if "Segment 3 of 3" in user:
                return r2v_prompt("UNIQUE_PROMPT_THREE")
            raise AssertionError(user[:200])

        videos = [
            nodes._DirectorVideoFrames(slot=1, frames="frames-1", frame_uris=("uri-video-1",)),
            nodes._DirectorVideoFrames(slot=2, frames="frames-2", frame_uris=("uri-video-2a", "uri-video-2b")),
        ]
        title, segments, rebalanced, debug, record = nodes._plan_director_with_independent_segment_refine(
            complete,
            mode=R2V,
            brief="A night office story. Do not copy the beat tokens.",
            reference_text="Keep the phone voice.",
            schedule=(10.0, 10.0, 10.0),
            image_references=((1, "uri-picture-1"), (2, "uri-picture-2")),
            director_videos=videos,
            audio_slots=(1,),
            video_audio_slots=(2,),
            lora_triggers=(),
            prompt_example=None,
            quality=BALANCED,
            requested_max_tokens=4096,
            seed=7,
        )

        self.assertEqual(title, "Night office")
        self.assertFalse(rebalanced)
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(call["seed"] == 7 for call in calls))
        self.assertEqual(calls[0]["max_tokens"], max(512 + 3 * 480, 4096))
        self.assertTrue(all(call["max_tokens"] == 4096 for call in calls[1:]))
        self.assertEqual([segment.duration_seconds for segment in segments], [8.0, 12.0, 10.0])
        self.assertEqual(
            [segment.continuity_from_prev for segment in segments],
            [False, True, False],
        )
        self.assertEqual(segments[0].picture_slots, (1,))
        self.assertEqual(segments[1].picture_slots, (1,))
        self.assertEqual(segments[2].picture_slots, ())  # empty PICTURES → no stills
        self.assertEqual(segments[0].video_slots, (1,))
        self.assertEqual(segments[1].video_slots, (2,))
        self.assertEqual(segments[1].video_audio_slots, (2,))
        self.assertEqual(segments[2].audio_slots, (1,))
        self.assertEqual(segments[2].video_slots, ())
        self.assertIn("UNIQUE_PROMPT_ONE", segments[0].prompt)
        self.assertIn("UNIQUE_PROMPT_TWO", segments[1].prompt)
        self.assertNotIn("UNIQUE_PROMPT_ONE", calls[2]["user"])
        self.assertNotIn("UNIQUE_PROMPT_TWO", calls[1]["user"])
        self.assertIn("DIALOGUE_TWO", calls[2]["user"])
        self.assertIn("HANDOFF_ONE", calls[2]["user"])
        self.assertIn("BEAT_TWO", calls[2]["user"])
        self.assertIn("BEAT_THREE", calls[2]["user"])
        self.assertNotIn("DIALOGUE_ONE", calls[2]["user"])
        self.assertNotIn("DIALOGUE_THREE", calls[2]["user"])
        self.assertIn("hard cut", calls[3]["user"])
        self.assertNotIn("uri-video-1", json.dumps(calls[2]["parts"]))
        self.assertIn("uri-video-2a", json.dumps(calls[2]["parts"]))
        self.assertIn("uri-picture-1", json.dumps(calls[2]["parts"]))
        self.assertNotIn("uri-picture-2", json.dumps(calls[2]["parts"]))
        self.assertNotIn("uri-picture-1", json.dumps(calls[3]["parts"]))
        self.assertNotIn("uri-picture-2", json.dumps(calls[3]["parts"]))
        self.assertNotIn("uri-video-2a", json.dumps(calls[3]["parts"]))
        self.assertIn("<Audio 1>", calls[3]["user"])
        self.assertIn("per_segment_refine:", debug)
        self.assertIn("segment 2/3 | refine=independent | attempt=1 | continuity=true", debug)
        self.assertIn("segment 3/3 | refine=independent | attempt=1 | continuity=false", debug)
        self.assertEqual(record["passes"][1]["videos"], [2])
        self.assertGreater(record["passes"][0]["prompt_words"], 10)

        _rendered, plan_json = nodes._director_long_plan_text(
            title,
            R2V,
            30.0,
            segments,
            prompt_refinement=nodes.DIRECTOR_PROMPT_REFINEMENT_PER_SEGMENT,
            refine_record=record,
        )
        plan = json.loads(plan_json)
        self.assertEqual(plan["prompt_refinement"], nodes.DIRECTOR_PROMPT_REFINEMENT_PER_SEGMENT)
        self.assertEqual(plan["segments"][1]["dialogue"], "DIALOGUE_TWO I can hear you")
        self.assertIn("HANDOFF_TWO", plan["segments"][1]["handoff"])
        self.assertEqual(len(plan["refine"]["passes"]), 3)

        groups, packer = nodes._pack_director_long_groups(
            R2V,
            segments,
            {1: "image-1", 2: "image-2"},
            {1: "frames-1", 2: "frames-2"},
            {2: "paired-audio-2"},
            {1: "audio-1"},
        )
        self.assertEqual(packer, "legacy_fallback")
        self.assertFalse(groups[0]["continuityFromPrev"])
        self.assertTrue(groups[1]["continuityFromPrev"])
        self.assertFalse(groups[2]["continuityFromPrev"])
        self.assertEqual(set(groups[1]["ref_images"]), {0})
        self.assertEqual(groups[2]["ref_images"], {})
        self.assertEqual(set(groups[1]["ref_videos"]), {1})
        self.assertEqual(set(groups[2]["ref_videos"]), set())
        self.assertIn("UNIQUE_PROMPT_TWO", groups[1]["prompt"])

    def test_duration_rebalance_happens_before_refine_and_preserves_beats(self) -> None:
        seen_durations: list[str] = []

        def complete(messages, max_tokens, seed):
            del max_tokens, seed
            system = messages[0]["content"]
            user = messages[1]["content"][0]["text"]
            if "STRUCTURE mode" in system:
                return structure_plan().replace("DURATION_SECONDS: 8", "DURATION_SECONDS: 12").replace(
                    "DURATION_SECONDS: 10", "DURATION_SECONDS: 12"
                )
            seen_durations.append(user.split("Duration: ", 1)[1][:6])
            if "Segment 1 of 3" in user:
                return r2v_prompt("A")
            if "Segment 2 of 3" in user:
                return r2v_prompt("B")
            return r2v_prompt("C")

        _title, segments, rebalanced, debug, _record = nodes._plan_director_with_independent_segment_refine(
            complete,
            mode=R2V,
            brief="Story",
            reference_text="",
            schedule=(10.0, 10.0, 10.0),
            image_references=((1, "pic"),),
            director_videos=(
                nodes._DirectorVideoFrames(slot=1, frames="frames-1", frame_uris=("uri-video-1",)),
                nodes._DirectorVideoFrames(slot=2, frames="frames-2", frame_uris=("uri-video-2",)),
            ),
            audio_slots=(1,),
            video_audio_slots=(2,),
            lora_triggers=(),
            prompt_example=None,
            quality=BALANCED,
            requested_max_tokens=4096,
            seed=1,
        )
        self.assertTrue(rebalanced)
        self.assertAlmostEqual(sum(segment.duration_seconds for segment in segments), 30.0, places=3)
        self.assertTrue(all(4.0 <= segment.duration_seconds <= 15.0 for segment in segments))
        self.assertIn("BEAT_TWO", segments[1].beat)
        self.assertTrue(all(item.startswith("10.000") for item in seen_durations))
        self.assertIn("durations_rebalanced=true", debug)

    def test_missing_fields_retry_once_then_accept(self) -> None:
        segment_two_calls = 0

        def complete(messages, max_tokens, seed):
            nonlocal segment_two_calls
            del max_tokens, seed
            system = messages[0]["content"]
            user = messages[1]["content"][0]["text"]
            if "STRUCTURE mode" in system:
                return structure_plan()
            if "Segment 2 of 3" in user:
                segment_two_calls += 1
                if segment_two_calls == 1:
                    return "This is not an H3 prompt."
                return r2v_prompt("RECOVERED")
            if "Segment 1 of 3" in user:
                return r2v_prompt("ONE")
            return r2v_prompt("THREE")

        _title, segments, _rebalanced, debug, record = nodes._plan_director_with_independent_segment_refine(
            complete,
            mode=R2V,
            brief="Story",
            reference_text="",
            schedule=(10.0, 10.0, 10.0),
            image_references=((1, "pic"),),
            director_videos=(
                nodes._DirectorVideoFrames(slot=1, frames="frames-1", frame_uris=("uri-video-1",)),
                nodes._DirectorVideoFrames(slot=2, frames="frames-2", frame_uris=("uri-video-2",)),
            ),
            audio_slots=(1,),
            video_audio_slots=(2,),
            lora_triggers=(),
            prompt_example=None,
            quality=BALANCED,
            requested_max_tokens=4096,
            seed=3,
        )
        self.assertEqual(segment_two_calls, 2)
        self.assertIn("RECOVERED", segments[1].prompt)
        self.assertEqual(record["passes"][1]["attempts"], 2)
        self.assertEqual(record["passes"][0]["attempts"], 1)
        self.assertIn("segment 2/3 | refine=independent | attempt=2", debug)

    def test_structure_failure_does_not_emit_partial_prompts(self) -> None:
        calls = 0

        def complete(messages, max_tokens, seed):
            nonlocal calls
            del messages, max_tokens, seed
            calls += 1
            return "I will not follow the format."

        with self.assertRaises(nodes.PromptRefinerError) as raised:
            nodes._plan_director_with_independent_segment_refine(
                complete,
                mode=R2V,
                brief="Story",
                reference_text="",
                schedule=(10.0, 10.0),
                image_references=(),
                director_videos=(),
                audio_slots=(),
                video_audio_slots=(),
                lora_triggers=(),
                prompt_example=None,
                quality=BALANCED,
                requested_max_tokens=4096,
                seed=1,
            )
        self.assertIn("structure pass failed", str(raised.exception))
        self.assertEqual(calls, 2)

    def test_single_group_skips_structure_and_refines_once(self) -> None:
        calls = []

        def complete(messages, max_tokens, seed):
            calls.append((messages[0]["content"], max_tokens, seed))
            self.assertNotIn("STRUCTURE mode", messages[0]["content"])
            self.assertIn("exactly one R2V group", messages[0]["content"])
            user = messages[1]["content"][0]["text"]
            self.assertIn("full project brief", user)
            self.assertIn("Meet at the station.", user)
            return r2v_prompt("ONLY")

        title, segments, rebalanced, debug, record = nodes._plan_director_with_independent_segment_refine(
            complete,
            mode=R2V,
            brief="Meet at the station.",
            reference_text="",
            schedule=(12.0,),
            image_references=((1, "pic"),),
            director_videos=(),
            audio_slots=(),
            video_audio_slots=(),
            lora_triggers=("exact_trigger",),
            prompt_example=None,
            quality=BALANCED,
            requested_max_tokens=4096,
            seed=9,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], 4096)
        self.assertEqual(title, "Meet at the station.")
        self.assertFalse(rebalanced)
        self.assertEqual(segments[0].duration_seconds, 12.0)
        self.assertFalse(segments[0].continuity_from_prev)
        self.assertIn("exact_trigger", segments[0].prompt)
        self.assertIn("skipped_single_group", debug)
        self.assertEqual(record["structure_attempts"], 0)
        self.assertTrue(record["structure_skipped"])

    def test_fl2v_visuals_follow_first_and_last_frames(self) -> None:
        images = ((1, "first"), (2, "last"), (3, "extra"))
        opening, _videos = nodes._segment_visual_references(FL2V, 1, 3, (), (), images, ())
        middle, _videos = nodes._segment_visual_references(FL2V, 2, 3, (), (), images, ())
        ending, _videos = nodes._segment_visual_references(FL2V, 3, 3, (), (), images, ())
        only, _videos = nodes._segment_visual_references(FL2V, 1, 1, (), (), images, ())
        self.assertEqual([slot for slot, _uri in opening], [1, 3])
        self.assertEqual([slot for slot, _uri in middle], [3])
        self.assertEqual([slot for slot, _uri in ending], [2, 3])
        self.assertEqual([slot for slot, _uri in only], [1, 2, 3])

    def test_legacy_marked_plan_still_rebalances_and_keeps_continuity(self) -> None:
        plan = """
TITLE: Legacy
=== SEGMENT 1 ===
DURATION_SECONDS: 14
CONTINUITY_FROM_PREV: false
PICTURES: 1
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
PROMPT:
subject_definitions:
Hero.
summary:
Opens.
retention_analysis:
Keep the room.
detailed_description:
[Shot 1] A hand sets down a cup.
overall_soundscape:
Room tone.
non_diegetic_music:
N/A
=== END SEGMENT ===
=== SEGMENT 2 ===
DURATION_SECONDS: 14
CONTINUITY_FROM_PREV: false
PICTURES: 1
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
PROMPT:
subject_definitions:
Hero.
summary:
Continues after a cut.
retention_analysis:
Keep the room.
detailed_description:
[Shot 1] The camera finds the window.
overall_soundscape:
Room tone.
non_diegetic_music:
N/A
=== END SEGMENT ===
""".strip()
        title, segments, rebalanced = nodes._validate_long_director_plan(
            plan,
            R2V,
            (15.0, 15.0),
            image_slots=(1,),
            video_slots=(),
            video_audio_slots=(),
            audio_slots=(),
            lora_triggers=(),
        )
        self.assertEqual(title, "Legacy")
        self.assertTrue(rebalanced)
        self.assertAlmostEqual(sum(segment.duration_seconds for segment in segments), 30.0, places=3)
        self.assertFalse(segments[0].continuity_from_prev)
        self.assertFalse(segments[1].continuity_from_prev)
        self.assertIn("<Picture 1>", segments[0].prompt)
        self.assertEqual(segments[0].picture_slots, (1,))

    def test_prompt_fallback_from_old_marked_output_still_refines(self) -> None:
        old = """
TITLE: Old shape
=== SEGMENT 1 ===
DURATION_SECONDS: 10
CONTINUITY_FROM_PREV: false
PICTURES: 1
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
PROMPT:
subject_definitions:
<Picture 1> hero.
summary:
First beat only.
retention_analysis:
<Picture 1>: fully_preserved.
detailed_description:
[Shot 1] OLD_BEAT_ONE stands up.
overall_soundscape:
Room tone.
non_diegetic_music:
N/A
=== END SEGMENT ===
=== SEGMENT 2 ===
DURATION_SECONDS: 10
CONTINUITY_FROM_PREV: true
PICTURES: 1
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
PROMPT:
subject_definitions:
<Picture 1> hero.
summary:
Second beat only.
retention_analysis:
<Picture 1>: fully_preserved.
detailed_description:
[Shot 1] OLD_BEAT_TWO sits.
overall_soundscape:
Room tone.
non_diegetic_music:
N/A
=== END SEGMENT ===
""".strip()
        calls = []

        def complete(messages, max_tokens, seed):
            del max_tokens, seed
            system = messages[0]["content"]
            user = messages[1]["content"][0]["text"]
            calls.append(system)
            if "STRUCTURE mode" in system:
                return old
            self.assertIn("SINGLE-SEGMENT", system)
            if "Segment 1 of 2" in user:
                self.assertIn("OLD_BEAT_ONE", user)
                return r2v_prompt("FRESH_ONE")
            self.assertIn("OLD_BEAT_TWO", user)
            self.assertNotIn("FRESH_ONE", user)
            return r2v_prompt("FRESH_TWO")

        _title, segments, _rebalanced, _debug, _record = nodes._plan_director_with_independent_segment_refine(
            complete,
            mode=R2V,
            brief="Story",
            reference_text="",
            schedule=(10.0, 10.0),
            image_references=((1, "pic"),),
            director_videos=(),
            audio_slots=(),
            video_audio_slots=(),
            lora_triggers=(),
            prompt_example=None,
            quality=BALANCED,
            requested_max_tokens=2048,
            seed=1,
        )
        self.assertEqual(len(calls), 3)
        self.assertIn("FRESH_ONE", segments[0].prompt)
        self.assertIn("FRESH_TWO", segments[1].prompt)
        self.assertNotIn("OLD_BEAT_ONE", segments[0].prompt)

    def test_total_duration_accepts_240_seconds_and_rejects_241(self) -> None:
        schedule = nodes._director_duration_schedule(240.0, 10.0)
        self.assertEqual(len(schedule), 24)
        self.assertEqual(schedule, (10.0,) * 24)
        self.assertAlmostEqual(sum(schedule), 240.0, places=3)

        # A total above the old 120s ceiling, including one past 180s, stays intact.
        over_old_ceiling = nodes._director_duration_schedule(180.0, 10.0)
        self.assertEqual(len(over_old_ceiling), 18)
        self.assertAlmostEqual(sum(over_old_ceiling), 180.0, places=3)

        with self.assertRaises(nodes.PromptRefinerError) as over_limit:
            nodes._director_duration_schedule(241.0, 10.0)
        message = str(over_limit.exception)
        self.assertIn("total_duration_seconds", message)
        self.assertIn("240", message)
        self.assertIn("241", message)
        self.assertIn("not clamped", message)

        with self.assertRaises(nodes.PromptRefinerError) as under_limit:
            nodes._director_duration_schedule(3.9, 10.0)
        self.assertIn("total_duration_seconds", str(under_limit.exception))

        with self.assertRaises(nodes.PromptRefinerError) as bad_segment:
            nodes._director_duration_schedule(240.0, 16.0)
        self.assertIn("segment_duration_seconds", str(bad_segment.exception))

        self.assertEqual(nodes.MAX_DIRECTOR_TOTAL_SECONDS, 240.0)
        self.assertEqual(nodes.MIN_DIRECTOR_SEGMENT_SECONDS, 4.0)
        self.assertEqual(nodes.MAX_DIRECTOR_SEGMENT_SECONDS, 15.0)
        self.assertEqual(nodes.MAX_CONFIGURABLE_REFERENCE_MEDIA_SECONDS, 120.0)
        required = nodes.Qwen38LongVideoDirectorPlanner.INPUT_TYPES()["required"]
        self.assertEqual(required["total_duration_seconds"][1]["max"], 240.0)
        self.assertEqual(required["total_duration_seconds"][1]["min"], 4.0)
        self.assertEqual(required["segment_duration_seconds"][1]["min"], 4.0)
        self.assertEqual(required["segment_duration_seconds"][1]["max"], 15.0)

    def _plan_until_duration(self, **overrides: object) -> None:
        planner = nodes.Qwen38LongVideoDirectorPlanner()
        kwargs = dict(
            brief="A complete night-office story.",
            director_mode=R2V,
            total_duration_seconds=60.0,
            segment_duration_seconds=10.0,
            model_name="missing.gguf",
            mmproj_name="none",
            vision_handler="auto (Qwen3-VL)",
            context_length=32768,
            gpu_layers=-1,
            temperature=0.4,
            top_p=0.9,
            max_tokens=4096,
            seed=1,
            keep_model_loaded=False,
        )
        kwargs.update(overrides)
        planner.plan(**kwargs)

    def test_external_total_duration_overrides_widget_and_rejects_241(self) -> None:
        input_types = nodes.Qwen38LongVideoDirectorPlanner.INPUT_TYPES()
        self.assertNotIn("external_total_duration_seconds", input_types["required"])
        socket = input_types["optional"]["external_total_duration_seconds"]
        self.assertEqual(socket[0], "FLOAT")
        self.assertTrue(socket[1]["forceInput"])
        self.assertNotIn("min", socket[1])
        self.assertNotIn("max", socket[1])
        self.assertEqual(input_types["required"]["segment_duration_seconds"][1]["min"], 4.0)
        self.assertEqual(input_types["required"]["segment_duration_seconds"][1]["max"], 15.0)

        overridden, source = nodes._effective_total_duration_seconds(60.0, 90)
        self.assertEqual(overridden, 90.0)
        self.assertEqual(source, "external_total_duration_seconds")
        self.assertEqual(len(nodes._director_duration_schedule(overridden, 10.0)), 9)

        from_int, int_source = nodes._effective_total_duration_seconds(60.0, 48)
        self.assertEqual(from_int, 48.0)
        self.assertEqual(int_source, "external_total_duration_seconds")

        at_cap, cap_source = nodes._effective_total_duration_seconds(60.0, 240)
        self.assertEqual(cap_source, "external_total_duration_seconds")
        capped_schedule = nodes._director_duration_schedule(at_cap, 10.0)
        self.assertEqual(len(capped_schedule), 24)
        self.assertAlmostEqual(sum(capped_schedule), 240.0, places=3)

        for missing in (None, "", "   "):
            fallback, fallback_source = nodes._effective_total_duration_seconds(60.0, missing)
            self.assertEqual(fallback, 60.0)
            self.assertEqual(fallback_source, "total_duration_seconds")

        illegal, illegal_source = nodes._effective_total_duration_seconds(60.0, 241)
        self.assertEqual(illegal, 241.0)
        self.assertEqual(illegal_source, "external_total_duration_seconds")
        with self.assertRaises(nodes.PromptRefinerError) as over_limit:
            nodes._director_duration_schedule(illegal, 10.0)
        message = str(over_limit.exception)
        self.assertIn("total_duration_seconds", message)
        self.assertIn("240", message)
        self.assertIn("241", message)
        self.assertIn("not clamped", message)
        self.assertNotIn("60", message)

        below, below_source = nodes._effective_total_duration_seconds(60.0, 3)
        self.assertEqual(below_source, "external_total_duration_seconds")
        with self.assertRaises(nodes.PromptRefinerError) as under_limit:
            nodes._director_duration_schedule(below, 10.0)
        self.assertIn("3", str(under_limit.exception))

        with self.assertRaises(nodes.PromptRefinerError) as bad_type:
            nodes._effective_total_duration_seconds(60.0, "soon")
        self.assertIn("external_total_duration_seconds", str(bad_type.exception))

        seen: dict[str, float] = {}
        real_schedule = nodes._director_duration_schedule

        def capture(total: float, segment: float) -> tuple[float, ...]:
            seen["total"] = total
            seen["segment"] = segment
            raise nodes.PromptRefinerError("stop-after-duration")

        nodes._director_duration_schedule = capture
        try:
            for external, expected in ((90, 90.0), (48, 48.0), (None, 60.0)):
                with self.assertRaises(nodes.PromptRefinerError) as stopped:
                    self._plan_until_duration(external_total_duration_seconds=external)
                self.assertEqual(str(stopped.exception), "stop-after-duration")
                self.assertEqual(seen["total"], expected)
                self.assertEqual(seen["segment"], 10.0)
            with self.assertRaises(nodes.PromptRefinerError) as stopped_default:
                self._plan_until_duration()
            self.assertEqual(str(stopped_default.exception), "stop-after-duration")
            self.assertEqual(seen["total"], 60.0)
        finally:
            nodes._director_duration_schedule = real_schedule

        with self.assertRaises(nodes.PromptRefinerError) as planned_over_limit:
            self._plan_until_duration(total_duration_seconds=60.0, external_total_duration_seconds=241)
        planned_message = str(planned_over_limit.exception)
        self.assertIn("total_duration_seconds", planned_message)
        self.assertIn("241", planned_message)
        self.assertIn("not clamped", planned_message)
        self.assertNotIn("stop-after-duration", planned_message)

    def test_widget_default_and_cache_key(self) -> None:
        required = nodes.Qwen38LongVideoDirectorPlanner.INPUT_TYPES()["required"]
        names = list(required)
        self.assertLess(names.index("planning_quality"), names.index("prompt_refinement"))
        self.assertLess(names.index("prompt_refinement"), names.index("anchor_all_pictures"))
        self.assertLess(names.index("anchor_all_pictures"), names.index("total_duration_seconds"))
        self.assertEqual(
            required["prompt_refinement"][1]["default"],
            nodes.DIRECTOR_PROMPT_REFINEMENT_PER_SEGMENT,
        )

        def cache_key(refinement: str) -> str:
            return nodes._director_draft_cache_key(
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
                prompt_refinement=refinement,
                seed=1,
                example_preset="None",
                lora_presets=(),
                video_frame_samples=3,
                reference_text="",
                manual_lora_trigger_words="",
                assets={},
            )

        self.assertNotEqual(
            cache_key(nodes.DIRECTOR_PROMPT_REFINEMENT_PER_SEGMENT),
            cache_key(nodes.DIRECTOR_PROMPT_REFINEMENT_SINGLE_SHOT),
        )
        tighten = nodes._director_draft_cache_key(
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
            anchor_all_pictures=False,
        )
        full_anchor = nodes._director_draft_cache_key(
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
            anchor_all_pictures=True,
        )
        self.assertNotEqual(tighten, full_anchor)
        with self.assertRaises(nodes.PromptRefinerError):
            nodes._director_prompt_refinement_mode("both at once")

    def test_fl2v_refine_uses_three_fields_and_boundary_frames(self) -> None:
        calls = []

        def complete(messages, max_tokens, seed):
            del max_tokens, seed
            system = messages[0]["content"]
            user = messages[1]["content"][0]["text"]
            calls.append((system, user, messages[1]["content"]))
            if "STRUCTURE mode" in system:
                return """
TITLE: Doorway
=== SEGMENT 1 ===
DURATION_SECONDS: 6
CONTINUITY_FROM_PREV: false
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
BEAT_OPEN the door and step through.
DIALOGUE:
NONE
HANDOFF:
HANDOFF_OPEN one foot across the threshold.
=== END SEGMENT ===
=== SEGMENT 2 ===
DURATION_SECONDS: 6
CONTINUITY_FROM_PREV: true
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
BEAT_END arrive at the window.
DIALOGUE:
NONE
HANDOFF:
HANDOFF_END stopped at the glass.
=== END SEGMENT ===
""".strip()
            if "Segment 1 of 2" in user:
                return fl2v_prompt("OPENING")
            return fl2v_prompt("ENDING")

        _title, segments, rebalanced, debug, _record = nodes._plan_director_with_independent_segment_refine(
            complete,
            mode=FL2V,
            brief="Walk from the door to the window.",
            reference_text="",
            schedule=(6.0, 6.0),
            image_references=((1, "uri-first"), (2, "uri-last"), (4, "uri-extra")),
            director_videos=(),
            audio_slots=(),
            video_audio_slots=(),
            lora_triggers=(),
            prompt_example=None,
            quality=BALANCED,
            requested_max_tokens=4096,
            seed=4,
        )
        self.assertFalse(rebalanced)
        self.assertEqual(len(calls), 3)
        self.assertIn("exactly one FL2V group", calls[1][0])
        opening_blob = json.dumps(calls[1][2])
        ending_blob = json.dumps(calls[2][2])
        self.assertIn("uri-first", opening_blob)
        self.assertNotIn("uri-last", opening_blob)
        self.assertIn("uri-extra", opening_blob)
        self.assertIn("uri-last", ending_blob)
        self.assertNotIn("uri-first", ending_blob)
        self.assertIn("HANDOFF_OPEN", calls[2][1])
        self.assertIn("OPENING", segments[0].prompt)
        self.assertIn("integrated_multimodal_description:", segments[1].prompt)
        self.assertTrue(segments[1].continuity_from_prev)
        self.assertIn("segment 2/2 | refine=independent | attempt=1 | continuity=true", debug)
        groups, _packer = nodes._pack_director_long_groups(
            FL2V,
            segments,
            {1: "first-image", 2: "last-image", 4: "extra-image"},
            {},
            {},
            {},
        )
        self.assertEqual(groups[0]["first_frame"], "first-image")
        self.assertIsNone(groups[0]["last_frame"])
        self.assertIsNone(groups[1]["first_frame"])
        self.assertEqual(groups[1]["last_frame"], "last-image")
        self.assertTrue(groups[1]["continuityFromPrev"])

    def test_single_shot_request_still_attaches_visuals(self) -> None:
        messages = nodes._build_long_director_request(
            R2V,
            "Brief",
            "",
            (10.0, 10.0),
            ((1, "uri-picture-1"),),
            (nodes._DirectorVideoFrames(slot=1, frames="frames", frame_uris=("uri-video-1",)),),
            (),
            (),
            (),
            None,
            BALANCED,
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Produce exactly 2 segments", messages[0]["content"])
        blob = json.dumps(messages[1]["content"])
        self.assertIn("uri-picture-1", blob)
        self.assertIn("uri-video-1", blob)


def _png_size(uri: str) -> tuple[int, int]:
    prefix = "data:image/png;base64,"
    if not uri.startswith(prefix):
        raise AssertionError("expected a PNG data URI")
    with Image.open(io.BytesIO(base64.b64decode(uri[len(prefix) :]))) as image:
        return image.size


def _beat_plan(indexes: list[int], duration: float, *, empty_beat_at: int | None = None) -> str:
    blocks = ["TITLE: Drift"]
    for index in indexes:
        beat = (
            ""
            if index == empty_beat_at
            else f"BEAT_{index} starts, changes, and ends in a readable pose."
        )
        continuity = "false" if index == indexes[0] else "true"
        blocks.append(
            f"""=== SEGMENT {index} ===
DURATION_SECONDS: {duration}
CONTINUITY_FROM_PREV: {continuity}
PICTURES: 1
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
{beat}
DIALOGUE:
NONE
HANDOFF:
HANDOFF_{index} holds the ending pose.
=== END SEGMENT ==="""
        )
    return "\n".join(blocks)


def _segments_from_plan(text: str, schedule: tuple[float, ...]):
    _title, cards = nodes._parse_director_structure_output(text)
    segments, rebalanced, _misses = nodes._segments_from_structure_cards(
        cards,
        R2V,
        schedule,
        (1,),
        (),
        (),
        (),
    )
    return segments, rebalanced



class SegmentCapAndSoftBudgetTests(unittest.TestCase):
    def test_rebalance_respects_widget_segment_cap(self) -> None:
        from dataclasses import replace as dc_replace

        segments, _ = _segments_from_plan(_beat_plan(list(range(1, 6)), 10), (9.0,) * 5)
        heavy = [dc_replace(seg, duration_seconds=14.0) for seg in segments]
        out = nodes._rebalance_director_segment_durations(heavy, 45.0, max_segment_seconds=10.0)
        self.assertAlmostEqual(sum(seg.duration_seconds for seg in out), 45.0, places=3)
        self.assertTrue(all(4.0 <= seg.duration_seconds <= 10.0 for seg in out))

    def test_shots_1_to_9_with_seg15_do_not_make_135(self) -> None:
        brief = "[STORY_BEATS]\n1|45|1-9|arrive|LANDMARK\n[/STORY_BEATS]"
        schedule, plan = nodes._story_duration_schedule(
            brief, budget_seconds=45.0, max_segment_seconds=15.0,
            allow_duration_overrun=True,
        )
        self.assertIsNotNone(plan)
        self.assertAlmostEqual(sum(schedule), 45.0, places=3)
        self.assertEqual(plan.budget_seconds, 45.0)
        self.assertLess(len(schedule), 9)


class TruncatedStructureTests(unittest.TestCase):
    def test_trailing_segment_without_end_preserves_fields(self) -> None:
        plan = _beat_plan(list(range(1, 10)), 15).rsplit("=== END SEGMENT ===", 1)[0]
        self.assertEqual(plan.count("=== END SEGMENT ==="), 8)
        _title, cards = nodes._parse_director_structure_output(plan)
        self.assertEqual(len(cards), 9)
        self.assertEqual(cards[-1]["index"], 9)
        self.assertEqual(cards[-1]["duration_raw"], "15")
        self.assertIn("BEAT_9", cards[-1]["beat"])
        self.assertEqual(cards[-1]["dialogue"], "NONE")
        self.assertIn("HANDOFF_9", cards[-1]["handoff"])

    def test_trailing_junk_without_beat_is_ignored(self) -> None:
        plan = _beat_plan([1, 2], 10)
        expected = nodes._parse_director_structure_output(plan)
        for junk in ("Unfinished text", "=== SEGMENT 3 ===\nDURATION_SECONDS: 10\nBEAT:\nDIALOGUE: NONE"):
            with self.subTest(junk=junk):
                self.assertEqual(nodes._parse_director_structure_output(plan + "\n" + junk), expected)

    def test_only_last_open_segment_is_recovered(self) -> None:
        plan = _beat_plan([1], 10)
        plan += "\n=== SEGMENT 2 ===\nBEAT: Middle placeholder."
        plan += "\n=== SEGMENT 3 ===\nBEAT: Final placeholder."
        _title, cards = nodes._parse_director_structure_output(plan)
        self.assertEqual([card["index"] for card in cards], [1, 3])
        self.assertEqual(cards[-1]["beat"], "Final placeholder.")
        self.assertEqual(cards[-1]["dialogue"], "")
        self.assertEqual(cards[-1]["handoff"], "")
        self.assertEqual(cards[-1]["duration_raw"], "")

    def test_open_segment_accepts_prompt_fallback(self) -> None:
        _title, cards = nodes._parse_director_structure_output(
            "=== SEGMENT 1 ===\nPROMPT: A placeholder crosses the room."
        )
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["beat"], "A placeholder crosses the room.")

    def test_impossible_rebalance_reports_count_and_totals(self) -> None:
        segments, _rebalanced = _segments_from_plan(_beat_plan(list(range(1, 9)), 10), (10.0,) * 8)
        for total in (135, 31):
            with self.subTest(total=total):
                with self.assertRaises(nodes.PromptRefinerError) as error:
                    nodes._rebalance_director_segment_durations(segments, total)
                message = str(error.exception)
                self.assertIn("8 Director groups", message)
                self.assertIn("32 to 120 seconds", message)
                self.assertIn(f"requested total is {total} seconds", message)


class StructureSegmentCountTests(unittest.TestCase):
    def test_off_by_one_cards_are_accepted_and_rebalanced(self) -> None:
        schedule = (10.0,) * 12
        short, short_rebalanced = _segments_from_plan(_beat_plan(list(range(1, 12)), 10), schedule)
        self.assertTrue(short_rebalanced)
        self.assertEqual([segment.index for segment in short], list(range(1, 12)))
        self.assertAlmostEqual(sum(segment.duration_seconds for segment in short), 120.0, places=3)
        self.assertTrue(all(4.0 <= segment.duration_seconds <= 15.0 for segment in short))
        self.assertIn("BEAT_11", short[-1].beat)
        self.assertIn("HANDOFF_1", short[0].handoff)

        long, long_rebalanced = _segments_from_plan(_beat_plan(list(range(1, 14)), 8), schedule)
        self.assertTrue(long_rebalanced)
        self.assertEqual(len(long), 13)
        self.assertEqual([segment.index for segment in long], list(range(1, 14)))
        self.assertAlmostEqual(sum(segment.duration_seconds for segment in long), 120.0, places=3)
        self.assertTrue(all(4.0 <= segment.duration_seconds <= 15.0 for segment in long))

    def test_larger_count_mismatch_and_incoherent_indexes_still_fail(self) -> None:
        schedule = (10.0,) * 12
        with self.assertRaises(nodes.PromptRefinerError) as mismatch:
            _segments_from_plan(_beat_plan(list(range(1, 11)), 10), schedule)
        self.assertIn("exactly 12 groups", str(mismatch.exception))

        with self.assertRaises(nodes.PromptRefinerError) as incoherent:
            _segments_from_plan(_beat_plan(list(range(1, 11)) + [12], 10), schedule)
        self.assertIn("segment numbers", str(incoherent.exception))
        self.assertIn("1–11", str(incoherent.exception))

        with self.assertRaises(nodes.PromptRefinerError) as empty_beat:
            _segments_from_plan(_beat_plan(list(range(1, 12)), 10, empty_beat_at=5), schedule)
        self.assertIn("segment 5 has no BEAT", str(empty_beat.exception))

    def test_off_by_one_structure_refines_the_returned_group_count(self) -> None:
        plan = _beat_plan(list(range(1, 12)), 10)

        def complete(messages, max_tokens, seed):
            del max_tokens, seed
            system = messages[0]["content"]
            user = messages[1]["content"][0]["text"]
            if "STRUCTURE mode" in system:
                return plan
            self.assertIn("Segment ", user)
            self.assertIn(" of 11", user)
            return r2v_prompt(user.split("Segment ", 1)[1][:2])

        _title, segments, rebalanced, debug, _record = nodes._plan_director_with_independent_segment_refine(
            complete,
            mode=R2V,
            brief="A two-minute office story.",
            reference_text="",
            schedule=(10.0,) * 12,
            image_references=((1, "uri-picture-1"),),
            director_videos=(),
            audio_slots=(),
            video_audio_slots=(),
            lora_triggers=(),
            prompt_example=None,
            quality=BALANCED,
            requested_max_tokens=4096,
            seed=11,
        )
        self.assertTrue(rebalanced)
        self.assertEqual(len(segments), 11)
        self.assertAlmostEqual(sum(segment.duration_seconds for segment in segments), 120.0, places=3)
        self.assertTrue(all(4.0 <= segment.duration_seconds <= 15.0 for segment in segments))
        self.assertIn("durations_rebalanced=true", debug)
        self.assertIn("segment 11/11", debug)

    def test_json_plan_accepts_off_by_one_and_rejects_a_larger_gap(self) -> None:
        def payload(count: int, indexes: list[int] | None = None) -> str:
            segments = []
            for position in range(count):
                index = (indexes or list(range(1, count + 1)))[position]
                segments.append(
                    {
                        "index": index,
                        "duration_seconds": 10,
                        "prompt": r2v_prompt(f"JSON_{index}"),
                        "assets": {"pictures": [1], "videos": [], "video_audios": [], "audios": []},
                        "continuity_from_prev": index > 1,
                    }
                )
            return json.dumps({"title": "JSON drift", "segments": segments})

        title, segments, rebalanced = nodes._validate_long_director_plan(
            payload(3),
            R2V,
            (10.0, 10.0),
            (1,),
            (),
            (),
            (),
            (),
        )
        self.assertEqual(title, "JSON drift")
        self.assertTrue(rebalanced)
        self.assertEqual([segment.index for segment in segments], [1, 2, 3])
        self.assertAlmostEqual(sum(segment.duration_seconds for segment in segments), 20.0, places=3)
        self.assertTrue(all(4.0 <= segment.duration_seconds <= 15.0 for segment in segments))
        self.assertIn("JSON_3", segments[2].prompt)

        with self.assertRaises(nodes.PromptRefinerError) as mismatch:
            nodes._validate_long_director_plan(
                payload(4),
                R2V,
                (10.0, 10.0),
                (1,),
                (),
                (),
                (),
                (),
            )
        self.assertIn("exactly 2 groups", str(mismatch.exception))

        with self.assertRaises(nodes.PromptRefinerError) as incoherent:
            nodes._validate_long_director_plan(
                payload(2, indexes=[1, 3]),
                R2V,
                (10.0, 10.0),
                (1,),
                (),
                (),
                (),
                (),
            )
        self.assertIn("segment numbers 1, 3", str(incoherent.exception))


class StructureVisionSplitTests(unittest.TestCase):
    def test_structure_request_has_picture_labels_and_no_still_image_url(self) -> None:
        videos = (
            nodes._DirectorVideoFrames(slot=1, frames="frames", frame_uris=("uri-video-1",)),
        )
        messages = nodes._build_long_director_structure_request(
            R2V,
            "Brief with seven stills.",
            "",
            (10.0, 10.0),
            ((1, "uri-picture-1"), (7, "uri-picture-7")),
            videos,
            (),
            (),
        )
        content = messages[1]["content"]
        text = content[0]["text"]
        self.assertIn("<Picture 1>", text)
        self.assertIn("<Picture 7>", text)
        urls = [
            part["image_url"]["url"]
            for part in content
            if part.get("type") == "image_url"
        ]
        self.assertNotIn("uri-picture-1", urls)
        self.assertNotIn("uri-picture-7", urls)
        self.assertEqual(urls, ["uri-video-1"])

        stills_only = nodes._build_long_director_structure_request(
            R2V,
            "Brief",
            "",
            (10.0, 10.0),
            ((1, "uri-picture-1"), (2, "uri-picture-2")),
            (),
            (),
            (),
        )
        still_content = stills_only[1]["content"]
        self.assertTrue(all(part.get("type") != "image_url" for part in still_content))
        self.assertIn("<Picture 1>", still_content[0]["text"])
        self.assertIn("<Picture 2>", still_content[0]["text"])

        segment = nodes._DirectorLongSegment(
            index=1,
            duration_seconds=10.0,
            prompt="",
            picture_slots=(1, 7),
            video_slots=(),
            video_audio_slots=(),
            audio_slots=(),
            continuity_from_prev=False,
            beat="One beat.",
            dialogue="",
            handoff="Hold the pose.",
        )
        refine = nodes._build_long_director_segment_request(
            R2V,
            "Brief",
            "",
            (segment,),
            segment,
            ((1, "uri-picture-1"), (7, "uri-picture-7")),
            (),
            (),
            None,
            BALANCED,
        )
        refine_urls = [
            part["image_url"]["url"]
            for part in refine[1]["content"]
            if part.get("type") == "image_url"
        ]
        self.assertEqual(refine_urls, ["uri-picture-1", "uri-picture-7"])


class VisionStillImageScaleTests(unittest.TestCase):
    def test_still_references_use_still_max_side(self) -> None:
        self.assertEqual(nodes.STILL_IMAGE_MAX_SIDE, 512)
        self.assertEqual(nodes.VIDEO_FRAME_MAX_SIDE, 1024)
        # 1536×1024 is the still size that failed mtmd when several were attached.
        array = np.zeros((1024, 1536, 3), dtype=np.uint8)
        uri = nodes._image_to_data_uri(array)
        self.assertEqual(_png_size(uri), (512, 341))

        batch = (array.astype(np.float32) / 255.0)[np.newaxis, ...]
        self.assertEqual(_png_size(nodes._image_to_data_uri(batch)), (512, 341))

        small = np.zeros((64, 80, 3), dtype=np.uint8)
        self.assertEqual(_png_size(nodes._image_to_data_uri(small)), (80, 64))

        inputs = nodes._get_image_inputs({"reference_image_3": array, "reference_image_1": None})
        self.assertEqual(inputs[0][0], 3)
        self.assertEqual(_png_size(inputs[0][1]), (512, 341))

        video_uri = nodes._array_to_data_uri(array, max_side=nodes.VIDEO_FRAME_MAX_SIDE)
        self.assertEqual(_png_size(video_uri), (1024, 683))
        self.assertNotEqual(_png_size(video_uri), _png_size(uri))

        frames = np.zeros((2, 1024, 1536, 3), dtype=np.uint8)
        videos = nodes._get_director_video_frame_inputs(
            {"reference_video_frames_1": frames},
            sample_count=1,
        )
        self.assertEqual(_png_size(videos[0].frame_uris[0]), (1024, 683))


class LlamaPerformanceKwargTests(unittest.TestCase):
    def test_optional_llama_flags_follow_the_installed_signature(self) -> None:
        names = ["n_ctx", "n_batch", "n_ubatch", "n_gpu_layers", "flash_attn", "swa_full", "verbose"]
        cpu = nodes._llama_performance_kwargs(32768, 0, names)
        self.assertEqual(cpu["n_batch"], 2048)
        self.assertEqual(cpu["n_ubatch"], 1024)
        self.assertNotIn("flash_attn", cpu)
        self.assertTrue(cpu["swa_full"])

        gpu = nodes._llama_performance_kwargs(32768, -1, names)
        self.assertTrue(gpu["flash_attn"])
        self.assertLessEqual(gpu["n_ubatch"], gpu["n_batch"])
        self.assertLessEqual(gpu["n_batch"], 32768)

        self.assertEqual(
            nodes._llama_performance_kwargs(32768, -1, ["n_ctx", "n_gpu_layers", "verbose"]),
            {},
        )
        tiny = nodes._llama_performance_kwargs(512, -1, names)
        self.assertNotIn("n_batch", tiny)
        self.assertNotIn("n_ubatch", tiny)
        self.assertTrue(tiny["flash_attn"])

        class Declared:
            def __init__(self, model_path, n_ctx=512, n_batch=512, verbose=True, **kwargs):
                del model_path, n_ctx, n_batch, verbose, kwargs

        explicit = nodes._llama_explicit_parameter_names(Declared)
        self.assertIn("n_batch", explicit)
        self.assertNotIn("flash_attn", explicit)
        self.assertNotIn("kwargs", explicit)
        self.assertNotIn("swa_full", nodes._llama_performance_kwargs(32768, -1, explicit))

    def test_rejected_attention_flags_fall_back_without_hiding_load_errors(self) -> None:
        class FlashRejected:
            def __init__(self, **kwargs):
                if kwargs.get("flash_attn") or kwargs.get("swa_full"):
                    raise RuntimeError("flash attention / swa_full is not supported")
                self.kwargs = kwargs

        kept = nodes._instantiate_llama(
            FlashRejected,
            {"model_path": "m", "n_ctx": 32768},
            {"n_batch": 2048, "n_ubatch": 1024, "flash_attn": True, "swa_full": True},
        )
        self.assertEqual(kept.kwargs["n_batch"], 2048)
        self.assertEqual(kept.kwargs["n_ubatch"], 1024)
        self.assertNotIn("flash_attn", kept.kwargs)
        self.assertNotIn("swa_full", kept.kwargs)

        class OldApi:
            def __init__(self, **kwargs):
                if "n_batch" in kwargs or "flash_attn" in kwargs or "swa_full" in kwargs:
                    raise TypeError("unexpected keyword argument")
                self.kwargs = kwargs

        plain = nodes._instantiate_llama(
            OldApi,
            {"model_path": "m"},
            {"n_batch": 2048, "flash_attn": True},
        )
        self.assertEqual(plain.kwargs, {"model_path": "m"})

        class MissingModel:
            def __init__(self, **kwargs):
                del kwargs
                raise RuntimeError("model file not found")

        with self.assertRaises(RuntimeError) as missing:
            nodes._instantiate_llama(MissingModel, {"model_path": "m"}, {"n_batch": 2048})
        self.assertIn("model file not found", str(missing.exception))




class R2VAudioSelectionTests(unittest.TestCase):
    """Empty AUDIOS attaches no standalone audio. Explicit tags still hang."""

    def test_empty_or_missing_audios_attach_nothing(self) -> None:
        _pictures, _videos, _paired, audios = nodes._select_director_segment_assets(
            R2V,
            {"pictures": [1], "videos": [], "video_audios": [], "audios": []},
            1,
            (1,),
            (),
            (),
            (1, 2),
        )
        self.assertEqual(audios, ())
        missing = nodes._select_director_segment_assets(
            R2V,
            {},
            2,
            (),
            (),
            (),
            (1, 2),
        )[3]
        self.assertEqual(missing, ())
        groups, _packer = nodes._pack_director_long_groups(
            R2V,
            [
                nodes._DirectorLongSegment(
                    index=1,
                    duration_seconds=8.0,
                    prompt=r2v_prompt("SILENT", picture=False),
                    picture_slots=(),
                    video_slots=(),
                    video_audio_slots=(),
                    audio_slots=audios,
                    continuity_from_prev=False,
                )
            ],
            {},
            {},
            {},
            {1: "audio-1", 2: "audio-2"},
        )
        self.assertEqual(groups[0]["ref_audios"], {})

    def test_explicit_audio_selection_is_kept(self) -> None:
        _pictures, _videos, _paired, audios = nodes._select_director_segment_assets(
            R2V,
            {"pictures": [], "videos": [], "video_audios": [], "audios": [2]},
            1,
            (),
            (),
            (),
            (1, 2),
        )
        self.assertEqual(audios, (2,))
        groups, _packer = nodes._pack_director_long_groups(
            R2V,
            [
                nodes._DirectorLongSegment(
                    index=1,
                    duration_seconds=8.0,
                    prompt=r2v_prompt("VOICE", picture=False),
                    picture_slots=(),
                    video_slots=(),
                    video_audio_slots=(),
                    audio_slots=audios,
                    continuity_from_prev=False,
                )
            ],
            {},
            {},
            {},
            {1: "audio-1", 2: "audio-2"},
        )
        self.assertEqual(groups[0]["ref_audios"], {1: "audio-2"})


class StoryShotAudioBindingTests(unittest.TestCase):
    """NV Global @reference_audio_N (no voice-line) must still populate card audios."""

    def _plan_one_shot(self):
        beat = nodes._StoryBeatSpec(
            index=1,
            estimated_seconds=8.0,
            shot_start=1,
            shot_end=1,
            label="b1",
            body="",
            key_dialogue=(),
        )
        return nodes._CoverageKnifePlan(
            beats=(beat,),
            budget_seconds=8.0,
            group_count=1,
            estimated_sum=8.0,
            compressed=False,
            allocations=(8.0,),
            card_counts=(1,),
            segment_to_beat=(1,),
        )

    def _card(self):
        return {
            "index": 1,
            "duration_raw": "8",
            "continuity_raw": "false",
            "beat": "placeholder",
            "dialogue": "",
            "handoff": "end",
            "assets": {"pictures": [], "videos": [], "video_audios": [], "audios": []},
        }

    def test_no_brief_audio_tags_leave_connected_slots_unhung(self) -> None:
        brief = (
            "[STORY_BEATS]\n"
            "1|8|1-1|hold|\n"
            "[/STORY_BEATS]\n"
            "分镜1使用参考: <Picture 1>\n"
            "人物停下，没有对白。\n"
        )
        out = nodes._apply_story_shot_contract([self._card()], brief, self._plan_one_shot())
        self.assertEqual(out[0]["assets"]["audios"], [])
        _pictures, _videos, _paired, audios = nodes._select_director_segment_assets(
            R2V,
            out[0]["assets"],
            1,
            (1,),
            (),
            (),
            (1, 2),
        )
        self.assertEqual(audios, ())
        groups, _packer = nodes._pack_director_long_groups(
            R2V,
            [
                nodes._DirectorLongSegment(
                    index=1,
                    duration_seconds=8.0,
                    prompt=r2v_prompt("QUIET", picture=False),
                    picture_slots=(),
                    video_slots=(),
                    video_audio_slots=(),
                    audio_slots=audios,
                    continuity_from_prev=False,
                )
            ],
            {},
            {},
            {},
            {1: "leftover-1", 2: "leftover-2"},
        )
        self.assertEqual(groups[0]["ref_audios"], {})

    def test_single_group_without_audio_tags_ignores_connected_pins(self) -> None:
        silent = nodes._single_group_structure_segment(
            R2V,
            (8.0,),
            "人物停下，没有对白。",
            (),
            (),
            (),
            (1, 2),
        )
        self.assertEqual(silent.audio_slots, ())
        tagged = nodes._single_group_structure_segment(
            R2V,
            (8.0,),
            "对白用 @reference_audio_2 voice.wav",
            (),
            (),
            (),
            (1, 2),
        )
        self.assertEqual(tagged.audio_slots, (2,))

    def test_global_at_reference_audio_fills_empty_shot_audios(self) -> None:
        brief = (
            "Global\n"
            "@reference_audio_1 hero_voice.wav\n"
            "@reference_image_1 hero.png\n"
            "\n"
            "分镜1使用参考: <Picture 1>\n"
            "角色说「你好」\n"
        )
        out = nodes._apply_story_shot_contract([self._card()], brief, self._plan_one_shot())
        self.assertEqual(out[0]["assets"]["audios"], [1])
        self.assertIn(1, out[0]["assets"]["pictures"])

    def test_shot_at_reference_audio_is_collected(self) -> None:
        brief = (
            "Global\n"
            "@reference_audio_1 a.wav\n"
            "@reference_audio_2 b.wav\n"
            "\n"
            "分镜1使用参考: @reference_audio_2 b.wav\n"
            "台词「开始」\n"
        )
        out = nodes._apply_story_shot_contract([self._card()], brief, self._plan_one_shot())
        # Shot-local @reference_audio_2 wins over empty; global is not merged on top
        # when the shot already named an audio tag.
        self.assertEqual(out[0]["assets"]["audios"], [2])

    def test_shot_contract_lifts_exact_dialogue_without_beat_duplication(self) -> None:
        brief = (
            "分镜1使用参考:\n"
            "柳二姐抬眼说「（声音慵懒）蛇不就在这吗？来，摸摸看。」然后靠近。\n"
        )
        out = nodes._apply_story_shot_contract([self._card()], brief, self._plan_one_shot())
        self.assertEqual(out[0]["dialogue"], "（声音慵懒）蛇不就在这吗？来，摸摸看。")
        self.assertNotIn("蛇不就在这吗", out[0]["beat"])
        self.assertIn("然后靠近", out[0]["beat"])

    def test_shot_contract_preserves_structure_dialogue_when_shot_has_none(self) -> None:
        card = self._card()
        card["dialogue"] = "柳二姐: （声音慵懒）蛇不就在这吗？"
        brief = "分镜1使用参考:\n柳二姐抬眼，然后靠近。\n"
        out = nodes._apply_story_shot_contract([card], brief, self._plan_one_shot())
        # The shot slice has no quotes, so Structure's leftover line is not spoken here.
        self.assertEqual(out[0]["dialogue"], "")
        self.assertIn("柳二姐抬眼", out[0]["beat"])

    def test_later_shot_without_quotes_drops_structure_dialogue(self) -> None:
        brief = (
            "[STORY_BEATS]\n"
            "1|16|1-2|talk|\n"
            "[/STORY_BEATS]\n"
            "分镜1使用参考:\n"
            "她说「小六……看来你还没有尽兴呢。让姐姐好好伺候你……」\n"
            "分镜2使用参考:\n"
            "她的手停在袜夹上，镜头推向嘴唇，然后淡出。\n"
        )
        beat = nodes._story_beat_from_fields(1, 16.0, 1, 2, "talk", "")
        plan = nodes._CoverageKnifePlan(
            beats=(beat,),
            budget_seconds=16.0,
            group_count=2,
            estimated_sum=16.0,
            compressed=False,
            allocations=(8.0, 8.0),
            card_counts=(2,),
            segment_to_beat=(1, 1),
        )
        spoken = "小六……看来你还没有尽兴呢。让姐姐好好伺候你……"
        cards = []
        for index in (1, 2):
            card = self._card()
            card["index"] = index
            card["dialogue"] = spoken
            cards.append(card)
        out = nodes._apply_story_shot_contract(cards, brief, plan)
        self.assertIn("尽兴", out[0]["dialogue"])
        self.assertEqual(out[1]["dialogue"], "")
        invented = (
            "detailed_description:\n"
            "[Brief Shot 4] She speaks softly: (S1) <d>[Chinese] (S1) <d>[Chinese] "
            f"{spoken}</d></d> The camera stays on her face.\n"
            "overall_soundscape:\n"
            "Room tone.\n"
        )
        stripped = nodes._strip_unassigned_spoken_lines(invented)
        self.assertNotIn("<d", stripped.lower())
        self.assertNotIn("尽兴", stripped)
        self.assertIn("The camera stays on her face", stripped)
        self.assertIn("overall_soundscape:", stripped)

    def test_assigned_dialogue_wrappers_do_not_nest(self) -> None:
        spoken = "让姐姐好好伺候你……"
        assigned = f"(S1) <d>[Chinese] (S1) <d>[Chinese] {spoken}</d></d>"
        prompt = (
            "detailed_description:\n"
            "She speaks: (S1) <d>[Chinese] (S1) <d>[Chinese] "
            "小六……看来你还没有尽兴呢。让姐姐好好伺候你……</d></d>\n"
            "overall_soundscape:\n"
            "Room tone.\n"
        )
        out = nodes._force_assigned_dialogue_in_prompt(prompt, assigned)
        self.assertIn(f"<d>[Chinese] {spoken}</d>", out)
        self.assertNotIn("</d></d>", out)
        self.assertNotIn("<d>[Chinese] (S1)", out)
        self.assertNotIn("<d>[Chinese] <d>", out.lower())
        self.assertEqual(out.lower().count("<d"), 1)
        self.assertEqual(out.lower().count("</d>"), 1)


class R2VPictureSelectionTests(unittest.TestCase):
    """Empty PICTURES attaches no stills; anchor_all_pictures still force-attaches every connected still."""

    def test_model_selected_pictures_are_kept(self) -> None:
        pictures, _videos, _paired, _audios = nodes._select_director_segment_assets(
            R2V,
            {"pictures": [1, 3], "videos": [], "video_audios": [], "audios": []},
            1,
            (1, 2, 3),
            (),
            (),
            (),
        )
        self.assertEqual(pictures, (1, 3))
        with self.assertRaises(nodes.PromptRefinerError):
            nodes._select_director_segment_assets(
                R2V,
                {"pictures": [4], "videos": [], "video_audios": [], "audios": []},
                1,
                (1, 2, 3),
                (),
                (),
                (),
            )

    def test_empty_or_missing_pictures_attach_no_stills(self) -> None:
        connected = tuple(range(1, 10))
        pictures, videos, _paired, audios = nodes._select_director_segment_assets(
            R2V,
            {"pictures": [], "videos": [], "video_audios": [], "audios": []},
            2,
            connected,
            (1,),
            (),
            (1, 2),
        )
        self.assertEqual(pictures, ())
        self.assertEqual(videos, ())
        self.assertEqual(audios, ())
        missing_pictures, missing_videos, _paired, missing_audios = nodes._select_director_segment_assets(
            R2V,
            {},
            3,
            connected,
            (1,),
            (),
            (3,),
        )
        self.assertEqual(missing_pictures, ())
        self.assertEqual(missing_videos, ())
        self.assertEqual(missing_audios, ())

        segment = nodes._DirectorLongSegment(
            index=1,
            duration_seconds=8.0,
            prompt=r2v_prompt("BLANK"),
            picture_slots=pictures,
            video_slots=(),
            video_audio_slots=(),
            audio_slots=audios,
            continuity_from_prev=False,
        )
        groups, _packer = nodes._pack_director_long_groups(
            R2V,
            [segment],
            {slot: f"image-{slot}" for slot in connected},
            {},
            {},
            {1: "audio-1", 2: "audio-2"},
        )
        self.assertEqual(groups[0]["ref_images"], {})
        self.assertEqual(groups[0]["ref_audios"], {})

    def test_blank_shot_reference_does_not_fan_wire_connected_stills(self) -> None:
        brief = (
            "[STORY_BEATS]\n"
            "1|8|1-1|hold|\n"
            "[/STORY_BEATS]\n"
            "<Picture 1> 主角甲, <Picture 2> 客厅全景, <Picture 9> 红伞道具\n"
            "分镜1使用参考:\n"
            "人物停下，没有指定参考图。\n"
        )
        beat = nodes._story_beat_from_fields(1, 8.0, 1, 1, "hold", "")
        plan = nodes._CoverageKnifePlan(
            beats=(beat,),
            budget_seconds=8.0,
            group_count=1,
            estimated_sum=8.0,
            compressed=False,
            allocations=(8.0,),
            card_counts=(1,),
            segment_to_beat=(1,),
        )
        card = {
            "index": 1,
            "duration_raw": "8",
            "continuity_raw": "false",
            "beat": "placeholder",
            "dialogue": "",
            "handoff": "end",
            "assets": {"pictures": [], "videos": [], "video_audios": [], "audios": []},
        }
        contracted = nodes._apply_story_shot_contract([card], brief, plan)
        self.assertEqual(contracted[0]["assets"]["pictures"], [])
        connected = tuple(range(1, 10))
        pictures, _videos, _paired, _audios = nodes._select_director_segment_assets(
            R2V,
            contracted[0]["assets"],
            1,
            connected,
            (),
            (),
            (),
        )
        self.assertEqual(pictures, ())
        anchored, _videos, _paired, _audios = nodes._select_director_segment_assets(
            R2V,
            contracted[0]["assets"],
            1,
            connected,
            (),
            (),
            (),
            anchor_all_pictures=True,
        )
        self.assertEqual(anchored, connected)
        groups, _packer = nodes._pack_director_long_groups(
            R2V,
            [
                nodes._DirectorLongSegment(
                    index=1,
                    duration_seconds=8.0,
                    prompt=r2v_prompt("BLANK", picture=False),
                    picture_slots=pictures,
                    video_slots=(),
                    video_audio_slots=(),
                    audio_slots=(),
                    continuity_from_prev=False,
                )
            ],
            {slot: f"image-{slot}" for slot in connected},
            {},
            {},
            {},
        )
        self.assertEqual(groups[0]["ref_images"], {})

    def test_at_reference_image_shot_line_fills_pictures(self) -> None:
        brief = (
            "[STORY_BEATS]\n"
            "1|8|1-1|hold|\n"
            "[/STORY_BEATS]\n"
            "@reference_image_9 unused.png\n"
            "分镜1使用参考: @reference_image_1, @reference_image_2\n"
            "人物站在办公室。\n"
        )
        beat = nodes._story_beat_from_fields(1, 8.0, 1, 1, "hold", "")
        plan = nodes._CoverageKnifePlan(
            beats=(beat,),
            budget_seconds=8.0,
            group_count=1,
            estimated_sum=8.0,
            compressed=False,
            allocations=(8.0,),
            card_counts=(1,),
            segment_to_beat=(1,),
        )
        card = {
            "index": 1,
            "duration_raw": "8",
            "continuity_raw": "false",
            "beat": "placeholder",
            "dialogue": "",
            "handoff": "end",
            "assets": {"pictures": [], "videos": [], "video_audios": [], "audios": []},
        }
        contracted = nodes._apply_story_shot_contract([card], brief, plan)
        self.assertEqual(contracted[0]["assets"]["pictures"], [1, 2])
        pictures, _videos, _paired, _audios = nodes._select_director_segment_assets(
            R2V,
            contracted[0]["assets"],
            1,
            (1, 2, 3),
            (),
            (),
            (),
        )
        self.assertEqual(pictures, (1, 2))
        groups, _packer = nodes._pack_director_long_groups(
            R2V,
            [
                nodes._DirectorLongSegment(
                    index=1,
                    duration_seconds=8.0,
                    prompt=r2v_prompt("OFFICE"),
                    picture_slots=pictures,
                    video_slots=(),
                    video_audio_slots=(),
                    audio_slots=(),
                    continuity_from_prev=False,
                )
            ],
            {1: "image-1", 2: "image-2", 3: "image-3"},
            {},
            {},
            {},
        )
        self.assertEqual(set(groups[0]["ref_images"]), {0, 1})

        blank = (
            "[STORY_BEATS]\n"
            "1|8|1-1|hold|\n"
            "[/STORY_BEATS]\n"
            "@reference_image_1 hero.png\n"
            "@reference_image_2 office.png\n"
            "分镜1使用参考:\n"
            "人物停下。\n"
        )
        empty = nodes._apply_story_shot_contract([card], blank, plan)
        self.assertEqual(empty[0]["assets"]["pictures"], [])

    def test_prompt_cited_picture_is_packed_when_linked(self) -> None:
        prompt = (
            "subject_definitions:\n"
            "<Picture 1> subject, fully_preserved.\n"
            "<Picture 2> office environment, fully_preserved.\n"
            "summary:\n"
            "The subject enters the office.\n"
            "retention_analysis:\n"
            "<Picture 1>: identity.\n"
            "<Picture 2>: office.\n"
            "detailed_description:\n"
            "[Shot 1] The subject crosses the room.\n"
            "overall_soundscape:\n"
            "Room tone.\n"
            "non_diegetic_music:\n"
            "N/A"
        )
        self.assertEqual(nodes._picture_slots_for_materialize((1,), prompt, (1, 2)), (1, 2))
        self.assertEqual(nodes._picture_slots_for_materialize((1,), prompt, (1,)), (1,))
        silent = r2v_prompt("QUIET", picture=False)
        self.assertEqual(nodes._picture_slots_for_materialize((), silent, (1, 2, 3)), ())
        self.assertEqual(
            nodes._picture_slots_for_materialize((), silent + "\n@reference_image_2 office.png", (1, 2)),
            (2,),
        )
        plan = f"""
TITLE: Cited still
=== SEGMENT 1 ===
DURATION_SECONDS: 8
CONTINUITY_FROM_PREV: false
PICTURES: 1
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
PROMPT:
{prompt}
=== END SEGMENT ===
""".strip()
        _title, segments, _rebalanced = nodes._validate_long_director_plan(
            plan,
            R2V,
            (8.0,),
            image_slots=(1, 2),
            video_slots=(),
            video_audio_slots=(),
            audio_slots=(),
            lora_triggers=(),
        )
        self.assertEqual(segments[0].picture_slots, (1, 2))
        groups, _packer = nodes._pack_director_long_groups(
            R2V,
            segments,
            {1: "image-1", 2: "image-2"},
            {},
            {},
            {},
        )
        self.assertEqual(set(groups[0]["ref_images"]), {0, 1})
        specs = nodes._director_ui_group_specs(R2V, segments)
        self.assertEqual(specs[0]["pictures"], [1, 2])

    def test_omitted_environment_picture_is_injected_and_wired(self) -> None:
        prompt = (
            "subject_definitions:\n"
            "<Subject 1> Hero, identified from <Picture 1>, must retain her face.\n"
            "<Environment> Interior office space, implied by the lighting and context.\n"
            "summary:\n"
            "The face stays close, then the action shifts.\n"
            "retention_analysis:\n"
            "<Picture 1>: Preserves the hero.\n"
            "detailed_description:\n"
            "[Shot 1] The camera holds a close-up.\n"
            "overall_soundscape:\n"
            "Room tone.\n"
            "non_diegetic_music:\n"
            "N/A"
        )
        final = nodes._finalize_director_segment_prompt(
            prompt,
            engine=nodes.FULL_REFERENCE_ENGINE,
            picture_slots=(1, 2),
            lora_triggers=(),
            index=2,
        )
        environment = next(line for line in final.splitlines() if "<Environment>" in line)
        self.assertIn("<Picture 2>", environment)
        self.assertIn("<Environment>", environment)
        self.assertNotRegex(environment, r"<Subject\s+2>")
        retention = nodes._h3_field_body(final, "retention_analysis:")
        definitions = nodes._h3_field_body(final, "subject_definitions:")
        self.assertIn("<Picture 2>", definitions)
        self.assertIn("<Picture 2>", retention)
        slots = nodes._picture_slots_for_materialize((1, 2), final, (1, 2))
        self.assertEqual(slots, (1, 2))
        segment = nodes._DirectorLongSegment(
            index=2,
            duration_seconds=8.0,
            prompt=final,
            picture_slots=slots,
            video_slots=(),
            video_audio_slots=(),
            audio_slots=(),
            continuity_from_prev=True,
        )
        groups, _packer = nodes._pack_director_long_groups(
            R2V, [segment], {1: "image-1", 2: "image-2"}, {}, {}, {},
        )
        self.assertEqual(set(groups[0]["ref_images"]), {0, 1})
        specs = nodes._director_ui_group_specs(R2V, [segment])
        self.assertEqual(specs[0]["pictures"], [1, 2])

        bare = (
            "subject_definitions:\n"
            "<Environment> Interior office space.\n"
            "summary:\n"
            "Holds.\n"
            "retention_analysis:\n"
            "No still was assigned.\n"
            "detailed_description:\n"
            "[Shot 1] The camera holds.\n"
            "overall_soundscape:\n"
            "Room tone.\n"
            "non_diegetic_music:\n"
            "N/A"
        )
        untouched = nodes._finalize_director_segment_prompt(
            bare,
            engine=nodes.FULL_REFERENCE_ENGINE,
            picture_slots=(),
            lora_triggers=(),
            index=1,
        )
        self.assertNotIn("<Picture", untouched)
        self.assertEqual(nodes._picture_slots_for_materialize((), untouched, (1, 2)), ())

        card = {
            "index": 1,
            "duration_raw": "8",
            "continuity_raw": "false",
            "beat": "分镜1使用参考: <Picture 1>, <Picture 2>\n人物在办公室。",
            "dialogue": "",
            "handoff": "end",
            "assets": {"pictures": [1], "videos": [], "video_audios": [], "audios": []},
        }
        segments, _rebalanced, _misses = nodes._segments_from_structure_cards(
            [card], R2V, (8.0,), (1, 2), (), (), (),
        )
        self.assertEqual(segments[0].picture_slots, (1, 2))

    def test_at_reference_image_labels_bind_scene_and_pack_keys(self) -> None:
        brief = (
            "[STORY_BEATS]\n"
            "1|16|1-2|talk|\n"
            "[/STORY_BEATS]\n"
            "Global reference description:\n"
            "@reference_image_1 小程, @reference_image_2 赵凤红, @reference_image_3 办公室.\n"
            "分镜1使用参考:\n"
            "小程在办公室里看见赵凤红。\n"
            "分镜2使用参考:\n"
            "赵凤红靠近镜头。\n"
        )
        normalized = nodes._normalize_director_asset_links(
            brief, {1: "a", 2: "b", 3: "c"}, {}, {}, {},
        )
        self.assertIn("<Picture 3> 办公室", normalized)
        self.assertNotIn("@reference_image_3", normalized)
        beat = nodes._story_beat_from_fields(1, 16.0, 1, 2, "talk", "")
        plan = nodes._CoverageKnifePlan(
            beats=(beat,),
            budget_seconds=16.0,
            group_count=2,
            estimated_sum=16.0,
            compressed=False,
            allocations=(8.0, 8.0),
            card_counts=(2,),
            segment_to_beat=(1, 1),
        )
        cards = []
        for index in (1, 2):
            cards.append({
                "index": index,
                "duration_raw": "8",
                "continuity_raw": "false" if index == 1 else "true",
                "beat": "placeholder",
                "dialogue": "",
                "handoff": "end",
                "assets": {"pictures": [], "videos": [], "video_audios": [], "audios": []},
            })
        out = nodes._apply_story_shot_contract(cards, brief, plan)
        self.assertEqual(out[0]["assets"]["pictures"], [1, 2, 3])
        self.assertEqual(out[1]["assets"]["pictures"], [2])
        self.assertFalse(nodes._is_scene_picture_label("赵凤红"))
        self.assertFalse(nodes._is_scene_picture_label("小程"))
        self.assertTrue(nodes._is_scene_picture_label("办公室"))

        prompt = (
            "subject_definitions:\n"
            "<Subject 1> Heroine, identified from <Picture 2>, must retain her face.\n"
            "<Environment> Office interior, implied by the window light.\n"
            "summary:\n"
            "She stays in the office.\n"
            "retention_analysis:\n"
            "<Picture 2>: identity.\n"
            "detailed_description:\n"
            "[Shot 1] The camera holds.\n"
            "overall_soundscape:\n"
            "Room tone.\n"
            "non_diegetic_music:\n"
            "N/A"
        )
        slots = nodes._picture_slots_for_materialize((2,), prompt, (1, 2, 3))
        scene = nodes._scene_slots_for_bare_environment(prompt, "", brief, (1, 2, 3), slots)
        self.assertEqual(scene, (3,))
        slots = tuple(sorted(set(slots) | set(scene)))
        bound = nodes._ensure_full_reference_picture_labels(prompt, slots)
        environment = next(line for line in bound.splitlines() if "<Environment>" in line)
        self.assertIn("identified from <Picture 3>", environment)
        self.assertIn("<Picture 2>", bound)
        segment = nodes._DirectorLongSegment(
            index=2,
            duration_seconds=8.0,
            prompt=bound,
            picture_slots=slots,
            video_slots=(),
            video_audio_slots=(),
            audio_slots=(),
            continuity_from_prev=True,
        )
        groups, _packer = nodes._pack_director_long_groups(
            R2V,
            [segment],
            {1: "image-1", 2: "image-2", 3: "image-3"},
            {},
            {},
            {},
        )
        self.assertEqual(groups[0]["ref_images"][1], "image-2")
        self.assertEqual(groups[0]["ref_images"][2], "image-3")
        self.assertNotIn(0, groups[0]["ref_images"])
        specs = nodes._director_ui_group_specs(R2V, [segment])
        self.assertEqual(specs[0]["pictures"], [2, 3])

        quiet = (
            "subject_definitions:\n"
            "<Subject 1> Heroine.\n"
            "summary:\n"
            "She waits.\n"
            "retention_analysis:\n"
            "No still.\n"
            "detailed_description:\n"
            "[Shot 1] She waits.\n"
            "overall_soundscape:\n"
            "Room tone.\n"
            "non_diegetic_music:\n"
            "N/A"
        )
        self.assertEqual(
            nodes._scene_slots_for_bare_environment(quiet, "人物停下。", brief, (1, 2, 3), ()),
            (),
        )

    def test_anchor_all_pictures_forces_every_connected_still(self) -> None:
        pictures, _videos, _paired, _audios = nodes._select_director_segment_assets(
            R2V,
            {"pictures": [1, 3], "videos": [], "video_audios": [], "audios": []},
            1,
            (1, 2, 3),
            (),
            (),
            (),
            anchor_all_pictures=True,
        )
        self.assertEqual(pictures, (1, 2, 3))

    def test_structure_system_wording_follows_toggle(self) -> None:
        tighten = nodes._long_director_structure_system(R2V, (10.0, 10.0))
        self.assertIn("Assign only the pictures this beat uses", tighten)
        self.assertIn("@reference_image_N", tighten)
        self.assertNotIn("project-wide anchor and may be listed on every segment", tighten)
        full = nodes._long_director_structure_system(
            R2V, (10.0, 10.0), anchor_all_pictures=True
        )
        self.assertIn("project-wide anchor and may be listed on every segment", full)
        self.assertNotIn("Assign only the pictures this beat uses", full)

    def test_marked_plan_honors_subset_unless_anchor_all(self) -> None:
        plan = """
TITLE: Subset stills
=== SEGMENT 1 ===
DURATION_SECONDS: 10
CONTINUITY_FROM_PREV: false
PICTURES: 1, 3
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
PROMPT:
subject_definitions:
Hero.
summary:
Opens.
retention_analysis:
Keep the look.
detailed_description:
[Shot 1] A hand sets down a cup.
overall_soundscape:
Room tone.
non_diegetic_music:
N/A
=== END SEGMENT ===
=== SEGMENT 2 ===
DURATION_SECONDS: 10
CONTINUITY_FROM_PREV: true
PICTURES: 2
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
PROMPT:
subject_definitions:
Hero.
summary:
Continues.
retention_analysis:
Keep the room.
detailed_description:
[Shot 1] The camera finds the window.
overall_soundscape:
Room tone.
non_diegetic_music:
N/A
=== END SEGMENT ===
""".strip()
        _title, segments, _rebalanced = nodes._validate_long_director_plan(
            plan,
            R2V,
            (10.0, 10.0),
            image_slots=(1, 2, 3),
            video_slots=(),
            video_audio_slots=(),
            audio_slots=(),
            lora_triggers=(),
        )
        self.assertEqual(segments[0].picture_slots, (1, 3))
        self.assertEqual(segments[1].picture_slots, (2,))

        _title, anchored, _rebalanced = nodes._validate_long_director_plan(
            plan,
            R2V,
            (10.0, 10.0),
            image_slots=(1, 2, 3),
            video_slots=(),
            video_audio_slots=(),
            audio_slots=(),
            lora_triggers=(),
            anchor_all_pictures=True,
        )
        self.assertEqual(anchored[0].picture_slots, (1, 2, 3))
        self.assertEqual(anchored[1].picture_slots, (1, 2, 3))

    def test_marked_empty_pictures_line_packs_no_ref_images(self) -> None:
        prompt = r2v_prompt("EMPTY", picture=False)
        plan = f"""
TITLE: No stills
=== SEGMENT 1 ===
DURATION_SECONDS: 8
CONTINUITY_FROM_PREV: false
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS: 1
PROMPT:
{prompt}
=== END SEGMENT ===
=== SEGMENT 2 ===
DURATION_SECONDS: 8
CONTINUITY_FROM_PREV: true
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
PROMPT:
{prompt}
=== END SEGMENT ===
""".strip()
        _title, segments, _rebalanced = nodes._validate_long_director_plan(
            plan,
            R2V,
            (8.0, 8.0),
            image_slots=tuple(range(1, 10)),
            video_slots=(),
            video_audio_slots=(),
            audio_slots=(1,),
            lora_triggers=(),
        )
        self.assertEqual(segments[0].picture_slots, ())
        self.assertEqual(segments[1].picture_slots, ())
        self.assertEqual(segments[0].video_slots, ())
        self.assertEqual(segments[0].video_audio_slots, ())
        self.assertEqual(segments[0].audio_slots, (1,))
        self.assertEqual(segments[1].audio_slots, ())
        groups, _packer = nodes._pack_director_long_groups(
            R2V,
            segments,
            {slot: f"image-{slot}" for slot in range(1, 10)},
            {},
            {},
            {1: "audio-1"},
        )
        self.assertEqual(groups[0]["ref_images"], {})
        self.assertEqual(groups[1]["ref_images"], {})
        self.assertEqual(groups[0]["ref_audios"], {0: "audio-1"})
        self.assertEqual(groups[1]["ref_audios"], {})

    def test_planner_exposes_anchor_all_pictures_widget(self) -> None:
        required = nodes.Qwen38LongVideoDirectorPlanner.INPUT_TYPES()["required"]
        self.assertIn("anchor_all_pictures", required)
        self.assertEqual(required["anchor_all_pictures"][0], "BOOLEAN")
        self.assertFalse(required["anchor_all_pictures"][1]["default"])
        tooltip = required["anchor_all_pictures"][1]["tooltip"]
        self.assertIn("空＝不挂图", tooltip)
        self.assertIn("no stills", tooltip)
        self.assertNotIn("falls back to all connected stills", tooltip)




def _template_block(system: str) -> str:
    start = system.index("Exact return template")
    end = system.index("=== END SEGMENT ===", start)
    return system[start:end]


def _words(count: int) -> str:
    return " ".join(f"line{index}" for index in range(count))


def _dialogue_plan(rows: list[tuple[float, str]]) -> str:
    blocks = ["TITLE: Dialogue rhythm"]
    for index, (duration, dialogue) in enumerate(rows, start=1):
        continuity = "false" if index == 1 else "true"
        blocks.append(
            f"""=== SEGMENT {index} ===
DURATION_SECONDS: {duration}
CONTINUITY_FROM_PREV: {continuity}
PICTURES: 1
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
BEAT_{index} starts, changes, and ends in a readable pose.
DIALOGUE:
{dialogue or "NONE"}
HANDOFF:
HANDOFF_{index} holds the ending pose.
=== END SEGMENT ==="""
        )
    return "\n".join(blocks)


class DialogueDurationTests(unittest.TestCase):
    def test_templates_use_a_range_placeholder_instead_of_the_equal_average(self) -> None:
        schedule = (10.0,) * 6
        for system in (
            nodes._long_director_structure_system(R2V, schedule),
            nodes._long_director_system(R2V, schedule, BALANCED),
        ):
            template = _template_block(system)
            self.assertIn("DURATION_SECONDS: <4-15>", template)
            self.assertNotRegex(template, r"DURATION_SECONDS:\s*\d")
            self.assertNotIn("10.000", template)
            self.assertIn("near 15", system)
            self.assertIn("near 4", system)
            self.assertIn("make their sum exactly", system)
            self.assertIn("60.000 seconds", system)
            self.assertIn("only the average used to estimate that group count", system)

        uneven = nodes._long_director_structure_system(R2V, (4.0, 15.0, 8.0))
        uneven_template = _template_block(uneven)
        self.assertIn("DURATION_SECONDS: <4-15>", uneven_template)
        self.assertNotIn("DURATION_SECONDS: 9", uneven_template)
        self.assertNotIn("4.000", uneven_template)
        self.assertNotIn("15.000", uneven_template)

    def test_segment_duration_tooltip_is_a_count_estimator(self) -> None:
        tooltip = nodes.Qwen38LongVideoDirectorPlanner.INPUT_TYPES()["required"]["segment_duration_seconds"][1][
            "tooltip"
        ]
        self.assertIn("group count", tooltip)
        self.assertIn("not a target or forced duration", tooltip)
        self.assertIn("STORY_BEATS", tooltip)

    def test_cjk_dialogue_is_not_counted_as_one_english_word_per_character(self) -> None:
        english = _words(40)
        cjk = "字" * 40
        self.assertEqual(nodes._dialogue_speech_units(english), 40.0)
        self.assertAlmostEqual(nodes._dialogue_speech_units(cjk), 20.0)
        self.assertAlmostEqual(nodes._dialogue_speech_units("你好 hello"), 2.0)
        self.assertEqual(nodes._dialogue_speech_units("NONE"), 1.0)
        self.assertEqual(nodes._dialogue_speech_units(""), 0.0)

    def test_near_equal_durations_follow_uneven_dialogue(self) -> None:
        # Observed failure: 60s at a 10s average came back as six copies of 10.0.
        cjk = "字" * 40
        plan = _dialogue_plan(
            [
                (10.0, _words(40)),
                (10.0, ""),
                (10.2, "Look"),
                (9.8, cjk),
                (10.0, ""),
                (10.0, _words(40)),
            ]
        )
        segments, rebalanced = _segments_from_plan(plan, (10.0,) * 6)
        durations = [segment.duration_seconds for segment in segments]
        self.assertTrue(rebalanced)
        self.assertEqual(len(durations), 6)
        self.assertAlmostEqual(sum(durations), 60.0, places=3)
        self.assertTrue(all(4.0 <= value <= 15.0 for value in durations))
        self.assertGreater(durations[0], durations[3])
        self.assertGreater(durations[3], durations[1])
        self.assertGreater(durations[5], durations[2])
        self.assertGreater(durations[0], 12.0)
        self.assertLess(durations[1], 9.0)
        self.assertAlmostEqual(durations[0], durations[5], delta=0.01)
        self.assertEqual(segments[0].dialogue, _words(40))
        self.assertEqual(segments[1].dialogue, "")

    def test_matching_unequal_rhythm_is_preserved(self) -> None:
        plan = _dialogue_plan(
            [
                (14.0, _words(40)),
                (5.0, ""),
                (11.0, _words(20)),
            ]
        )
        segments, rebalanced = _segments_from_plan(plan, (10.0, 10.0, 10.0))
        self.assertFalse(rebalanced)
        self.assertEqual([segment.duration_seconds for segment in segments], [14.0, 5.0, 11.0])

    def test_equal_durations_without_dialogue_imbalance_stay_equal(self) -> None:
        plan = _dialogue_plan([(10.0, ""), (10.0, ""), (10.0, "")])
        segments, rebalanced = _segments_from_plan(plan, (10.0, 10.0, 10.0))
        self.assertFalse(rebalanced)
        self.assertEqual([segment.duration_seconds for segment in segments], [10.0, 10.0, 10.0])

    def test_inverted_durations_are_retargeted_to_dialogue(self) -> None:
        plan = _dialogue_plan(
            [
                (5.0, _words(40)),
                (14.0, ""),
                (11.0, "Look"),
            ]
        )
        segments, rebalanced = _segments_from_plan(plan, (10.0, 10.0, 10.0))
        durations = [segment.duration_seconds for segment in segments]
        self.assertTrue(rebalanced)
        self.assertAlmostEqual(sum(durations), 30.0, places=3)
        self.assertTrue(all(4.0 <= value <= 15.0 for value in durations))
        self.assertGreater(durations[0], durations[1])
        self.assertGreater(durations[0], durations[2])


if __name__ == "__main__":
    unittest.main()


class CoverageKnifeTests(unittest.TestCase):
    def test_dialogue_landmark_ignores_delivery_cue_but_keeps_spoken_words(self) -> None:
        cue_landmark = nodes._normalize_dialogue_needle("（声音慵懒）蛇不就在这吗")
        panting_landmark = nodes._normalize_dialogue_needle("（娇喘）啊……弟弟好大")
        spoken = nodes._normalize_dialogue_needle("柳二姐「蛇不就在这吗？来，摸摸看。」")
        self.assertEqual(cue_landmark, "蛇不就在这吗")
        self.assertEqual(panting_landmark, "啊弟弟好大")
        self.assertIn(cue_landmark, spoken)
        # Parentheses without delivery-cue words remain semantically significant.
        self.assertIn("明天", nodes._normalize_dialogue_needle("我们（明天）出发"))

    """Host-side beat floor + proportional compression when budget < Σest."""

    JOB18_EST = (45.0, 20.0, 30.0, 40.0, 50.0, 30.0, 45.0)
    JOB18_BUDGET = 216.0
    JOB18_GROUPS = 22

    SINGLE_BEAT_BRIEF = "[STORY_BEATS]\n1|45|1-9|arrive|LANDMARK\n[/STORY_BEATS]"

    def test_single_beat_block_and_headers_are_detected(self) -> None:
        for brief in (
            self.SINGLE_BEAT_BRIEF,
            "节拍1 (est 45s) 分镜1-9\nLANDMARK",
            "Beat 1 (est 45s) 分镜1-9\nLANDMARK",
        ):
            with self.subTest(brief=brief):
                beats = nodes._parse_story_beats_from_brief(brief)
                self.assertEqual(len(beats), 1)
                self.assertEqual(beats[0].estimated_seconds, 45.0)
                self.assertEqual((beats[0].shot_start, beats[0].shot_end), (1, 9))
                plan = nodes._build_coverage_knife_plan(brief, budget_seconds=45.0, group_count=5)
                self.assertIsNotNone(plan)
                self.assertEqual(plan.card_counts, (5,))

    def test_overrun_follows_beat_est_not_shot_range(self) -> None:
        """Shots 1-9 are coverage only; overrun uses beat estimated_seconds (45), not 9*segment."""
        schedule, plan = nodes._story_duration_schedule(
            self.SINGLE_BEAT_BRIEF, budget_seconds=45.0, max_segment_seconds=10.0,
            allow_duration_overrun=True,
        )
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.beats), 1)
        self.assertAlmostEqual(sum(schedule), 45.0, places=3)
        self.assertEqual(plan.budget_seconds, 45.0)
        self.assertEqual(plan.card_counts, (5,))
        schedule15, plan15 = nodes._story_duration_schedule(
            self.SINGLE_BEAT_BRIEF, budget_seconds=45.0, max_segment_seconds=15.0,
            allow_duration_overrun=True,
        )
        self.assertAlmostEqual(sum(schedule15), 45.0, places=3)
        self.assertEqual(plan15.budget_seconds, 45.0)
        self.assertLess(sum(schedule15), 135.0)

    def test_single_beat_unlocked_durations_follow_dialogue(self) -> None:
        schedule, plan = nodes._story_duration_schedule(
            self.SINGLE_BEAT_BRIEF, budget_seconds=45.0, max_segment_seconds=10.0,
            allow_duration_overrun=True,
        )
        assert plan is not None
        self.assertEqual(plan.card_counts, (5,))
        _title, cards = nodes._parse_director_structure_output(self._cards_for_plan(plan))
        for card in cards:
            card["dialogue"] = "NONE"
        cards[0]["dialogue"] = "LANDMARK " + _words(40)
        cards[2]["dialogue"] = _words(36)
        cards[4]["dialogue"] = _words(36)
        cards[1]["dialogue"] = "ok"
        cards[3]["dialogue"] = "ok"
        segments, rebalanced, misses = nodes._segments_from_structure_cards(
            cards, R2V, schedule, (), (), (), (),
            coverage_plan=plan, locked_durations=False,
            max_segment_seconds=10.0,
        )
        durations = [seg.duration_seconds for seg in segments]
        self.assertEqual(misses, [])
        self.assertTrue(rebalanced)
        self.assertEqual(len(segments), plan.card_counts[0])
        self.assertAlmostEqual(sum(durations), 45.0, places=2)
        self.assertTrue(all(4.0 <= value <= 10.0 for value in durations))
        heavy = (durations[0] + durations[2] + durations[4]) / 3
        light = (durations[1] + durations[3]) / 2
        self.assertGreater(heavy, light)

        locked, rebalanced, misses = nodes._segments_from_structure_cards(
            cards, R2V, schedule, (), (), (), (),
            coverage_plan=plan, locked_durations=True,
            max_segment_seconds=10.0,
        )
        self.assertEqual([seg.duration_seconds for seg in locked], list(schedule))
        self.assertFalse(rebalanced)
        self.assertEqual(misses, [])

    def test_single_beat_unlocked_keeps_llm_durations_when_dialogue_flat(self) -> None:
        """Coverage equal-seed must not flatten a single beat when unlock is on.

        Overrun follows beat est (45s → 5 cards). Mild dialogue (spread < 8 units),
        but structure cards already chose an uneven rhythm. Retarget correctly stays
        off; unlocked path should still keep DURATION_SECONDS instead of N×(budget/N).
        """
        schedule, plan = nodes._story_duration_schedule(
            self.SINGLE_BEAT_BRIEF, budget_seconds=45.0, max_segment_seconds=10.0,
            allow_duration_overrun=True,
        )
        assert plan is not None
        self.assertEqual(len(schedule), 5)
        self.assertAlmostEqual(sum(schedule), 45.0, places=3)
        llm_durations = (10.0, 10.0, 9.0, 8.0, 8.0)
        blocks = ["TITLE: Flat dialogue single beat"]
        for index, duration in enumerate(llm_durations, start=1):
            continuity = "false" if index == 1 else "true"
            dialogue = "LANDMARK" if index == 1 else "NONE"
            blocks.append(
                f"""=== SEGMENT {index} ===
DURATION_SECONDS: {duration:g}
CONTINUITY_FROM_PREV: {continuity}
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
Story card {index}.
DIALOGUE:
{dialogue}
HANDOFF:
Handoff {index}.
=== END SEGMENT ==="""
            )
        _title, cards = nodes._parse_director_structure_output("\n".join(blocks))
        units = [
            nodes._dialogue_speech_units(card.get("dialogue") or "")
            for card in sorted(cards, key=lambda item: int(item["index"]))
        ]
        self.assertLess(max(units) - min(units), nodes._DIALOGUE_REDISTRIBUTE_MIN_UNITS)
        segments, rebalanced, misses = nodes._segments_from_structure_cards(
            cards, R2V, schedule, (), (), (), (),
            coverage_plan=plan, locked_durations=False,
            max_segment_seconds=10.0,
        )
        durations = [seg.duration_seconds for seg in segments]
        self.assertEqual(misses, [])
        self.assertEqual(len(durations), 5)
        self.assertAlmostEqual(sum(durations), 45.0, places=2)
        self.assertTrue(all(4.0 <= value <= 10.0 for value in durations))
        self.assertNotEqual(durations, list(schedule))
        self.assertGreaterEqual(durations[0], durations[-1])
        self.assertTrue(all(4.0 <= value <= 10.0 for value in durations))
        self.assertAlmostEqual(durations[0], 10.0, places=2)
        self.assertAlmostEqual(durations[-1], 8.0, places=2)

    def test_overrun_may_shrink_below_widget_via_beat_est(self) -> None:
        brief = "[STORY_BEATS]\n1|30|1-9|arrive|LANDMARK\n[/STORY_BEATS]"
        schedule, plan = nodes._story_duration_schedule(
            brief, budget_seconds=45.0, max_segment_seconds=10.0,
            allow_duration_overrun=True,
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan.budget_seconds, 30.0)
        self.assertAlmostEqual(sum(schedule), 30.0, places=3)

    def test_single_beat_without_overrun_keeps_requested_budget(self) -> None:
        schedule, plan = nodes._story_duration_schedule(
            self.SINGLE_BEAT_BRIEF, budget_seconds=45.0, max_segment_seconds=10.0,
            allow_duration_overrun=False,
        )
        self.assertEqual(schedule, (9.0,) * 5)
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.beats), 1)
        self.assertEqual(plan.budget_seconds, 45.0)
        self.assertEqual(plan.card_counts, (5,))

    def test_two_beats_overrun_still_expands_to_estimate_sum(self) -> None:
        brief = "[STORY_BEATS]\n1|40|1-2|arrive|INTRO\n2|30|3-4|exit|EXIT\n[/STORY_BEATS]"
        self.assertEqual(len(nodes._parse_story_beats_from_brief(brief)), 2)
        schedule, plan = nodes._story_duration_schedule(
            brief, budget_seconds=45.0, max_segment_seconds=10.0,
            allow_duration_overrun=True,
        )
        self.assertEqual(schedule, (10.0,) * 7)
        self.assertIsNotNone(plan)
        self.assertFalse(plan.compressed)
        self.assertEqual(plan.budget_seconds, 70.0)
        self.assertEqual(plan.card_counts, (4, 3))
        self.assertIsNotNone(nodes._build_coverage_knife_plan(brief, budget_seconds=70.0, group_count=7))

    def test_huge_shot_range_does_not_inflate_or_trip_ceiling(self) -> None:
        """Shot 1-61 must not create a shot_floor; only huge beat est hits the ceiling."""
        brief = "[STORY_BEATS]\n1|45|1-61|arrive|LANDMARK\n[/STORY_BEATS]"
        schedule, plan = nodes._story_duration_schedule(
            brief, budget_seconds=45.0, max_segment_seconds=10.0,
            allow_duration_overrun=True,
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan.budget_seconds, 45.0)
        self.assertAlmostEqual(sum(schedule), 45.0, places=3)
        huge = "[STORY_BEATS]\n1|700|1-2|arrive|LANDMARK\n[/STORY_BEATS]"
        with self.assertRaisesRegex(nodes.PromptRefinerError, "safety ceiling"):
            nodes._story_duration_schedule(
                huge, budget_seconds=45.0, max_segment_seconds=10.0,
                allow_duration_overrun=True,
            )

    def _story_beats_brief(self) -> str:
        return """
[STORY_BEATS]
1|45|1-8|arrive|LANDMARK_B1_INTRO
2|20|9-12|gossip|LANDMARK_B2_DEPTH
3|30|13-22|snake|LANDMARK_B3_SNAKE
4|40|23-32|tease|LANDMARK_B4_TEASE
5|50|33-45|climax|LANDMARK_B5_CLIMAX
6|30|46-55|round2|LANDMARK_B6_SPARE
7|45|56-60|exit|LANDMARK_B7_PITY;LANDMARK_B7_PLACE
[/STORY_BEATS]
Story body uses placeholders only: （露骨动作）（露骨台词）.
""".strip()

    def _cards_for_plan(self, plan: nodes._CoverageKnifePlan, *, omit_beat6_dialogue: bool = False) -> str:
        blocks = ["TITLE: Coverage knife demo"]
        for seg_index, beat_idx in enumerate(plan.segment_to_beat, start=1):
            beat = plan.beats[beat_idx - 1]
            if omit_beat6_dialogue and beat.index == 6:
                dialogue = "NONE"
            else:
                # Put every landmark on the first card of the beat; later cards stay NONE.
                first_of_beat = seg_index == 1 or plan.segment_to_beat[seg_index - 2] != beat_idx
                if first_of_beat and beat.key_dialogue:
                    dialogue = " / ".join(beat.key_dialogue)
                else:
                    dialogue = "NONE"
            continuity = "false" if seg_index == 1 else "true"
            # Seed an over-long early duration to prove compression keeps trailing cards.
            duration = 10 if beat_idx <= 5 else 4
            blocks.append(
                f"""=== SEGMENT {seg_index} ===
DURATION_SECONDS: {duration}
CONTINUITY_FROM_PREV: {continuity}
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
Story beat {beat.index} card {seg_index}: {beat.label}. Placeholders only （露骨动作）.
DIALOGUE:
{dialogue}
HANDOFF:
Ending state for beat {beat.index} card {seg_index}.
=== END SEGMENT ==="""
            )
        return "\n".join(blocks)

    def test_proportional_compression_when_budget_below_est_sum(self) -> None:
        allocations, compressed = nodes._scale_beat_allocations(self.JOB18_EST, self.JOB18_BUDGET)
        self.assertTrue(compressed)
        self.assertAlmostEqual(sum(allocations), self.JOB18_BUDGET, places=3)
        # Each beat keeps a positive share; ratios match estimates.
        for est, alloc in zip(self.JOB18_EST, allocations):
            self.assertAlmostEqual(alloc / est, self.JOB18_BUDGET / sum(self.JOB18_EST), places=3)

    def test_seven_beats_get_at_least_one_card_including_trailing(self) -> None:
        plan = nodes._build_coverage_knife_plan(
            self._story_beats_brief(),
            budget_seconds=self.JOB18_BUDGET,
            group_count=self.JOB18_GROUPS,
        )
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(len(plan.beats), 7)
        self.assertTrue(plan.compressed)
        self.assertEqual(sum(plan.card_counts), self.JOB18_GROUPS)
        self.assertTrue(all(count >= 1 for count in plan.card_counts))
        # Trailing beats must not be zeroed when early/climax are heavy.
        self.assertGreaterEqual(plan.card_counts[5], 1)  # beat 6
        self.assertGreaterEqual(plan.card_counts[6], 1)  # beat 7
        self.assertIn(7, plan.segment_to_beat)
        self.assertEqual(plan.segment_to_beat[-1], 7)

    def test_structure_soft_misses_missing_landmark_dialogue(self) -> None:
        plan = nodes._build_coverage_knife_plan(
            self._story_beats_brief(),
            budget_seconds=self.JOB18_BUDGET,
            group_count=self.JOB18_GROUPS,
        )
        assert plan is not None
        bad = self._cards_for_plan(plan, omit_beat6_dialogue=True)
        _title, cards = nodes._parse_director_structure_output(bad)
        misses = nodes._validate_structure_coverage(cards, plan)
        self.assertTrue(any(item.startswith("beat6:") for item in misses))
        debug = nodes._coverage_knife_debug_summary(plan, landmark_misses=misses)
        self.assertIn("coverage_landmark_miss=", debug)
        self.assertIn("beat6:", debug)

    def test_landmark_in_beat_field_only_passes(self) -> None:
        plan = nodes._build_coverage_knife_plan(
            self._story_beats_brief(),
            budget_seconds=self.JOB18_BUDGET,
            group_count=self.JOB18_GROUPS,
        )
        assert plan is not None
        blocks = ["TITLE: Landmark in BEAT only"]
        for seg_index, beat_idx in enumerate(plan.segment_to_beat, start=1):
            beat = plan.beats[beat_idx - 1]
            first_of_beat = seg_index == 1 or plan.segment_to_beat[seg_index - 2] != beat_idx
            landmark_text = " / ".join(beat.key_dialogue) if first_of_beat and beat.key_dialogue else ""
            continuity = "false" if seg_index == 1 else "true"
            beat_body = f"Story beat {beat.index} card {seg_index}: {beat.label}."
            if landmark_text:
                beat_body = f"{beat_body} Includes {landmark_text}"
            blocks.append(
                f"""=== SEGMENT {seg_index} ===
DURATION_SECONDS: 10
CONTINUITY_FROM_PREV: {continuity}
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
{beat_body}
DIALOGUE:
NONE
HANDOFF:
Ending state for beat {beat.index} card {seg_index}.
=== END SEGMENT ==="""
            )
        _title, cards = nodes._parse_director_structure_output("\n".join(blocks))
        misses = nodes._validate_structure_coverage(cards, plan)
        self.assertEqual(misses, [])

    def test_placeholder_landmark_is_ignored(self) -> None:
        brief = """
[STORY_BEATS]
1|20|1-4|a|REAL_LANDMARK_A
2|20|5-8|b|（露骨台词）
[/STORY_BEATS]
""".strip()
        plan = nodes._build_coverage_knife_plan(brief, budget_seconds=40.0, group_count=4)
        assert plan is not None
        self.assertIn("REAL_LANDMARK_A", plan.beats[0].key_dialogue)
        self.assertEqual(plan.beats[1].key_dialogue, ())
        self.assertTrue(nodes._is_placeholder_landmark("（露骨台词）"))
        self.assertTrue(nodes._is_placeholder_landmark("(explicit dialogue)"))
        blocks = ["TITLE: Placeholder landmark"]
        for seg_index, beat_idx in enumerate(plan.segment_to_beat, start=1):
            beat = plan.beats[beat_idx - 1]
            first_of_beat = seg_index == 1 or plan.segment_to_beat[seg_index - 2] != beat_idx
            dialogue = "REAL_LANDMARK_A" if beat.index == 1 and first_of_beat else "NONE"
            continuity = "false" if seg_index == 1 else "true"
            blocks.append(
                f"""=== SEGMENT {seg_index} ===
DURATION_SECONDS: 10
CONTINUITY_FROM_PREV: {continuity}
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
Story beat {beat.index} card {seg_index}.
DIALOGUE:
{dialogue}
HANDOFF:
Handoff {seg_index}.
=== END SEGMENT ==="""
            )
        _title, cards = nodes._parse_director_structure_output("\n".join(blocks))
        misses = nodes._validate_structure_coverage(cards, plan)
        self.assertEqual(misses, [])

    def test_parenthetical_key_split_is_not_a_landmark(self) -> None:
        brief = """
[STORY_BEATS]
1|20|1-2|arrival|角色甲（画外;平静）;"人群笑声;请往前走
[/STORY_BEATS]
""".strip()
        beats = nodes._parse_story_beats_block(brief)
        self.assertEqual(beats[0].key_dialogue, ("请往前走",))
        self.assertEqual(
            nodes._split_landmark_list("角色甲（画外;平静）;请往前走"),
            ["角色甲（画外;平静）", "请往前走"],
        )
        for token in ("角色甲（画外", "平静）", '"人群笑声', "角色甲（画外;平静）"):
            self.assertTrue(nodes._is_non_dialogue_landmark(token))
        kept = nodes._parse_story_beats_block(
            "[STORY_BEATS]\n1|20|1-2|talk|你好世界;另一句完整台词\n[/STORY_BEATS]"
        )
        self.assertEqual(kept[0].key_dialogue, ("你好世界", "另一句完整台词"))
        cues_only = nodes._parse_story_beats_block(
            "[STORY_BEATS]\n1|20|1-2|arrival|角色甲（画外;平静）;\"人群笑声\n[/STORY_BEATS]"
        )
        self.assertEqual(cues_only[0].key_dialogue, ())

    def _repair_two_cards(self, key_dialogue: tuple[str, ...]):
        beat = nodes._StoryBeatSpec(
            index=1,
            estimated_seconds=20.0,
            shot_start=1,
            shot_end=2,
            label="arrival",
            body="",
            key_dialogue=key_dialogue,
        )
        plan = nodes._CoverageKnifePlan(
            beats=(beat,),
            budget_seconds=20.0,
            group_count=2,
            estimated_sum=20.0,
            compressed=False,
            allocations=(10.0, 10.0),
            card_counts=(2,),
            segment_to_beat=(1, 1),
        )
        structure = """
TITLE: Fragment landmarks
=== SEGMENT 1 ===
DURATION_SECONDS: 12
CONTINUITY_FROM_PREV: false
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
Alpha action. Beta action.
DIALOGUE:
请往前走。
HANDOFF:
Hold at the doorway.
=== END SEGMENT ===
=== SEGMENT 2 ===
DURATION_SECONDS: 8
CONTINUITY_FROM_PREV: true
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
Second card holds.
DIALOGUE:
NONE
HANDOFF:
Still at the doorway.
=== END SEGMENT ===
""".strip()
        _title, cards = nodes._parse_director_structure_output(structure)
        audit: dict = {"schedule": (10.0, 10.0), "outputs": [], "repairs": []}

        def complete(_messages, _max_tokens, _seed):
            return json.dumps({
                "parts": [
                    {"beat": "Alpha action.", "dialogue": "", "handoff": "Between actions."},
                    {"beat": "Beta action.", "dialogue": "请往前走。", "handoff": "Hold at the doorway."},
                ]
            })

        return nodes._repair_structure_duration_cards(
            cards, (10.0, 10.0), plan, complete, 1024, 0, audit,
            max_segment_seconds=10.0,
        )

    def test_repaired_structure_ignores_direction_fragments(self) -> None:
        fragments = ("角色甲（画外", "平静）", '"人群笑声')
        repaired, resolved, _plan = self._repair_two_cards(fragments)
        self.assertEqual(len(repaired), 3)
        self.assertAlmostEqual(sum(resolved), 20.0, places=3)
        misses = nodes._validate_structure_coverage(repaired, _plan)
        self.assertEqual(misses, [])

    def test_repaired_structure_still_requires_real_landmarks(self) -> None:
        with self.assertRaisesRegex(nodes.PromptRefinerError, "missing dialogue landmarks: beat1:MUST_KEEP_LINE_ZX9"):
            self._repair_two_cards(("MUST_KEEP_LINE_ZX9",))

    def test_moan_label_does_not_abort_repaired_structure(self) -> None:
        self.assertTrue(nodes._is_non_dialogue_landmark("赵凤红呻吟"))
        self.assertFalse(nodes._is_non_dialogue_landmark("你别呻吟"))
        self.assertFalse(nodes._is_non_dialogue_landmark("啊……小程……你真会玩"))
        brief = (
            "[STORY_BEATS]\n"
            "1|15|1-4|beat|啊……小程……你真会玩;赵凤红呻吟\n"
            "[/STORY_BEATS]\n"
        )
        beats = nodes._parse_story_beats_block(brief)
        self.assertIn("啊……小程……你真会玩", beats[0].key_dialogue)
        self.assertNotIn("赵凤红呻吟", beats[0].key_dialogue)
        repaired, resolved, plan = self._repair_two_cards(("赵凤红呻吟",))
        self.assertEqual(len(repaired), 3)
        self.assertAlmostEqual(sum(resolved), 20.0, places=3)
        self.assertFalse(any("赵凤红呻吟" in item for item in nodes._validate_structure_coverage(repaired, plan)))
        spoken = " ".join(str(card.get("dialogue") or "") for card in repaired)
        self.assertIn("请往前走", spoken)

    def test_structure_allocates_cards_and_keeps_trailing_duration_floor(self) -> None:
        plan = nodes._build_coverage_knife_plan(
            self._story_beats_brief(),
            budget_seconds=self.JOB18_BUDGET,
            group_count=self.JOB18_GROUPS,
        )
        assert plan is not None
        good = self._cards_for_plan(plan, omit_beat6_dialogue=False)
        _title, cards = nodes._parse_director_structure_output(good)
        cards[0]["dialogue"] += " " + _words(40)
        schedule = nodes._director_duration_schedule(self.JOB18_BUDGET, 10.0)
        self.assertEqual(len(schedule), self.JOB18_GROUPS)
        segments, rebalanced, misses = nodes._segments_from_structure_cards(
            cards,
            R2V,
            schedule,
            (),
            (),
            (),
            (),
            coverage_plan=plan,
            locked_durations=False,
        )
        self.assertEqual(misses, [])
        self.assertTrue(rebalanced)
        self.assertEqual(len(segments), self.JOB18_GROUPS)
        self.assertGreater(segments[0].duration_seconds, segments[1].duration_seconds)
        self.assertAlmostEqual(sum(seg.duration_seconds for seg in segments), self.JOB18_BUDGET, places=2)
        # Every host-mapped beat range still has ≥1 card after packing.
        for beat_idx, count in enumerate(plan.card_counts, start=1):
            owned = [seg for seg, mapped in zip(segments, plan.segment_to_beat) if mapped == beat_idx]
            self.assertEqual(len(owned), count)
            self.assertGreaterEqual(len(owned), 1)
        # Trailing beats keep a meaningful floor (not wiped by early 15s cards).
        for beat_idx in (6, 7):
            total = sum(
                seg.duration_seconds
                for seg, mapped in zip(segments, plan.segment_to_beat)
                if mapped == beat_idx
            )
            self.assertGreaterEqual(total, plan.card_counts[beat_idx - 1] * nodes.MIN_DIRECTOR_SEGMENT_SECONDS - 0.05)
            self.assertGreaterEqual(total, min(plan.allocations[beat_idx - 1], 12.0) - 1.5)


    def test_structure_prompt_does_not_lock_equal_host_seconds(self) -> None:
        """STORY_BEATS overrun seeds from beat est, but the structure prompt must not force copy."""
        schedule, plan = nodes._story_duration_schedule(
            self.SINGLE_BEAT_BRIEF, budget_seconds=45.0, max_segment_seconds=10.0,
            allow_duration_overrun=True,
        )
        self.assertEqual(len(schedule), 5)
        self.assertAlmostEqual(sum(schedule), 45.0, places=3)
        assert plan is not None
        system = nodes._long_director_structure_system(
            R2V, schedule, coverage_plan=plan, max_segment_seconds=10.0
        )
        self.assertNotIn("LOCKED DURATION SCHEDULE", system)
        self.assertNotIn("Copy the corresponding host duration exactly", system)
        template = _template_block(system)
        self.assertIn("DURATION_SECONDS: <4-10>", template)
        self.assertNotRegex(template, r"DURATION_SECONDS:\s*\d")
        self.assertNotIn("10.000", template)
        self.assertIn("DURATION_SECONDS: <4-10>", system)
        self.assertIn("only the average used to estimate that group count", system)
        self.assertIn("make their sum exactly", system)
        self.assertIn("near 10", system)
        self.assertIn("near 4", system)
        self.assertIn("45.000 seconds", system)

    def test_repair_keeps_variable_llm_durations_against_equal_host_seed(self) -> None:
        """Unlocked repair keeps variable card times under the widget segment ceiling."""
        schedule, plan = nodes._story_duration_schedule(
            self.SINGLE_BEAT_BRIEF, budget_seconds=45.0, max_segment_seconds=10.0,
            allow_duration_overrun=True,
        )
        assert plan is not None
        self.assertEqual(len(schedule), 5)
        llm_durations = (10.0, 8.0, 10.0, 9.0, 8.0)
        self.assertAlmostEqual(sum(llm_durations), 45.0, places=3)
        blocks = ["TITLE: Variable LLM rhythm"]
        for index, duration in enumerate(llm_durations, start=1):
            continuity = "false" if index == 1 else "true"
            dialogue = "LANDMARK" if index == 1 else "NONE"
            blocks.append(
                f"""=== SEGMENT {index} ===
DURATION_SECONDS: {duration:g}
CONTINUITY_FROM_PREV: {continuity}
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
Story card {index}.
DIALOGUE:
{dialogue}
HANDOFF:
Handoff {index}.
=== END SEGMENT ==="""
            )
        _title, cards = nodes._parse_director_structure_output("\n".join(blocks))
        audit: dict = {"schedule": schedule, "outputs": [], "repairs": []}

        def _complete(*_args, **_kwargs):
            raise AssertionError("repair must not call complete when cards already fit")

        repaired, resolved, out_plan = nodes._repair_structure_duration_cards(
            cards, schedule, plan, _complete, 1024, 0, audit,
        )
        self.assertEqual(out_plan, plan)
        self.assertEqual(audit["repairs"], [])
        self.assertEqual(list(resolved), list(llm_durations))
        self.assertEqual(
            [float(card["duration_raw"]) for card in repaired],
            list(llm_durations),
        )
        self.assertTrue(all(4.0 <= value <= 10.0 for value in resolved))

    def test_repair_accepts_extra_natural_parts_for_overflow(self) -> None:
        """18s overflow with max=15 may return 3 natural parts; accept and rebalance."""
        schedule = (18.0,)
        # Nonempty DIALOGUE so empty-dialogue clamp does not apply; split-repair runs.
        structure = """TITLE: Overflow three beats
=== SEGMENT 1 ===
DURATION_SECONDS: 18
CONTINUITY_FROM_PREV: false
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
First beat action. Second beat action. Third beat action.
DIALOGUE:
LINE_A hello. LINE_B there. LINE_C friend.
HANDOFF:
Final handoff state.
=== END SEGMENT ==="""
        _title, cards = nodes._parse_director_structure_output(structure)
        self.assertEqual(len(cards), 1)
        audit: dict = {"schedule": schedule, "outputs": [], "repairs": []}
        expected_prompt_count = len(nodes._director_duration_schedule(18.0, 15.0))
        self.assertEqual(expected_prompt_count, 2)

        def _complete(messages, _max_tokens, _seed):
            user = messages[-1]["content"]
            # Three natural parts with verbatim dialogue slices.
            payload = {
                "parts": [
                    {
                        "beat": "First beat action.",
                        "dialogue": "LINE_A hello.",
                        "handoff": "At first beat end.",
                    },
                    {
                        "beat": "Second beat action.",
                        "dialogue": "LINE_B there.",
                        "handoff": "At second beat end.",
                    },
                    {
                        "beat": "Third beat action.",
                        "dialogue": "LINE_C friend.",
                        "handoff": "Final handoff state.",
                    },
                ]
            }
            # Confirm the repair prompt asked for the schedule part count.
            joined = " ".join(
                str(m.get("content") or "") for m in messages if m.get("role") == "system"
            )
            self.assertIn(f"exactly {expected_prompt_count} parts", joined)
            self.assertIn("exactly 2 parts", user)
            return json.dumps(payload)

        repaired, resolved, out_plan = nodes._repair_structure_duration_cards(
            cards, schedule, None, _complete, 1024, 0, audit,
            max_segment_seconds=15.0,
        )
        self.assertIsNone(out_plan)
        self.assertEqual(len(repaired), 3)
        self.assertEqual(len(resolved), 3)
        self.assertAlmostEqual(sum(resolved), 18.0, places=3)
        self.assertTrue(all(4.0 <= value <= 15.0 for value in resolved))
        self.assertEqual(audit["repairs"][0].get("accepted_part_count"), 3)
        self.assertAlmostEqual(sum(audit["repairs"][0]["rebuilt_durations"]), 18.0, places=3)

    def test_repair_exact_schedule_part_count_still_works(self) -> None:
        """Matching the schedule part count keeps the original split_times."""
        schedule = (18.0,)
        structure = """TITLE: Overflow two beats
=== SEGMENT 1 ===
DURATION_SECONDS: 18
CONTINUITY_FROM_PREV: false
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
First beat action. Second beat action.
DIALOGUE:
LINE_A hello. LINE_B there.
HANDOFF:
Final handoff state.
=== END SEGMENT ==="""
        _title, cards = nodes._parse_director_structure_output(structure)
        audit: dict = {"schedule": schedule, "outputs": [], "repairs": []}
        split = nodes._director_duration_schedule(18.0, 15.0)
        self.assertEqual(len(split), 2)

        def _complete(messages, _max_tokens, _seed):
            return json.dumps({
                "parts": [
                    {
                        "beat": "First beat action.",
                        "dialogue": "LINE_A hello.",
                        "handoff": "Mid handoff.",
                    },
                    {
                        "beat": "Second beat action.",
                        "dialogue": "LINE_B there.",
                        "handoff": "Final handoff state.",
                    },
                ]
            })

        repaired, resolved, _out_plan = nodes._repair_structure_duration_cards(
            cards, schedule, None, _complete, 1024, 0, audit,
            max_segment_seconds=15.0,
        )
        self.assertEqual(len(repaired), 2)
        self.assertEqual(list(resolved), list(split))
        self.assertNotIn("accepted_part_count", audit["repairs"][0])

    def test_repair_clamps_empty_everything_duration_over_max(self) -> None:
        """18s card with empty DIALOGUE and no BEAT quotes clamps; no LLM repair."""
        schedule = (15.0,)
        structure = """TITLE: Clamp empty everything
=== SEGMENT 1 ===
DURATION_SECONDS: 18
CONTINUITY_FROM_PREV: false
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
Wide shot of the square. A visitor approaches the guide near the gate, then they walk on.
DIALOGUE:
NONE
HANDOFF:
Both figures continue down the path.
=== END SEGMENT ==="""
        _title, cards = nodes._parse_director_structure_output(structure)
        self.assertEqual(len(cards), 1)
        self.assertEqual(nodes._clean_structure_dialogue(cards[0].get("dialogue") or ""), "")
        self.assertEqual(nodes._beat_quoted_speech_text(cards[0]["beat"]), "")
        self.assertEqual(nodes._structure_card_speech_seconds("", cards[0]["beat"]), 0.0)
        audit: dict = {"schedule": schedule, "outputs": [], "repairs": []}
        calls: list = []

        def _complete(messages, _max_tokens, _seed):
            calls.append(messages)
            raise AssertionError("empty-everything clamp must not call complete")

        repaired, resolved, out_plan = nodes._repair_structure_duration_cards(
            cards, schedule, None, _complete, 1024, 0, audit,
            max_segment_seconds=15.0,
        )
        self.assertIsNone(out_plan)
        self.assertEqual(calls, [])
        self.assertEqual(audit["repairs"], [])
        self.assertEqual(len(repaired), 1)
        self.assertEqual(float(repaired[0]["duration_raw"]), 15.0)
        self.assertIn("visitor approaches", repaired[0]["beat"])
        self.assertEqual(nodes._clean_structure_dialogue(repaired[0].get("dialogue") or ""), "")
        self.assertEqual(list(resolved), [15.0])

    def test_repair_skips_dialogue_verbatim_when_original_empty(self) -> None:
        """Empty original DIALOGUE skips verbatim check when split-repair still runs."""
        schedule = (18.0,)
        # BEAT-embedded quotes make speech countable → split (not clamp); field stays empty.
        structure = """TITLE: Empty dialogue skip path
=== SEGMENT 1 ===
DURATION_SECONDS: 18
CONTINUITY_FROM_PREV: false
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
First action beat. Hero says "Please show me the quiet path to the gate today." Second action beat. Guide answers "Follow me along this road and stay close now."
DIALOGUE:
NONE
HANDOFF:
Final handoff state.
=== END SEGMENT ==="""
        _title, cards = nodes._parse_director_structure_output(structure)
        self.assertEqual(nodes._clean_structure_dialogue(cards[0].get("dialogue") or ""), "")
        self.assertGreater(nodes._structure_card_speech_seconds("", cards[0]["beat"]), 0.0)
        audit: dict = {"schedule": schedule, "outputs": [], "repairs": []}

        def _complete(messages, _max_tokens, _seed):
            return json.dumps({
                "parts": [
                    {
                        "beat": 'First action beat. Hero says "Please show me the quiet path to the gate today."',
                        "dialogue": "invented from beat quotes",
                        "handoff": "Mid state.",
                    },
                    {
                        "beat": 'Second action beat. Guide answers "Follow me along this road and stay close now."',
                        "dialogue": "",
                        "handoff": "Final handoff state.",
                    },
                ]
            })

        repaired, resolved, _out = nodes._repair_structure_duration_cards(
            cards, schedule, None, _complete, 1024, 0, audit,
            max_segment_seconds=15.0,
        )
        self.assertEqual(len(audit["repairs"]), 1)
        self.assertNotIn("changed or skipped dialogue", audit["repairs"][0].get("error", ""))
        self.assertEqual(len(repaired), 2)
        self.assertAlmostEqual(sum(resolved), 18.0, places=3)
        # Empty-origin: host replaces model lifts with BEAT quotes in DIALOGUE.
        joined = "\n".join(
            nodes._clean_structure_dialogue(card.get("dialogue") or "") for card in repaired
        )
        self.assertIn("Please show me the quiet path to the gate today.", joined)
        self.assertIn("Follow me along this road and stay close now.", joined)
        self.assertNotEqual(joined.strip(), "")
        self.assertNotIn("invented from beat quotes", joined)

    def test_repair_splits_beat_embedded_quotes_with_empty_dialogue(self) -> None:
        """Empty DIALOGUE but quoted speech in BEAT must overflow-split, not clamp."""
        schedule = (18.0,)
        structure = """TITLE: Beat quotes split
=== SEGMENT 1 ===
DURATION_SECONDS: 18
CONTINUITY_FROM_PREV: false
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
Open on the square. Hero says "Please show me the quiet path to the village gate today now." Guide answers "Follow me along this road and stay close beside me please." They walk on together.
DIALOGUE:
NONE
HANDOFF:
Both continue down the path.
=== END SEGMENT ==="""
        _title, cards = nodes._parse_director_structure_output(structure)
        self.assertEqual(nodes._clean_structure_dialogue(cards[0].get("dialogue") or ""), "")
        beat_speech = nodes._structure_card_speech_seconds("", cards[0]["beat"])
        self.assertGreater(beat_speech, 0.0)
        audit: dict = {"schedule": schedule, "outputs": [], "repairs": []}
        calls: list = []

        def _complete(messages, _max_tokens, _seed):
            calls.append(messages)
            beat = cards[0]["beat"]
            # Split beat into two contiguous verbatim slices at a natural boundary.
            mid = beat.index("Guide answers")
            return json.dumps({
                "parts": [
                    {
                        "beat": beat[:mid].rstrip(),
                        "dialogue": "",
                        "handoff": "Hero has just spoken.",
                    },
                    {
                        "beat": beat[mid:].lstrip(),
                        "dialogue": "",
                        "handoff": "Both continue down the path.",
                    },
                ]
            })

        repaired, resolved, out_plan = nodes._repair_structure_duration_cards(
            cards, schedule, None, _complete, 1024, 0, audit,
            max_segment_seconds=15.0,
        )
        self.assertIsNone(out_plan)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(audit["repairs"]), 1)
        self.assertEqual(len(repaired), 2)
        self.assertAlmostEqual(sum(resolved), 18.0, places=3)
        self.assertTrue(all(4.0 <= value <= 15.0 for value in resolved))
        # Spoken lines lift into DIALOGUE; BEAT keeps narrative without quote duplication.
        joined_dialogue = "\n".join(
            nodes._clean_structure_dialogue(card.get("dialogue") or "") for card in repaired
        )
        self.assertIn("Please show me", joined_dialogue)
        self.assertIn("Follow me along", joined_dialogue)
        self.assertTrue(any(nodes._clean_structure_dialogue(c.get("dialogue") or "") for c in repaired))
        joined_beat = " ".join(card["beat"] for card in repaired)
        self.assertIn("Open on the square", joined_beat)
        self.assertIn("They walk on together", joined_beat)
        # Quotes stripped from BEAT so refine does not double-speak.
        self.assertNotIn("Please show me", joined_beat)
        self.assertNotIn("Follow me along", joined_beat)

    def test_repair_preserves_d_tag_dialogue_and_ignores_speaker_label_landmark(self) -> None:
        """Regression: BEAT <d> lines survive a 17.111s split; 女声 is not a quote."""
        schedule = (10.0, 10.0)
        structure = """TITLE: Tagged dialogue repair
=== SEGMENT 1 ===
DURATION_SECONDS: 10
CONTINUITY_FROM_PREV: false
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
The woman approaches the doorway and pauses.
DIALOGUE:
NONE
HANDOFF:
She looks into the room.
=== END SEGMENT ===
=== SEGMENT 2 ===
DURATION_SECONDS: 17.111
CONTINUITY_FROM_PREV: true
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
She enters and says <d>[Chinese] 你终于来了，我一直在这里等你。</d> 随后他转身回答 <d>[Chinese] 路上耽搁了一会儿，我们现在开始吧。</d>
DIALOGUE:
NONE
HANDOFF:
They face each other beside the table.
=== END SEGMENT ==="""
        _title, cards = nodes._parse_director_structure_output(structure)
        self.assertEqual(
            nodes._beat_quoted_speech_units(cards[1]["beat"]),
            ["你终于来了，我一直在这里等你。", "路上耽搁了一会儿，我们现在开始吧。"],
        )
        story_beat = nodes._story_beat_from_fields(
            1, 27.111, None, None, "dialogue scene", structure, ["女声"]
        )
        self.assertEqual(story_beat.key_dialogue, ())
        plan = nodes._CoverageKnifePlan(
            beats=(story_beat,), budget_seconds=20.0, group_count=2,
            estimated_sum=27.111, compressed=True, allocations=(20.0,),
            card_counts=(2,), segment_to_beat=(1, 1),
        )
        audit: dict = {"schedule": schedule, "outputs": [], "repairs": []}

        def _complete(messages, _max_tokens, _seed):
            payload = json.loads(messages[-1]["content"])
            beat = payload["beat"]
            cut = beat.index("随后")
            return json.dumps({"parts": [
                {"beat": beat[:cut].rstrip(), "dialogue": "", "handoff": "The first line ends."},
                {"beat": beat[cut:].lstrip(), "dialogue": "", "handoff": payload["handoff"]},
            ]}, ensure_ascii=False)

        repaired, resolved, _out_plan = nodes._repair_structure_duration_cards(
            cards, schedule, plan, _complete, 1024, 0, audit,
            max_segment_seconds=15.0, allow_duration_overrun=True,
        )
        dialogue = "\n".join(card.get("dialogue") or "" for card in repaired)
        beat = "\n".join(card.get("beat") or "" for card in repaired)
        self.assertIn("你终于来了，我一直在这里等你。", dialogue)
        self.assertIn("路上耽搁了一会儿，我们现在开始吧。", dialogue)
        self.assertNotIn("<d>", beat)
        self.assertAlmostEqual(sum(resolved), audit["expanded_total_seconds"], places=3)
        self.assertEqual(audit["overflow_budget_expand_detail"]["overflow_segments"], [2])

    def test_repair_expands_soft_target_when_speech_overflows_exceed_schedule(self) -> None:
        """Overrun ON: speech overflows beyond 45s expand soft target and split without truncating."""
        schedule = (15.0, 15.0, 15.0)
        # 63 Latin words / 2.25 wps = 28.000s; 50 words = 22.222s (mirrors 27.8+22 diagnostic).
        heavy_a = " ".join(f"alpha{i:02d}" for i in range(62))
        heavy_b = " ".join(f"bravo{i:02d}" for i in range(50))
        self.assertAlmostEqual(
            nodes._dialogue_speech_units(heavy_a) / nodes._DIALOGUE_WORDS_PER_SECOND,
            62 / nodes._DIALOGUE_WORDS_PER_SECOND,
            places=3,
        )
        self.assertAlmostEqual(
            nodes._dialogue_speech_units(heavy_b) / nodes._DIALOGUE_WORDS_PER_SECOND,
            50 / nodes._DIALOGUE_WORDS_PER_SECOND,
            places=3,
        )
        structure = f"""TITLE: Soft overrun overflow budget
=== SEGMENT 1 ===
DURATION_SECONDS: 10
CONTINUITY_FROM_PREV: false
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
Wide establishing shot of the quiet square at dusk.
DIALOGUE:
NONE
HANDOFF:
Camera settles on the approaching visitor.
=== END SEGMENT ===
=== SEGMENT 2 ===
DURATION_SECONDS: 15
CONTINUITY_FROM_PREV: true
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
Visitor greets the guides near the gate and offers a friendly introduction.
DIALOGUE:
{heavy_a}
HANDOFF:
Visitor stands facing the guides with luggage at their feet.
=== END SEGMENT ===
=== SEGMENT 3 ===
DURATION_SECONDS: 15
CONTINUITY_FROM_PREV: true
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
A guide welcomes the visitor and walks them toward the village path.
DIALOGUE:
{heavy_b}
HANDOFF:
Both walk away down the path as the others watch.
=== END SEGMENT ==="""
        _title, cards = nodes._parse_director_structure_output(structure)
        self.assertEqual(len(cards), 3)
        speech_a = nodes._structure_card_speech_seconds(cards[1].get("dialogue") or "", cards[1]["beat"])
        speech_b = nodes._structure_card_speech_seconds(cards[2].get("dialogue") or "", cards[2]["beat"])
        self.assertAlmostEqual(speech_a, 62 / nodes._DIALOGUE_WORDS_PER_SECOND, places=3)
        self.assertAlmostEqual(speech_b, 50 / nodes._DIALOGUE_WORDS_PER_SECOND, places=3)
        overflow_sum = speech_a + speech_b
        self.assertGreater(overflow_sum, sum(schedule))
        audit: dict = {"schedule": schedule, "outputs": [], "repairs": []}

        def _complete(messages, _max_tokens, _seed):
            user = json.loads(messages[-1]["content"])
            beat = str(user["beat"])
            dialogue = str(user["dialogue"])
            durations = list(user["durations"])
            part_count = int(user["part_count"])
            self.assertEqual(part_count, len(durations))
            if part_count != 2:
                raise AssertionError(f"fixture expects 2-way split, got {part_count}")
            # Split on whole words so each half's speech fits the assigned duration.
            words = dialogue.split()
            mid_words = len(words) // 2
            dialogue_a = " ".join(words[:mid_words])
            dialogue_b = " ".join(words[mid_words:])
            mid_beat = max(1, len(beat) // 2)
            beat_cut = beat.rfind(" ", 0, mid_beat)
            if beat_cut <= 0:
                beat_cut = mid_beat
            return json.dumps({
                "parts": [
                    {
                        "beat": beat[:beat_cut].rstrip() or beat[:mid_beat],
                        "dialogue": dialogue_a,
                        "handoff": "Mid handoff state.",
                    },
                    {
                        "beat": beat[beat_cut:].lstrip() or beat[mid_beat:],
                        "dialogue": dialogue_b,
                        "handoff": str(user.get("handoff") or "Final handoff."),
                    },
                ]
            })

        repaired, resolved, out_plan = nodes._repair_structure_duration_cards(
            cards, schedule, None, _complete, 1024, 0, audit,
            max_segment_seconds=15.0,
            allow_duration_overrun=True,
        )
        self.assertIsNone(out_plan)
        self.assertTrue(audit.get("overflow_budget_expand"))
        preferred = overflow_sum + 15.0  # one ordinary card at effective max
        self.assertAlmostEqual(audit["expanded_total_seconds"], preferred, places=3)
        self.assertAlmostEqual(sum(resolved), preferred, places=3)
        self.assertGreater(sum(resolved), sum(schedule))
        self.assertGreaterEqual(len(repaired), 4)  # ordinary + split overflows
        self.assertTrue(all(4.0 <= value <= 15.0 for value in resolved))
        joined = " ".join(
            nodes._clean_structure_dialogue(card.get("dialogue") or "")
            for card in repaired
        )
        for token in heavy_a.split():
            self.assertIn(token, joined)
        for token in heavy_b.split():
            self.assertIn(token, joined)
        self.assertEqual(len(audit["repairs"]), 2)


    def test_repair_beat_coverage_split_keeps_embedded_quotes_when_lift_overruns(self) -> None:
        """27.778s-style empty DIALOGUE: BEAT cut across parts; quotes lifted to DIALOGUE.

        Mirrors diagnostic seg2: speech lives in BEAT quotes (~28s), one lifted block
        alone exceeds a half-slot. Coverage check must accept the BEAT split (not
        whole-card equality). Host partitions all quotes into part DIALOGUE — no silent drop.
        """
        # Short tokens keep each quoted span <=120 chars (beat-quote scanner limit).
        # 25 + 19 + 19 = 63 Latin words @ 2.25 wps = 28.000s (over 15 -> split).
        line_a = " ".join(f"a{i:02d}" for i in range(25))
        line_b1 = " ".join(f"b{i:02d}" for i in range(19))
        line_b2 = " ".join(f"c{i:02d}" for i in range(19))
        line_b = f"{line_b1} {line_b2}"
        self.assertLessEqual(len(line_a), 120)
        self.assertLessEqual(len(line_b1), 120)
        self.assertLessEqual(len(line_b2), 120)
        beat = (
            f'Visitor greets the guides. Hero says "{line_a}" '
            f'Then the guide steps forward. Guide answers "{line_b1}" '
            f'Guide continues "{line_b2}" They hold still under the tree.'
        )
        schedule = (15.0,)
        structure = f"""TITLE: Beat coverage empty dialogue
=== SEGMENT 1 ===
DURATION_SECONDS: 15
CONTINUITY_FROM_PREV: false
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
{beat}
DIALOGUE:
NONE
HANDOFF:
Visitor stands facing the guides.
=== END SEGMENT ==="""
        _title, cards = nodes._parse_director_structure_output(structure)
        self.assertEqual(nodes._clean_structure_dialogue(cards[0].get("dialogue") or ""), "")
        speech = nodes._structure_card_speech_seconds("", cards[0]["beat"])
        self.assertAlmostEqual(speech, 63 / nodes._DIALOGUE_WORDS_PER_SECOND, places=3)
        self.assertGreater(speech, 15.0)
        audit: dict = {"schedule": schedule, "outputs": [], "repairs": []}
        source_beat = cards[0]["beat"]
        mid = source_beat.index("Then the guide")

        def _complete(messages, _max_tokens, _seed):
            # Attempt-1 diagnostic shape: contiguous BEAT slices keep quotes;
            # dialogue lift puts the long block alone in part 2 (over half-slot).
            return json.dumps({
                "parts": [
                    {
                        "beat": source_beat[:mid].rstrip(),
                        "dialogue": line_a,
                        "handoff": "Hero has just spoken.",
                    },
                    {
                        "beat": source_beat[mid:].lstrip(),
                        "dialogue": line_b,
                        "handoff": "Visitor stands facing the guides.",
                    },
                ]
            })

        repaired, resolved, _out = nodes._repair_structure_duration_cards(
            cards, schedule, None, _complete, 1024, 0, audit,
            max_segment_seconds=15.0,
            allow_duration_overrun=True,
        )
        self.assertEqual(len(audit["repairs"]), 1)
        self.assertIsNone(audit["repairs"][0].get("error"))
        self.assertEqual(len(repaired), 2)
        self.assertAlmostEqual(sum(resolved), speech, places=2)
        self.assertTrue(all(4.0 <= value <= 15.0 for value in resolved))
        # All countable quote words land in DIALOGUE across parts — no silent drop.
        joined_dialogue = " ".join(
            nodes._clean_structure_dialogue(card.get("dialogue") or "") for card in repaired
        )
        for token in line_a.split():
            self.assertIn(token, joined_dialogue)
        for token in line_b1.split():
            self.assertIn(token, joined_dialogue)
        for token in line_b2.split():
            self.assertIn(token, joined_dialogue)
        self.assertTrue(all(
            nodes._dialogue_speech_units(nodes._clean_structure_dialogue(c.get("dialogue") or ""))
            / nodes._DIALOGUE_WORDS_PER_SECOND
            <= value + 0.02
            for c, value in zip(repaired, resolved)
        ))
        # BEAT narrative coverage preserved; spoken quote bodies stripped from BEAT.
        joined_beat = " ".join(card["beat"] for card in repaired)
        self.assertIn("Visitor greets the guides", joined_beat)
        self.assertIn("They hold still under the tree", joined_beat)
        self.assertNotIn("a00", joined_beat)
        self.assertNotIn("b00", joined_beat)
        self.assertNotIn("c00", joined_beat)

    def test_repair_empty_dialogue_beat_quotes_fill_part_dialogues(self) -> None:
        """Empty DIALOGUE + BEAT quotes: after overflow repair every quote reaches DIALOGUE."""
        schedule = (18.0,)
        line_a = "Please show me the quiet path to the village gate today now."
        line_b = "Follow me along this road and stay close beside me please."
        beat = (
            f'Open on the square. Hero says "{line_a}" '
            f'Guide answers "{line_b}" They walk on together.'
        )
        structure = f"""TITLE: Empty dialogue quote lift
=== SEGMENT 1 ===
DURATION_SECONDS: 18
CONTINUITY_FROM_PREV: false
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
{beat}
DIALOGUE:
NONE
HANDOFF:
Both continue down the path.
=== END SEGMENT ==="""
        _title, cards = nodes._parse_director_structure_output(structure)
        self.assertEqual(nodes._clean_structure_dialogue(cards[0].get("dialogue") or ""), "")
        units = nodes._beat_quoted_speech_units(cards[0]["beat"])
        self.assertEqual(units, [line_a, line_b])
        audit: dict = {"schedule": schedule, "outputs": [], "repairs": []}
        source_beat = cards[0]["beat"]
        mid = source_beat.index("Guide answers")

        def _complete(messages, _max_tokens, _seed):
            return json.dumps({
                "parts": [
                    {
                        "beat": source_beat[:mid].rstrip(),
                        "dialogue": "",
                        "handoff": "Hero has just spoken.",
                    },
                    {
                        "beat": source_beat[mid:].lstrip(),
                        "dialogue": "",
                        "handoff": "Both continue down the path.",
                    },
                ]
            })

        repaired, resolved, _out = nodes._repair_structure_duration_cards(
            cards, schedule, None, _complete, 1024, 0, audit,
            max_segment_seconds=15.0,
        )
        self.assertEqual(len(repaired), 2)
        self.assertTrue(all(4.0 <= value <= 15.0 for value in resolved))
        dialogues = [
            nodes._clean_structure_dialogue(card.get("dialogue") or "") for card in repaired
        ]
        self.assertTrue(all(dialogues))
        joined = "\n".join(dialogues)
        self.assertIn(line_a, joined)
        self.assertIn(line_b, joined)
        # Narrative BEAT coverage across parts; quote bodies not left only-in-BEAT.
        joined_beat = " ".join(card["beat"] for card in repaired)
        self.assertIn("Open on the square", joined_beat)
        self.assertIn("They walk on together", joined_beat)
        self.assertNotIn(line_a, joined_beat)
        self.assertNotIn(line_b, joined_beat)


    def test_partition_long_cjk_quote_is_contiguous_prefix_remainder(self) -> None:
        """One long BEAT quote split across parts must be prefix|suffix with no restart.

        Regression: adjacent Director groups duplicated the opening phrase when a
        single quote exceeded one duration slot — part B restarted the whole line
        instead of continuing only the leftover remainder.
        """
        quote = (
            "哎呦，姐姐最喜欢这嘴甜的弟弟了。既然你都这么说了，姐姐我自然要亲自领你去了。"
            "这条路虽远，但风景正好，你跟紧姐姐，莫要掉队。"
        )
        # Two unequal slots so part1 takes a mid-utterance prefix of the one unit.
        assigned = nodes._partition_quote_units_by_duration(
            [quote], [3.0, 6.0], ceiling=6.0,
        )
        self.assertEqual(len(assigned), 2)
        part_a, part_b = assigned
        self.assertTrue(part_a.strip())
        self.assertTrue(part_b.strip())
        norm_a = part_a.replace(" ", "")
        norm_b = part_b.replace(" ", "")
        norm_q = quote.replace(" ", "")
        # Contiguous coverage of the original line.
        self.assertEqual(norm_a + norm_b, norm_q)
        # No restart: part B must not begin with part A's spoken prefix.
        self.assertFalse(norm_b.startswith(norm_a))
        # Opening phrase lands only in part A when the cut falls after it.
        opening = "哎呦，姐姐最喜欢这嘴甜的弟弟"
        if opening in norm_a:
            self.assertNotIn(opening, norm_b)

    
    def test_split_speech_prefers_chinese_punct_at_or_before_budget(self) -> None:
        """Mid-unit cut lands on 。！？；， nearest the budget when marks exist."""
        quote = (
            "哎呦，姐姐最喜欢这嘴甜的弟弟了。既然你都这么说了，姐姐我自然要亲自领你去了。"
            "这条路虽远，但风景正好，你跟紧姐姐，莫要掉队。"
        )
        # Budget just past the first period: cut after 。 not mid-phrase before 了。
        head, tail = nodes._split_speech_text_at_seconds(quote, 3.2)
        self.assertTrue(head.endswith("。"))
        self.assertEqual(head + tail, quote)
        self.assertFalse(tail.startswith(head))
        self.assertTrue(tail.startswith("既然"))

        # Budget near a clause comma: prefer ， nearest budget over a coarser token cut.
        head2, tail2 = nodes._split_speech_text_at_seconds(quote, 5.0)
        self.assertTrue(head2.endswith("，") or head2.endswith("。"))
        self.assertEqual(head2 + tail2, quote)
        self.assertFalse(tail2.replace(" ", "").startswith(head2.replace(" ", "")))
        # Nearest preferred mark at/before 5.0s is the clause comma after 说了.
        self.assertTrue(head2.endswith("说了，"))

    def test_partition_long_cjk_quote_prefers_punct_boundaries(self) -> None:
        """Duration partition mid-splits end on punct when the slot budget allows."""
        quote = (
            "哎呦，姐姐最喜欢这嘴甜的弟弟了。既然你都这么说了，姐姐我自然要亲自领你去了。"
            "这条路虽远，但风景正好，你跟紧姐姐，莫要掉队。"
        )
        assigned = nodes._partition_quote_units_by_duration(
            [quote], [3.2, 8.0], ceiling=8.0,
        )
        self.assertEqual(len(assigned), 2)
        part_a, part_b = assigned
        norm_a = part_a.replace(" ", "")
        norm_b = part_b.replace(" ", "")
        self.assertEqual(norm_a + norm_b, quote.replace(" ", ""))
        self.assertFalse(norm_b.startswith(norm_a))
        self.assertTrue(part_a.endswith("。"))
        self.assertTrue(part_b.startswith("既然"))
        # Final part keeps the leftover speech untruncated.
        self.assertTrue(part_b.endswith("队。"))


    def test_force_assigned_dialogue_replaces_brief_restart_in_refine_prompt(self) -> None:
        """Refine must not keep a full-line <d> when structure assigned only a slice."""
        assigned = "既然你都这么说了，姐姐我自然要亲自领你去了。"
        prompt = (
            "detailed_description:\\n"
            "[Shot 1] She smiles and says: "
            "<d>[Chinese] 哎呦，姐姐最喜欢这嘴甜的弟弟了。既然你都这么说了，姐姐我自然要亲自领你去了。</d>\\n\\n"
            "overall_soundscape:\\n"
            "Courtyard tone.\\n"
        )
        out = nodes._force_assigned_dialogue_in_prompt(prompt, assigned)
        self.assertIn(assigned, out)
        self.assertNotIn("哎呦，姐姐最喜欢这嘴甜的弟弟了", out)
        self.assertIn("<d>[Chinese]", out)

    def test_repair_empty_origin_long_quote_parts_have_no_dialogue_prefix_overlap(self) -> None:
        """Empty-origin overflow: host partition keeps adjacent DIALOGUE non-overlapping."""
        quote = (
            "哎呦，姐姐最喜欢这嘴甜的弟弟了。既然你都这么说了，姐姐我自然要亲自领你去了。"
            "这条路虽远，但风景正好，你跟紧姐姐，莫要掉队，走到尽头便是那座小院啊。"
            "那里还有新鲜瓜果和清茶，咱们慢慢走也不迟。"
        )
        self.assertLessEqual(len(quote), 120)
        self.assertGreater(
            nodes._speech_seconds_of_text(quote),
            nodes.MAX_DIRECTOR_SEGMENT_SECONDS,
        )
        beat = f"堂内。姐姐笑着说「{quote}」然后起身带路。"
        structure = f"""TITLE: Long CJK quote partition
=== SEGMENT 1 ===
DURATION_SECONDS: 15
CONTINUITY_FROM_PREV: false
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
{beat}
DIALOGUE:
NONE
HANDOFF:
两人离开堂内。
=== END SEGMENT ==="""
        _title, cards = nodes._parse_director_structure_output(structure)
        schedule = (15.0,)
        audit: dict = {"schedule": schedule, "outputs": [], "repairs": []}
        source = cards[0]["beat"]
        mid = source.index("然后起身")

        def _complete(messages, _max_tokens, _seed):
            return json.dumps({
                "parts": [
                    {"beat": source[:mid].rstrip(), "dialogue": "", "handoff": "姐姐刚说完。"},
                    {"beat": source[mid:].lstrip(), "dialogue": "", "handoff": "两人离开堂内。"},
                ]
            }, ensure_ascii=False)

        repaired, resolved, _out = nodes._repair_structure_duration_cards(
            cards, schedule, None, _complete, 1024, 0, audit,
            max_segment_seconds=15.0,
            allow_duration_overrun=True,
        )
        self.assertGreaterEqual(len(repaired), 2)
        dialogues = [
            nodes._clean_structure_dialogue(card.get("dialogue") or "")
            for card in repaired
        ]
        self.assertTrue(all(dialogues))
        norms = [text.replace(" ", "") for text in dialogues]
        self.assertEqual("".join(norms), quote.replace(" ", ""))
        for left, right in zip(norms, norms[1:]):
            self.assertFalse(right.startswith(left))
            # Shared opening phrase must not appear in both adjacent parts.
            opening = "哎呦，姐姐最喜欢这嘴甜的弟弟"
            if opening in left:
                self.assertNotIn(opening, right)



    def test_repair_punct_cut_over_max_tail_splits_into_more_parts(self) -> None:
        """Punct-preferring partition leaving a >max tail must add parts, not abort.

        Mirrors diagnostic seg2: empty DIALOGUE, two long CJK BEAT quotes (~27.8s),
        2-way host schedule. Punct cut parks most of quote2 in part2 (>15s). Ensure-fit
        raises durations to speech and re-splits over-max parts at punctuation.
        """
        q1 = (
            "各位漂亮姐姐好，我是新来这里上任的，大家叫我小程就可以。"
            "不知道村部怎么走，还望哪位神仙姐姐行个方便，给我指条路。"
        )
        q2 = (
            "哎呦，姐姐最喜欢这嘴甜的弟弟了。既然你都这么说了，姐姐我自然要亲自领你去了。"
            "记住，以后叫姐姐柳二姐。都是一个村里住着的，咱们又这么投缘，"
            "以后有啥事和姐姐说，姐姐一定帮你。"
        )
        self.assertLessEqual(len(q1), 120)
        self.assertLessEqual(len(q2), 120)
        speech_total = nodes._speech_seconds_of_text(q1) + nodes._speech_seconds_of_text(q2)
        self.assertGreater(nodes._speech_seconds_of_text(q2), nodes.MAX_DIRECTOR_SEGMENT_SECONDS)
        self.assertAlmostEqual(speech_total, 27.778, places=2)

        # Show the 2-slot punct partition alone would leave an over-max tail.
        two_slot = nodes._director_duration_schedule(speech_total, 15.0)
        self.assertEqual(len(two_slot), 2)
        partitioned = nodes._partition_quote_units_by_duration([q1, q2], two_slot, ceiling=15.0)
        self.assertTrue(
            any(nodes._speech_seconds_of_text(p) > 15.02 for p in partitioned),
            "fixture requires an over-max tail after punct partition",
        )

        beat = (
            f"分镜4: 小程向妇女们自我介绍。小程「{q1}」"
            f"分镜5: 妇女们的目光。"
            f"分镜6: 柳二姐起身走近。柳二姐「{q2}」"
        )
        schedule = (15.0, 15.0, 15.0)
        structure = f"""TITLE: Punct cut duration fit
=== SEGMENT 1 ===
DURATION_SECONDS: 12
CONTINUITY_FROM_PREV: false
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
Wide establishing shot of the village entrance at dusk.
DIALOGUE:
NONE
HANDOFF:
Camera settles on the approaching visitor.
=== END SEGMENT ===
=== SEGMENT 2 ===
DURATION_SECONDS: 15
CONTINUITY_FROM_PREV: true
PICTURES: 1
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
{beat}
DIALOGUE:
NONE
HANDOFF:
Xiaocheng stands with his suitcase beside him.
=== END SEGMENT ===
=== SEGMENT 3 ===
DURATION_SECONDS: 15
CONTINUITY_FROM_PREV: true
PICTURES: 1, 2
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
They begin walking down the village path together.
DIALOGUE:
NONE
HANDOFF:
Both walk away down the path as the others watch.
=== END SEGMENT ==="""
        _title, cards = nodes._parse_director_structure_output(structure)
        audit: dict = {"schedule": schedule, "outputs": [], "repairs": []}
        source = cards[1]["beat"]
        mid = source.index("分镜5")

        def _complete(messages, _max_tokens, _seed):
            user = json.loads(messages[1]["content"])
            part_count = int(user["part_count"])
            self.assertEqual(part_count, 2)
            return json.dumps({
                "parts": [
                    {
                        "beat": source[:mid].rstrip(),
                        "dialogue": "",
                        "handoff": "Xiaocheng finishes his introduction.",
                    },
                    {
                        "beat": source[mid:].lstrip(),
                        "dialogue": "",
                        "handoff": "Xiaocheng stands with his suitcase beside him.",
                    },
                ]
            }, ensure_ascii=False)

        repaired, resolved, _out = nodes._repair_structure_duration_cards(
            cards, schedule, None, _complete, 1024, 0, audit,
            max_segment_seconds=15.0,
            allow_duration_overrun=True,
        )
        self.assertEqual(len(audit["repairs"]), 1)
        self.assertIsNone(audit["repairs"][0].get("error"))
        # Over-max tail forced extra parts beyond the initial 2-way schedule.
        self.assertGreaterEqual(audit["repairs"][0].get("fitted_part_count", 0), 3)
        seg2_cards = [
            card for card in repaired
            if "小程" in (card.get("dialogue") or "") or "姐姐" in (card.get("dialogue") or "")
            or "柳二" in (card.get("dialogue") or "")
        ]
        # All speech words survive across DIALOGUE parts; no prefix overlap.
        dialogues = [
            nodes._clean_structure_dialogue(card.get("dialogue") or "")
            for card in repaired
            if nodes._clean_structure_dialogue(card.get("dialogue") or "")
        ]
        self.assertGreaterEqual(len(dialogues), 3)
        joined = "".join(text.replace(" ", "") for text in dialogues)
        self.assertIn(q1.replace(" ", ""), joined)
        self.assertIn(q2.replace(" ", ""), joined)
        norms = [text.replace(" ", "") for text in dialogues]
        for left, right in zip(norms, norms[1:]):
            self.assertFalse(right.startswith(left))
        for card, duration in zip(repaired, resolved):
            speech = nodes._speech_seconds_of_text(
                nodes._clean_structure_dialogue(card.get("dialogue") or "")
            )
            if speech > 0.02:
                self.assertLessEqual(speech, duration + 0.02)
                self.assertLessEqual(duration, 15.0 + 0.001)
                self.assertGreaterEqual(duration, speech - 0.02)
        self.assertTrue(all(4.0 <= value <= 15.0 for value in resolved))


    def test_repair_still_errors_when_speech_overflows_exceed_schedule_without_overrun(self) -> None:

        """Overrun OFF: same overflows against 45 still error and ask to raise total / enable overrun."""
        schedule = (15.0, 15.0, 15.0)
        heavy_a = " ".join(f"alpha{i:02d}" for i in range(62))
        heavy_b = " ".join(f"bravo{i:02d}" for i in range(50))
        structure = f"""TITLE: Hard budget overflow
=== SEGMENT 1 ===
DURATION_SECONDS: 10
CONTINUITY_FROM_PREV: false
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
Wide establishing shot of the quiet square at dusk.
DIALOGUE:
NONE
HANDOFF:
Camera settles on the approaching visitor.
=== END SEGMENT ===
=== SEGMENT 2 ===
DURATION_SECONDS: 15
CONTINUITY_FROM_PREV: true
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
Visitor greets the guides near the gate and offers a friendly introduction.
DIALOGUE:
{heavy_a}
HANDOFF:
Visitor stands facing the guides with luggage at their feet.
=== END SEGMENT ===
=== SEGMENT 3 ===
DURATION_SECONDS: 15
CONTINUITY_FROM_PREV: true
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
A guide welcomes the visitor and walks them toward the village path.
DIALOGUE:
{heavy_b}
HANDOFF:
Both walk away down the path as the others watch.
=== END SEGMENT ==="""
        _title, cards = nodes._parse_director_structure_output(structure)
        audit: dict = {"schedule": schedule, "outputs": [], "repairs": []}

        def _complete(*_args, **_kwargs):
            raise AssertionError("repair must fail before calling complete when overrun is off")

        with self.assertRaises(nodes.PromptRefinerError) as ctx:
            nodes._repair_structure_duration_cards(
                cards, schedule, None, _complete, 1024, 0, audit,
                max_segment_seconds=15.0,
                allow_duration_overrun=False,
            )
        message = str(ctx.exception)
        self.assertIn("Cannot safely repair", message)
        self.assertIn("allow_duration_overrun", message)
        self.assertIn("target=45.000s", message)
        self.assertNotIn("overflow_budget_expand", audit)
        self.assertEqual(audit["repairs"], [])

    def test_modest_dialogue_imbalance_retargets_flat_equal_split(self) -> None:
        """Backup: lowered MIN_UNITS should retarget when LLM echoes host equals."""
        schedule, plan = nodes._story_duration_schedule(
            self.SINGLE_BEAT_BRIEF, budget_seconds=45.0, max_segment_seconds=10.0,
            allow_duration_overrun=True,
        )
        assert plan is not None
        _title, cards = nodes._parse_director_structure_output(self._cards_for_plan(plan))
        # Backup for lowered MIN_UNITS: spread in [5, 8) with ideal_spread >= 2.
        # Empty-vs-light cannot clear the ideal_spread gate below ~14 units, so use
        # two non-empty loads (18 vs 12 words) and force equal host-echo durations.
        for card in cards:
            card["duration_raw"] = "10"
            card["dialogue"] = " ".join(f"pad{i}" for i in range(12))
        cards[0]["dialogue"] = "LANDMARK " + " ".join(f"w{i}" for i in range(18))
        units = [
            nodes._dialogue_speech_units(card.get("dialogue") or "")
            for card in sorted(cards, key=lambda item: int(item["index"]))
        ]
        self.assertGreaterEqual(max(units) - min(units), nodes._DIALOGUE_REDISTRIBUTE_MIN_UNITS)
        self.assertLess(max(units) - min(units), 8.0)
        segments, rebalanced, misses = nodes._segments_from_structure_cards(
            cards, R2V, schedule, (), (), (), (),
            coverage_plan=plan, locked_durations=False,
        max_segment_seconds=10.0,
        )
        durations = [seg.duration_seconds for seg in segments]
        self.assertEqual(misses, [])
        self.assertTrue(rebalanced)
        self.assertAlmostEqual(sum(durations), 45.0, places=2)
        self.assertTrue(all(4.0 <= value <= 10.0 for value in durations))
        self.assertGreater(durations[0], durations[1])

    def test_header_beat_labels_are_detected_without_machine_block(self) -> None:
        brief = """
节拍1 (est 45s) 分镜1-8
角色「LANDMARK_H1」
节拍2 (est 20s) 分镜9-12
角色「LANDMARK_H2」
节拍3 (est 30s) 分镜13-22
角色「LANDMARK_H3」
""".strip()
        beats = nodes._parse_story_beats_from_brief(brief)
        self.assertEqual([beat.index for beat in beats], [1, 2, 3])
        self.assertEqual(beats[0].estimated_seconds, 45.0)
        self.assertIn("LANDMARK_H1", beats[0].key_dialogue)

    def test_plain_shot_list_without_beat_labels_keeps_knife_off(self) -> None:
        brief = "分镜1: 开场\n分镜2: 路过\n分镜3: 结束"
        self.assertEqual(nodes._parse_story_beats_from_brief(brief), [])
        self.assertIsNone(
            nodes._build_coverage_knife_plan(brief, budget_seconds=60.0, group_count=6)
        )


class AudioVoiceLabelTests(unittest.TestCase):
    """Standalone <Audio N> is a character voice ref, never ambient/Foley."""

    def test_extract_keeps_handwritten_speaker_labels(self) -> None:
        brief = (
            "Global\n"
            "@reference_audio_1 小程声音参考, @reference_audio_2 柳二姐声音参考.\n"
            "小程「你好」\n"
            "柳二姐「嗯」\n"
        )
        labels = nodes._extract_audio_voice_labels(brief)
        self.assertEqual(nodes._speaker_from_audio_label(labels[1]), "小程")
        self.assertEqual(nodes._speaker_from_audio_label(labels[2]), "柳二姐")

    def test_normalize_preserves_trailing_voice_label(self) -> None:
        brief = "@reference_audio_1 小程声音参考, @reference_image_1 face.png"
        out = nodes._normalize_director_asset_links(brief, {1: "img"}, {}, {1: "audio"}, {})
        self.assertIn("<Audio 1> 小程声音参考", out)
        self.assertIn("<Picture 1>", out)
        self.assertNotIn("@reference_audio_1", out)

    def test_uuid_filename_is_not_a_speaker_name(self) -> None:
        uuid = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        brief = f"@reference_audio_1 {uuid}.wav\n小程「过来」\n"
        labels = nodes._extract_audio_voice_labels(brief)
        self.assertEqual(labels[1], f"{uuid}.wav")
        self.assertEqual(nodes._speaker_from_audio_label(labels[1]), "")
        bindings = nodes._resolve_audio_speaker_bindings((1,), labels, brief)
        self.assertEqual(bindings.get(1), "小程")

    def test_inventory_forbids_ambient_even_for_uuid(self) -> None:
        uuid = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        brief = f"@reference_audio_1 {uuid}\n妇女「坐下」\n"
        lines = nodes._director_long_inventory_lines(
            (),
            (),
            (1,),
            (),
            audio_labels=nodes._extract_audio_voice_labels(brief),
            context_text=brief,
        )
        joined = "\n".join(lines)
        self.assertIn("<Audio 1>", joined)
        self.assertIn("voice/timbre", joined)
        self.assertIn("NOT ambient", joined)
        self.assertIn("妇女", joined)
        self.assertNotRegex(joined, r"(?i)cicada|seeds\s+Foley|Ambient village|瓜子")
        self.assertIn("NOT ambient noise, Foley, or SFX", joined)

    def test_ensure_rewrites_ambient_and_not_used_when_dialogue_present(self) -> None:
        prompt = (
            "subject_definitions:\nS1.\n\n"
            "summary:\nBeat.\n\n"
            "retention_analysis:\n"
            "<Audio 1>/<Audio 2>: Not used in this segment; no audio cue assigned.\n"
            "<Audio 1>: Ambient village/cicada bed.\n"
            "<Audio 2>: seeds Foley.\n\n"
            "detailed_description:\n"
            "小程 says <d>[Chinese] 你好</d> then 柳二姐 <d>[Chinese] 嗯</d>.\n\n"
            "overall_soundscape:\nWind.\n\n"
            "non_diegetic_music:\nN/A\n"
        )
        labels = {
            1: "小程声音参考",
            2: "柳二姐声音参考",
        }
        out = nodes._ensure_full_reference_audio_voice_labels(
            prompt,
            (1, 2),
            labels,
            context_text="小程「你好」\n柳二姐「嗯」",
        )
        self.assertNotRegex(out, r"(?i)Not used in this segment")
        self.assertNotRegex(out, r"(?i)cicada|seeds Foley|Ambient village")
        self.assertIn("小程", out)
        self.assertIn("柳二姐", out)
        self.assertRegex(out, r"<Audio 1>: retain voice/timbre")
        self.assertRegex(out, r"<Audio 2>: retain voice/timbre")

    def test_story_shot_binds_handwritten_voice_to_quoted_speaker(self) -> None:
        beat = nodes._StoryBeatSpec(
            index=1,
            estimated_seconds=8.0,
            shot_start=1,
            shot_end=1,
            label="b1",
            body="",
            key_dialogue=(),
        )
        plan = nodes._CoverageKnifePlan(
            beats=(beat,),
            budget_seconds=8.0,
            group_count=1,
            estimated_sum=8.0,
            compressed=False,
            allocations=(8.0,),
            card_counts=(1,),
            segment_to_beat=(1,),
        )
        card = {
            "index": 1,
            "duration_raw": "8",
            "continuity_raw": "false",
            "beat": "placeholder",
            "dialogue": "",
            "handoff": "end",
            "assets": {"pictures": [], "videos": [], "video_audios": [], "audios": []},
        }
        brief = (
            "Global\n"
            "@reference_audio_1 小程声音参考\n"
            "@reference_audio_2 柳二姐声音参考\n"
            "\n"
            "分镜1使用参考: <Picture 1>\n"
            "小程「你好啊」\n"
        )
        normalized = nodes._normalize_director_asset_links(
            brief, {1: "img"}, {}, {1: "a", 2: "b"}, {}
        )
        out = nodes._apply_story_shot_contract([card], normalized, plan)
        self.assertIn(1, out[0]["assets"]["audios"])


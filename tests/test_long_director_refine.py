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
        # Unlock keeps structure DURATION_SECONDS (8/12/10) vs equal host seed → rebalanced.
        self.assertTrue(rebalanced)
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
        self.assertEqual(segments[2].picture_slots, (1, 2))  # empty PICTURES → all connected
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
        self.assertIn("uri-picture-1", json.dumps(calls[3]["parts"]))
        self.assertIn("uri-picture-2", json.dumps(calls[3]["parts"]))
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
        self.assertEqual(set(groups[2]["ref_images"]), {0, 1})
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


class R2VPictureSelectionTests(unittest.TestCase):
    """Tighten mode keeps per-segment stills; full-anchor restores force-all."""

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

    def test_empty_picture_selection_falls_back_to_all_connected(self) -> None:
        pictures, _videos, _paired, _audios = nodes._select_director_segment_assets(
            R2V,
            {"pictures": [], "videos": [], "video_audios": [], "audios": []},
            2,
            (1, 2, 3),
            (),
            (),
            (),
        )
        self.assertEqual(pictures, (1, 2, 3))
        missing = nodes._select_director_segment_assets(
            R2V,
            {},
            3,
            (1, 2),
            (),
            (),
            (),
        )[0]
        self.assertEqual(missing, (1, 2))

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

    def test_planner_exposes_anchor_all_pictures_widget(self) -> None:
        required = nodes.Qwen38LongVideoDirectorPlanner.INPUT_TYPES()["required"]
        self.assertIn("anchor_all_pictures", required)
        self.assertEqual(required["anchor_all_pictures"][0], "BOOLEAN")
        self.assertFalse(required["anchor_all_pictures"][1]["default"])




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

    def test_single_beat_overrun_expands_to_cover_shots(self) -> None:
        schedule, plan = nodes._story_duration_schedule(
            self.SINGLE_BEAT_BRIEF, budget_seconds=45.0, max_segment_seconds=10.0,
            allow_duration_overrun=True,
        )
        self.assertEqual(schedule, (10.0,) * 9)
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.beats), 1)
        self.assertEqual(plan.budget_seconds, 90.0)
        self.assertEqual(plan.card_counts, (9,))

    def test_single_beat_unlocked_durations_follow_dialogue(self) -> None:
        schedule, plan = nodes._story_duration_schedule(
            self.SINGLE_BEAT_BRIEF, budget_seconds=45.0, max_segment_seconds=10.0,
            allow_duration_overrun=True,
        )
        assert plan is not None
        _title, cards = nodes._parse_director_structure_output(self._cards_for_plan(plan))
        cards[0]["dialogue"] += " " + _words(40)
        cards[3]["dialogue"] = "字" * 40
        cards[5]["dialogue"] = _words(40)
        segments, rebalanced, misses = nodes._segments_from_structure_cards(
            cards, R2V, schedule, (), (), (), (),
            coverage_plan=plan, locked_durations=False,
        )
        durations = [seg.duration_seconds for seg in segments]
        self.assertEqual(misses, [])
        self.assertTrue(rebalanced)
        self.assertEqual(len(segments), plan.card_counts[0])
        self.assertAlmostEqual(sum(durations), 90.0, places=2)
        self.assertGreater(durations[0], durations[1])
        self.assertGreater(durations[3], durations[1])
        self.assertGreater(durations[5], durations[1])

        locked, rebalanced, misses = nodes._segments_from_structure_cards(
            cards, R2V, schedule, (), (), (), (),
            coverage_plan=plan, locked_durations=True,
        )
        self.assertEqual([seg.duration_seconds for seg in locked], list(schedule))
        self.assertFalse(rebalanced)
        self.assertEqual(misses, [])

    def test_single_beat_unlocked_keeps_llm_durations_when_dialogue_flat(self) -> None:
        """Coverage equal-seed must not flatten a single beat when unlock is on.

        Job29-style: overrun → 9×10 host schedule, mild dialogue (spread < 8 units),
        but structure cards already chose an uneven rhythm. Retarget correctly stays
        off; unlocked path should still keep DURATION_SECONDS instead of N×(budget/N).
        """
        schedule, plan = nodes._story_duration_schedule(
            self.SINGLE_BEAT_BRIEF, budget_seconds=45.0, max_segment_seconds=10.0,
            allow_duration_overrun=True,
        )
        assert plan is not None
        self.assertEqual(schedule, (10.0,) * 9)
        llm_durations = (12.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 8.0)
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
        )
        durations = [seg.duration_seconds for seg in segments]
        self.assertEqual(misses, [])
        self.assertEqual(len(durations), 9)
        self.assertAlmostEqual(sum(durations), 90.0, places=2)
        self.assertNotEqual(durations, list(schedule))
        self.assertGreater(durations[0], durations[-1])
        self.assertAlmostEqual(durations[0], 12.0, places=2)
        self.assertAlmostEqual(durations[-1], 8.0, places=2)

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

    def test_shot_floor_over_expanded_cap_is_rejected(self) -> None:
        brief = "[STORY_BEATS]\n1|45|1-61|arrive|LANDMARK\n[/STORY_BEATS]"
        with self.assertRaisesRegex(nodes.PromptRefinerError, "safety ceiling"):
            nodes._story_duration_schedule(
                brief, budget_seconds=45.0, max_segment_seconds=10.0,
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
            duration = 15 if beat_idx <= 5 else 4
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
        """STORY_BEATS overrun still seeds N x 10s, but the structure prompt must not force copy."""
        schedule, plan = nodes._story_duration_schedule(
            self.SINGLE_BEAT_BRIEF, budget_seconds=45.0, max_segment_seconds=10.0,
            allow_duration_overrun=True,
        )
        self.assertEqual(schedule, (10.0,) * 9)
        assert plan is not None
        system = nodes._long_director_structure_system(R2V, schedule, coverage_plan=plan)
        self.assertNotIn("LOCKED DURATION SCHEDULE", system)
        self.assertNotIn("Copy the corresponding host duration exactly", system)
        template = _template_block(system)
        self.assertIn("DURATION_SECONDS: <4-15>", template)
        self.assertNotRegex(template, r"DURATION_SECONDS:\s*\d")
        self.assertNotIn("10.000", template)
        self.assertIn("DURATION_SECONDS: <4-15>", system)
        self.assertIn("only the average used to estimate that group count", system)
        self.assertIn("make their sum exactly", system)
        self.assertIn("near 15", system)
        self.assertIn("near 4", system)
        self.assertIn("90.000 seconds", system)

    def test_repair_keeps_variable_llm_durations_against_equal_host_seed(self) -> None:
        """Speech that fits a long card must not overflow just because the host seed is 10s."""
        schedule, plan = nodes._story_duration_schedule(
            self.SINGLE_BEAT_BRIEF, budget_seconds=45.0, max_segment_seconds=10.0,
            allow_duration_overrun=True,
        )
        assert plan is not None
        self.assertEqual(schedule, (10.0,) * 9)
        llm_durations = (14.0, 8.0, 10.0, 12.0, 8.0, 10.0, 10.0, 10.0, 8.0)
        self.assertAlmostEqual(sum(llm_durations), 90.0, places=3)
        blocks = ["TITLE: Variable LLM rhythm"]
        heavy = _words(28)  # ~12.4s of speech at 2.25 wps; fits 14s, exceeds equal 10s seed
        for index, duration in enumerate(llm_durations, start=1):
            continuity = "false" if index == 1 else "true"
            dialogue = heavy if index == 1 else ("LANDMARK" if index == 2 else "NONE")
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
        )
        durations = [seg.duration_seconds for seg in segments]
        self.assertEqual(misses, [])
        self.assertTrue(rebalanced)
        self.assertAlmostEqual(sum(durations), 90.0, places=2)
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

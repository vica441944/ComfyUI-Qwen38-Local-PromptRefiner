"""Planner tests that do not load a GGUF model.

Per-segment refinement is checked with a fake completion function.  Legacy
single-shot parsing, duration rebalance, and Director group packing stay covered
so the new path cannot silently change those contracts.
"""

from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path


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
        self.assertEqual(segments[0].picture_slots, (1, 2))
        self.assertEqual(segments[1].picture_slots, (1, 2))
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
        self.assertIn("uri-picture-2", json.dumps(calls[2]["parts"]))
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
        self.assertEqual(set(groups[1]["ref_images"]), {0, 1})
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
        self.assertLess(names.index("prompt_refinement"), names.index("total_duration_seconds"))
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


if __name__ == "__main__":
    unittest.main()

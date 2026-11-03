"""Spatial continuity locks for HANDOFF and CONTINUITY_FROM_PREV.

LOOK, landmark matching, and duration soft-overrun stay on their existing paths.
These tests only add the screen / axis / anchor contract.
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
import spatial_continuity as spatial  # noqa: E402


R2V = nodes.DIRECTOR_LONG_R2V_MODE
BALANCED = nodes.DIRECTOR_LONG_QUALITY_PROFILES[nodes.DIRECTOR_LONG_QUALITY_BALANCED]

SAMPLE = {
    "scene_blocking": {
        "anchors": {"desk": "center", "door": "north wall"},
        "characters_relative": {"Ivo": "beside desk", "Neri": "beside door"},
        "axis": "Ivo to Neri",
        "working_side": "south",
        "default_screen_left_right": {"Ivo": "left", "Neri": "right"},
    },
    "shot": {
        "screen_left_right": {"Ivo": "left"},
        "axis_side": "south",
        "inherits_axis": True,
        "exit_side": "none",
        "enter_side": "none",
        "spatial_anchor": "desk",
        "eyeline_to": "screen-right",
        "camera_vs_axis": "same",
        "depth": {"Ivo": "foreground"},
    },
}

TEXT_BLOCK = """
[SPATIAL_CONTINUITY]
scene_blocking.anchors: desk=center; door=north wall
scene_blocking.characters_relative: Ivo=beside desk; Neri=beside door
scene_blocking.axis: Ivo to Neri
scene_blocking.working_side: south
scene_blocking.default_screen_left_right: Ivo=left; Neri=right
shot.screen_left_right: Ivo=left
shot.axis_side: south
shot.inherits_axis: true
shot.exit_side: none
shot.spatial_anchor: desk
shot.eyeline_to: screen-right
shot.camera_vs_axis: same
shot.depth: Ivo=foreground
[/SPATIAL_CONTINUITY]
""".strip()


def _marked(payload: dict) -> str:
    return "[SPATIAL_CONTINUITY]\n" + json.dumps(payload) + "\n[/SPATIAL_CONTINUITY]"


def _look(prompt: str) -> str:
    return prompt.split("summary:", 1)[0]


def _action(prompt: str) -> str:
    for label in ("detailed_description:", "integrated_multimodal_description:"):
        if label in prompt:
            return prompt.split(label, 1)[1].split("overall_soundscape:", 1)[0]
    return ""


class ParseSpatialContinuityTests(unittest.TestCase):
    def test_sample_json_prefers_shot_then_scene_defaults(self) -> None:
        catalog = spatial.parse_spatial_continuity(_marked(SAMPLE))
        locks = catalog.resolve(None)
        self.assertIsNotNone(locks)
        assert locks is not None
        sentence = locks.opening_sentence()
        self.assertIn("Ivo screen-left", sentence)
        self.assertNotIn("Neri screen-right", sentence)
        self.assertIn("working side south", sentence)
        self.assertIn("axis Ivo to Neri", sentence)
        self.assertIn("spatial anchor: desk", sentence)
        self.assertNotIn("desk (center)", sentence)
        self.assertNotRegex(sentence, r"(?i)xyz|seedance|x\s*=|y\s*=|z\s*=")

    def test_text_block_matches_sample_locks(self) -> None:
        from_json = spatial.parse_spatial_continuity(_marked(SAMPLE)).resolve(None)
        from_text = spatial.parse_spatial_continuity(TEXT_BLOCK).resolve(None)
        assert from_json is not None and from_text is not None
        self.assertEqual(from_json.opening_sentence(), from_text.opening_sentence())

    def test_omitted_shot_fields_fall_back_to_scene_blocking(self) -> None:
        payload = {"scene_blocking": SAMPLE["scene_blocking"]}
        locks = spatial.parse_spatial_continuity(_marked(payload)).resolve(None)
        assert locks is not None
        sentence = locks.opening_sentence()
        self.assertIn("Ivo screen-left", sentence)
        self.assertIn("Neri screen-right", sentence)
        self.assertIn("working side south", sentence)
        self.assertIn("axis Ivo to Neri", sentence)
        self.assertIn("desk (center)", sentence)

    def test_numbered_shot_overrides_and_does_not_leak(self) -> None:
        brief = _marked({"scene_blocking": SAMPLE["scene_blocking"]}) + """
分镜1使用参考: @reference_image_1
Ivo waits at the desk.
分镜2使用参考: @reference_image_1
{"screen_left_right": {"Ivo": "right", "Neri": "left"}, "axis_side": "north", "spatial_anchor": "door", "inherits_axis": false, "motion_vector": "toward the door"}
"""
        catalog = spatial.parse_spatial_continuity(brief)
        first = spatial.locks_for_text(catalog, "分镜1使用参考: @reference_image_1\nIvo waits.", end=True)
        second = spatial.locks_for_text(catalog, "分镜2使用参考: @reference_image_1", end=False)
        assert first is not None and second is not None
        self.assertIn("Ivo screen-left", first.opening_sentence())
        self.assertIn("Neri screen-right", first.opening_sentence())
        self.assertIn("working side south", first.opening_sentence())
        self.assertIn("Ivo screen-right", second.opening_sentence())
        self.assertIn("Neri screen-left", second.opening_sentence())
        self.assertIn("working side north", second.opening_sentence())
        self.assertNotIn("working side south", second.opening_sentence())
        self.assertNotIn("Ivo to Neri", second.opening_sentence())
        self.assertIn("spatial anchor: door", second.opening_sentence())
        self.assertEqual(second.optional_notes[0][0], "motion_vector")

    def test_raw_shot_json_outside_the_marker_is_read(self) -> None:
        brief = json.dumps(SAMPLE)
        locks = spatial.parse_spatial_continuity(brief).resolve(None)
        assert locks is not None
        self.assertIn("spatial anchor: desk", locks.opening_sentence())


class HandoffAndOpeningRepairTests(unittest.TestCase):
    def _sample_locks(self) -> spatial.SpatialLocks:
        locks = spatial.parse_spatial_continuity(_marked(SAMPLE)).resolve(None)
        assert locks is not None
        return locks

    def test_handoff_appends_missing_locks_and_keeps_pose(self) -> None:
        locks = self._sample_locks()
        pose = "Right hand lowers the cup, eyes toward the window."
        repaired = spatial.repair_handoff(pose, locks)
        self.assertTrue(repaired.startswith(pose))
        self.assertIn("screen left/right: Ivo screen-left", repaired)
        self.assertIn("working side south", repaired)
        self.assertIn("axis Ivo to Neri", repaired)
        self.assertIn("spatial anchor: desk", repaired)
        self.assertNotRegex(repaired, r"(?i)\b(?:xyz|seedance)\b|x\s*=|y\s*=|z\s*=")
        self.assertNotRegex(repaired, r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+")
        self.assertEqual(spatial.repair_handoff(repaired, locks), repaired)

    def test_complete_handoff_is_left_unchanged(self) -> None:
        locks = self._sample_locks()
        handoff = (
            "Cup down. Spatial continuity: screen left/right: Ivo screen-left; "
            "axis/working side: working side south, axis Ivo to Neri; spatial anchor: desk."
        )
        self.assertEqual(spatial.repair_handoff(handoff, locks), handoff)

    def test_optional_passthrough_is_appended_and_not_required(self) -> None:
        payload = json.loads(json.dumps(SAMPLE))
        payload["shot"]["receiver_in"] = "Neri from the doorway"
        payload["shot"]["handoff_out"] = "none"
        payload["shot"]["spatial_bridge"] = "across the desk edge"
        locks = spatial.parse_spatial_continuity(_marked(payload)).resolve(None)
        assert locks is not None
        pose = "Right hand lowers the cup, eyes toward the window."
        repaired = spatial.repair_handoff(pose, locks)
        self.assertIn("receiver_in Neri from the doorway", repaired)
        self.assertIn("spatial_bridge across the desk edge", repaired)
        self.assertNotIn("handoff_out", repaired)
        self.assertNotIn("motion_vector", repaired)
        held = (
            "Right hand lowers the cup. screen left/right: Ivo screen-left. "
            "axis/working side: working side south, axis Ivo to Neri. spatial anchor: desk."
        )
        with_optional = spatial.repair_handoff(held, locks)
        self.assertTrue(with_optional.startswith(held))
        self.assertIn("receiver_in Neri from the doorway", with_optional)

    def test_continuity_first_sentence_restates_locks_without_touching_look(self) -> None:
        locks = self._sample_locks()
        prompt = (
            "subject_definitions:\n"
            "<Subject 1> Ivo, fully_preserved wardrobe, same face.\n\n"
            "summary:\n"
            "The cup stays down.\n\n"
            "retention_analysis:\n"
            "<Picture 1>: fully_preserved.\n\n"
            "detailed_description:\n"
            "[Shot 1] Ivo lifts the cup.\n"
            "<d>[Chinese] 请往前走</d>\n\n"
            "overall_soundscape:\n"
            "Room tone.\n\n"
            "non_diegetic_music:\n"
            "N/A\n"
        )
        repaired = spatial.repair_continuity_opening(prompt, locks, continuity=True)
        self.assertEqual(_look(prompt), _look(repaired))
        action = _action(repaired)
        first = action.split(".", 1)[0]
        self.assertIn("Ivo screen-left", first)
        self.assertIn("south", first)
        self.assertIn("desk", first)
        self.assertIn("Ivo to Neri", first)
        self.assertIn("[Shot 1]", repaired)
        self.assertIn("Ivo lifts the cup.", repaired)
        self.assertLess(repaired.index("[Shot 1]"), repaired.index("Ivo lifts the cup."))
        self.assertIn("<d>[Chinese] 请往前走</d>", repaired)
        self.assertIn("\n\noverall_soundscape:", repaired)
        self.assertEqual(spatial.repair_continuity_opening(repaired, locks, continuity=True), repaired)
        self.assertEqual(spatial.repair_continuity_opening(prompt, locks, continuity=False), prompt)

    def test_opening_repair_uses_fl2v_field_and_skips_a_sentence_that_already_locks(self) -> None:
        locks = self._sample_locks()
        sentence = locks.opening_sentence()
        prompt = (
            "integrated_multimodal_description:\n"
            f"[Shot 1] {sentence} Ivo stays with the cup.\n\n"
            "overall_soundscape:\n"
            "Room tone.\n\n"
            "non_diegetic_music:\n"
            "N/A\n"
        )
        self.assertEqual(
            spatial.repair_continuity_opening(prompt, locks, continuity=True),
            prompt,
        )


class PlannerContractTests(unittest.TestCase):
    def test_prompts_require_the_three_locks_and_keep_duration_wording(self) -> None:
        structure = nodes._long_director_structure_system(R2V, (10.0, 10.0))
        segment = nodes._long_director_segment_system(R2V, BALANCED)
        legacy = nodes._long_director_system(R2V, (10.0, 10.0), BALANCED)
        for system in (structure, segment, legacy):
            self.assertIn("screen left/right", system)
            self.assertIn("spatial anchor", system)
            self.assertIn("working side", system)
            self.assertIn("Do not invent absolute XY/Z coordinates.", system)
            self.assertNotIn("Seedance", system)
        self.assertIn("HANDOFF", structure)
        self.assertIn("make their sum exactly", structure)
        self.assertIn("make their sum exactly", legacy)
        self.assertIn("first sentence", segment)
        self.assertIn("not subject_definitions", segment)

    def test_segment_request_states_the_opening_locks_only_when_continuing(self) -> None:
        brief = "Office scene.\n" + _marked(SAMPLE)
        opening = nodes._DirectorLongSegment(
            index=1,
            duration_seconds=10.0,
            prompt="",
            picture_slots=(),
            video_slots=(),
            video_audio_slots=(),
            audio_slots=(),
            continuity_from_prev=False,
            beat="分镜1使用参考: @reference_image_1\nIvo sits.",
            dialogue="",
            handoff="Cup down.",
        )
        continued = nodes._DirectorLongSegment(
            index=2,
            duration_seconds=10.0,
            prompt="",
            picture_slots=(),
            video_slots=(),
            video_audio_slots=(),
            audio_slots=(),
            continuity_from_prev=True,
            beat="分镜1使用参考: @reference_image_1\nIvo stays seated.",
            dialogue="",
            handoff="Still at the desk.",
        )
        continued_request = nodes._build_long_director_segment_request(
            R2V, brief, "", (opening, continued), continued, (), (), (), None, BALANCED,
        )
        hard_request = nodes._build_long_director_segment_request(
            R2V, brief, "", (opening, continued), opening, (), (), (), None, BALANCED,
        )
        continued_user = continued_request[1]["content"][0]["text"]
        hard_user = hard_request[1]["content"][0]["text"]
        self.assertIn("must restate these spatial locks", continued_user)
        self.assertIn("Ivo screen-left", continued_user)
        self.assertIn("working side south", continued_user)
        self.assertIn("spatial anchor: desk", continued_user)
        self.assertNotIn("must restate these spatial locks", hard_user)

    def test_independent_refine_repairs_handoff_and_opening_without_duration_or_look_edits(self) -> None:
        brief = "Two quiet beats.\n" + _marked(SAMPLE)
        structure = """
TITLE: Spatial office
=== SEGMENT 1 ===
DURATION_SECONDS: 10
CONTINUITY_FROM_PREV: false
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
分镜1使用参考: @reference_image_1
Ivo sets the cup down.
DIALOGUE:
NONE
HANDOFF:
Right hand lowers the cup, eyes toward the window.
=== END SEGMENT ===
=== SEGMENT 2 ===
DURATION_SECONDS: 10
CONTINUITY_FROM_PREV: true
PICTURES:
VIDEOS:
VIDEO_AUDIOS:
AUDIOS:
BEAT:
分镜1使用参考: @reference_image_1
Ivo stays with the cup.
DIALOGUE:
请往前走
HANDOFF:
Eyes stay on the window.
=== END SEGMENT ===
""".strip()
        look = "<Subject 1> Ivo, fully_preserved wardrobe, same face."
        continued = (
            "subject_definitions:\n"
            f"{look}\n\n"
            "summary:\n"
            "The cup stays down.\n\n"
            "retention_analysis:\n"
            "No still.\n\n"
            "detailed_description:\n"
            "[Shot 1] Ivo keeps the cup in his right hand.\n"
            "<d>[Chinese] 请往前走</d>\n\n"
            "overall_soundscape:\n"
            "Room tone.\n\n"
            "non_diegetic_music:\n"
            "N/A"
        )

        def complete(messages, max_tokens, seed):
            del max_tokens, seed
            system = messages[0]["content"]
            user = messages[1]["content"][0]["text"]
            if "STRUCTURE mode" in system:
                self.assertIn("SPATIAL CONTINUITY", user)
                self.assertIn("screen left/right", system)
                return structure
            if "Segment 2 of 2" in user:
                self.assertIn("must restate these spatial locks", user)
                return continued
            return continued.replace("keeps the cup in his right hand", "sets the cup down")

        _title, segments, rebalanced, debug, _record = nodes._plan_director_with_independent_segment_refine(
            complete,
            mode=R2V,
            brief=brief,
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
            seed=3,
        )
        self.assertFalse(rebalanced)
        self.assertEqual([segment.duration_seconds for segment in segments], [10.0, 10.0])
        self.assertFalse(segments[0].continuity_from_prev)
        self.assertTrue(segments[1].continuity_from_prev)
        self.assertIn("Right hand lowers the cup, eyes toward the window.", segments[0].handoff)
        self.assertIn("Ivo screen-left", segments[0].handoff)
        self.assertIn("working side south", segments[0].handoff)
        self.assertIn("spatial anchor: desk", segments[0].handoff)
        self.assertIn("Eyes stay on the window.", segments[1].handoff)
        action = _action(segments[1].prompt)
        first = action.split(".", 1)[0]
        self.assertIn("Ivo screen-left", first)
        self.assertIn("south", first)
        self.assertIn("desk", first)
        self.assertIn(look, segments[1].prompt)
        self.assertEqual(_look(segments[1].prompt), _look(continued))
        self.assertIn("请往前走", segments[1].prompt)
        self.assertIn("[Brief Shot 1] Ivo keeps the cup in his right hand.", segments[1].prompt)
        self.assertLess(segments[1].prompt.index("spatial anchor: desk"), segments[1].prompt.index("[Brief Shot 1]"))
        self.assertIn("spatial_continuity=on", debug)

    def test_spatial_block_does_not_change_duration_or_landmark_contracts(self) -> None:
        beats = """[STORY_BEATS]
1 | 20s | 1-2 | opening | 请往前走
2 | 12s | 3-3 | ending | 停在门口
[/STORY_BEATS]
分镜1使用参考: @reference_image_1
请往前走
分镜2使用参考: @reference_image_1
停一下
分镜3使用参考: @reference_image_1
停在门口
"""
        plain, plain_plan = nodes._story_duration_schedule(
            beats, budget_seconds=40, max_segment_seconds=15, allow_duration_overrun=True,
        )
        with_spatial, spatial_plan = nodes._story_duration_schedule(
            beats + "\n" + _marked(SAMPLE),
            budget_seconds=40,
            max_segment_seconds=15,
            allow_duration_overrun=True,
        )
        self.assertEqual(plain, with_spatial)
        assert plain_plan is not None and spatial_plan is not None
        self.assertEqual(plain_plan.allocations, spatial_plan.allocations)
        self.assertEqual(plain_plan.card_counts, spatial_plan.card_counts)
        self.assertEqual(plain_plan.compressed, spatial_plan.compressed)
        self.assertEqual(plain_plan.budget_seconds, spatial_plan.budget_seconds)
        self.assertTrue(nodes._is_non_dialogue_landmark("赵凤红呻吟"))
        self.assertFalse(nodes._is_non_dialogue_landmark("你别呻吟"))
        self.assertTrue(nodes._is_placeholder_landmark("（露骨台词）"))
        self.assertIn(
            "coverage_landmark_miss=",
            nodes._format_coverage_landmark_miss(["beat1:请往前走"]),
        )


# Shape of uploads/qwen38-director-structure-o2710nqd.json: the structure card's
# DIALOGUE was a flattened spatial schema, and BEAT carried a mangled block.
_SCHEMA_DIALOGUE = """anchors
房门
走廊尽头右侧，半开状态，月光从门缝透出
走廊
青石板地面，上方瓦檐漏下冷月光
axis
小六面向房门，房门为轴心点
axis_cross_shots
characters_relative
小六
站在房门前，面向门（画面右侧），四分之三侧面朝向镜头
柳二姐
门内炕上侧卧，画外，头朝炕里
default_screen_left_right
小六
left
房门
right
working_side
走廊侧，镜头在小六身后偏左
screen_left_right
小六
left
房门
right
axis_side
走廊侧
inherits_axis
exit_side
none
enter_side
none
spatial_anchor
房门
""".strip()

_REAL_LINE = "(S1) <d>[Chinese] 男人就没一个好玩意</d>"


class SpatialDumpIsNotSpeechTests(unittest.TestCase):
    def test_schema_dump_is_empty_dialogue_and_real_lines_stay(self) -> None:
        self.assertEqual(nodes._clean_structure_dialogue(_SCHEMA_DIALOGUE), "")
        self.assertEqual(nodes._clean_structure_dialogue("NONE"), "")
        self.assertEqual(nodes._clean_structure_dialogue(_REAL_LINE), _REAL_LINE)
        self.assertGreater(nodes._structure_card_speech_seconds(_REAL_LINE, ""), 0.5)

    def test_marker_in_beat_does_not_count_schema_quotes_as_speech(self) -> None:
        beat = (
            "分镜3使用参考:<Picture 1>\n"
            "分镜3: 小六关门并侧身闪入。「请往前走」\n"
            "[SPATIAL_CONTINUITY]\n"
            "For HANDOFF and CONTINUITY_FROM_PREV, first restate screen_left_right, "
            "axis_side and spatial_anchor.\n"
            'shot 3: {"receiver_in": "", "facing": "侧面对镜头", "spatial_anchor": "房门"}\n'
        )
        schema_only = (
            "[SPATIAL_CONTINUITY]\n"
            'shot 3: {"facing": "侧面对镜头", "receiver_in": "从门口"}\n'
        )
        self.assertEqual(nodes._beat_quoted_speech_text(schema_only), "")
        self.assertEqual(nodes._structure_card_speech_seconds("", schema_only), 0.0)
        spoken = nodes._structure_card_speech_seconds(_SCHEMA_DIALOGUE, beat)
        quote_only = nodes._structure_card_speech_seconds("", "「请往前走」")
        self.assertAlmostEqual(spoken, quote_only, places=3)
        self.assertLess(spoken, 3.0)
        self.assertNotIn("侧面对镜头", nodes._beat_quoted_speech_text(beat))

    def test_shot_blocks_keep_prose_and_locks_and_drop_the_schema(self) -> None:
        brief = """[STORY_BEATS]
1 | 8.5s | 1-2 | corridor | 男人就没一个好玩意
2 | 8.5s | 3-4 | room | 请往前走
[/STORY_BEATS]
分镜1使用参考: @reference_image_1
分镜1: 走廊听门。「男人就没一个好玩意」
分镜2使用参考: @reference_image_1
分镜2: 小六侧耳。
分镜3使用参考: @reference_image_1
分镜3: 小六关门并侧身闪入。「请往前走」
分镜4使用参考: @reference_image_1
分镜4: 小六进入卧室。
[SPATIAL_CONTINUITY]
scene_blocking.anchors: 房门=走廊尽头右侧
scene_blocking.axis: 小六面向房门
scene_blocking.working_side: 走廊侧
scene_blocking.default_screen_left_right: 小六=left; 房门=right
shot.3.screen_left_right: 小六=left
shot.3.axis_side: 走廊侧
shot.3.spatial_anchor: 房门
[/SPATIAL_CONTINUITY]
"""
        blocks = nodes._brief_shot_blocks(brief)
        self.assertNotIn("[SPATIAL_CONTINUITY]", blocks[4])
        self.assertNotIn("anchors", blocks[4])
        self.assertIn("小六进入卧室", blocks[4])
        self.assertIn("请往前走", blocks[3])
        locks = spatial.locks_for_text(spatial.parse_spatial_continuity(brief), "分镜3使用参考", end=True)
        assert locks is not None
        self.assertIn("房门", locks.opening_sentence())
        self.assertIn("走廊侧", locks.opening_sentence())
        self.assertIn("小六 screen-left", locks.opening_sentence())

        plan = nodes._build_coverage_knife_plan(brief, budget_seconds=17.0, group_count=2)
        assert plan is not None
        cards = [
            dict(
                index=1, duration_raw="10", continuity_raw="false",
                assets={"pictures": [], "videos": [], "video_audios": [], "audios": []},
                beat="Wide corridor.", dialogue=_REAL_LINE, handoff="Smirk.",
            ),
            dict(
                index=2, duration_raw="7", continuity_raw="true",
                assets={"pictures": [1, 2, 3], "videos": [], "video_audios": [], "audios": []},
                beat="Close-up.", dialogue="NONE", handoff="Inside the room.",
            ),
        ]
        assigned = nodes._apply_story_shot_contract(cards, brief, plan)
        later = assigned[1]
        self.assertNotIn("[SPATIAL_CONTINUITY]", later["beat"])
        self.assertNotIn("screen_left_right", later["dialogue"])
        self.assertNotIn("anchors", later["dialogue"])
        self.assertIn("请往前走", later["dialogue"])
        self.assertLess(nodes._structure_card_speech_seconds(later["dialogue"], later["beat"]), 3.0)

    def test_diagnostic_card_does_not_expand_budget_or_partition_schema(self) -> None:
        beat = (
            "分镜3使用参考:<Picture 1>,<Picture 2>,<Picture 3>\n"
            "分镜3: 中景。小六在画面左侧，右手伸向房门。「请往前走」\n"
            "分镜4使用参考:<Picture 2>,<Picture 3>\n"
            "分镜4: 全景。小六进入卧室。\n"
            "[SPATIAL_CONTINUITY]\n"
            "scene_blocking 5e52751b-42c2-44fb-b3b5-8238797a31f1: {:}\n"
            'shot 4: {"receiver_in": "", "facing": "侧面对镜头", "spatial_anchor": "炕"}\n'
        )
        cards = [
            dict(
                index=1, duration_raw="10", continuity_raw="false",
                assets={"pictures": [2], "videos": [], "video_audios": [], "audios": [1]},
                beat="Wide static shot in a moonlit stone corridor.",
                dialogue=_REAL_LINE,
                handoff="Smirk fully formed.",
            ),
            dict(
                index=2, duration_raw="7", continuity_raw="true",
                assets={"pictures": [1, 2, 3], "videos": [], "video_audios": [], "audios": []},
                beat=beat,
                dialogue=_SCHEMA_DIALOGUE,
                handoff="Xiao Cheng has crossed the threshold into the bedroom.",
            ),
        ]

        def complete(*_args):
            raise AssertionError("spatial schema must not enter local dialogue repair")

        audit: dict = {"repairs": []}
        fixed, durations, _plan = nodes._repair_structure_duration_cards(
            cards, (8.5, 8.5), None, complete, 4096, 1, audit,
            max_segment_seconds=15, allow_duration_overrun=True,
        )
        self.assertNotIn("overflow_budget_expand", audit)
        self.assertEqual(audit["repairs"], [])
        self.assertEqual(durations, (10.0, 7.0))
        self.assertEqual(fixed[1]["dialogue"], _SCHEMA_DIALOGUE)
        self.assertEqual(nodes._clean_structure_dialogue(fixed[1]["dialogue"]), "")
        self.assertLess(
            nodes._structure_card_speech_seconds(fixed[1]["dialogue"], fixed[1]["beat"]),
            3.0,
        )

    def test_real_spoken_dialogue_still_partitions(self) -> None:
        spoken = " ".join(["word"] * 40)
        source = dict(
            index=1, duration_raw="8", continuity_raw="false",
            assets={"pictures": [], "videos": [], "video_audios": [], "audios": []},
            beat="Opens the door. Steps through.",
            dialogue=spoken,
            handoff="Inside.",
        )

        def complete(messages, _tokens, _seed):
            payload = json.loads(messages[1]["content"])
            self.assertNotIn("anchors", payload["dialogue"])
            self.assertIn("word", payload["dialogue"])
            words = payload["dialogue"].split()
            mid = len(words) // 2
            return json.dumps({"parts": [
                {
                    "beat": "Opens the door.",
                    "dialogue": " ".join(words[:mid]),
                    "handoff": "Pause.",
                },
                {
                    "beat": "Steps through.",
                    "dialogue": " ".join(words[mid:]),
                    "handoff": payload["handoff"],
                },
            ]})

        audit: dict = {"repairs": []}
        fixed, durations, _plan = nodes._repair_structure_duration_cards(
            [source], (8.0,), None, complete, 4096, 1, audit,
            max_segment_seconds=15, allow_duration_overrun=True,
        )
        self.assertTrue(audit["repairs"])
        joined = " ".join(nodes._clean_structure_dialogue(card["dialogue"]) for card in fixed)
        self.assertEqual(joined.split(), spoken.split())
        self.assertGreater(sum(durations), 8.0)
        self.assertIn("word", joined)
        self.assertNotIn("spatial_anchor", joined)

    def test_unclosed_spatial_block_still_parses_locks(self) -> None:
        brief = """分镜3使用参考: @reference_image_1
分镜3: 关门。「请往前走」
[SPATIAL_CONTINUITY]
scene_blocking.anchors: 房门=走廊尽头
scene_blocking.axis: 小六面向房门
scene_blocking.working_side: 走廊侧
scene_blocking.default_screen_left_right: 小六=left
shot.axis_side: 走廊侧
shot.spatial_anchor: 房门
shot.screen_left_right: 小六=left
"""
        locks = spatial.parse_spatial_continuity(brief).resolve(3)
        assert locks is not None
        self.assertIn("房门", locks.opening_sentence())
        self.assertIn("走廊侧", locks.opening_sentence())
        block = nodes._brief_shot_blocks(brief)[3]
        self.assertIn("请往前走", block)
        self.assertNotIn("[SPATIAL_CONTINUITY]", block)
        self.assertNotIn("anchors", block)


if __name__ == "__main__":
    unittest.main()

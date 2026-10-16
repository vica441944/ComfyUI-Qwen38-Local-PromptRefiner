"""Host duration contracts and lossless local repair, without a model/GPU."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
from test_long_director_refine import nodes, R2V, BALANCED, r2v_prompt


def cards(durations):
    return [dict(index=i, duration_raw=str(d), continuity_raw="false" if i % 2 else "true",
                 assets={"pictures": [1], "videos": [1], "video_audios": [1], "audios": []},
                 beat=f"Action_{i}_a. Action_{i}_b.", dialogue=f"Line_{i}_a. Line_{i}_b.",
                 handoff=f"End_{i}.") for i, d in enumerate(durations, 1)]


def marked(items):
    return "TITLE: Duration regression\n" + "\n".join(
        f"=== SEGMENT {c['index']} ===\nDURATION_SECONDS: {c['duration_raw']}\n"
        f"CONTINUITY_FROM_PREV: {c['continuity_raw']}\nPICTURES: 1\nVIDEOS: 1\nVIDEO_AUDIOS: 1\n"
        f"BEAT:\n{c['beat']}\nDIALOGUE:\n{c['dialogue']}\nHANDOFF:\n{c['handoff']}\n=== END SEGMENT ==="
        for c in items
    )


def split_reply(messages, _tokens, _seed):
    source = json.loads(messages[1]["content"])
    parts = len(source["durations"])
    beat = source["beat"].split(". ")
    dialogue = source["dialogue"].split(". ") if source["dialogue"] else []
    assert parts == 2
    return json.dumps({"parts": [
        dict(beat=beat[0] + ".", dialogue=dialogue[0] + "." if dialogue else "", handoff="Hold the first action."),
        dict(beat=beat[1], dialogue=dialogue[1] if dialogue else "", handoff=source["handoff"]),
    ]})


class DurationContractTests(unittest.TestCase):
    def repair(self, items, schedule, complete=split_reply, coverage=None):
        audit = {"outputs": [marked(items)], "repairs": []}
        result = nodes._repair_structure_duration_cards(items, tuple(schedule), coverage, complete, 4096, 7, audit)
        return (*result, audit)

    def assert_valid(self, durations, total):
        self.assertTrue(all(4 <= d <= 15 for d in durations))
        self.assertAlmostEqual(sum(durations), total, places=3)

    def test_180_and_240_host_duration_is_authoritative(self):
        for total in (180, 240):
            with self.subTest(total=total):
                schedule = nodes._director_duration_schedule(total, 10)
                source = cards([12] * len(schedule))
                fixed, durations, _, audit = self.repair(source, schedule, complete=lambda *a: self.fail("Unneeded repair"))
                segments, _, _ = nodes._segments_from_structure_cards(
                    fixed, R2V, durations, [1], [1], [1], [], locked_durations=True,
                )
                self.assertEqual(tuple(s.duration_seconds for s in segments), schedule)
                self.assert_valid(durations, total)
                self.assertEqual(audit["repairs"], [])

    def test_260_is_rejected_without_clamping(self):
        with self.assertRaisesRegex(nodes.PromptRefinerError, "260.*not clamped"):
            nodes._director_duration_schedule(260, 10)

    def test_schedule_does_not_make_an_under_four_second_tail(self):
        self.assertEqual(nodes._director_duration_schedule(7, 4), (7,))
        for total in (4, 7.999, 16, 30.001, 180.123, 239.999, 240):
            for average in (4, 7, 10, 15):
                with self.subTest(total=total, average=average):
                    self.assert_valid(nodes._director_duration_schedule(total, average), total)

    def test_story_beats_choose_group_count_and_max_is_only_a_ceiling(self):
        brief = """[STORY_BEATS]
1|45|1-8|arrival|
2|20|9-12|conversation|
3|30|13-22|departure|
4|40|23-32|search|
5|50|33-45|conflict|
6|30|46-55|aftermath|
7|45|56-60|ending|
[/STORY_BEATS]"""
        schedule, plan = nodes._story_duration_schedule(
            brief, budget_seconds=240, max_segment_seconds=15,
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan.card_counts, (3, 2, 2, 3, 4, 2, 3))
        self.assertEqual(len(schedule), 19)
        self.assert_valid(schedule, 240)
        self.assertTrue(all(value <= 15 for value in schedule))
        self.assertGreater(len(set(schedule)), 1)

    def test_story_complete_mode_may_expand_beyond_240_target(self):
        brief = """[STORY_BEATS]
1|100|1-10|opening|
2|100|11-20|middle|
3|110|21-30|ending|
[/STORY_BEATS]"""
        schedule, plan = nodes._story_duration_schedule(
            brief, budget_seconds=240, max_segment_seconds=15,
            allow_duration_overrun=True,
        )
        self.assertAlmostEqual(sum(schedule), 310, places=3)
        self.assertAlmostEqual(plan.budget_seconds, 310, places=3)
        self.assertFalse(plan.compressed)
        self.assertTrue(all(4 <= value <= 15 for value in schedule))

        compressed, compressed_plan = nodes._story_duration_schedule(
            brief, budget_seconds=240, max_segment_seconds=15,
            allow_duration_overrun=False,
        )
        self.assertAlmostEqual(sum(compressed), 240, places=3)
        self.assertTrue(compressed_plan.compressed)

    def test_story_complete_mode_has_a_600_second_safety_ceiling(self):
        brief = "[STORY_BEATS]\n1|301|1-2|opening|\n2|301|3-4|ending|\n[/STORY_BEATS]"
        with self.assertRaisesRegex(nodes.PromptRefinerError, "safety ceiling.*600"):
            nodes._story_duration_schedule(
                brief, budget_seconds=240, max_segment_seconds=15,
                allow_duration_overrun=True,
            )

    def test_explicit_final_shots_are_losslessly_bound_to_cards(self):
        brief = """[STORY_BEATS]
1|45|56-60|ending|final line
[/STORY_BEATS]
<Audio 1> 小程声线参考, <Audio 2> 柳二姐声线参考
分镜56使用参考:<Picture 1>,<Picture 2>
分镜56: 整理衣物。柳二姐「开始走吧。」
分镜57使用参考:<Picture 1>,<Picture 2>
分镜57: 两人沿村道行走。
分镜58使用参考:<Picture 2>
分镜58: 柳二姐停下。柳二姐「今天到此为止。」
分镜59使用参考:<Picture 1>,<Picture 2>
分镜59: 柳二姐贴近耳畔。
分镜60使用参考:<Picture 1>,<Picture 2>,<Picture 7>
分镜60: 柳二姐离开。小程「这里不会无聊。」"""
        beat = nodes._story_beat_from_fields(1, 45, 56, 60, "ending", "final line")
        plan = nodes._CoverageKnifePlan(
            beats=(beat,), budget_seconds=45, group_count=3, estimated_sum=45,
            compressed=False, allocations=(45,), card_counts=(3,), segment_to_beat=(1, 1, 1),
        )
        source = cards([15, 15, 15])
        result = nodes._apply_story_shot_contract(source, brief, plan)
        joined = "\n".join(card["beat"] for card in result)
        for shot in range(56, 61):
            self.assertEqual(joined.count(f"分镜{shot}使用参考"), 1)
        self.assertIn("分镜59", result[1]["beat"])
        self.assertIn("分镜60", result[2]["beat"])
        self.assertEqual(result[2]["assets"]["pictures"], [1, 2, 7])
        self.assertEqual(result[1]["assets"]["audios"], [2])
        self.assertEqual(result[2]["assets"]["audios"], [1])
        self.assertTrue(all(card["dialogue"] == "" for card in result))
        self.assertIn("这里不会无聊", result[2]["beat"])

    def test_declared_missing_shot_fails_with_diagnostic(self):
        brief = "[STORY_BEATS]\n1|10|57-58|ending|\n[/STORY_BEATS]\n分镜57使用参考:<Picture 1>\n分镜57: only"
        beat = nodes._story_beat_from_fields(1, 10, 57, 58, "ending", "")
        plan = nodes._CoverageKnifePlan(
            beats=(beat,), budget_seconds=10, group_count=1, estimated_sum=10,
            compressed=False, allocations=(10,), card_counts=(1,), segment_to_beat=(1,),
        )
        with self.assertRaisesRegex(nodes.PromptRefinerError, "missing: 58"):
            nodes._apply_story_shot_contract(cards([10]), brief, plan)

    def test_final_refine_cannot_drop_assigned_late_shots(self):
        prompt = r2v_prompt("[Brief Shot 59] The first assigned action. [Brief Shot 60] The final action.")
        accepted = nodes._finalize_director_segment_prompt(
            prompt, engine=nodes.FULL_REFERENCE_ENGINE, picture_slots=(1,),
            lora_triggers=(), index=19, required_shot_ids=(59, 60),
        )
        self.assertIn("[Brief Shot 60]", accepted)
        recovered = nodes._finalize_director_segment_prompt(
            r2v_prompt("[Brief Shot 59] Only one action."),
            engine=nodes.FULL_REFERENCE_ENGINE, picture_slots=(1,),
            lora_triggers=(), index=19, required_shot_ids=(59, 60),
            shot_contract_text=(
                "分镜59使用参考:<Picture 1>\n分镜59: first\n"
                "分镜60使用参考:<Picture 1>\n分镜60: final action"
            ),
        )
        self.assertIn("[Brief Shot 60] final action", recovered)
        self.assertLess(recovered.index("[Brief Shot 60]"), recovered.index("overall_soundscape:"))

    def test_native_english_shots_are_relabelled_without_duplicate_brief_text(self):
        prompt = r2v_prompt("[Shot 1] English action one. [Shot 2] English action two. [Shot 3] English action three.")
        contract = (
            "分镜1使用参考:<Picture 1>\n分镜1: 中文动作一\n"
            "分镜2使用参考:<Picture 1>\n分镜2: 中文动作二\n"
            "分镜3使用参考:<Picture 1>\n分镜3: 中文动作三"
        )
        result = nodes._finalize_director_segment_prompt(
            prompt, engine=nodes.FULL_REFERENCE_ENGINE, picture_slots=(1,),
            lora_triggers=(), index=1, required_shot_ids=(1, 2, 3),
            shot_contract_text=contract,
        )
        self.assertIn("[Brief Shot 1] English action one", result)
        self.assertIn("[Brief Shot 3] English action three", result)
        self.assertNotIn("中文动作", result)

    def test_global_scene_picture_is_added_only_to_matching_shots(self):
        brief = """[STORY_BEATS]
1|10|1-2|arrival|
[/STORY_BEATS]
<Picture 1> 小程, <Picture 5> 红菊村村口, <Picture 6> 通往村部的小路/苞米地边缘, <Picture 7> 村部门口
分镜1使用参考:<Picture 1>
分镜1: 小程站在红菊村村口。
分镜2使用参考:<Picture 1>
分镜2: 小程沿苞米地行走，最后抵达村部门口。"""
        beat = nodes._story_beat_from_fields(1, 10, 1, 2, "arrival", "")
        plan = nodes._CoverageKnifePlan(
            beats=(beat,), budget_seconds=10, group_count=2, estimated_sum=10,
            compressed=False, allocations=(10,), card_counts=(2,), segment_to_beat=(1, 1),
        )
        result = nodes._apply_story_shot_contract(cards([5, 5]), brief, plan)
        self.assertEqual(result[0]["assets"]["pictures"], [1, 5])
        self.assertEqual(result[1]["assets"]["pictures"], [1, 6, 7])

    def test_valid_contract_needs_no_repair_or_card_mutation(self):
        source = cards([8, 12, 10])
        fixed, durations, _, audit = self.repair(source, (8, 12, 10), lambda *a: self.fail("Unexpected call"))
        self.assertEqual(fixed, source)
        self.assertEqual(durations, (8, 12, 10))
        self.assertEqual(audit["repairs"], [])

    def test_speech_overflow_with_a_legal_duration_is_detected(self):
        source = cards([10, 10, 10])
        source[0]["dialogue"] = " ".join(["word"] * 36)
        with self.assertRaisesRegex(nodes.PromptRefinerError, "Local repair failed for segment 1, 16.000s"):
            self.repair(source, (10, 10, 10), lambda *a: '{"parts":[]}')

    def test_21_seconds_split_without_shortening_or_content_loss(self):
        source = cards([21, 9, 10])
        fixed, durations, _, audit = self.repair(source, (15, 12, 13))
        self.assert_valid(durations, 40)
        self.assertEqual(durations[:2], (10.5, 10.5))
        self.assertEqual(fixed[0]["continuity_raw"], source[0]["continuity_raw"])
        self.assertEqual(fixed[1]["continuity_raw"], "true")
        self.assertEqual(fixed[1]["handoff"], source[0]["handoff"])
        for field in ("beat", "dialogue"):
            self.assertEqual(" ".join(c[field] for c in fixed[:2]), source[0][field])
            self.assertEqual([c[field] for c in fixed[2:]], [c[field] for c in source[1:]])
        self.assertEqual([c["index"] for c in fixed], [1, 2, 3, 4])
        self.assertEqual(fixed[2]["continuity_raw"], source[1]["continuity_raw"])
        self.assertEqual([c["assets"] for c in fixed], [source[0]["assets"]] * 4)
        self.assertEqual(len(audit["repairs"]), 1)

    def test_multiple_oversized_cards_and_short_tail(self):
        source = cards([21, 16, 10, 10])
        fixed, durations, _, audit = self.repair(source, (15,) * 4)
        self.assert_valid(durations, 60)
        self.assertEqual(durations[:4], (10.5, 10.5, 8.0, 8.0))  # No 15+1 tail.
        self.assertEqual(len(fixed), 6)
        self.assertEqual(len(audit["repairs"]), 2)
        for field in ("beat", "dialogue"):
            self.assertEqual(" ".join(c[field] for c in fixed), " ".join(c[field] for c in source))

    def test_segment_nine_overflow_in_180_and_240_second_plans(self):
        for total in (180, 240):
            with self.subTest(total=total):
                schedule = nodes._director_duration_schedule(total, 10)
                source = cards([10] * len(schedule))
                source[8]["duration_raw"] = "21"
                fixed, durations, _, audit = self.repair(source, schedule)
                self.assert_valid(durations, total)
                self.assertEqual(durations[8:10], (10.5, 10.5))
                self.assertEqual(audit["repairs"][0]["segment"], 9)
                self.assertEqual(len(fixed), len(source) + 1)
                for field in ("beat", "dialogue"):
                    self.assertEqual(" ".join(c[field] for c in fixed), " ".join(c[field] for c in source))

    def test_all_cards_overflow_but_exact_total_is_feasible(self):
        fixed, durations, _, _ = self.repair(cards([16, 16, 16]), (12,) * 4)
        self.assert_valid(durations, 48)
        self.assertEqual(len(fixed), 6)

    def test_infeasible_repair_reports_segment_duration_and_reason(self):
        with self.assertRaisesRegex(nodes.PromptRefinerError, "segment 1: 21.000s.*target=20.000s"):
            self.repair(cards([21, 4]), (10, 10), lambda *a: self.fail("No budget for model repair"))

    def test_repair_cannot_starve_other_dialogue(self):
        source = cards([21, 10, 10])
        source[1]["dialogue"] = " ".join(["word"] * 20)
        with self.assertRaisesRegex(nodes.PromptRefinerError, "dialogue would not fit"):
            self.repair(source, (10, 10, 10), lambda *a: self.fail("Infeasible speech budget"))

    def test_bad_partitions_rejected_with_two_local_attempts(self):
        for bad in ("drop", "duplicate", "reorder"):
            calls = []
            def complete(messages, tokens, seed):
                reply = json.loads(split_reply(messages, tokens, seed))
                if bad == "drop":
                    reply["parts"][1]["dialogue"] = ""
                elif bad == "duplicate":
                    reply["parts"][1]["beat"] = reply["parts"][0]["beat"]
                else:
                    reply["parts"].reverse()
                calls.append(messages)
                return json.dumps(reply)
            with self.subTest(bad=bad), self.assertRaisesRegex(nodes.PromptRefinerError, "segment 1, 21.000s"):
                self.repair(cards([21, 9, 10]), (15, 12, 13), complete)
            self.assertEqual(len(calls), 2)

    def test_coverage_membership_preserved_after_split(self):
        brief = "[STORY_BEATS]\n1|30|1-2|entry|\n2|10|3-4|exit|\n[/STORY_BEATS]"
        coverage = nodes._build_coverage_knife_plan(brief, budget_seconds=40, group_count=3)
        source = cards([21, 10, 9])
        source[0]["beat"] = "entry. Arrival."
        source[2]["beat"] = "exit. Departure."
        fixed, durations, updated, _ = self.repair(source, (15, 15, 10), coverage=coverage)
        self.assert_valid(durations, 40)
        self.assertEqual(updated.segment_to_beat, (1, 1, 1, 2))
        self.assertEqual(updated.card_counts, (3, 1))
        self.assertEqual(nodes._validate_structure_coverage(fixed, updated), [])

    def test_duration_parser_does_not_turn_minus_or_range_into_positive_seconds(self):
        for raw in ("-10", "10-21", "nan", "infinity", "garbage10", "0", "21"):
            with self.subTest(raw=raw), self.assertRaises(nodes.PromptRefinerError):
                nodes._parse_duration_token(raw, 9)

    def test_full_pipeline_local_repair_records_raw_and_keeps_packing_contract(self):
        calls = []
        def complete(messages, tokens, seed):
            system = messages[0]["content"]
            calls.append(system)
            if "STRUCTURE mode" in system:
                self.assertIn("LOCKED DURATION SCHEDULE", system)
                return marked(cards([21, 9, 10]))
            if "Repair only" in system:
                return split_reply(messages, tokens, seed)
            return r2v_prompt("Walk")
        with tempfile.TemporaryDirectory() as directory, patch.object(nodes.tempfile, "tempdir", directory):
            _, segments, repaired, debug, record = nodes._plan_director_with_independent_segment_refine(
                complete, mode=R2V, brief="Enter, talk, then exit.", reference_text="", schedule=(15, 12, 13),
                image_references=((1, "image"),),
                director_videos=(nodes._DirectorVideoFrames(slot=1, frames="video", frame_uris=()),),
                audio_slots=(), video_audio_slots=(1,),
                lora_triggers=(), prompt_example=None, quality=BALANCED, requested_max_tokens=4096, seed=7,
            )
            self.assertTrue(repaired)
            self.assertEqual(record["structure_attempts"], 1)
            self.assertEqual(record["structure_repairs"], 1)
            saved = json.loads(Path(record["structure_diagnostic"]).read_text(encoding="utf-8"))
            self.assertIn("DURATION_SECONDS: 21", saved["outputs"][0])
            self.assertEqual(sum("STRUCTURE mode" in s for s in calls), 1)
            self.assertEqual(len(segments), 4)
            self.assert_valid([s.duration_seconds for s in segments], 40)
            self.assertTrue(all(isinstance(s, nodes._DirectorLongSegment) for s in segments))

    def test_failure_logs_raw_without_regenerating_all_cards(self):
        calls = []
        def complete(*args):
            calls.append(args)
            return marked(cards([21, 4])).replace("PICTURES: 1", "PICTURES:").replace(
                "VIDEOS: 1", "VIDEOS:").replace("VIDEO_AUDIOS: 1", "VIDEO_AUDIOS:")
        with tempfile.TemporaryDirectory() as directory, patch.object(nodes.tempfile, "tempdir", directory):
            with self.assertRaisesRegex(nodes.PromptRefinerError, "Raw structure diagnostic"):
                nodes._plan_director_with_independent_segment_refine(
                    complete, mode=R2V, brief="Small story.", reference_text="", schedule=(10, 10),
                    image_references=(), director_videos=(), audio_slots=(), video_audio_slots=(),
                    lora_triggers=(), prompt_example=None, quality=BALANCED, requested_max_tokens=4096, seed=7,
                )
            self.assertEqual(len(calls), 1)
            logs = list(Path(directory).glob("qwen38-director-structure-*.json"))
            self.assertEqual(len(logs), 1)
            saved = json.loads(logs[0].read_text(encoding="utf-8"))
            self.assertIn("DURATION_SECONDS: 21", saved["outputs"][0])
            self.assertIn("segment 1", saved["error"])


if __name__ == "__main__":
    unittest.main()

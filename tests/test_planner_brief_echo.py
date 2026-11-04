"""Planner detailed_description must not keep a raw brief dump in Brief Shot fields.

Group 2 can leave an early shot in English and then paste later 分镜 blocks:
picture-name lists, 特写/中近景 cues, and the unrefined brief. Those bodies are
rejected so the segment refine retries instead of publishing the dump.
"""

from __future__ import annotations

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
BALANCED = nodes.DIRECTOR_LONG_QUALITY_PROFILES[nodes.DIRECTOR_LONG_QUALITY_BALANCED]

BRIEF = """分镜5使用参考:<Picture 1> 小程
分镜5: 小程站在门口。
分镜6使用参考:<Picture 1> 小程 <Picture 2> 柳二姐
分镜6: 中近景。柳二姐站在画面中央，正对镜头。特写鞋面。
<Picture 2> 柳二姐 <Picture 1> 小程 <Picture 2> 柳二姐的卧室 小程
分镜7使用参考:<Picture 2> 柳二姐
分镜7: 特写。手停在门把上，正对镜头。
<Picture 2> 柳二姐 <Picture 1> 小程 <Picture 1> 小程
分镜8使用参考:<Picture 1> 小程 <Picture 2> 柳二姐
分镜8: 全景。两人停在床边，特写转到门口。
<Picture 1> 小程 <Picture 2> 柳二姐 <Picture 1> 小程
"""

DUMP_MARK = "<Picture 2> 柳二姐 <Picture 1> 小程 <Picture 2> 柳二姐的卧室 小程"
ENGLISH_SHOTS = (
    "[Brief Shot 5] At 00:00.000, the camera holds on the man in the doorway.\n"
    "[Brief Shot 6] At 00:03.000, a medium close-up finds her at the center of the frame.\n"
    "[Brief Shot 7] At 00:06.000, the lens drops to her hand on the latch.\n"
    "[Brief Shot 8] At 00:09.000, a wide view settles both figures beside the bed."
)


def r2v(detail: str) -> str:
    return (
        "subject_definitions:\n"
        "<Picture 1> 小程, fully_preserved.\n"
        "\n"
        "summary:\n"
        "The doorway beat continues from the previous group.\n"
        "\n"
        "retention_analysis:\n"
        "<Picture 1>: fully_preserved.\n"
        "<Picture 2>: fully_preserved.\n"
        "\n"
        "detailed_description:\n"
        f"{detail}\n"
        "\n"
        "overall_soundscape:\n"
        "Room tone and a faint electrical hum.\n"
        "\n"
        "non_diegetic_music:\n"
        "N/A"
    )


class PlannerBriefShotDumpTests(unittest.TestCase):
    def test_picture_name_pile_is_a_raw_dump_and_a_short_action_is_not(self) -> None:
        block = (
            "分镜6使用参考:<Picture 1> 小程 <Picture 2> 柳二姐\n"
            "分镜6: 中近景。柳二姐站在画面中央，正对镜头。特写鞋面。\n"
            f"{DUMP_MARK}"
        )
        body = "中近景。柳二姐站在画面中央，正对镜头。特写鞋面。\n" + DUMP_MARK
        self.assertTrue(nodes._shot_body_is_raw_brief_dump(body, block))
        self.assertFalse(
            nodes._shot_body_is_raw_brief_dump(
                "final action",
                "分镜60使用参考:<Picture 1>\n分镜60: final action",
            )
        )
        self.assertFalse(
            nodes._shot_body_is_raw_brief_dump(
                "At 00:03.000, a medium close-up finds her at the center of the frame.",
                block,
            )
        )

    def test_omitted_later_shots_are_not_filled_with_the_brief(self) -> None:
        with self.assertRaises(nodes.PromptRefinerError) as caught:
            nodes._finalize_director_segment_prompt(
                r2v("[Brief Shot 5] At 00:00.000, the camera holds on the man in the doorway."),
                engine=nodes.FULL_REFERENCE_ENGINE,
                picture_slots=(1, 2),
                lora_triggers=(),
                index=2,
                required_shot_ids=(5, 6, 7, 8),
                shot_contract_text=BRIEF,
            )
        message = str(caught.exception)
        self.assertIn("raw brief", message)
        self.assertIn("6", message)
        self.assertNotIn(DUMP_MARK, message)

    def test_model_written_brief_shot_dump_is_rejected(self) -> None:
        detail = (
            "[Brief Shot 5] At 00:00.000, the camera holds on the man in the doorway.\n"
            "[Brief Shot 6] 中近景。柳二姐站在画面中央，正对镜头。特写鞋面。\n"
            f"{DUMP_MARK}"
        )
        with self.assertRaises(nodes.PromptRefinerError) as caught:
            nodes._finalize_director_segment_prompt(
                r2v(detail),
                engine=nodes.FULL_REFERENCE_ENGINE,
                picture_slots=(1, 2),
                lora_triggers=(),
                index=2,
                required_shot_ids=(5, 6),
                shot_contract_text=BRIEF,
            )
        self.assertIn("raw brief", str(caught.exception))
        self.assertIn("6", str(caught.exception))

    def test_english_brief_shots_with_one_picture_label_stay(self) -> None:
        detail = (
            "[Brief Shot 5] At 00:00.000, the camera holds on <Picture 1> as he waits in the doorway.\n"
            "[Brief Shot 6] At 00:03.000, a medium close-up finds her at the center of the frame."
        )
        prompt = nodes._finalize_director_segment_prompt(
            r2v(detail),
            engine=nodes.FULL_REFERENCE_ENGINE,
            picture_slots=(1, 2),
            lora_triggers=(),
            index=2,
            required_shot_ids=(5, 6),
            shot_contract_text=BRIEF,
        )
        narrative = nodes._h3_field_body(prompt, "detailed_description:")
        self.assertIn("[Brief Shot 5]", narrative)
        self.assertIn("medium close-up", narrative)
        self.assertNotIn("中近景", narrative)
        self.assertNotIn(DUMP_MARK, narrative)

    def test_segment_request_forbids_pasting_the_brief(self) -> None:
        segment = nodes._DirectorLongSegment(
            index=1,
            duration_seconds=12.0,
            prompt="",
            picture_slots=(1, 2),
            video_slots=(),
            video_audio_slots=(),
            audio_slots=(),
            continuity_from_prev=True,
            beat=BRIEF,
            dialogue="",
            handoff="Her hand stays on the latch.",
        )
        messages = nodes._build_long_director_segment_request(
            R2V,
            BRIEF,
            "",
            (segment,),
            segment,
            ((1, "pic-1"), (2, "pic-2")),
            (),
            (),
            None,
            BALANCED,
        )
        text = messages[1]["content"][0]["text"]
        self.assertTrue(text.startswith("/no_think\n"))
        self.assertIn("Do not paste the brief", text)
        self.assertIn("[Brief Shot 6]", text)
        self.assertIn("Segment 1 of 1", text)

    def test_segment_refine_retries_instead_of_publishing_the_dump(self) -> None:
        calls: list[str] = []

        def complete(messages, _max_tokens, _seed):
            user = messages[-1]["content"]
            if isinstance(user, list):
                user = user[0]["text"]
            calls.append(user)
            if len(calls) == 1:
                return r2v("[Brief Shot 5] At 00:00.000, the camera holds on the man in the doorway.")
            self.assertIn("raw brief", user)
            return r2v(ENGLISH_SHOTS)

        _title, segments, _rebalanced, _debug, record = nodes._plan_director_with_independent_segment_refine(
            complete,
            mode=R2V,
            brief=BRIEF,
            reference_text="",
            schedule=(12.0,),
            image_references=((1, "pic-1"), (2, "pic-2")),
            director_videos=(),
            audio_slots=(),
            video_audio_slots=(),
            lora_triggers=(),
            prompt_example=None,
            quality=BALANCED,
            requested_max_tokens=4096,
            seed=2,
        )
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0].startswith("/no_think\n"))
        narrative = nodes._h3_field_body(segments[0].prompt, "detailed_description:")
        self.assertIn("[Brief Shot 6]", narrative)
        self.assertIn("[Brief Shot 8]", narrative)
        self.assertIn("medium close-up", narrative)
        self.assertNotIn(DUMP_MARK, narrative)
        self.assertNotIn("中近景", narrative)
        self.assertNotIn("使用参考", narrative)
        self.assertEqual(record["passes"][0]["attempts"], 2)


if __name__ == "__main__":
    unittest.main()

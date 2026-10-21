"""Unit tests for nvstudio-beat-pack parse / clear allowlist."""

from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import beat_pack


def _valid_manifest(**overrides):
    data = {
        "format": "nvstudio-beat-pack",
        "version": 1,
        "beat_id": "beat-1",
        "brief": "Hero walks. @reference_image_1",
        "external_total_duration_seconds": 40,
        "segment_duration_seconds": 10,
        "refs": [
            {"slot": "reference_image_1", "path": "refs/hero.png", "kind": "image"},
        ],
    }
    data.update(overrides)
    return data


class BeatPackParseTests(unittest.TestCase):
    def test_parse_defaults(self):
        parsed = beat_pack.parse_beat_pack_manifest(_valid_manifest())
        self.assertEqual(parsed["format"], beat_pack.BEAT_PACK_FORMAT)
        self.assertTrue(parsed["clear_director_groups"])
        self.assertFalse(parsed["allow_duration_overrun"])
        self.assertEqual(
            parsed["director_execution"],
            "Preview plan only (Director blocked)",
        )

    def test_reject_wrong_format(self):
        with self.assertRaises(ValueError):
            beat_pack.parse_beat_pack_manifest(_valid_manifest(format="mmxpack"))

    def test_reject_newer_version(self):
        with self.assertRaises(ValueError):
            beat_pack.parse_beat_pack_manifest(_valid_manifest(version=2))

    def test_reject_bad_segment(self):
        with self.assertRaises(ValueError):
            beat_pack.parse_beat_pack_manifest(_valid_manifest(segment_duration_seconds=20))

    def test_clear_allowlist_exact(self):
        self.assertEqual(
            beat_pack.CLEAR_DIRECTOR_GROUP_CLASS_TYPES,
            frozenset({
                "MiniMaxH3DirectorGroupImageToVideo",
                "MiniMaxH3DirectorGroupReferenceToVideo",
                "MiniMaxH3DirectorGroupsCombine",
            }),
        )
        banned = {
            "MiniMaxH3Director",
            "Qwen38LongVideoDirectorPlanner",
            "LoadImage",
            "VAEDecode",
            "KSampler",
            "UNETLoader",
            "CLIPLoader",
        }
        self.assertTrue(banned.isdisjoint(beat_pack.CLEAR_DIRECTOR_GROUP_CLASS_TYPES))

    def test_resolve_refs_relative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "refs").mkdir()
            media = root / "refs" / "hero.png"
            media.write_bytes(b"png")
            path = beat_pack.resolve_ref_path("refs/hero.png", root)
            self.assertEqual(path, media.resolve())

    def test_load_zip_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack_dir = root / "pack"
            (pack_dir / "refs").mkdir(parents=True)
            (pack_dir / "refs" / "hero.png").write_bytes(b"png")
            (pack_dir / "manifest.json").write_text(
                json.dumps(_valid_manifest()),
                encoding="utf-8",
            )
            zip_path = root / "beat.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.write(pack_dir / "manifest.json", "manifest.json")
                zf.write(pack_dir / "refs" / "hero.png", "refs/hero.png")
            extracted = root / "extracted"
            extracted.mkdir()
            manifest, pack_root = beat_pack.load_beat_pack(zip_path, extracted)
            self.assertEqual(manifest["beat_id"], "beat-1")
            self.assertTrue((pack_root / "refs" / "hero.png").is_file())
            input_root = root / "input"
            input_root.mkdir()
            result = beat_pack.import_media(manifest, pack_root, input_root)
            self.assertEqual(result["refs"][0]["slot"], "reference_image_1")
            self.assertTrue(Path(result["refs"][0]["path"]).is_file())
            self.assertTrue(result["refs"][0]["input_path"].startswith("qwen38_beat_packs/"))




class TitleAliasTests(unittest.TestCase):
    def test_aliases_image(self):
        slot = "reference_image_2"
        self.assertEqual(
            beat_pack.slot_title_aliases(slot),
            ["Ref_i_2", "Ref 2", "Ref_2"],
        )
        for title in (
            "reference_image_2",
            "Load reference_image_2",
            "Ref_i_2",
            "hero Ref_i_2 still",
            "Ref 2",
            "Ref_2",
            "NV Ref_2 plate",
        ):
            self.assertTrue(beat_pack.title_matches_slot(title, slot), title)

    def test_aliases_video_audio_not_generic_ref(self):
        self.assertEqual(beat_pack.slot_title_aliases("reference_video_3"), ["Ref_v_3"])
        self.assertEqual(beat_pack.slot_title_aliases("reference_audio_1"), ["Ref_a_1"])
        self.assertTrue(beat_pack.title_matches_slot("Ref_v_3", "reference_video_3"))
        self.assertTrue(beat_pack.title_matches_slot("clip Ref_v_3", "reference_video_3"))
        self.assertTrue(beat_pack.title_matches_slot("Ref_a_1", "reference_audio_1"))
        # Generic Ref N / Ref_N is image-only.
        self.assertFalse(beat_pack.title_matches_slot("Ref 3", "reference_video_3"))
        self.assertFalse(beat_pack.title_matches_slot("Ref_3", "reference_video_3"))
        self.assertFalse(beat_pack.title_matches_slot("Ref 1", "reference_audio_1"))
        self.assertFalse(beat_pack.title_matches_slot("Ref_1", "reference_audio_1"))
        # Cross-kind typed aliases must not collide.
        self.assertFalse(beat_pack.title_matches_slot("Ref_i_3", "reference_video_3"))
        self.assertFalse(beat_pack.title_matches_slot("Ref_v_2", "reference_image_2"))

    def test_unknown_slot_has_no_aliases(self):
        self.assertEqual(beat_pack.slot_title_aliases("reference_video_frames_1"), [])
        self.assertFalse(beat_pack.title_matches_slot("Ref_v_1", "reference_video_frames_1"))


if __name__ == "__main__":
    unittest.main()

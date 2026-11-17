"""CPU-only tests against read-only Director sources; no Comfy startup."""

import importlib
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

ROOT = Path(__file__).resolve().parents[1]
DIRECTOR = ROOT.parent / "ComfyUI_MiniMaxH3_Director" / "director"
sys.path.insert(0, str(ROOT))
import director_face_refine_mem as memory


def package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


package("mem_reference", DIRECTOR / "face_refine")
reference = importlib.import_module("mem_reference.stitch")
package("mem_test", DIRECTOR.parent)
package("mem_test.director", DIRECTOR)
package("mem_test.director.face_refine", DIRECTOR / "face_refine")
memory.register_patch()
stitch = importlib.import_module("mem_test.director.face_refine.stitch")
inject = importlib.import_module("mem_test.director.face_refine.inject")


class MemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.base = torch.rand(40, 8, 8, 3)
        self.crops = torch.rand(40, 8, 8, 3)
        self.transform = {"boxes": [(0, 0, 8, 8)] * 40, "canvas": (8, 8), "src_size": (8, 8)}

    def test_stitch_equal_without_clone(self):
        for region in ("full_crop", "face_only", "face_ellipse"):
            pack = {"paste_region": region}
            expected = reference.stitch_faces(self.base, self.crops, self.transform, pack)
            working = self.base.clone()
            with patch.object(torch.Tensor, "clone", side_effect=RuntimeError("not enough memory")), patch.object(stitch, "_promptrefiner_release_models") as cleanup:
                actual = stitch.stitch_faces(working, self.crops, self.transform, pack)
            cleanup.assert_called_once()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(actual.data_ptr(), working.data_ptr())

    def test_empty_stitch_without_clone(self):
        with patch.object(torch.Tensor, "clone", side_effect=RuntimeError("not enough memory")), patch.object(stitch, "_promptrefiner_release_models"):
            actual = stitch.stitch_faces(self.base, self.crops[:0], self.transform, {})
        self.assertEqual(actual.data_ptr(), self.base.data_ptr())

    def test_fade_equal_edges_only_without_clone(self):
        for length, head, tail in ((40, 5, 6), (30, 5, 6), (40, 30, 30), (40, 0, 0)):
            base = self.base[:length]
            expected = reference.fade_stitch_at_seams(self.crops, base, head_frames=head, tail_frames=tail)
            working = self.crops.clone()
            with patch.object(torch.Tensor, "clone", side_effect=RuntimeError("not enough memory")):
                actual = stitch.fade_stitch_at_seams(working, base, head_frames=head, tail_frames=tail)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if head == 5:
                torch.testing.assert_close(actual[5:length - 6], self.crops[5:length - 6], rtol=0, atol=0)

    def test_independent_base_preserves_source(self):
        original = self.base.clone()
        with patch.object(torch.Tensor, "clone", side_effect=RuntimeError("not enough memory")):
            base = memory._independent_base(self.base)
        self.assertNotEqual(base.data_ptr(), self.base.data_ptr())
        base.zero_()
        torch.testing.assert_close(self.base, original)

    def test_inject_cleanup_before_unchanged_encode(self):
        events = []

        class NestedTensor:
            def __init__(self, members):
                self.members = members

            def unbind(self):
                return self.members

        comfy = types.ModuleType("comfy")
        nested = types.ModuleType("comfy.nested_tensor")
        nested.NestedTensor = NestedTensor
        comfy.nested_tensor = nested
        encoded = torch.rand(1, 3, 40, 8, 8)

        class VAE:
            def encode(inner, images):
                events.append("encode")
                torch.testing.assert_close(images, self.base)
                return encoded

        latent = {"samples": NestedTensor((torch.zeros_like(encoded), torch.ones(1))), "extra": 42}
        with patch.dict(sys.modules, {"comfy": comfy, "comfy.nested_tensor": nested}), patch.object(inject, "_promptrefiner_release_models", side_effect=lambda: events.append("cleanup")):
            actual = inject.inject_video_latent(latent, self.base, VAE())
        self.assertEqual(events, ["cleanup", "encode"])
        self.assertIs(actual["samples"].members[0], encoded)
        self.assertIs(actual["samples"].members[1], latent["samples"].members[1])
        self.assertEqual(actual["extra"], 42)

    def test_runtime_rewrite_and_existing_alias(self):
        # Compile just this function to avoid importing Director's application.
        import ast
        source = (DIRECTOR / "face_refine" / "runtime.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "apply_segment_face_refine"]
        module = types.ModuleType("alias_test.director.face_refine.runtime")
        module.__dict__.update(torch=torch, Any=object)
        exec(compile(tree, str(DIRECTOR / "face_refine" / "runtime.py"), "exec"), module.__dict__)
        alias = module.apply_segment_face_refine
        memory._patch(module)
        self.assertIs(alias, module.apply_segment_face_refine)
        self.assertIn("_promptrefiner_independent_base", alias.__code__.co_names)
        memory._patch(module)


if __name__ == "__main__":
    unittest.main(verbosity=2)

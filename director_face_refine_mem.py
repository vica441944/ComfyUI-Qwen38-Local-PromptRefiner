"""Late, process-local memory fixes; Director source files stay untouched."""

import gc
import importlib.abc
import importlib.machinery
import inspect
import logging
import sys


log = logging.getLogger(__name__)
_TARGETS = {
    "director.face_refine.stitch",
    "director.face_refine.inject",
    "director.face_refine.runtime",
}
_patched = set()


def _release_models():
    gc.collect()
    try:
        import comfy.model_management as mm
    except ImportError:
        return
    mm.unload_all_models()
    mm.soft_empty_cache()
    gc.collect()


def _independent_base(frames):
    base = frames[..., :3].contiguous().float().cpu()
    if base.untyped_storage().data_ptr() == frames.untyped_storage().data_ptr():
        out = base.new_empty(base.shape)
        for start in range(0, len(base), 32):
            out[start:start + 32].copy_(base[start:start + 32])
        return out
    return base


def _rewrite(function, replacements):
    source = inspect.getsource(function)
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError(f"Director {function.__name__} source changed: {before!r}")
        source = source.replace(before, after, 1)
    namespace = {}
    exec(compile(source, function.__code__.co_filename, "exec"), function.__globals__, namespace)
    return namespace[function.__name__].__code__


def _patch(module):
    name = module.__name__
    if name in _patched:
        return
    changes = []
    try:
        if name.endswith("director.face_refine.stitch"):
            changes.append((module.stitch_faces, _rewrite(module.stitch_faces, [
                ('    boxes = transform["boxes"]', '    _promptrefiner_release_models()\n    boxes = transform["boxes"]'),
                ('return base_images[..., :3].clone()', 'return base_images[..., :3]'),
                ('out = base_images[..., :3].clone()', 'out = base_images[..., :3]'),
            ])))
            changes.append((module.fade_stitch_at_seams, _rewrite(module.fade_stitch_at_seams, [
                ('out = stitched[:n, ..., :3].clone()', 'out = stitched[:n, ..., :3]'),
                ('src = base[:n, ..., :3].to(device=out.device, dtype=out.dtype)', 'src = base[:n, ..., :3]'),
                ('src[:head] * w', 'src[:head].to(device=out.device, dtype=out.dtype) * w'),
                ('src[-tail:] * w', 'src[-tail:].to(device=out.device, dtype=out.dtype) * w'),
                ('merged = stitched.clone()', 'merged = stitched'),
            ])))
        elif name.endswith("director.face_refine.inject"):
            changes.append((module.inject_video_latent, _rewrite(module.inject_video_latent, [
                ('    encoded = vae.encode(images[..., :3])', '    _promptrefiner_release_models()\n    encoded = vae.encode(images[..., :3])'),
            ])))
        elif name.endswith("director.face_refine.runtime"):
            changes.append((module.apply_segment_face_refine, _rewrite(module.apply_segment_face_refine, [
                ('base = frames[..., :3].contiguous().float().cpu()', 'base = _promptrefiner_independent_base(frames)'),
            ])))
        else:
            return
    except (OSError, TypeError, ValueError, SyntaxError, AttributeError) as exc:
        log.warning("PromptRefiner: Director memory patch not applied: %s", exc)
        return
    module._promptrefiner_release_models = _release_models
    module._promptrefiner_independent_base = _independent_base
    for function, code in changes:
        # Preserve references already imported by runtime/executor_core.
        function.__code__ = code
    _patched.add(name)
    if name.endswith("director.face_refine.stitch"):
        log.info("PromptRefiner: patched Director face-refine stitch to avoid full-clip CPU clones")
    else:
        log.info("PromptRefiner: patched %s memory handling", name)


class _Loader(importlib.abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not any(fullname == suffix or fullname.endswith('.' + suffix) for suffix in _TARGETS):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader)
        return spec


def register_patch():
    if not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    for name, module in list(sys.modules.items()):
        if module is not None and any(name == suffix or name.endswith('.' + suffix) for suffix in _TARGETS):
            _patch(module)

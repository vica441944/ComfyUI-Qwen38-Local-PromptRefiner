"""Local NVStudio beat-pack validation and media import."""

import asyncio
import json
import math
from pathlib import Path, PureWindowsPath
import re
import shutil
import tempfile
import zipfile


BEAT_PACK_FORMAT = "nvstudio-beat-pack"
CLEAR_DIRECTOR_GROUP_CLASS_TYPES = frozenset({
    "MiniMaxH3DirectorGroupImageToVideo",
    "MiniMaxH3DirectorGroupReferenceToVideo",
    "MiniMaxH3DirectorGroupsCombine",
})
DIRECTOR_EXECUTION_CHOICES = (
    "Preview plan only (Director blocked)",
    "Release cached plan to Director",
    "Plan and run Director (one queue)",
)


def parse_beat_pack_manifest(data: dict) -> dict:
    if not isinstance(data, dict) or data.get("format") != BEAT_PACK_FORMAT:
        raise ValueError("format must be nvstudio-beat-pack")
    if type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("Only beat pack version 1 is supported")
    result = dict(data)
    if not isinstance(result.get("brief"), str):
        raise ValueError("brief must be a string")
    for name, minimum, maximum in (
        ("external_total_duration_seconds", 4, math.inf),
        ("segment_duration_seconds", 4, 15),
    ):
        value = result.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"Invalid {name}: expected {minimum}–{maximum} seconds")
    result.setdefault("director_execution", DIRECTOR_EXECUTION_CHOICES[0])
    if result["director_execution"] not in DIRECTOR_EXECUTION_CHOICES:
        raise ValueError("Unknown director_execution choice")
    for name, default in (("clear_director_groups", True), ("allow_duration_overrun", False)):
        result.setdefault(name, default)
        if type(result[name]) is not bool:
            raise ValueError(f"{name} must be a boolean")
    refs = result.get("refs", [])
    if not isinstance(refs, list):
        raise ValueError("refs must be a list")
    slots = set()
    for ref in refs:
        if not isinstance(ref, dict) or not isinstance(ref.get("slot"), str) or not re.fullmatch(r"reference_(image|video|audio)_\d+", ref["slot"]):
            raise ValueError("Invalid reference slot")
        if ref["slot"] in slots:
            raise ValueError(f"Duplicate reference slot: {ref['slot']}")
        slots.add(ref["slot"])
        if ref.get("kind") not in ("image", "video", "audio") or not ref["slot"].startswith(f"reference_{ref['kind']}_"):
            raise ValueError("Reference kind must match its slot")
        if not isinstance(ref.get("path"), str) or not ref["path"].strip():
            raise ValueError("Reference path must be a nonempty string")
    result["refs"] = [dict(ref) for ref in refs]
    return result


def _contained_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative.replace("\\", "/")).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes pack/input directory: {relative}")
    return path


def resolve_ref_path(path, pack_root, input_root=None):
    """Resolve pack refs, explicit absolute paths, or Comfy input-relative paths."""
    normalized = path.replace("\\", "/")
    if Path(path).is_absolute():
        return Path(path).resolve()
    if PureWindowsPath(path).drive:
        raise ValueError(f"Unsupported absolute or drive-relative path: {path}")
    if normalized.startswith("refs/"):
        candidate = _contained_path(Path(pack_root) / "refs", normalized[5:])
        if candidate.exists() or input_root is None:
            return candidate
    if input_root is None:
        raise ValueError("Relative pack media must be under refs/")
    return _contained_path(input_root, normalized)


def load_beat_pack(source, extraction_root):
    """Load a folder, manifest, or ZIP into an explicitly owned temporary root."""
    source = Path(source)
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            for info in archive.infolist():
                _contained_path(extraction_root, info.filename)
                if PureWindowsPath(info.filename).drive or (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("ZIP contains an unsafe path or symbolic link")
            archive.extractall(extraction_root)
        manifests = list(Path(extraction_root).rglob("manifest.json"))
        if len(manifests) != 1:
            raise ValueError("Pack must contain exactly one manifest.json")
        source = manifests[0]
    elif source.is_dir():
        source = source / "manifest.json"
    return parse_beat_pack_manifest(json.loads(source.read_text(encoding="utf-8-sig"))), source.parent


def import_media(manifest, pack_root, input_root):
    input_root = Path(input_root).resolve()
    sources = [(ref, resolve_ref_path(ref["path"], pack_root, input_root)) for ref in manifest["refs"]]
    for ref, source in sources:
        if not source.is_file():
            raise ValueError(f"Missing media for {ref['slot']}: {ref['path']}")
    destination = None
    refs = []
    try:
        for ref, source in sources:
            if source.is_relative_to(input_root):
                target = source
            else:
                if destination is None:
                    parent = input_root / "qwen38_beat_packs"
                    parent.mkdir(parents=True, exist_ok=True)
                    destination = Path(tempfile.mkdtemp(prefix="beat_", dir=parent))
                target = destination / f"{ref['slot']}{source.suffix}"
                shutil.copyfile(source, target)
            refs.append({**ref, "input_path": target.relative_to(input_root).as_posix(), "path": str(target)})
    except OSError:
        if destination is not None:
            shutil.rmtree(destination)
        raise
    return {"manifest": manifest, "refs": refs}



def slot_title_aliases(slot: str) -> list[str]:
    """Return Load* title aliases for a canonical reference_* slot.

    Canonical remains reference_image_N / reference_video_N / reference_audio_N.
    Aliases are only for finding Load* nodes by title:
      Ref_i_N ↔ image, Ref_v_N ↔ video, Ref_a_N ↔ audio
      "Ref N" / "Ref_N" → image only (video/audio use Ref_v_ / Ref_a_)
    """
    match = re.fullmatch(r"reference_(image|video|audio)_(\d+)", str(slot or "").strip(), flags=re.IGNORECASE)
    if not match:
        return []
    kind = match.group(1).lower()
    index = match.group(2)
    if kind == "image":
        return [f"Ref_i_{index}", f"Ref {index}", f"Ref_{index}"]
    if kind == "video":
        return [f"Ref_v_{index}"]
    if kind == "audio":
        return [f"Ref_a_{index}"]
    return []


def _alias_pattern(alias: str) -> re.Pattern[str]:
    """Build a case-insensitive whole-token pattern for one alias."""
    text = str(alias)
    if re.fullmatch(r"Ref \d+", text, flags=re.IGNORECASE):
        index = text.split()[-1]
        body = rf"Ref[\s_]+{re.escape(index)}"
    else:
        body = re.escape(text)
    return re.compile(rf"(?:^|[^A-Za-z0-9_]){body}(?:[^A-Za-z0-9_]|$)", flags=re.IGNORECASE)


def title_matches_slot(title: str, slot: str) -> bool:
    """True when a Load* node title matches canonical slot or an allowed alias."""
    raw_title = str(title or "").strip()
    raw_slot = str(slot or "").strip()
    if not raw_title or not raw_slot:
        return False
    if raw_title == raw_slot:
        return True
    # Preserve prior behavior: word-ish / substring match on canonical reference_* .
    canonical = re.compile(
        rf"(?:^|[^A-Za-z0-9_]){re.escape(raw_slot)}(?:[^A-Za-z0-9_]|$)",
        flags=re.IGNORECASE,
    )
    if canonical.search(raw_title) or raw_slot.lower() in raw_title.lower():
        return True
    for alias in slot_title_aliases(raw_slot):
        if raw_title.lower() == alias.lower():
            return True
        if _alias_pattern(alias).search(raw_title):
            return True
        if alias.lower() in raw_title.lower():
            return True
    return False

_ROUTES_REGISTERED = False


def register_routes():
    """Register optional ComfyUI HTTP route; safe no-op outside Comfy."""
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return True
    try:
        from server import PromptServer
        import folder_paths
        from aiohttp import web
    except ImportError:
        return False
    server = getattr(PromptServer, "instance", None)
    if server is None or not hasattr(server, "routes"):
        return False

    @server.routes.post("/qwen38/promptrefiner/import_beat_pack")
    async def import_beat_pack(request):
        try:
            with tempfile.TemporaryDirectory(prefix="qwen38_beat_upload_") as temporary:
                root = Path(temporary)
                if request.content_type == "application/json":
                    manifest = parse_beat_pack_manifest(await request.json())
                    pack_root = root
                else:
                    reader = await request.multipart()
                    source = None
                    async for part in reader:
                        if part.name not in ("pack", "file") or not part.filename:
                            raise ValueError("Expected pack or file upload")
                        relative = "upload.zip" if part.name == "pack" else part.filename
                        target = _contained_path(root / "files", relative)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if target.exists():
                            raise ValueError("Duplicate upload path")
                        with target.open("wb") as output:
                            while chunk := await part.read_chunk():
                                output.write(chunk)
                        if part.name == "pack" or target.name == "manifest.json":
                            if source is not None:
                                raise ValueError("Select exactly one pack manifest")
                            source = target
                    if source is None:
                        raise ValueError("Missing manifest.json")
                    manifest, pack_root = await asyncio.to_thread(load_beat_pack, source, root / "extracted")
                result = await asyncio.to_thread(import_media, manifest, pack_root, folder_paths.get_input_directory())
                return web.json_response(result)
        except (ValueError, OSError, zipfile.BadZipFile) as error:
            return web.json_response({"error": str(error)}, status=400)

    _ROUTES_REGISTERED = True
    return True

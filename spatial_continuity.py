"""NVStudio spatial-continuity locks for Long Video Planner HANDOFF.

Shot fields win. Missing screen left/right, axis/working side, or a spatial
anchor fall back to ``scene_blocking`` defaults. Gaps are filled with a short
clause, the same way an empty HANDOFF falls back to pose and prop state.

This module does not invent absolute XY/Z positions, and it does not edit
LOOK (``subject_definitions`` / wardrobe retention), dialogue landmarks, or
duration budgets.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


HANDOFF_SPATIAL_RULE = (
    "When the brief supplies spatial continuity ([SPATIAL_CONTINUITY], scene_blocking, or per-shot "
    "spatial fields), HANDOFF must also name screen left/right, the axis or working side, and at least "
    "one spatial anchor, alongside pose, prop state, camera heading, and audio state. Prefer "
    "shot.screen_left_right, shot.axis_side, and shot.spatial_anchor; otherwise use "
    "scene_blocking.default_screen_left_right, scene_blocking.axis or working_side, and one "
    "scene_blocking.anchors entry. Pass through receiver_in, handoff_out, motion_vector, and "
    "spatial_bridge when the brief includes them. Do not invent absolute XY/Z coordinates."
)

CONTINUITY_OPENING_RULE = (
    "When CONTINUITY_FROM_PREV is true, the first sentence of the continued shot description "
    "(detailed_description or integrated_multimodal_description, not subject_definitions) must restate "
    "the spatial locks: screen left/right, axis or working side, and one spatial anchor. "
    "Do not invent absolute XY/Z coordinates."
)

OPTIONAL_PASSTHROUGH_KEYS = (
    "receiver_in",
    "handoff_out",
    "motion_vector",
    "spatial_bridge",
)

_SCENE_KEYS = (
    "anchors",
    "characters_relative",
    "axis",
    "working_side",
    "default_screen_left_right",
    *OPTIONAL_PASSTHROUGH_KEYS,
)
_SHOT_KEYS = (
    "screen_left_right",
    "axis_side",
    "inherits_axis",
    "exit_side",
    "enter_side",
    "spatial_anchor",
    "eyeline_to",
    "camera_vs_axis",
    "depth",
    "axis",
    "working_side",
    "anchors",
    *OPTIONAL_PASSTHROUGH_KEYS,
)
_SHOT_ONLY_KEYS = frozenset({
    "screen_left_right",
    "axis_side",
    "inherits_axis",
    "exit_side",
    "enter_side",
    "spatial_anchor",
    "eyeline_to",
    "camera_vs_axis",
    "depth",
    *OPTIONAL_PASSTHROUGH_KEYS,
})
_SCENE_ONLY_KEYS = frozenset({
    "anchors",
    "characters_relative",
    "working_side",
    "default_screen_left_right",
})
_MAPPING_KEYS = frozenset({
    "anchors",
    "characters_relative",
    "screen_left_right",
    "default_screen_left_right",
    "depth",
})
_ID_KEYS = frozenset({"id", "shot_id", "index", "shot", "分镜"})
_EMPTY_TOKEN = re.compile(r"(?i)^(none|n/?a|null|nil|无|-|—)$")
_SPATIAL_OPEN = re.compile(r"\[\s*SPATIAL_CONTINUITY\s*\]", flags=re.IGNORECASE)
_SPATIAL_CLOSE = re.compile(r"\[\s*/\s*SPATIAL_CONTINUITY\s*\]", flags=re.IGNORECASE)
_NEXT_SHOT_HEADER = re.compile(r"\n\s*分镜\s*\d+\s*使用参考", flags=re.IGNORECASE)
# Lines that are schema keys, not spoken dialogue. A DIALOGUE field that leads
# with these (NV's flattened scene_blocking / shot dump) is metadata.
SPATIAL_METADATA_KEYS = frozenset({
    "anchors",
    "axis",
    "axis_cross_shots",
    "axis_side",
    "camera_vs_axis",
    "characters_relative",
    "default_screen_left_right",
    "depth",
    "enter_side",
    "exit_side",
    "eyeline_to",
    "facing",
    "handoff_out",
    "inherits_axis",
    "motion_vector",
    "receiver_in",
    "scene_blocking",
    "screen_left_right",
    "spatial_anchor",
    "spatial_bridge",
    "working_side",
})
_SPATIAL_KEY_LINE = re.compile(
    r"^(?:(?:scene_blocking|shot|分镜)\b[\w\s.:#-]*?)?"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:[:=：]\s*\S.*)?$",
    flags=re.IGNORECASE,
)
_SHOT_HEADER = re.compile(
    r"(?im)^\s*分镜\s*(?P<index>\d+)\s*使用参考\s*[=:：]\s*(?P<references>[^\n]*)\s*$"
)
_SCOPED_LINE = re.compile(
    r"(?im)^\s*(?:"
    r"(?P<scope>scene_blocking)\s*[.#]\s*(?P<key>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<scope2>shot|分镜)\s*[.#]?\s*(?P<sid>\d+)\s*[.#]\s*(?P<key2>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<scope3>shot|分镜)\s*[.#]\s*(?P<key3>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<key4>[A-Za-z_][A-Za-z0-9_]*)"
    r")\s*[:=：]\s*(?P<value>\S.*?)\s*$"
)
_SHOT_ID_IN_TEXT = re.compile(
    r"(?i)(?:分镜|brief\s*shot)\s*#?\s*(\d+)|(?<![A-Za-z0-9_.])shot\s*[.#]\s*(\d+)"
)
_ACTION_FIELD = re.compile(
    r"(?im)^(detailed_description|integrated_multimodal_description)\s*:\s*"
)
_NEXT_H3_FIELD = re.compile(
    r"(?im)^(subject_definitions|summary|retention_analysis|detailed_description|"
    r"integrated_multimodal_description|overall_soundscape|non_diegetic_music)\s*:"
)
_SHOT_ONE = re.compile(r"^(\[\s*Shot\s+1\s*\]\s*)", flags=re.IGNORECASE)


def _norm(text: str) -> str:
    lowered = (text or "").casefold().replace("_", " ")
    return re.sub(r"[\s\-]+", " ", lowered).strip()


def _is_empty(value: object) -> bool:
    if value is None or isinstance(value, bool):
        return value is None
    if isinstance(value, str):
        token = value.strip()
        return not token or _EMPTY_TOKEN.fullmatch(token) is not None
    if isinstance(value, (list, dict, tuple)) and len(value) == 0:
        return True
    return False


def _parse_boolish(value: object) -> object:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"true", "yes", "on", "1"}:
        return True
    if token in {"false", "no", "off", "0"}:
        return False
    return value


def _parse_mapping(value: object) -> object:
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if not raw:
        return raw
    if raw.startswith("{") or raw.startswith("["):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    if ";" in raw or "；" in raw or "|" in raw:
        parts = re.split(r"\s*[;；|]\s*", raw)
    elif raw.count("=") >= 1 and "," in raw:
        parts = re.split(r"\s*,\s*", raw)
    elif "=" in raw or re.search(r":\s*\S", raw):
        parts = [raw]
    else:
        return raw
    mapping: dict[str, str] = {}
    for part in parts:
        if not part.strip():
            continue
        match = re.match(r"\s*([^:=]+?)\s*[=:]\s*(.+?)\s*$", part)
        if not match:
            return raw
        mapping[match.group(1).strip()] = match.group(2).strip()
    return mapping or raw


def _coerce_field(key: str, value: object) -> object:
    if key == "inherits_axis":
        return _parse_boolish(value)
    if key in _MAPPING_KEYS:
        parsed = _parse_mapping(value)
        return parsed
    if isinstance(value, str):
        return " ".join(value.split())
    return value


def _side_phrase(name: object, side: object) -> str:
    label = str(name).strip()
    token = _norm(str(side))
    if "left" in token or token in {"l", "左", "画面左"}:
        rendered = "screen-left"
    elif "right" in token or token in {"r", "右", "画面右"}:
        rendered = "screen-right"
    else:
        rendered = str(side).strip()
    return f"{label} {rendered}".strip()


def _screen_parts(value: object) -> tuple[str, tuple[str, ...]]:
    if _is_empty(value):
        return "", ()
    if isinstance(value, dict):
        phrases = [
            _side_phrase(name, side)
            for name, side in value.items()
            if not _is_empty(side)
        ]
        return ", ".join(phrases), tuple(phrases)
    if isinstance(value, list):
        phrases = []
        for item in value:
            if isinstance(item, dict):
                name = item.get("name") or item.get("id") or item.get("subject") or ""
                side = item.get("side") or item.get("screen") or ""
                if not _is_empty(name) and not _is_empty(side):
                    phrases.append(_side_phrase(name, side))
            elif not _is_empty(item):
                phrases.append(str(item).strip())
        return ", ".join(phrases), tuple(phrases)
    text = str(value).strip()
    return text, (text,) if text else ()


def _anchor_parts(value: object) -> tuple[str, str]:
    """Return (display, required name). Display may include a place; the name is the lock token."""

    if _is_empty(value):
        return "", ""
    if isinstance(value, dict) and value:
        key, place = next(iter(value.items()))
        name = str(key).strip()
        if _is_empty(place):
            return name, name
        return f"{name} ({place})", name
    if isinstance(value, list) and value:
        return _anchor_parts(value[0])
    text = str(value).strip()
    name = re.split(r"\s*[\(（:=]", text, maxsplit=1)[0].strip()
    return text, name or text


def _first_anchor(anchors: object) -> object:
    if _is_empty(anchors):
        return None
    if isinstance(anchors, dict):
        key, place = next(iter(anchors.items()))
        return {str(key): place}
    if isinstance(anchors, list):
        return anchors[0]
    return anchors


def _format_note(value: object) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _token_present(text: str, token: str) -> bool:
    norm_text = _norm(text)
    norm_token = _norm(token)
    if not norm_token:
        return False
    if re.search(r"[a-z0-9]", norm_token):
        return re.search(
            rf"(?<![a-z0-9]){re.escape(norm_token)}(?![a-z0-9])",
            norm_text,
        ) is not None
    return norm_token in norm_text


def _phrase_present(text: str, phrase: str) -> bool:
    norm_phrase = _norm(phrase)
    if not norm_phrase:
        return False
    return norm_phrase in _norm(text)


def _screen_phrase_present(text: str, phrase: str) -> bool:
    if _phrase_present(text, phrase):
        return True
    parts = _norm(phrase).split()
    if len(parts) < 2:
        return _token_present(text, phrase)
    name, side = parts[0], parts[-1]
    if side not in {"left", "right"}:
        return False
    return re.search(
        rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])"
        rf"(?:[^a-z0-9]+[a-z0-9]+){{0,5}}[^a-z0-9]+{side}(?![a-z0-9])",
        _norm(text),
    ) is not None


@dataclass(frozen=True)
class SpatialLocks:
    """Resolved screen, axis/working-side, and anchor text for one continuation."""

    screen_text: str = ""
    screen_phrases: tuple[str, ...] = ()
    axis_text: str = ""
    axis_tokens: tuple[str, ...] = ()
    anchor_text: str = ""
    anchor_name: str = ""
    optional_notes: tuple[tuple[str, str], ...] = ()

    def has_lock(self) -> bool:
        return bool(self.screen_text or self.axis_text or self.anchor_text)

    def has_content(self) -> bool:
        return self.has_lock() or bool(self.optional_notes)

    def screen_satisfied(self, text: str) -> bool:
        if not self.screen_phrases:
            return True
        return all(_screen_phrase_present(text, phrase) for phrase in self.screen_phrases)

    def axis_satisfied(self, text: str) -> bool:
        if not self.axis_tokens:
            return True
        return all(
            _token_present(text, token) if " " not in _norm(token) else _phrase_present(text, token)
            for token in self.axis_tokens
        )

    def anchor_satisfied(self, text: str) -> bool:
        if not self.anchor_name:
            return True
        return _token_present(text, self.anchor_name)

    def satisfied(self, text: str) -> bool:
        return (
            self.screen_satisfied(text)
            and self.axis_satisfied(text)
            and self.anchor_satisfied(text)
        )

    def opening_sentence(self) -> str:
        bits: list[str] = []
        if self.screen_text:
            bits.append(f"screen left/right: {self.screen_text}")
        if self.axis_text:
            bits.append(f"axis/working side: {self.axis_text}")
        if self.anchor_text:
            bits.append(f"spatial anchor: {self.anchor_text}")
        if not bits:
            return ""
        sentence = "; ".join(bits).strip()
        if sentence[-1] not in ".。!！?？":
            sentence += "."
        return sentence


@dataclass
class SpatialCatalog:
    """Scene defaults plus per-shot overrides parsed from one brief."""

    scene: dict = field(default_factory=dict)
    shots: dict[int, dict] = field(default_factory=dict)
    unscoped: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.scene and not self.shots and not self.unscoped

    def resolve(self, shot_id: int | None) -> SpatialLocks | None:
        shot: dict = {}
        if shot_id is not None and shot_id in self.shots:
            shot = dict(self.unscoped)
            shot.update(self.shots[shot_id])
        elif shot_id is not None:
            shot = dict(self.unscoped)
        elif self.unscoped:
            shot = dict(self.unscoped)
        elif len(self.shots) == 1:
            shot = dict(next(iter(self.shots.values())))
        return _locks_from_fields(shot, self.scene)


def _inherits_scene_axis(shot: dict) -> bool:
    if "inherits_axis" not in shot:
        return True
    flag = shot.get("inherits_axis")
    if isinstance(flag, bool):
        return flag
    return _parse_boolish(flag) is not False


def _locks_from_fields(shot: dict, scene: dict) -> SpatialLocks | None:
    screen_value = shot.get("screen_left_right")
    if _is_empty(screen_value):
        screen_value = scene.get("default_screen_left_right")
    screen_text, screen_phrases = _screen_parts(screen_value)

    axis_side = shot.get("axis_side")
    if _is_empty(axis_side):
        axis_side = shot.get("working_side")
    if _is_empty(axis_side):
        axis_side = scene.get("working_side")
    axis_line = shot.get("axis")
    if _is_empty(axis_line) and _inherits_scene_axis(shot):
        axis_line = scene.get("axis")
    axis_bits: list[str] = []
    axis_tokens: list[str] = []
    if not _is_empty(axis_side):
        side_text = str(axis_side).strip()
        axis_bits.append(f"working side {side_text}")
        axis_tokens.append(side_text)
    if not _is_empty(axis_line):
        line_text = str(axis_line).strip()
        axis_bits.append(f"axis {line_text}")
        axis_tokens.append(line_text)
    axis_text = ", ".join(axis_bits)

    anchor_value = shot.get("spatial_anchor")
    if _is_empty(anchor_value):
        anchor_value = _first_anchor(shot.get("anchors"))
    if _is_empty(anchor_value):
        anchor_value = _first_anchor(scene.get("anchors"))
    anchor_text, anchor_name = _anchor_parts(anchor_value)

    notes: list[tuple[str, str]] = []
    for key in OPTIONAL_PASSTHROUGH_KEYS:
        value = shot.get(key)
        if _is_empty(value):
            value = scene.get(key)
        if _is_empty(value):
            continue
        notes.append((key, _format_note(value)))

    locks = SpatialLocks(
        screen_text=screen_text,
        screen_phrases=screen_phrases,
        axis_text=axis_text,
        axis_tokens=tuple(axis_tokens),
        anchor_text=anchor_text,
        anchor_name=anchor_name,
        optional_notes=tuple(notes),
    )
    if not locks.has_content():
        return None
    return locks


def _merge_scene(catalog: SpatialCatalog, payload: dict) -> None:
    for key in _SCENE_KEYS:
        if key not in payload or _is_empty(payload.get(key)):
            continue
        catalog.scene[key] = _coerce_field(key, payload.get(key))


def _merge_shot(catalog: SpatialCatalog, shot_id: int | None, payload: dict) -> None:
    cleaned: dict = {}
    allowed = set(_SHOT_KEYS)
    for key, value in payload.items():
        if key in _ID_KEYS or key in {"scene_blocking", "shots", "shot"}:
            continue
        if key not in allowed:
            continue
        if _is_empty(value):
            continue
        cleaned[key] = _coerce_field(key, value)
    if not cleaned:
        return
    if shot_id is None:
        catalog.unscoped.update(cleaned)
        return
    bucket = catalog.shots.setdefault(int(shot_id), {})
    bucket.update(cleaned)


def _shot_id_from(payload: dict) -> int | None:
    for key in ("shot_id", "id", "index", "分镜"):
        if key not in payload:
            continue
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return int(value)
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    shot = payload.get("shot")
    if isinstance(shot, int) and not isinstance(shot, bool):
        return int(shot)
    if isinstance(shot, str) and shot.strip().isdigit():
        return int(shot.strip())
    return None


def _looks_like_scene(payload: dict) -> bool:
    return any(key in payload and not _is_empty(payload.get(key)) for key in _SCENE_ONLY_KEYS)


def _looks_like_shot(payload: dict) -> bool:
    return any(key in payload and not _is_empty(payload.get(key)) for key in _SHOT_ONLY_KEYS)


def _ingest_object(catalog: SpatialCatalog, payload: dict, forced_shot_id: int | None = None) -> None:
    if not isinstance(payload, dict):
        return
    scene = payload.get("scene_blocking")
    if isinstance(scene, dict):
        _merge_scene(catalog, scene)
    shot = payload.get("shot")
    if isinstance(shot, dict):
        _merge_shot(catalog, forced_shot_id if forced_shot_id is not None else _shot_id_from(shot), shot)
    shots = payload.get("shots")
    if isinstance(shots, dict):
        for key, value in shots.items():
            if not isinstance(value, dict):
                continue
            sid = int(key) if str(key).strip().isdigit() else _shot_id_from(value)
            _merge_shot(catalog, sid, value)
    elif isinstance(shots, list):
        for value in shots:
            if isinstance(value, dict):
                _merge_shot(catalog, _shot_id_from(value), value)
    wrapped = any(key in payload for key in ("scene_blocking", "shot", "shots"))
    if wrapped:
        return
    if forced_shot_id is not None and (_looks_like_shot(payload) or _looks_like_scene(payload)):
        _merge_shot(catalog, forced_shot_id, payload)
        return
    if _looks_like_scene(payload):
        _merge_scene(catalog, payload)
    if _looks_like_shot(payload):
        _merge_shot(catalog, _shot_id_from(payload), payload)


def _json_values(text: str) -> list[object]:
    values: list[object] = []
    decoder = json.JSONDecoder()
    index = 0
    source = text or ""
    while index < len(source):
        start = source.find("{", index)
        list_start = source.find("[", index)
        if start < 0 and list_start < 0:
            break
        if start < 0 or (list_start >= 0 and list_start < start):
            start = list_start
        try:
            value, offset = decoder.raw_decode(source[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        values.append(value)
        index = start + max(offset, 1)
    return values


def _ingest_text_json(catalog: SpatialCatalog, text: str, forced_shot_id: int | None = None) -> None:
    for value in _json_values(text):
        if isinstance(value, dict):
            _ingest_object(catalog, value, forced_shot_id)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _ingest_object(catalog, item, forced_shot_id)


def _apply_line(
    catalog: SpatialCatalog,
    scope: str | None,
    sid: str | None,
    key: str,
    value: str,
    *,
    default_shot_id: int | None,
) -> None:
    key_name = key.strip()
    shot_id = int(sid) if sid and sid.isdigit() else default_shot_id
    scope_name = (scope or "").lower()
    if scope_name == "scene_blocking" or (
        scope_name == "" and default_shot_id is None and key_name in _SCENE_ONLY_KEYS
    ):
        _merge_scene(catalog, {key_name: value})
        return
    if scope_name in {"shot", "分镜"} or key_name in _SHOT_ONLY_KEYS or default_shot_id is not None:
        _merge_shot(catalog, shot_id if scope_name != "scene_blocking" else None, {key_name: value})
        return
    if key_name in {"axis", "anchors", "working_side"} and default_shot_id is None and scope_name == "":
        _merge_scene(catalog, {key_name: value})


def _parse_lines(catalog: SpatialCatalog, text: str, default_shot_id: int | None) -> None:
    known = set(_SHOT_KEYS) | set(_SCENE_KEYS)
    skipped = {
        "duration_seconds",
        "continuity_from_prev",
        "pictures",
        "videos",
        "video_audios",
        "audios",
        "beat",
        "dialogue",
        "handoff",
        "prompt",
        "title",
    }
    for match in _SCOPED_LINE.finditer(text or ""):
        key = match.group("key") or match.group("key2") or match.group("key3") or match.group("key4") or ""
        if key.lower() in skipped or key not in known:
            continue
        scope = match.group("scope") or match.group("scope2") or match.group("scope3")
        _apply_line(
            catalog,
            scope,
            match.group("sid"),
            key,
            match.group("value") or "",
            default_shot_id=default_shot_id,
        )


def _spatial_spans(text: str) -> list[tuple[int, int, str]]:
    """Closed blocks, or an unclosed marker running until the next shot or EOF.

    NV sometimes appends ``[SPATIAL_CONTINUITY]`` to the last ``分镜`` with no
    closer. That tail is still spatial metadata, not the shot's spoken lines.
    """

    spans: list[tuple[int, int, str]] = []
    source = text or ""
    cursor = 0
    while cursor < len(source):
        match = _SPATIAL_OPEN.search(source, cursor)
        if not match:
            break
        rest_start = match.end()
        rest = source[rest_start:]
        close = _SPATIAL_CLOSE.search(rest)
        nxt = _NEXT_SHOT_HEADER.search(rest)
        if close and (nxt is None or close.start() <= nxt.start()):
            body_end = rest_start + close.start()
            end = rest_start + close.end()
        elif nxt is not None:
            body_end = rest_start + nxt.start()
            end = body_end
        else:
            body_end = len(source)
            end = len(source)
        spans.append((match.start(), end, source[rest_start:body_end]))
        cursor = max(end, match.end())
    return spans


def strip_spatial_continuity_text(text: str) -> str:
    """Drop spatial blocks so shot prose and DIALOGUE cannot absorb the schema."""

    source = text or ""
    spans = _spatial_spans(source)
    if not spans:
        return source
    pieces: list[str] = []
    cursor = 0
    for start, end, _body in spans:
        pieces.append(source[cursor:start])
        cursor = end
    pieces.append(source[cursor:])
    return "".join(pieces)


def _is_spatial_key_line(line: str) -> bool:
    match = _SPATIAL_KEY_LINE.match((line or "").strip())
    if not match:
        return False
    return match.group(1).lower() in SPATIAL_METADATA_KEYS


def dialogue_is_spatial_metadata(text: str) -> bool:
    """True when a DIALOGUE field is a spatial schema dump, not spoken lines.

    NV's flattened export puts keys such as ``anchors`` / ``axis`` /
    ``screen_left_right`` / ``spatial_anchor`` on their own lines. That text
    must not be budgeted or verbatim-partitioned as speech.
    """

    raw = strip_spatial_continuity_text(text or "").strip()
    if not raw:
        return bool(_SPATIAL_OPEN.search(text or ""))
    if _SPATIAL_OPEN.search(raw):
        return True
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return False
    if _is_spatial_key_line(lines[0]):
        return True
    hits = 0
    for line in lines:
        if not _is_spatial_key_line(line):
            continue
        hits += 1
        if hits >= 3:
            return True
    return False


def _blank_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    chars = list(text)
    for start, end in spans:
        for index in range(max(0, start), min(len(chars), end)):
            chars[index] = " "
    return "".join(chars)


def parse_spatial_continuity(brief: str) -> SpatialCatalog:
    """Read ``[SPATIAL_CONTINUITY]``, embedded shot JSON, and per-shot lines.

    Shot-level keys stay separate from ``scene_blocking`` so resolve can prefer
    the shot and fall back per field. JSON inside a ``分镜`` block stays on that
    shot instead of becoming the scene-wide default.
    """

    catalog = SpatialCatalog()
    text = brief or ""
    if not text.strip():
        return catalog
    spatial_spans = _spatial_spans(text)
    marker_spans = [(start, end) for start, end, _body in spatial_spans]
    for _start, _end, body in spatial_spans:
        _ingest_text_json(catalog, body)
        _parse_lines(catalog, body, default_shot_id=None)
    shot_spans: list[tuple[int, int]] = []
    shot_blocks: list[tuple[int, str]] = []
    matches = list(_SHOT_HEADER.finditer(text))
    for position, header in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        shot_spans.append((header.start(), end))
        shot_blocks.append((int(header.group("index")), text[header.start():end]))
    outside = _blank_spans(text, marker_spans + shot_spans)
    _ingest_text_json(catalog, outside)
    _parse_lines(catalog, outside, default_shot_id=None)
    for shot_id, block in shot_blocks:
        local = strip_spatial_continuity_text(block)
        _parse_lines(catalog, local, default_shot_id=shot_id)
        _ingest_text_json(catalog, local, forced_shot_id=shot_id)
    return catalog


def shot_ids_in_text(text: str) -> tuple[int, ...]:
    found: list[int] = []
    for match in _SHOT_ID_IN_TEXT.finditer(text or ""):
        raw = match.group(1) or match.group(2)
        if not raw:
            continue
        shot_id = int(raw)
        if shot_id not in found:
            found.append(shot_id)
    return tuple(found)


def locks_for_text(catalog: SpatialCatalog, text: str, *, end: bool) -> SpatialLocks | None:
    """Pick the shot this card starts on, or the shot it ends on."""

    if catalog.is_empty:
        return None
    ids = shot_ids_in_text(text)
    if not ids:
        return catalog.resolve(None)
    return catalog.resolve(ids[-1] if end else ids[0])


def format_spatial_brief(catalog: SpatialCatalog) -> str:
    """Instruction block for a structure or legacy plan request. Empty when unused."""

    if catalog.is_empty:
        return ""
    lines = [
        "SPATIAL CONTINUITY (parsed from the brief; shot fields override scene_blocking defaults):",
    ]
    if catalog.scene:
        lines.append(
            "- scene_blocking: " + json.dumps(catalog.scene, ensure_ascii=False, sort_keys=True)
        )
    default = catalog.resolve(None)
    if catalog.unscoped:
        lines.append("- shot: " + json.dumps(catalog.unscoped, ensure_ascii=False, sort_keys=True))
    if default is not None and default.opening_sentence():
        lines.append(f"- resolved default locks: {default.opening_sentence()}")
    for shot_id in sorted(catalog.shots):
        locks = catalog.resolve(shot_id)
        rendered = locks.opening_sentence() if locks is not None else ""
        lines.append(
            f"- shot {shot_id}: "
            + json.dumps(catalog.shots[shot_id], ensure_ascii=False, sort_keys=True)
            + (f". Resolved locks: {rendered}" if rendered else "")
        )
    lines.append(
        "HANDOFF must include screen left/right, axis or working side, and one spatial anchor "
        "together with pose, prop state, camera heading, and audio state. "
        "When CONTINUITY_FROM_PREV is true, the first sentence of the continued shot description "
        "must restate those three locks. "
        "Repeat receiver_in, handoff_out, motion_vector, or spatial_bridge only when supplied above. "
        "Do not invent absolute XY/Z coordinates."
    )
    return "\n".join(lines)


def _ensure_sentence(text: str) -> str:
    body = (text or "").strip()
    if not body:
        return ""
    if body[-1] not in ".。!！?？":
        return body + "."
    return body


def repair_handoff(handoff: str, locks: SpatialLocks | None) -> str:
    """Append any missing spatial locks. Pose and prop wording already present stays."""

    if locks is None or not locks.has_content():
        return handoff
    base = handoff or ""
    probe = base.strip()
    missing: list[str] = []
    if locks.screen_text and not locks.screen_satisfied(probe):
        missing.append(f"screen left/right: {locks.screen_text}")
    if locks.axis_text and not locks.axis_satisfied(probe):
        missing.append(f"axis/working side: {locks.axis_text}")
    if locks.anchor_text and not locks.anchor_satisfied(probe):
        missing.append(f"spatial anchor: {locks.anchor_text}")
    for key, note in locks.optional_notes:
        if _token_present(probe, key) and _phrase_present(probe, note):
            continue
        if _phrase_present(probe, f"{key} {note}"):
            continue
        missing.append(f"{key} {_format_note(note)}")
    if not missing:
        return handoff
    if not probe or _EMPTY_TOKEN.fullmatch(probe):
        stem = "Continue from this group's ending pose, prop state, and camera direction."
    else:
        stem = _ensure_sentence(probe)
    clause = "; ".join(missing)
    return f"{stem} Spatial continuity: {clause}."


def _action_body_span(prompt: str) -> tuple[int, int] | None:
    """Body span of the shot-description field. LOOK fields are not selected."""

    match = None
    for candidate in _ACTION_FIELD.finditer(prompt or ""):
        match = candidate
        if candidate.group(1).lower() == "detailed_description":
            break
    if match is None:
        return None
    tail = prompt[match.end():]
    nxt = _NEXT_H3_FIELD.search(tail)
    body_end = match.end() + (nxt.start() if nxt else len(tail))
    return match.end(), body_end


def _first_sentence(body: str) -> str:
    prefix = re.match(r"\s*", body or "").group(0)
    core = (body or "")[len(prefix):]
    shot = _SHOT_ONE.match(core)
    if shot:
        core = core[shot.end():]
    core = re.split(r"\[\s*Shot\s+\d+", core, maxsplit=1, flags=re.IGNORECASE)[0]
    parts = re.split(r"(?<=[.!?。！？])\s+", core.strip(), maxsplit=1)
    return parts[0] if parts else core.strip()


def repair_continuity_opening(prompt: str, locks: SpatialLocks | None, *, continuity: bool) -> str:
    """Restate spatial locks in the first shot-description sentence when continuing.

    ``subject_definitions`` and ``retention_analysis`` are left untouched so the
    LOOK contract stays as the refine pass wrote it.
    """

    if not continuity or locks is None or not locks.has_lock():
        return prompt
    sentence = locks.opening_sentence()
    if not sentence:
        return prompt
    span = _action_body_span(prompt or "")
    if span is None:
        return prompt
    start, end = span
    body = prompt[start:end]
    if locks.satisfied(_first_sentence(body)):
        return prompt
    prefix = re.match(r"\s*", body).group(0)
    trailing_match = re.search(r"\s*$", body)
    trailing = trailing_match.group(0) if trailing_match else ""
    core_end = len(body) - len(trailing)
    core = body[len(prefix):core_end] if core_end >= len(prefix) else ""
    shot = _SHOT_ONE.match(core)
    if shot:
        rest = core[shot.end():].lstrip()
        core = f"{shot.group(1)}{sentence} {rest}".rstrip()
    else:
        # Keep a leading [Brief Shot N] glued to its action. The lock sentence
        # is still the first sentence of the continued description.
        core = f"{sentence} {core.lstrip()}".rstrip()
    return prompt[:start] + prefix + core + trailing + prompt[end:]

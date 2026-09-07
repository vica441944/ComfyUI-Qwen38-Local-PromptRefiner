"""ComfyUI nodes for direct local Qwen-VL GGUF prompt refinement.

This package deliberately talks to llama.cpp through llama-cpp-python.  It does
not use Ollama or any remote API.  The Qwen GGUF and its matching mmproj GGUF
are loaded from ComfyUI's ``models/LLM`` directory.
"""

from __future__ import annotations

import base64
import gc
import inspect
import io
import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

import folder_paths


NODE_CATEGORY = "LLM/Prompt Refiner"
LLM_FOLDER_NAME = "LLM"
MODEL_SUFFIXES = {".gguf", ".ggml"}


ENGINE_TAGS = {
    "Auto (standard image)": "@image",
    "FLUX.2 / Klein": "@flux",
    "Ideogram JSON": "@ideogram",
    "360 panorama": "@pano360",
    "Ernie-ViLG": "@ernie",
    "MiniMax H3 text-to-video": "@minimax-t2v",
    "MiniMax H3 image-to-video": "@minimax-i2v",
    "MiniMax H3 360 video": "@minimax-pano",
    "MiniMax H3 full reference": "@minimax-fullref",
    "LTX video": "@ltx",
    "Video storyboard": "@storyboard",
    "ACE-Step music": "@acestep",
    "MiniMax music": "@minimax-music",
}

VISION_HANDLER_CHOICES = ["auto (Qwen3-VL)", "qwen3-vl", "mtmd", "qwen2.5-vl"]
FULL_REFERENCE_ENGINE = "MiniMax H3 full reference"
ROUTE_PROFILE_FILENAME = "qwen38_ollama_route_profile.txt"
LORA_PRESETS_FILENAME = "lora_trigger_presets.json"
LORA_PRESET_NONE = "None"
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_AUDIOS = 3
MIN_REFERENCE_MEDIA_SECONDS = 2.0
MAX_REFERENCE_MEDIA_SECONDS = 15.0
MAX_CONFIGURABLE_REFERENCE_MEDIA_SECONDS = 120.0
REFERENCE_DURATION_TOLERANCE_SECONDS = 0.05
VIDEO_FRAME_MAX_SIDE = 1024

FALLBACK_ROUTE_SYSTEM = """You are a local multi-engine generative-media prompt refiner. Return only the final
engine-native prompt. Do not include conversational filler, planning, hidden reasoning, markdown fences, or
commentary. Preserve explicit constraints and reference labels."""


class PromptRefinerError(RuntimeError):
    """A configuration error that should be shown clearly in the ComfyUI UI."""


def _llm_models_dir() -> Path:
    """Register and return ComfyUI's dedicated LLM model directory."""

    directory = Path(folder_paths.models_dir) / LLM_FOLDER_NAME
    directory.mkdir(parents=True, exist_ok=True)
    try:
        folder_paths.add_model_folder_path(LLM_FOLDER_NAME, str(directory))
    except Exception:
        # The path may already be registered by ComfyUI or another local node.
        pass
    return directory


def _model_files() -> list[str]:
    directory = _llm_models_dir()
    return sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES
    )


def _model_choices(projector: bool) -> list[str]:
    files = _model_files()
    selected = [
        name
        for name in files
        if ("mmproj" in name.lower() or "projector" in name.lower()) == projector
    ]
    if projector:
        return ["none"] + selected
    return selected or ["<put a GGUF model in models/LLM>"]


def _resolve_model_file(name: str, allow_none: bool = False) -> str | None:
    if allow_none and name == "none":
        return None
    if name.startswith("<"):
        raise PromptRefinerError(
            "No GGUF model was found. Put the Qwen GGUF in ComfyUI/models/LLM and refresh ComfyUI."
        )

    directory = _llm_models_dir().resolve()
    candidate = (directory / Path(name)).resolve()
    if directory not in candidate.parents or not candidate.is_file():
        raise PromptRefinerError(
            f"Model file '{name}' was not found under {directory}. Refresh the node after adding it."
        )
    return str(candidate)


def _array_to_data_uri(array: Any, max_side: int | None = None) -> str:
    """Convert an RGB(A) image array to an in-memory PNG data URI."""

    array = np.asarray(array)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise PromptRefinerError("Reference images must use H×W×RGB(A) format.")

    array = np.clip(array * 255.0 if array.dtype.kind == "f" else array, 0, 255).astype(np.uint8)
    mode = "RGBA" if array.shape[-1] == 4 else "RGB"
    image = Image.fromarray(array, mode=mode)
    if max_side and max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _image_to_data_uri(image_tensor: Any) -> str:
    """Convert a ComfyUI IMAGE tensor's first item to an in-memory PNG data URI."""

    if image_tensor is None:
        raise PromptRefinerError("A supplied reference image was empty.")

    try:
        array = image_tensor.detach().cpu().numpy()
    except AttributeError:
        array = np.asarray(image_tensor)

    if array.ndim == 4:
        array = array[0]
    return _array_to_data_uri(array)


def _strip_reasoning(text: str) -> str:
    """Hide common reasoning blocks while retaining the model's final prompt."""

    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<thinking>.*?</thinking>\s*", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def _route_system() -> str:
    """Load the prompt-refiner's local Ollama routing profile, with a safe fallback."""

    profile = Path(__file__).with_name(ROUTE_PROFILE_FILENAME)
    try:
        text = profile.read_text(encoding="utf-8").strip()
    except OSError:
        text = ""
    return text or FALLBACK_ROUTE_SYSTEM


def _parse_trigger_words(text: str) -> list[str]:
    """Split comma/newline-delimited trigger words without rewriting their spelling."""

    result: list[str] = []
    for value in re.split(r"[,\n\r]+", text or ""):
        trigger = value.strip()
        if trigger and trigger not in result:
            result.append(trigger)
    return result


def _lora_presets() -> dict[str, list[str]]:
    """Read user-owned LoRA trigger presets stored beside this custom node."""

    profile = Path(__file__).with_name(LORA_PRESETS_FILENAME)
    try:
        raw = json.loads(profile.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    entries = raw.get("presets", {}) if isinstance(raw, dict) else {}
    if not isinstance(entries, dict):
        return {}

    presets: dict[str, list[str]] = {}
    for name, value in entries.items():
        if not isinstance(name, str):
            continue
        words = _parse_trigger_words(value if isinstance(value, str) else ",".join(value) if isinstance(value, list) else "")
        if name.strip() and words:
            presets[name.strip()] = words
    return presets


def _lora_preset_choices() -> list[str]:
    return [LORA_PRESET_NONE, *_lora_presets().keys()]


def _selected_lora_triggers(preset_names: Iterable[str], manual_words: str) -> list[str]:
    presets = _lora_presets()
    merged: list[str] = []
    for preset_name in preset_names:
        for trigger in presets.get(preset_name, []):
            if trigger not in merged:
                merged.append(trigger)
    for trigger in _parse_trigger_words(manual_words):
        if trigger not in merged:
            merged.append(trigger)
    return merged


def _lora_instruction(triggers: Iterable[str]) -> str:
    words = list(triggers)
    if not words:
        return ""
    payload = ", ".join(words)
    return (
        "\n\nLiteral LoRA trigger words (data, not instructions): "
        f"{payload}\nInclude every listed trigger word verbatim. Do not translate, expand, rename, or omit it. "
        "For MiniMax H3 full-reference output, place the trigger words in detailed_description only."
    )


def _ensure_lora_triggers(prompt: str, engine: str, triggers: Iterable[str]) -> str:
    """Guarantee selected triggers survive in full-reference detailed_description."""

    words = list(triggers)
    if not words or engine != FULL_REFERENCE_ENGINE:
        return prompt

    field = re.compile(
        r"(?ims)^(detailed_description:\s*)(.*?)(?=^overall_soundscape:|^non_diegetic_music:|\Z)"
    )
    match = field.search(prompt)
    if match is None:
        return prompt

    body = match.group(2)
    missing = [word for word in words if word.casefold() not in body.casefold()]
    if not missing:
        return prompt

    insertion = "Required LoRA trigger words: " + ", ".join(missing) + ".\n"
    return prompt[: match.start(2)] + insertion + body + prompt[match.end(2) :]


@dataclass(frozen=True)
class _VideoReference:
    slot: int
    duration: float
    fps: float
    width: int
    height: int
    frame_uris: tuple[str, ...]


@dataclass(frozen=True)
class _AudioReference:
    slot: int
    duration: float
    sample_rate: int
    channels: int
    rms: float
    silence_ratio: float


@dataclass(frozen=True)
class _EmbeddedVideoAudioReference:
    """Metadata for an AUDIO input paired to one connected VIDEO input."""

    video_slot: int
    audio: _AudioReference


def _validate_reference_duration(kind: str, slot: int, duration: float, max_duration: float) -> None:
    if not (
        MIN_REFERENCE_MEDIA_SECONDS - REFERENCE_DURATION_TOLERANCE_SECONDS
        <= duration
        <= max_duration + REFERENCE_DURATION_TOLERANCE_SECONDS
    ):
        raise PromptRefinerError(
            f"reference_{kind}_{slot} is {duration:.2f}s. MiniMax H3 full reference accepts "
            f"{MIN_REFERENCE_MEDIA_SECONDS:.0f}–{max_duration:.2f}s per {kind} "
            f"(±{REFERENCE_DURATION_TOLERANCE_SECONDS:.2f}s media-timestamp tolerance)."
        )


def _video_source_path(video: Any, slot: int) -> tuple[str, bool]:
    """Return a readable local video path and whether it is a temporary export."""

    if isinstance(video, str) and os.path.isfile(video):
        return video, False
    if isinstance(video, dict):
        for key in ("file_path", "path", "filename"):
            value = video.get(key)
            if isinstance(value, str) and os.path.isfile(value):
                return value, False
    if not hasattr(video, "save_to"):
        raise PromptRefinerError(
            f"reference_video_{slot} must be connected to a native ComfyUI VIDEO output."
        )

    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    handle.close()
    try:
        saved = video.save_to(handle.name)
        if saved is False or not os.path.isfile(handle.name) or os.path.getsize(handle.name) <= 0:
            raise PromptRefinerError(f"reference_video_{slot} could not be exported for analysis.")
        return handle.name, True
    except Exception:
        try:
            os.remove(handle.name)
        except OSError:
            pass
        raise


def _analyze_video(
    video: Any, slot: int, sample_count: int, max_duration: float
) -> _VideoReference:
    """Read H3 reference-video metadata and uniformly sampled visual frames for Qwen-VL."""

    try:
        import cv2
    except ImportError as error:
        raise PromptRefinerError("OpenCV is required to analyze VIDEO inputs for this node.") from error

    path, is_temporary = _video_source_path(video, slot)
    capture = None
    try:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise PromptRefinerError(f"reference_video_{slot} could not be opened.")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise PromptRefinerError(f"reference_video_{slot} has invalid FPS, frame count, or dimensions.")

        duration = frame_count / fps
        _validate_reference_duration("video", slot, duration, max_duration)
        positions = np.linspace(0, frame_count - 1, min(sample_count, frame_count), dtype=int)
        frame_uris: list[str] = []
        for position in dict.fromkeys(int(value) for value in positions):
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = capture.read()
            if not ok:
                raise PromptRefinerError(f"reference_video_{slot} could not read sampled frame {position}.")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_uris.append(_array_to_data_uri(rgb, max_side=VIDEO_FRAME_MAX_SIDE))

        return _VideoReference(slot, duration, fps, width, height, tuple(frame_uris))
    finally:
        if capture is not None:
            capture.release()
        if is_temporary:
            try:
                os.remove(path)
            except OSError:
                pass


def _analyze_audio(audio: Any, slot: int, max_duration: float) -> _AudioReference:
    """Read H3 reference-audio metadata. Qwen-VL does not receive raw audio bytes."""

    if not isinstance(audio, dict):
        raise PromptRefinerError(f"reference_audio_{slot} must be connected to a native ComfyUI AUDIO output.")
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate") or audio.get("sampler_rate") or 0)
    if waveform is None or sample_rate <= 0:
        raise PromptRefinerError(f"reference_audio_{slot} is missing waveform or sample rate.")

    try:
        array = waveform.detach().cpu().numpy()
    except AttributeError:
        array = np.asarray(waveform)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise PromptRefinerError(f"reference_audio_{slot} contains multiple audio batches; split it first.")
        array = array[0]
    array = np.squeeze(array)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    elif array.ndim == 2 and array.shape[0] > 8 and array.shape[1] <= 8:
        array = array.T
    if array.ndim != 2 or not 1 <= array.shape[0] <= 8 or array.shape[1] <= 0:
        raise PromptRefinerError(f"reference_audio_{slot} has an unsupported channel layout.")

    if np.issubdtype(array.dtype, np.integer):
        maximum = max(abs(np.iinfo(array.dtype).min), np.iinfo(array.dtype).max)
        array = array.astype(np.float32) / float(maximum)
    else:
        array = array.astype(np.float32, copy=False)
    duration = array.shape[1] / float(sample_rate)
    _validate_reference_duration("audio", slot, duration, max_duration)
    return _AudioReference(
        slot=slot,
        duration=duration,
        sample_rate=sample_rate,
        channels=array.shape[0],
        rms=float(np.sqrt(np.mean(np.square(array), dtype=np.float64))),
        silence_ratio=float(np.mean(np.abs(array) < 0.005)),
    )


def _get_video_inputs(
    values: dict[str, Any], sample_count: int, max_duration: float
) -> list[_VideoReference]:
    return [
        _analyze_video(values[f"reference_video_{slot}"], slot, sample_count, max_duration)
        for slot in range(1, MAX_REFERENCE_VIDEOS + 1)
        if values.get(f"reference_video_{slot}") is not None
    ]


def _get_audio_inputs(values: dict[str, Any], max_duration: float) -> list[_AudioReference]:
    return [
        _analyze_audio(values[f"reference_audio_{slot}"], slot, max_duration)
        for slot in range(1, MAX_REFERENCE_AUDIOS + 1)
        if values.get(f"reference_audio_{slot}") is not None
    ]


def _get_video_audio_inputs(
    values: dict[str, Any], max_duration: float
) -> list[_EmbeddedVideoAudioReference]:
    """Read optional AUDIO inputs that are explicitly paired with VIDEO slots."""

    metadata: list[_EmbeddedVideoAudioReference] = []
    for slot in range(1, MAX_REFERENCE_VIDEOS + 1):
        audio = values.get(f"reference_video_audio_{slot}")
        if audio is None:
            continue
        try:
            audio_metadata = _analyze_audio(audio, slot, max_duration)
        except PromptRefinerError as error:
            message = str(error).replace(
                f"reference_audio_{slot}", f"reference_video_audio_{slot}"
            )
            raise PromptRefinerError(message) from error
        metadata.append(_EmbeddedVideoAudioReference(slot, audio_metadata))
    return metadata


def _normalize_asset_links(
    brief: str,
    image_references: Iterable[tuple[int, str]],
    video_references: Iterable[_VideoReference],
    audio_references: Iterable[_AudioReference],
    embedded_video_audio_references: Iterable[_EmbeddedVideoAudioReference] = (),
) -> str:
    """Resolve @reference_* aliases against actual connected inputs and tracks."""

    available = {
        "image": {slot for slot, _ in image_references},
        "video": {item.slot for item in video_references},
        "audio": {item.slot for item in audio_references},
    }
    labels = {"image": "Picture", "video": "Video", "audio": "Audio"}
    embedded_audio_slots = {item.video_slot for item in embedded_video_audio_references}
    embedded_pattern = re.compile(r"@reference_video_audio_([1-3])\b", re.IGNORECASE)

    def replace_embedded_audio(match: re.Match[str]) -> str:
        slot = int(match.group(1))
        if slot not in embedded_audio_slots:
            raise PromptRefinerError(
                f"{match.group(0)} was used in brief, but reference_video_audio_{slot} is not connected."
            )
        return f"<Video {slot}> paired audio track"

    brief = embedded_pattern.sub(replace_embedded_audio, brief)
    pattern = re.compile(r"@reference_(image|video|audio)_([1-9])\b", re.IGNORECASE)

    def replace(match: re.Match[str]) -> str:
        kind = match.group(1).lower()
        slot = int(match.group(2))
        if slot not in available[kind]:
            raise PromptRefinerError(
                f"{match.group(0)} was used in brief, but reference_{kind}_{slot} is not connected."
            )
        return f"<{labels[kind]} {slot}>"

    return pattern.sub(replace, brief)


def _get_image_inputs(values: dict[str, Any]) -> list[tuple[int, str]]:
    """Return image data paired with their actual ComfyUI input slot numbers."""

    image_uris: list[tuple[int, str]] = []
    for index in range(1, 10):
        image = values.get(f"reference_image_{index}")
        if image is not None:
            image_uris.append((index, _image_to_data_uri(image)))
    return image_uris


def _build_handler(mmproj_path: str, handler_choice: str) -> Any:
    """Create a llama-cpp-python vision handler without pinning a fragile API version."""

    try:
        from llama_cpp import llama_chat_format
    except ImportError as error:
        raise PromptRefinerError(
            "llama-cpp-python is not installed in ComfyUI's Python environment. See this node's README."
        ) from error

    requested_names = {
        "qwen3-vl": ["Qwen3VLChatHandler"],
        "mtmd": ["MTMDChatHandler"],
        "qwen2.5-vl": ["Qwen25VLChatHandler", "MTMDChatHandler"],
        "auto (Qwen3-VL)": ["Qwen3VLChatHandler", "MTMDChatHandler", "Qwen25VLChatHandler"],
    }[handler_choice]

    for name in requested_names:
        handler_class = getattr(llama_chat_format, name, None)
        if handler_class is None:
            continue

        try:
            if name == "Qwen3VLChatHandler":
                # Qwen3.8-VL uses Qwen3's multimodal template.  This explicitly
                # disables its reasoning mode so the node returns a prompt directly.
                return handler_class(
                    clip_model_path=mmproj_path,
                    verbose=False,
                    force_reasoning=False,
                )
            parameters = inspect.signature(handler_class).parameters
            if "clip_model_path" in parameters:
                return handler_class(clip_model_path=mmproj_path, verbose=False)
            if "mmproj_path" in parameters:
                return handler_class(mmproj_path=mmproj_path, verbose=False)
            return handler_class(mmproj_path, verbose=False)
        except (TypeError, ValueError) as error:
            last_error = error
            continue

    detail = locals().get("last_error")
    raise PromptRefinerError(
        "This llama-cpp-python build has no compatible Qwen-VL/MTMD chat handler. "
        "Upgrade llama-cpp-python to a recent GPU-enabled build."
        + (f" Details: {detail}" if detail else "")
    )


@dataclass(frozen=True)
class _RuntimeKey:
    model_path: str
    mmproj_path: str | None
    handler_choice: str
    n_ctx: int
    n_gpu_layers: int


@dataclass
class _Runtime:
    llm: Any
    handler: Any | None
    key: _RuntimeKey

    def close(self) -> None:
        for item in (self.llm, self.handler):
            close = getattr(item, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass


class _RuntimeCache:
    """One process-wide model cache so ComfyUI does not duplicate a 27B model."""

    _lock = threading.RLock()
    _runtime: _Runtime | None = None

    @classmethod
    def get(cls, key: _RuntimeKey) -> _Runtime:
        with cls._lock:
            if cls._runtime is not None and cls._runtime.key == key:
                return cls._runtime

            cls.unload()
            try:
                from llama_cpp import Llama
            except ImportError as error:
                raise PromptRefinerError(
                    "llama-cpp-python is not installed in ComfyUI's Python environment. See this node's README."
                ) from error

            handler = _build_handler(key.mmproj_path, key.handler_choice) if key.mmproj_path else None
            llm = Llama(
                model_path=key.model_path,
                chat_handler=handler,
                n_ctx=key.n_ctx,
                n_gpu_layers=key.n_gpu_layers,
                verbose=False,
            )
            cls._runtime = _Runtime(llm=llm, handler=handler, key=key)
            return cls._runtime

    @classmethod
    def unload(cls) -> None:
        with cls._lock:
            if cls._runtime is not None:
                cls._runtime.close()
                cls._runtime = None
                gc.collect()


def _build_request(
    engine: str,
    brief: str,
    reference_text: str,
    image_references: Iterable[tuple[int, str]],
    video_references: Iterable[_VideoReference],
    audio_references: Iterable[_AudioReference],
    embedded_video_audio_references: Iterable[_EmbeddedVideoAudioReference],
    lora_triggers: Iterable[str],
) -> list[dict[str, Any]]:
    images = list(image_references)
    videos = list(video_references)
    audios = list(audio_references)
    embedded_video_audios = list(embedded_video_audio_references)
    reference_lines = "\n".join(
        [f"<Picture {index}>: supplied through reference_image_{index}." for index, _ in images]
        + [
            f"<Video {item.slot}>: supplied through reference_video_{item.slot}; "
            f"{item.duration:.2f}s, {item.width}x{item.height}, {item.fps:.2f}fps; "
            f"{len(item.frame_uris)} visual samples are attached for local Qwen-VL analysis."
            for item in videos
        ]
        + [
            f"<Audio {item.slot}>: supplied through reference_audio_{item.slot}; "
            f"{item.duration:.2f}s, {item.sample_rate}Hz, {item.channels} channel(s), "
            f"RMS={item.rms:.4f}, silence={item.silence_ratio:.1%}. "
            "Raw audio is not available to this Qwen-VL model; use Additional reference notes for dialogue, voice, or music details."
            for item in audios
        ]
        + [
            f"<Video {item.video_slot}> paired audio track: supplied through reference_video_audio_{item.video_slot}; "
            f"{item.audio.duration:.2f}s, {item.audio.sample_rate}Hz, {item.audio.channels} channel(s). "
            "It belongs to the matching MiniMax H3 ref_video_audio port. Raw audio is not available to Qwen-VL; use Additional reference "
            "notes for exact dialogue, voice, or music details."
            for item in embedded_video_audios
        ]
    )
    extra = f"\nAdditional reference notes:\n{reference_text.strip()}" if reference_text.strip() else ""
    visual_note = (
        "\nThe local image inputs and sampled video frames below are authoritative visual references. Inspect them "
        "directly, preserve identity and motion when requested, and keep each <Picture N> or <Video N> label aligned "
        "with its actual reference input slot."
        if images or videos
        else ""
    )
    system = _route_system()
    if engine == FULL_REFERENCE_ENGINE:
        system += """

This is MiniMax H3 full-reference mode. Output ALL of these exact labels, in this exact order, with one blank line
between fields. Do not add or rename sections:
subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:

Use <Picture N>, <Video N>, <Audio N>, <Voice N>, and <Subject N> labels exactly as supplied. In
subject_definitions, define each referenced subject and preservation requirement. In retention_analysis, account for
every supplied visual reference. In detailed_description, write the observable H3 shot/action/camera/dialogue plan.
Use <d>[Language] exact dialogue</d> for spoken dialogue. Video samples are visual evidence for their matching
<Video N> labels. A paired video audio track belongs to its matching <Video N>, not a separate <Audio N> input.
Audio metadata is not a transcription: preserve supplied audio labels and use only supplied notes for dialogue,
voice, or music details. Include every field even when its value is N/A."""
    prompt = f"{ENGINE_TAGS[engine]}\n{brief.strip()}"
    if reference_lines:
        prompt += f"\n\nReference inventory:\n{reference_lines}"
    prompt += extra + visual_note + _lora_instruction(lora_triggers)

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index, uri in images:
        content.append({"type": "text", "text": f"Visual reference: <Picture {index}>."})
        content.append({"type": "image_url", "image_url": {"url": uri}})
    for video in videos:
        for frame_number, uri in enumerate(video.frame_uris, start=1):
            content.append(
                {
                    "type": "text",
                    "text": f"Visual sample {frame_number}/{len(video.frame_uris)} from <Video {video.slot}>.",
                }
            )
            content.append({"type": "image_url", "image_url": {"url": uri}})
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


class Qwen38LocalPromptRefiner:
    """Run a Qwen-VL GGUF directly from ComfyUI/models/LLM using llama.cpp."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "debug")
    FUNCTION = "refine"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        optional_images = {f"reference_image_{index}": ("IMAGE",) for index in range(1, 10)}
        optional_videos = {
            f"reference_video_{index}": (
                "VIDEO",
                {
                    "tooltip": "MiniMax H3 full-reference video. 2–15 seconds each; all video references total ≤15 seconds.",
                },
            )
            for index in range(1, MAX_REFERENCE_VIDEOS + 1)
        }
        optional_audios = {
            f"reference_audio_{index}": (
                "AUDIO",
                {
                    "tooltip": "MiniMax H3 full-reference audio. 2–15 seconds each; all audio references total ≤15 seconds.",
                },
            )
            for index in range(1, MAX_REFERENCE_AUDIOS + 1)
        }
        return {
            "required": {
                "brief": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "dynamicPrompts": False,
                        "tooltip": "Use @reference_image_1, @reference_video_1, @reference_video_audio_1, or @reference_audio_1 to link exact connected assets.",
                    },
                ),
                "engine": (list(ENGINE_TAGS.keys()), {"default": "Auto (standard image)"}),
                "model_name": (_model_choices(projector=False),),
                "mmproj_name": (_model_choices(projector=True), {"default": "none"}),
                "vision_handler": (VISION_HANDLER_CHOICES, {"default": "auto (Qwen3-VL)"}),
                "context_length": ("INT", {"default": 32768, "min": 2048, "max": 65536, "step": 1024}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 200, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 4096, "min": 64, "max": 16384, "step": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "lora_preset_1": (_lora_preset_choices(), {"default": LORA_PRESET_NONE}),
                "lora_preset_2": (_lora_preset_choices(), {"default": LORA_PRESET_NONE}),
                "lora_preset_3": (_lora_preset_choices(), {"default": LORA_PRESET_NONE}),
                "video_frame_samples": (
                    "INT",
                    {
                        "default": 5,
                        "min": 2,
                        "max": 8,
                        "step": 1,
                        "tooltip": "Uniform visual samples per reference video for Qwen-VL analysis.",
                    },
                ),
                "max_reference_media_seconds": (
                    "STRING",
                    {
                        "default": f"{MAX_REFERENCE_MEDIA_SECONDS:.0f}",
                        "multiline": False,
                        "tooltip": "Maximum duration in seconds for each reference video or audio and the total per media type. Leave blank for 15. Enter a value from 2 to 120; for example 16 or 20.5.",
                    },
                ),
            },
            "optional": {
                "reference_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Describe any <Audio N> dialogue, voice, or music content here; Qwen-VL receives audio metadata but not raw audio.",
                    },
                ),
                "manual_lora_trigger_words": ("STRING", {"multiline": True, "default": ""}),
                **optional_images,
                **optional_videos,
                **{
                    f"reference_video_audio_{index}": (
                        "AUDIO",
                        {
                            "tooltip": f"Audio paired with reference_video_{index}. Connect the matching audio output from your video loader; leave empty for a visual-only video reference.",
                        },
                    )
                    for index in range(1, MAX_REFERENCE_VIDEOS + 1)
                },
                **optional_audios,
            },
        }

    @classmethod
    def IS_CHANGED(cls, **_: Any) -> float:
        # Prompt refinement is intentionally evaluated on every queued execution.
        return float("nan")

    def refine(
        self,
        brief: str,
        engine: str,
        model_name: str,
        mmproj_name: str,
        vision_handler: str,
        context_length: int,
        gpu_layers: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        keep_model_loaded: bool,
        lora_preset_1: str = LORA_PRESET_NONE,
        lora_preset_2: str = LORA_PRESET_NONE,
        lora_preset_3: str = LORA_PRESET_NONE,
        video_frame_samples: int = 5,
        max_reference_media_seconds: float | str = MAX_REFERENCE_MEDIA_SECONDS,
        reference_text: str = "",
        manual_lora_trigger_words: str = "",
        **assets: Any,
    ) -> tuple[Any, ...]:
        if not brief.strip():
            raise PromptRefinerError("Write a prompt brief before running the node.")

        try:
            raw_max_media_seconds = str(max_reference_media_seconds).strip()
            max_media_seconds = (
                float(raw_max_media_seconds) if raw_max_media_seconds else MAX_REFERENCE_MEDIA_SECONDS
            )
        except (TypeError, ValueError) as error:
            raise PromptRefinerError(
                "max_reference_media_seconds must be a number from 2 to 120, for example 16 or 20.5."
            ) from error
        if not MIN_REFERENCE_MEDIA_SECONDS <= max_media_seconds <= MAX_CONFIGURABLE_REFERENCE_MEDIA_SECONDS:
            raise PromptRefinerError(
                f"max_reference_media_seconds must be between {MIN_REFERENCE_MEDIA_SECONDS:.0f} and "
                f"{MAX_CONFIGURABLE_REFERENCE_MEDIA_SECONDS:.0f}."
            )

        model_path = _resolve_model_file(model_name)
        mmproj_path = _resolve_model_file(mmproj_name, allow_none=True)
        has_video_inputs = any(
            assets.get(f"reference_video_{slot}") is not None
            for slot in range(1, MAX_REFERENCE_VIDEOS + 1)
        )
        has_audio_inputs = any(
            assets.get(f"reference_audio_{slot}") is not None
            for slot in range(1, MAX_REFERENCE_AUDIOS + 1)
        ) or any(
            assets.get(f"reference_video_audio_{slot}") is not None
            for slot in range(1, MAX_REFERENCE_VIDEOS + 1)
        )
        if (has_video_inputs or has_audio_inputs) and engine != FULL_REFERENCE_ENGINE:
            raise PromptRefinerError("reference_video and reference_audio inputs are available only in MiniMax H3 full reference mode.")

        image_references = _get_image_inputs(assets)
        sample_count = min(8, max(2, int(video_frame_samples)))
        video_references = _get_video_inputs(assets, sample_count, max_media_seconds)
        audio_references = _get_audio_inputs(assets, max_media_seconds)
        embedded_video_audio_references = _get_video_audio_inputs(assets, max_media_seconds)
        lora_triggers = _selected_lora_triggers(
            (lora_preset_1, lora_preset_2, lora_preset_3), manual_lora_trigger_words
        )
        if audio_references and not (image_references or video_references):
            raise PromptRefinerError("MiniMax H3 full reference requires at least one image or video when audio is supplied.")
        videos_by_slot = {item.slot: item for item in video_references}
        for item in embedded_video_audio_references:
            video = videos_by_slot.get(item.video_slot)
            if video is None:
                raise PromptRefinerError(
                    f"reference_video_audio_{item.video_slot} is connected, but reference_video_{item.video_slot} is not."
                )
            if abs(item.audio.duration - video.duration) > 0.5:
                raise PromptRefinerError(
                    f"reference_video_audio_{item.video_slot} is {item.audio.duration:.2f}s, but "
                    f"reference_video_{item.video_slot} is {video.duration:.2f}s. Paired video and audio must describe the same clip."
                )
        video_duration = sum(item.duration for item in video_references)
        audio_duration = sum(item.duration for item in audio_references)
        if video_duration > max_media_seconds + REFERENCE_DURATION_TOLERANCE_SECONDS:
            raise PromptRefinerError(
                f"Reference videos total {video_duration:.2f}s; the configured total limit is "
                f"{max_media_seconds:.2f}s."
            )
        if audio_duration > max_media_seconds + REFERENCE_DURATION_TOLERANCE_SECONDS:
            raise PromptRefinerError(
                f"Reference audios total {audio_duration:.2f}s; the configured total limit is "
                f"{max_media_seconds:.2f}s."
            )
        normalized_brief = _normalize_asset_links(
            brief,
            image_references,
            video_references,
            audio_references,
            embedded_video_audio_references,
        )
        if (image_references or video_references) and not mmproj_path:
            raise PromptRefinerError("Reference images or video samples require a matching mmproj GGUF selected from models/LLM.")

        key = _RuntimeKey(
            model_path=model_path,
            mmproj_path=mmproj_path,
            handler_choice=vision_handler,
            n_ctx=context_length,
            n_gpu_layers=gpu_layers,
        )
        runtime = _RuntimeCache.get(key)
        messages = _build_request(
            engine,
            normalized_brief,
            reference_text,
            image_references,
            video_references,
            audio_references,
            embedded_video_audio_references,
            lora_triggers,
        )

        try:
            response = runtime.llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
            )
            content = response["choices"][0]["message"]["content"]
            prompt = _ensure_lora_triggers(_strip_reasoning(content or ""), engine, lora_triggers)
            if not prompt:
                raise PromptRefinerError("The local model returned no final prompt.")
            debug = (
                f"local llama.cpp | model={Path(model_path).name} | "
                f"mmproj={Path(mmproj_path).name if mmproj_path else 'none'} | "
                f"images={len(image_references)} | videos={len(video_references)} ({video_duration:.2f}s) | "
                f"audios={len(audio_references)} ({audio_duration:.2f}s) | "
                f"paired_video_audios={len(embedded_video_audio_references)} | media_limit={max_media_seconds:.2f}s | "
                f"lora_triggers={len(lora_triggers)} | "
                f"cached={keep_model_loaded}"
            )
            return (prompt, debug)
        finally:
            if not keep_model_loaded:
                _RuntimeCache.unload()


NODE_CLASS_MAPPINGS = {"Qwen38LocalPromptRefiner": Qwen38LocalPromptRefiner}
NODE_DISPLAY_NAME_MAPPINGS = {"Qwen38LocalPromptRefiner": "Qwen3.8 Local Prompt Refiner (GGUF)"}

# Qwen3.8 Local Prompt Refiner for ComfyUI

This custom node runs a local Qwen-VL GGUF directly through `llama-cpp-python`. It never starts or calls Ollama. The node accepts a prompt brief and up to nine `IMAGE` references, then returns an engine-native prompt string ready to connect to a ComfyUI text input.

> Model weights and vision-projector files are intentionally not included. Download a compatible Qwen3.8-VL GGUF and matching `mmproj` separately, then place them in `ComfyUI/models/LLM`.

## Model downloads

- **Exact prompt-refiner package for Ollama:** [starnodes/qwen3.8-vl-27b-promptrefiner-abliterated](https://ollama.com/starnodes/qwen3.8-vl-27b-promptrefiner-abliterated)
- **Direct GGUF download for this node:** [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF)

The Hugging Face download is the compatible Abliterated base GGUF, rather than the Starnodes prompt-refiner package. This node includes the prompt-refiner routing profile separately. Download a main model and a matching vision projector from the same Hugging Face repository, for example `Qwen3.8-27B-ABLITERATED-Q5_K_M.gguf` plus `mmproj-Qwen3.8-27B-ABLITERATED-F16.gguf`. Configure the desired context length in the node itself (for example, `32768`); it is not determined by the filename.

## Install

1. Copy this folder into `ComfyUI\custom_nodes\ComfyUI-Qwen38-Local-PromptRefiner`.
2. Put the Qwen GGUF and its *matching* vision projector GGUF in `ComfyUI\models\LLM`.
   - Model example: `Huihui-Qwen3.8-27B-abliterated-Q4_K_L.gguf`
   - Projector example: `mmproj-model-bf16.gguf`
3. In the ComfyUI Python environment, install a GPU-enabled `llama-cpp-python` build appropriate for the installed CUDA/driver version. Then restart ComfyUI.

The generic package in `requirements.txt` is a CPU fallback. For NVIDIA CUDA on Windows, install the matching CUDA wheel instead; the `llama-cpp-python` project documents the available CUDA wheel indexes.

## Node usage

Add **LLM / Prompt Refiner → Qwen3.8 Local Prompt Refiner (GGUF)**.

- Select the model GGUF and matching `mmproj` GGUF from `models/LLM`.
- Set **GPU layers** to `-1` to offload all compatible layers to the GPU.
- Use **MiniMax H3 full reference** for the six-section full-reference output format.
- Attach up to nine `IMAGE` references, three native ComfyUI `VIDEO` references, and three `AUDIO` references. Slots retain their actual labels: `reference_image_2` becomes `<Picture 2>`, `reference_video_1` becomes `<Video 1>`, and `reference_audio_3` becomes `<Audio 3>`.
- Full-reference videos and audio must each be 2–15 seconds; videos together and audio together must each total no more than 15 seconds. Audio cannot be the only reference asset.
- The node uniformly samples 2–8 visual frames from each reference video (default 5) and sends those frames to Qwen-VL. Qwen3.8-VL does not directly receive or transcribe raw audio, so write required dialogue, voice identity, sound, or music details in `reference_text`, for example: `<Audio 1>: retain the supplied phone-call voice and wording.`
- In `brief`, use `@reference_image_1`, `@reference_video_1`, or `@reference_audio_1` to create a validated link to that exact connected port. The node converts these to `<Picture 1>`, `<Video 1>`, and `<Audio 1>` before inference; a link to an unconnected port raises a clear error.
- The `brief` text area has **@ autocomplete**: after connecting assets, type `@` and choose a displayed image, video, or audio slot with the mouse, arrow keys + Enter, or Tab. Only connected ports are shown.
- Leave **keep model loaded** disabled to release VRAM after each queue execution; enable it only when repeatedly prompting with the same model.

## LoRA trigger presets

The node can preserve LoRA trigger words without adding a seventh field to MiniMax H3 full-reference output.  Edit
`lora_trigger_presets.json` beside `nodes.py`, then restart ComfyUI (or reload custom nodes) so the preset names appear
in the three **lora preset** dropdowns:

```json
{
  "presets": {
    "My motion LoRA": ["exact_trigger_one", "exact_trigger_two"],
    "My style LoRA": ["exact_style_trigger"]
  }
}
```

Use up to three saved presets at once.  For a one-off LoRA, paste comma- or newline-separated values into
**manual lora trigger words**.  Trigger text is treated as literal data: the node asks the model to preserve each term
verbatim, and for MiniMax H3 full-reference output it inserts any missing terms inside `detailed_description` only.

## Compatibility notes

The node requires a current `llama-cpp-python` build with the generic `MTMDChatHandler` or the `Qwen25VLChatHandler` and a matching projector. Start with **auto (MTMD)**. If the selected build does not expose that handler, update `llama-cpp-python`; the node reports a clear error instead of silently falling back to text-only inference.

`mmproj` files are model-specific. Do not combine a projector from a different Qwen-VL conversion with the Qwen GGUF.

## Outputs

- `prompt`: local model's final prompt with common `<think>` / `<thinking>` blocks removed.
- `debug`: local execution summary, including model, projector, reference count, and cache setting.

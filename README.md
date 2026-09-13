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
- Use **MiniMax H3 first-last-frame (FL2V)** for first/last-frame video prompts. It keeps the existing image slots: `reference_image_1` is the optional first-frame keyframe, `reference_image_2` is the optional last-frame keyframe, and `reference_image_3`–`_9` are supplemental visual references. Connect either slot 1, slot 2, or both: a slot-2-only workflow is valid when you only want to constrain the ending frame.
- Attach up to nine `IMAGE` references, three native ComfyUI `VIDEO` references, and three `AUDIO` references. Slots retain their actual labels: `reference_image_2` becomes `<Picture 2>`, `reference_video_1` becomes `<Video 1>`, and `reference_audio_3` becomes `<Audio 3>`.
- A video can be visual-only, or can have a matched audio reference. For video plus its original audio, connect your loader's `VIDEO` output to `reference_video_1` and its matching `AUDIO` output to `reference_video_audio_1`. Connect those same original outputs to the Director H3 inputs `ref_video_0` and `ref_video_audio_0`. Leave `reference_video_audio_1` empty for a visual-only reference. The mappings are `1 → 0`, `2 → 1`, and `3 → 2` for the Director H3 ports. A paired audio input requires its matching video input and must describe the same clip (within 0.5 seconds).
- **max reference media seconds** is a numeric text field that defaults to 15 seconds (leave it blank to use 15), following MiniMax H3 full-reference guidance. Enter a value from 2 to 120 when your downstream workflow accepts a longer downloaded source; the value applies to each reference video/audio and to the total duration per media type. The node retains a 0.05-second tolerance for media timestamp rounding. Audio cannot be the only reference asset.
- The node uniformly samples 2–8 visual frames from each reference video (default 5) and sends those frames to Qwen-VL. Qwen3.8-VL does not directly receive or transcribe raw audio, so write required dialogue, voice identity, sound, or music details in `reference_text`, for example: `<Audio 1>: retain the supplied phone-call voice and wording.`
- In `brief`, use `@reference_image_1`, `@reference_video_1`, `@reference_video_audio_1`, or `@reference_audio_1` to create a validated link to that exact connected port. The node converts these to the matching `<Picture 1>`, `<Video 1>`, `<Video 1> paired audio track`, or `<Audio 1>` labels before inference; a link to an unconnected port raises a clear error.
- **external brief** is an optional `STRING` socket for a text/prompt node. A non-empty connected value takes priority over the editable `brief` widget, so you can retain and revise the brief for an earlier shot outside this node. After it is connected, the upstream editable text box receives the same `@` selection menu as this node's own brief widget; its choices are taken from the Qwen node's connected reference ports. The same `@reference_image_1`, `@reference_video_1`, `@reference_video_audio_1`, and `@reference_audio_1` tags are validated and mapped in external text.
- The `brief` text area has **@ autocomplete**: after connecting assets, type `@` and choose a displayed image, video, video-audio, or standalone audio slot with the mouse, arrow keys + Enter, or Tab. Only connected ports are shown.
- Leave **keep model loaded** disabled to release VRAM after each queue execution; enable it only when repeatedly prompting with the same model.

## Long Video Planner for MiniMax H3 Director

Add **LLM / Prompt Refiner → Qwen3.8 Long Video Planner → MiniMax H3 Director** when one story must span multiple
Director groups. The planner calls local Qwen once, splits the whole brief and dialogue into H3-safe 4–15 second
segments, then returns a complete `MMX_DIR_GROUP` list. It is compatible with the external-group API of
[AIMixer/ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director); do **not** put a
`Director Groups Combine` node between this planner and the Director when using the **direct** output path. The
separate **materialize native groups** workflow below deliberately creates a Combine node after you have reviewed a
plan.

1. Set **director mode**:
   - **MiniMax H3 Director FL2V (i2v_groups)** → connect `director_groups` directly to Director **i2v_groups** and
     set the Director task type to `fl2v`.
   - **MiniMax H3 Director R2V (r2v_groups)** → connect `director_groups` directly to Director **r2v_groups** and
     set the Director task type to `r2v`.
2. Set **total duration seconds** (4–120) and **segment duration seconds** (4–15). The latter is a preferred beat
   length: the planner creates `ceil(total / segment)` groups, then Qwen assigns each group a story-driven 4–15 second
   duration while keeping the exact total. For example, 60 seconds at a 10-second target produces six groups, but a
   natural rhythm can be `7 / 12 / 8 / 11 / 10 / 12` seconds instead of six mechanically equal clips.
   If Qwen makes a simple arithmetic mistake (for example, its six beats total 70 seconds for a 60-second project), the
   node automatically rescales those beats to the exact requested total while preserving their relative rhythm and the
   legal 4–15 second limit. It does not spend another Qwen inference merely to correct duration math.
   **planning quality** controls both prompt density and the H3 production standard, without exposing a `<think>`
   process. **Balanced (recommended)** produces a director-ready 180–280-word R2V beat: concrete camera, action,
   reference retention, sound, and an ending state for the next group. **Fast (compact)** preserves the older
   90–150-word output. **High quality (richer prompts)** uses 260–380 words per R2V beat and gives complex dialogue,
   visual references, sound, continuity, and shot timing more output budget. It is not merely longer prose: every
   profile now requires observable action, timestamped internal shots when a cut is warranted, stable speaker IDs,
   dialogue pacing, physical sound, and a usable continuity handoff. Use **context length = 32768** for Balanced or
   High quality when VRAM permits; 16K can constrain a multi-group plan with visual references. The effective token
   ceiling and a context recommendation are printed in `debug`. All profiles request zero reasoning tokens from
   compatible llama.cpp builds, so they return final prompts directly rather than spending time on visible `<think>`
   output.
3. Put the complete story and all dialogue in **brief**. The node preserves speaker IDs and splits dialogue at natural
   meaning boundaries rather than truncating it at a group boundary. Use `@reference_image_1`,
   `@reference_video_1`, `@reference_video_audio_1`, and `@reference_audio_1` to bind exact connected assets.
   You may instead connect a reusable text/prompt node to **external brief**; its non-empty text has priority and
   supports the exact same `@` asset tags.
4. In Director, enable **segment continuity / 段间引导**. Its default 22-frame overlap is appropriate for ordinary
   long-form continuation. The planner marks every later group as “reference previous segment” unless the brief asks
   for a hard scene break.

   The planner's `CONTINUITY_FROM_PREV` choice is transferred to each Director group: a direct continuation is on;
   a hard cut, new camera setup, time/location jump, insert, or explicit `硬切 / 镜头切到 / 新场景` is off. When at
   least one planned seam continues, the compatible Director UI turns on its master **segment continuity** switch and
   shows the normal per-group **引用上段** checkbox. You can still toggle any individual seam in Director after the
   plan has appeared; re-running the planner replaces those generated defaults.
5. The planner defaults to **Preview plan only (Director blocked)**. Queue once to create the editable Director cards;
   Director receives a silent execution blocker and does not sample. Review prompts, durations, references, and each
   **引用上段** switch. Then change **director execution** to **Release cached plan to Director** and queue again.
   Release never calls Qwen: it sends this Planner node's latest successful preview to Director. If the brief,
   references, duration, model settings, or other planner inputs changed since Preview, Release deliberately still
   uses the old preview and marks `release_last_preview_input_changed` in `debug`; choose Preview once more when you
   want those edits to take effect. The in-memory cache is cleared when ComfyUI restarts.

### Materialize native Director Groups (recommended for review)

After a successful **Preview plan only** run, the Qwen planner node shows a **生成原生 Director Groups** button (the
same command is available from its right-click menu). It converts the reviewed Qwen plan into the Director's own graph
nodes, without patching or depending on a modified Director frontend:

1. It creates one native `MiniMax H3 Director Group (Reference to Video)` node per R2V segment, or one native
   `MiniMax H3 Director Group (Image to Video)` node per FL2V segment.
2. It writes each segment's generated prompt and story-driven duration into that node, then copies only the exact
   `Picture / Video / paired Video Audio / Audio` wires that Qwen assigned to the segment. FL2V maps project Picture 1
   to only the first group’s first-frame input and project Picture 2 to only the last group’s last-frame input.
3. It creates a native `MiniMax H3 Director Groups Combine`, connects every new Group in chronological order, and
   connects the Combine output to Director `r2v_groups` or `i2v_groups`. The previous direct `director_groups` wire is
   removed so the Director sees only its own native graph structure.
4. It sets the Director task to R2V or FL2V, transfers Qwen's per-segment **引用上段** choices, and enables the
   Director master **段间引导** only when at least one planned seam is continuous.
5. It sets the Qwen planner node to **Never**. Therefore the next Queue runs the native Groups and Director without
   another Qwen inference. To revise the story later, change the Qwen node back to **Always**, run Preview again, then
   materialize the replacement plan. Clicking materialize again asks before replacing only the nodes tagged as generated
   by that same Qwen node; it never deletes your unrelated Director Groups.

This two-queue design is intentional: ComfyUI constructs its execution graph before a queue begins, so Group nodes
created after Qwen finishes cannot execute during that same queue. It lets you inspect and edit native Director group
cards, source-media wiring, durations, selection, and continuity before sampling. It is also robust to future Director
updates because this plugin creates Director's published native nodes instead of modifying Director files.

### FL2V long projects

- `reference_image_1` is the optional project first frame and is sent only to group 1.
- `reference_image_2` is the optional project final frame and is sent only to the final group.
- Intermediate groups are prompt-only; Director's segment continuity carries the previous generated picture and audio
  forward. Slot 1 only, slot 2 only, or both are valid.
- `reference_image_3`–`_9` can inform Qwen's planning, but Director FL2V accepts only the group first/last frames;
  choose R2V when every group needs multiple reference images, videos, or audios.

### R2V long projects and media wiring

- `reference_image_1`–`_9` are normal R2V image references.
- `reference_video_frames_1`–`_3` are **IMAGE frame batches**, matching the Director Group R2V input type. Connect the
  frame-batch `IMAGE` output of your video loader; do not connect a native `VIDEO` handle here. The same frame batch
  is passed to Director and sampled visually by Qwen.
- Connect matching source audio to `reference_video_audio_1`–`_3`; a paired audio slot requires the same-numbered video
  frame batch. Use `reference_audio_1`–`_3` for standalone audio references.
- Qwen assigns connected assets to each group in its plan. Review **segment prompts** or **plan json** before sampling;
  the actual Director groups receive exactly those assigned assets, using Director's internal zero-based mapping while
  prompts retain one-based `<Picture N>`, `<Video N>`, and `<Audio N>` labels.
- Every connected image is treated as a stable project reference for R2V: it is attached to every generated Director
  group. If the local model forgets a `<Picture N>` label, the node automatically adds visible `<Picture N>` statements
  to `subject_definitions` and `retention_analysis` before Director receives the prompt. This prevents a visually
  described character from silently losing its connected image reference.

Outputs:

- `director_groups`: ready-to-connect multi-group list for the correct Director port.
- `segment_prompts`: readable Group 1…N prompts with time ranges and continuity flags.
- `plan_json`: the serializable plan and asset assignment record, useful for review or copying.
- `debug`: local execution summary including the required Director port.

The planner uses a newline-safe marked response format internally instead of embedding every multi-line H3 prompt in
one large JSON string. Existing valid JSON plans remain readable, but new long plans are more resilient to dialogue,
quotes, and long multi-group output.

After the planner has run once, a compatible MiniMax H3 Director timeline also mirrors the dynamic plan: its external
group panel shows the actual group count, each group's duration and prompt, plus the connected image/video/audio
references. This is a **read-only preview** of a generated plan; edit the planner brief and queue it again to change
the groups. Before the first successful planner run there is no dynamic plan for Director to display.

## Prompt example library

`prompt_example_presets.json` beside `nodes.py` is a user-editable example library. It lets Qwen copy a selected
example's output structure, camera/detail density, and label placement without retraining or modifying the GGUF.
Each node run injects **only the selected example**, so a large personal library does not make ordinary runs slower.

Add one entry per tested online prompt. `name` values must be unique and `engine` is the **destination engine** selected
in the node's dropdown. `output_example` may be a copied prompt from another H3 mode: at execution, the plugin adapts
a three-field H3 example into six-field full-reference format (or the reverse) in memory, while preserving its useful
camera/action density. It never rewrites your library source file. A native example already using the destination
engine’s field layout remains the strongest option:

```json
{
  "presets": [
    {
      "name": "H3 full reference - office narrative",
      "engine": "MiniMax H3 full reference",
      "brief_example": "A night office scene with two supplied character references.",
      "output_example": "subject_definitions:\n<Subject 1> is derived from <Picture 1>, fully_preserved.\n\nsummary:\nA short night office scene.\n\nretention_analysis:\n<Picture 1>: fully_preserved.\n\ndetailed_description:\n[Shot 1] ...\n\noverall_soundscape:\nQuiet office ambience.\n\nnon_diegetic_music:\nN/A",
      "instruction": "Learn the six-field H3 full-reference structure and concise observational shot pacing."
    }
  ]
}
```

Restart ComfyUI or reload custom nodes after saving the file, then select the entry in **example preset**. A selected
example's `engine` must match the current engine; this prevents a FLUX example from contaminating an H3 run. For
example, a copied `integrated_multimodal_description` prompt saved with `engine: "MiniMax H3 full reference"` is
automatically wrapped into the six required fields before Qwen receives it. The `debug` output reports
`adapted_to_full_reference`, `adapted_to_three_field_h3`, or `native_*` so you can confirm what happened. The current
brief, connected `Picture` / `Video` / `Audio` references, selected LoRA triggers, and engine rules take priority. Do
not store an entire collection in one entry: add one good case per preset, and select just one at execution time.

### Directly pasting a multiline prompt

You may paste a copied prompt with normal line breaks, including dialogue quotation marks, directly between the quotes
of `output_example`. On load, the node automatically converts that **one field** into a valid JSON string in memory;
you do not need to manually replace every line break with `\n`. Keep the surrounding library metadata as normal JSON:
`name`, `engine`, `brief_example`, and `instruction` still require quoted JSON values, and `instruction` must remain
after `output_example`. If a preset still does not appear after a ComfyUI restart, the JSON is malformed outside of
`output_example` (for example, a missing comma or an unmatched quote in `name`).

If you accidentally paste another complete `{ "presets": [...] }` block below the first one, the node now merges the
consecutive documents in memory so existing presets keep working. It is still cleaner to add the new entry to the
first `presets` array; the plugin never rewrites your source library automatically.

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

Older workflows are migrated in the browser when they load. The migration restores widget values by their saved names,
so later plugin additions cannot shift `video_frame_samples`, `reference_text`, or `manual_lora_trigger_words` into one
another. After opening an old workflow once with the current plugin, save it to persist the corrected layout.

## Outputs

- `prompt`: local model's final prompt with common `<think>` / `<thinking>` blocks removed.
- `debug`: local execution summary, including model, projector, reference count, paired-video audio count, and cache setting.

# Local Prompt Refiner vs optional H3 rewriter — A/B fixtures

These folders are synthetic review fixtures. They compare the current **Qwen3.8 Local Prompt Refiner** (and, where noted, the Long Video Planner contract) with an optional short-segment **MiniMax-H3-Prompt-Rewriter** backend (`lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA`). Nothing here calls a model, loads a GGUF, or replaces the planner.

Read a case without a GPU. Run a model only when you want a live sample to score against the notes.

## What this pack is

- Chinese NV-style external briefs, using a character name plus `<Subject N>`.
- Notes for the local plugin’s current contract: Chinese prompt bodies, duration tags, subject binding, and no brief echo.
- Notes for an English six-section H3 rewrite, following the public full-reference guide `VIDEO_PROMPT_WRITING_GUIDE_ref_en.md` on the MiniMax-H3 model card.
- A CN→EN step **before** H3, and an EN→CN step (or a bilingual keep) **after** H3. Both bridges belong in PromptRefiner. NVStudio’s export language stays as it is.

## What this pack is not

- Not an H3 backend, client, or route. Do not treat these notes as implemented behavior.
- Not a Long Video Planner replacement. H3 rewrites one short segment (about 4–15 seconds). Case 03 is the only brief a planner would split, and its H3 notes cover the second segment only.
- Not a place for images, video, audio, production dumps, handwritten E01 shot plans, secrets, or weights. Reference slots are labels. Media stays on the reviewer machine.
- Not adult content. Every beat is fictional and SFW.

## Cases

| Case | Brief | Local path to score | H3 path to score |
| --- | --- | --- | --- |
| `01-night-market-lantern` | One person, 8 seconds, no dialogue | Local Prompt Refiner, full reference, output language **中文** | One English six-section rewrite of that same 8-second beat |
| `02-library-return` | Two people, 12 seconds, one spoken line, one voice-timbre label | Same local refiner path, plus verbatim dialogue and `<Audio 1>` | One English rewrite; dialogue stays the original Chinese line |
| `03-rooftop-two-beats` | Two beats, 8s + 10s, total 18s | Long Planner structure for both beats; refine notes for each card | H3 sees **only beat 2**, with beat 1’s handoff as context |

Each case folder contains:

- `brief.md` — paste this into **external brief**. It is the whole brief, not a commentary.
- `input.json` — the same brief plus language, duration, and empty reference placeholders for a future harness.
- `local_refiner_expected_notes.md` — what to accept or reject on the local path.
- `h3_rewrite_target_notes.md` — the high-level English six-section target, and where the two bridges sit.

`input.json` `"brief"` is `brief.md` with the final newline removed. The JSON does not embed media.

## Manual compare

1. Pick one case. Read `brief.md` and both notes before queueing anything.
2. Local side, when you do run ComfyUI: add **Qwen3.8 Local Prompt Refiner (GGUF)**, engine **MiniMax H3 full reference**, **output language** left at **中文**. Paste `brief.md` into **external brief** (a connected string wins over the widget). Leave image, video, and audio sockets empty unless you supply your own files outside this repo. Queue once. Score `prompt` against that case’s local notes. `debug` is only a runtime trace.
3. Case 03 local planner, separately: add **Qwen3.8 Long Video Planner → MiniMax H3 Director**, mode **R2V**, total duration **18**, segment duration **10** (the average used to estimate group count, not a forced length), **output language** **中文**, **Per segment**. Preview only. Score the two cards against the planner section of the local notes. Do not send that 18-second plan to H3 and call the H3 result a long plan.
4. H3 side is external to this plugin. If a Ref2VA rewriter is available, build its English instruction with the bridge described in `h3_rewrite_target_notes.md`, run that one short segment, then apply the return bridge. If no rewriter is available, review the notes only.
5. Compare facts, not which language won. After the return bridge, both sides should be scorable as Chinese bodies (or as an explicit bilingual pair). Raw H3 English is expected **before** that return bridge and is not a local-refiner failure.

## Score the same facts on both sides

- Six full-reference fields, in order: `subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music`.
- People referred to by the brief’s name or by the `<Subject N>` already bound to that name. A bare `主体` or `主角` (including `该主体`, `此主体`, `主人公`, `主体人物`, `主角人物`) is a failure. Do not guess Subject 1.
- The story is rewritten as shot prose. The output is not a paste of `brief.md`, the `[STORY_BEATS]` block, or a `分镜` / `使用参考` list.
- `[Shot 1]` has no timestamp. A later shot carries an `MM:SS.mmm` time inside that segment. Spoken words, if any, sit inside `<d>[Language] …</d>` and match the brief verbatim.
- Ambience and foley stay in `overall_soundscape`. Audience-only music is `N/A` unless the brief asks for a score. Dialogue is not repeated in those two fields.

## Bridge, in one line

Chinese external brief → PromptRefiner CN→EN (names, tags, duration, and dialogue preserved) → optional H3 English six-section rewrite → PromptRefiner EN→CN or bilingual keep → compare with the local Chinese body. NVStudio is not part of that bridge.

# H3 rewrite notes — 01 night market lantern

H3 here means the optional Ref2VA rewriter (`MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA`), guided by `VIDEO_PROMPT_WRITING_GUIDE_ref_en.md`. This case is one 8-second full-reference generation. It is not a long-video plan. This plugin does not call that model.

## Where the bridges sit

1. **Before H3, in PromptRefiner.** Turn `brief.md` into an English instruction. Keep `周晚`, `<Subject 1>`, `<Picture 1>`, and the 8-second duration unchanged. Do not translate those tags into “subject” or “image 1” in a way that drops the bound name. There is no dialogue to carry. Do not attach a picture from this repository; there is not one.
2. **H3.** One English six-section rewrite of that single beat. Task type in `summary` is `[reference generation]` because `<Picture 1>` identifies 周晚 and is not declared as a first or last frame.
3. **After H3, in PromptRefiner.** Either rewrite the natural-language bodies into Simplified Chinese, or keep the English rewrite beside a Chinese body. Field labels, `<Subject 1>`, `<Picture 1>`, `fully_preserved`, `N/A`, `[Shot N]`, and `MM:SS.mmm` stay as written. NVStudio’s export language is not changed.

Compare the returned Chinese body (or the bilingual pair) with `local_refiner_expected_notes.md`. Raw English before the return bridge is expected and is not a local-refiner defect.

## Six-section shape

Not a model output. A passing rewrite covers these facts in this order.

- `subject_definitions`: `<Subject 1>` is 周晚, the person in `<Picture 1>`, at a wooden night-market lantern stall. State the brief’s visible props: one lit warm-orange paper lantern, two unlit white paper lanterns. Do not add a second person.
- `summary`: begins with `[reference generation]`. One short English paragraph: she hangs the lit lantern on the rail during this 8-second clip. Use only labels already defined.
- `retention_analysis`: one line for `<Subject 1>`, marker `fully_preserved`, naming the shots she appears in. The marker stays the English token. Partial visibility of the stall is not `partially_preserved`.
- `detailed_description`: English. A style sentence may sit before `[Shot 1]`. `[Shot 1]` has no timestamp and shows the hang, the hand on the lantern base, and her gaze on the flame. A later shot, if any, uses `[Shot N] At MM:SS.mmm` inside the 8 seconds, with no cut denser than about three seconds. Camera movement is a sentence, not a keyword list. No `<d>` line.
- `overall_soundscape`: distant night-market murmur and the paper lantern tapping the rail. No dialogue and no music.
- `non_diegetic_music`: `N/A`.

## Expected differences from the local refiner

- The official guide asks for English section bodies and a generation-length `detailed_description` (often a few hundred English words). The local default is a shorter Simplified Chinese body. Length and language are not the A/B score. The score is the action, the binding, the clock, the sound split, and the absence of a brief paste.
- The local `summary` does not have to start with `[reference generation]`. The H3 rewrite does.
- Do not ask this rewriter to emit `DURATION_SECONDS`, `HANDOFF`, or a second Director group.

# H3 rewrite notes — 02 library return

Optional Ref2VA rewriter, one 12-second full-reference segment. Not a planner run. The backend is not part of this plugin.

## Where the bridges sit

1. **Before H3, in PromptRefiner.** English instruction from `brief.md`. Preserve `林舟`, `沈宁`, `<Subject 1>`, `<Subject 2>`, `<Picture 1>`, `<Picture 2>`, `<Audio 1>`, the 12-second duration, and the spoken line `这本书还在。` character for character. The bridge may explain that `<Audio 1>` is timbre only. It must not paraphrase the line into English.
2. **H3.** English six-section rewrite. `summary` task type is `[reference generation + audio reference]`: the pictures guide identity, and `<Audio 1>` guides 沈宁’s timbre without copying the signal. That is not `audio reuse` and not `fully_copy`.
3. **After H3, in PromptRefiner.** Translate the prose bodies to Simplified Chinese, or keep English and Chinese together. The `<d>[Chinese] 这本书还在。</d>` span is copied, not translated. NVStudio’s export language stays untouched.

## Six-section shape

Not a model output.

- `subject_definitions`: `<Subject 1>` is 林舟 in `<Picture 1>`. `<Subject 2>` is 沈宁 in `<Picture 2>`. `<Audio 1>` is the voice-timbre reference for `<Subject 2>` and reuses her speaker id, for example `(S2)`. Speaker order follows who actually speaks. Only 沈宁 speaks, so she may be `(S1)` if the rewriter numbers speakers by vocal order. Either stable id is acceptable if it is the same id in the definition and in the shot. Do not number a silent 林舟 as a speaker.
- `summary`: `[reference generation + audio reference]`, then the book pushed across the return desk and the one line. No new labels.
- `retention_analysis`: `<Subject 1>` and `<Subject 2>` use `fully_preserved` with the shots they appear in. `<Audio 1>` uses the audio marker `reference`, not `fully_copy`. Do not put `(S1)` or `(S2)` in this section.
- `detailed_description`: English action. `[Shot 1]` untimed: 林舟 pushes the dark-blue book, spine toward 沈宁. A later shot may cut to 沈宁’s hand on the corner and the line. The line is only `<d>[Chinese] 这本书还在。</d>` from `<Subject 2> (Sx)`. Camera, hand contact, and the desk are in the sentences. No second line of dialogue.
- `overall_soundscape`: page turns and the air vent. No quoted dialogue, no `<Audio 1>`, no music.
- `non_diegetic_music`: `N/A`.

## Expected differences from the local refiner

- H3 writes the section bodies in English and may run longer than the local Chinese body. Score binding, the verbatim line, the timbre-versus-foley split, and the single-segment clock.
- Local retention may be a short `<Picture N>: fully_preserved.` line. H3’s guide uses a longer `<Subject N> (appears in [Shot …]): fully_preserved - …` line. Both pass if the person and the marker are right.
- Do not send this 12-second beat through H3 as a multi-group Director plan.

# Local refiner notes — 02 library return

Score the **Qwen3.8 Local Prompt Refiner** on engine **MiniMax H3 full reference**, **output language** **中文**. One 12-second beat, two bound people, one spoken line, one voice-timbre label. The Long Video Planner is not required.

## Duration

- 12 seconds is one legal H3 segment (4–15). Do not split 林舟’s push and 沈宁’s line across two groups for this A/B.
- `[Shot 1]` has no timestamp. A later shot uses an `MM:SS.mmm` clock inside those 12 seconds. This brief does not request a hard cut, so cuts stay about three seconds apart.
- The spoken line is six Chinese characters. It fits the segment. Do not stretch it, subtitle it, or move it into `overall_soundscape`.

## Chinese body

- Natural-language bodies are Simplified Chinese: subject lines, `summary`, retention prose, shot prose, and soundscape.
- Labels that stay put: the six field names, `<Picture 1>`, `<Picture 2>`, `<Subject 1>`, `<Subject 2>`, `<Audio 1>`, `<d>`, `[Shot N]`, `fully_preserved`, `N/A`.
- `non_diegetic_music` is `N/A`.

## Subject binding

- `<Subject 1>` is 林舟 from `<Picture 1>`. `<Subject 2>` is 沈宁 from `<Picture 2>`.
- Every mention of either person uses that name or that `<Subject N>`. A bare `主体` or `主角` fails even if 林舟 and 沈宁 are named elsewhere in the same prompt.
- Do not assign 林舟’s action to `<Subject 2>`, and do not guess a third person.
- `<Audio 1>` is 沈宁’s voice-timbre reference only. It is not ambient, not the page-turn, and not a copied soundtrack. The retention line says so. `overall_soundscape` does not cite `<Audio 1>`.

## Dialogue

- The only spoken words are exactly `这本书还在。`
- They appear once, as `<d>[Chinese] 这本书还在。</d>`, spoken by 沈宁 / `<Subject 2>` with a stable speaker id such as `(S2)`.
- Do not translate the line. Do not add a reply for 林舟.

## No brief echo

- Shot prose shows the observable beat: evening community-library return desk, 林舟 slides the dark-blue book onto the counter with the spine toward 沈宁, 沈宁 sits behind the counter, left hand on the book corner, looks at 林舟, then speaks.
- Reject a paste of the duration line, the reference list, or the whole `brief.md`.
- Soundscape is page turns and the air-vent, in Chinese, with no dialogue and no score.

## Ending state

- The book is on the counter under 沈宁’s left hand. 林舟 is still on the public side of the desk. 沈宁 has finished the line.

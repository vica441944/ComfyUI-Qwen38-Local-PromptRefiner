# H3 rewrite notes — 03 rooftop, second beat only

Long Planner stays the owner of the 18-second split. The optional H3 rewriter (`MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA`, guide `VIDEO_PROMPT_WRITING_GUIDE_ref_en.md`) may rewrite **beat 2 only**, as one 10-second full-reference segment. Asking H3 for both beats, a `=== SEGMENT ===` plan, or an 18-second timeline is a failed compare, not a planner result.

## Where the bridges sit

1. **Before H3, in PromptRefiner.** English instruction for beat 2 only. Include the beat 1 handoff as continuity context, in English, without sending beat 1 as a second video to generate: can still in her right hand, she stands on the planter’s left, camera on the leaves. Preserve `陈夏`, `<Subject 1>`, `<Picture 1>`, `<Environment>`, `<Picture 2>`, and **10 seconds**. Do not include the `[STORY_BEATS]` block as text the model should echo.
2. **H3.** One English six-section rewrite. `summary` uses `[reference generation]`. `<Picture 1>` and `<Picture 2>` guide identity and the planter. They are not declared as first-frame or last-frame keyframes, so this is not `keyframe completion`.
3. **After H3, in PromptRefiner.** Chinese body, or bilingual keep. Structural tokens stay. No NVStudio language change. The returned Chinese group-2 prompt is what you set beside the local planner’s group-2 refine.

Beat 1 is scored only on the local planner notes.

## Six-section shape for beat 2

Not a model output.

- `subject_definitions`: `<Subject 1>` is 陈夏 in `<Picture 1>`, holding then releasing a silver watering can. `<Environment>` is the rooftop planter still in `<Picture 2>`, labeled `天台菜箱外景` (leaves, damp soil, wooden stool beside it). No second person.
- `summary`: `[reference generation]`. She sets the can on the stool, looks at the soil, then looks out at the buildings. Mention that the clip continues from the handoff rather than starting at the stairs.
- `retention_analysis`: `<Subject 1>` and the planter each `fully_preserved`, with the shots they appear in. A closer view of the soil does not become `partially_preserved`.
- `detailed_description`: English. The first sentence continues the handoff (can in the right hand, planter left side, camera toward the leaves). Then she sets the can on the stool, bends to the damp soil, and straightens toward the buildings. `[Shot 1]` has no timestamp. Any later `[Shot N] At MM:SS.mmm` falls inside 10 seconds. No dialogue.
- `overall_soundscape`: wind and distant traffic. No dialogue.
- `non_diegetic_music`: `N/A`.

## Expected differences from the local planner

- The planner also emits Chinese `BEAT` / `HANDOFF` cards and locked `DURATION_SECONDS`. H3 does not emit those cards. The 10-second length is an input to H3, not a field H3 is asked to re-decide.
- H3’s English body can be longer than the local Balanced band (about 180–280 English-word equivalents of Chinese). Do not fail the local group for being Chinese or shorter, and do not fail H3 for being English before the return bridge.
- A rewrite that replays the stair walk has borrowed beat 1 and fails this case.

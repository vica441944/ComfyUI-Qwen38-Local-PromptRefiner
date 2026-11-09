# Local refiner / planner notes — 03 rooftop two beats

This brief is the planner fixture. H3 A/B does not cover the whole 18 seconds. See `h3_rewrite_target_notes.md` for the one segment H3 is allowed to rewrite.

## Planner schedule

Use **Qwen3.8 Long Video Planner**, mode **R2V**, **output language** **中文**, **Per segment**, total duration **18**, segment duration **10**.

The `[STORY_BEATS]` rows are 8 seconds (`走到菜箱`) and 10 seconds (`放下水壶`). With total duration 18 and a segment widget of 10 or 15, those estimates fit without compression: one card each, allocations **8** seconds then **10** seconds, sum 18. The coverage knife names that map. A flat 9/9 split that ignores the two estimates should not be the schedule that refinement uses. A card over the segment cap, including a 21-second card, is rejected or split; this brief has no dialogue that would force a split.

- Group 1 `CONTINUITY_FROM_PREV` is false.
- Group 2 `CONTINUITY_FROM_PREV` is true. She is still at the planter; this is not a hard cut, a new location, or a `硬切`.
- Both beats show 陈夏 and the planter, so each card’s `PICTURES` line should list `1, 2`. Leave **Anchor all pictures** off. An empty `PICTURES` line attaches no stills and fails this brief. There is no video and no audio slot.
- `DIALOGUE` on both cards is `NONE`.
- Structure cards do not contain `subject_definitions` or `detailed_description`. Those appear only on the later per-segment refine.
- `BEAT` and `HANDOFF` sentences are Simplified Chinese. The labels `HANDOFF`, `BEAT`, `DIALOGUE`, `DURATION_SECONDS`, and `CONTINUITY_FROM_PREV` stay English.

## Group 1 content

- 陈夏 carries a silver watering can from the stair doorway to the planter and stops.
- Handoff, in Chinese prose under the English `HANDOFF:` label: the can is still in her right hand, she stands on the planter’s left, the camera faces the leaves.
- No spoken line.

## Group 2 content

- The refine starts from that handoff. It does not reintroduce the rooftop as a new scene.
- She sets the can on the wooden stool beside the planter, bends to look at the damp soil, then straightens and looks toward the distant buildings.
- Ending state: can on the stool, she is upright, gaze toward the buildings.
- `[Shot 1]` inside this group has no timestamp. A later shot’s `MM:SS.mmm` is measured inside the 10-second group, not from the start of the 18-second project, unless the text clearly says which clock it is. Prefer the in-group clock.
- Chinese bodies on all six fields. `non_diegetic_music` is `N/A`. Soundscape is wind and distant traffic only.

## Subject binding

- `<Subject 1>` is 陈夏 from `<Picture 1>`.
- `<Environment>` is the rooftop planter still, labeled `天台菜箱外景` on `<Picture 2>`. The planter is not a person and is not `<Subject 2>`.
- Bare `主体` or `主角` fails the brief before inference and fails a refined prompt. This fixture uses neither.

## No brief echo

- Reject a `detailed_description` that pastes `[STORY_BEATS]`, `1|8|-|走到菜箱`, `节拍1`, `节拍2`, or the handoff sentence unchanged as the whole shot body.
- The shot must be H3 prose: camera, hands, the can, the stool, the soil, the skyline. A one-line copy of `她把浇水壶放在菜箱边的木凳上` is not enough for the group 2 refine.

## Single-refiner misuse

Running only the Local Prompt Refiner on the entire 18-second brief produces one prompt, not two Director groups. That result cannot pass this planner checklist. It is out of scope for the H3 A/B as well.

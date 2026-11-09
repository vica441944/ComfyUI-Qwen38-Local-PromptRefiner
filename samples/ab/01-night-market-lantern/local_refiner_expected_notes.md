# Local refiner notes — 01 night market lantern

Score the **Qwen3.8 Local Prompt Refiner** on engine **MiniMax H3 full reference**, **output language** left at **中文**. This brief is one 8-second beat. It does not need the Long Video Planner. A one-group planner run may skip the structure pass and refine the brief directly; that single prompt is still judged by the list below.

## Duration

- The beat fits one segment. 8 seconds is inside the 4–15 second H3 segment range.
- `detailed_description` opens on `[Shot 1]` with no timestamp.
- A second shot, if the model cuts, is `[Shot 2] At MM:SS.mmm` (or the same clock inside a Chinese sentence, such as `在 00:03.000`). The time is strictly inside the 8 seconds and at least about three seconds after the start unless the brief asked for a hard cut. This brief does not ask for a hard cut.
- There is no `DURATION_SECONDS` line on the standalone refiner. Do not invent a multi-group timeline.

## Chinese body

- `summary`, the prose in `subject_definitions` and `retention_analysis`, `detailed_description`, and `overall_soundscape` are Simplified Chinese.
- Field labels stay English and in this order: `subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music`.
- `<Picture 1>`, `<Subject 1>`, `fully_preserved`, and `N/A` stay in that spelling.
- `non_diegetic_music` is the literal `N/A`. The brief asks for no audience score.

## Subject binding

- 周晚 is `<Subject 1>`, taken from `<Picture 1>`.
- The prompt uses `周晚` or `<Subject 1>` for her. It does not use a bare `主体` or `主角`, and it does not invent a second person.
- The two unlit white lanterns and the lit orange lantern are props, not extra people.
- No image file is in this repo. Wardrobe or face detail beyond the brief is optional and, if a reviewer later attaches a still, must follow that still. Missing hair color is not a fixture failure.

## No brief echo

- The shot prose states the visible action: she hangs the lit orange paper lantern on the rail, right hand stays on the lantern base, eyes go to the small flame, two unlit white lanterns remain on the stall.
- Reject a completion that pastes `brief.md` back, including the line `目标时长：8秒。单段。`, or that returns the brief unchanged.
- Reject a `分镜` / `使用参考` list. This brief has neither, so those lines should not appear.
- `overall_soundscape` is Chinese ambience and foley only: distant night-market murmur and the paper lantern tapping the rail. It does not repeat a plot summary and does not name `<Audio N>` (none is connected).

## Shot content to keep

- One continuous stall, night market, wood frame.
- Ending state a later beat could inherit: lantern is on the rail, her hand is still on the base, her gaze is on the flame.
- No dialogue tag. Do not invent a spoken line.

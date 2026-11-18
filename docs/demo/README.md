# Demo assets

`demo.gif` is a 75-second cut of a 9-minute session recording. It is a GIF
rather than an MP4 because GitHub renders images inline in a README and does
not embed video files.

## What the clip shows

| # | Source time | Content |
|---|---|---|
| 1 | 03:16 – 03:31 | Unverified request → identity interrupt |
| 2 | 03:56 – 04:26 | Phone number accepted, supervisor routes to `both`, merged answer |
| 3 | 07:35 – 08:05 | Follow-up answered from long-term memory |

## How it was produced

From the original `music-store-support-agents.mp4` (852×480, 30 fps, 9:06):

```bash
ffmpeg -i music-store-support-agents.mp4 -filter_complex "
  color=c=0x14161a:s=852x450:r=30:d=2.6,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='Music Store Support Agents':fontcolor=white:fontsize=36:x=(w-text_w)/2:y=(h-text_h)/2-26,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='LangGraph multi-agent support - live terminal demo':fontcolor=0x9aa4b2:fontsize=18:x=(w-text_w)/2:y=(h-text_h)/2+26[title];
  [0:v]trim=start=196:end=211,setpts=PTS-STARTPTS,crop=852:450:0:0[c1];
  [0:v]trim=start=236:end=266,setpts=PTS-STARTPTS,crop=852:450:0:0[c2];
  [0:v]trim=start=455:end=485,setpts=PTS-STARTPTS,crop=852:450:0:0[c3];
  [title][c1][c2][c3]concat=n=4:v=1:a=0,fps=12,split[a][b];
  [a]palettegen=max_colors=128:stats_mode=diff[p];
  [b][p]paletteuse=dither=bayer:bayer_scale=5
" -loop 0 demo.gif
```

The crop removes the OS taskbar; 12 fps and a 128-colour palette keep the
result at ~1.5 MB, which loads instantly on the repository page.

## Full recording

The 9-minute original is attached to the [`demo-v1` release](../../releases/tag/demo-v1)
rather than committed, since a 30 MB binary does not belong in git history.

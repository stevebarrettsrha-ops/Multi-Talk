# Multi-Talk

Audio-driven talking video, run locally. Give it a picture of one or two
people and what they say; get back a video of them saying it.

| Folder | What it is |
|---|---|
| [`multitalk-studio/`](multitalk-studio/) | The local app: setup, model downloads and the generator UI. **Start here.** |
| [`MultiTalk/`](MultiTalk/) | MeiGen's MultiTalk engine with offloading and tiled decode changes targeting an 8 GB card. |

To run it on Windows, double-click `multitalk-studio/run.bat`. On Linux or
macOS, run `multitalk-studio/run.sh`. Either one opens
<http://127.0.0.1:7806> and walks you through setup.

The [studio README](multitalk-studio/README.md) has the hardware notes for
an RTX 4060 (8 GB): 32 GB of system RAM, about 29 GB of downloads, and
minutes per clip.

The memory fixes reduce avoidable allocations; a full render with the real
weights on an 8 GB GPU has not been verified in this audit. Update both
folders, preserve your weights and data, and restart the studio.

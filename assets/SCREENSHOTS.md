# README images

Two images are used in the README. Both come from public models only, so nothing in them depends on a
private checkpoint. Check that before replacing either one: the UI shows the chosen model's file name.

**screenshot-ui.png** (hero image)

The web UI after one generation, with the "What EZ Comfy decided for you" panel open.

1. Run `python -m ez_comfy serve --port 7860` and open `http://127.0.0.1:7860`
2. Type a prompt, set the checkpoint to a public model such as `realvisxlV50_v50LightningBakedvae.safetensors`, click Generate
3. When the image appears, expand "What EZ Comfy decided for you" and capture the browser window (1440x1000)

**wan22-video-frames.jpg** (video section)

Four frames (first, one third, two thirds, last) from one Wan 2.2 text-to-video render, each resized to
320x320 and placed side by side (1280x320).

1. Render a clip: `python -m ez_comfy generate "<prompt>" --intent video --recipe video_wan22_t2v --timeout 1800`
2. Decode the MP4, take frames `0`, `n // 3`, `2 * n // 3` and `n - 1`, and paste them into one image

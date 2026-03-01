# Screenshot Guide

One screenshot is needed for the README hero image: **screenshot-ui.png**

**What:** The web UI with a completed generation showing an output image.

**How:**
1. Run `python -m ez_comfy serve --port 8088`
2. Open `http://127.0.0.1:8088`
3. Type a prompt, click Generate, wait for the result
4. Screenshot the full browser window showing the left panel + the generated image on the right

**Once captured:**
1. Save as `assets/screenshot-ui.png`
2. Remove the `assets/screenshot-*.png` line from `.gitignore`
3. `git add assets/screenshot-ui.png && git commit -m "Add UI screenshot"`

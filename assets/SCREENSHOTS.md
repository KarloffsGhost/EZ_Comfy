# Screenshot Guide

This folder holds the images referenced in README.md.
Add the four screenshots below and they will automatically appear in the README on GitHub.

---

## screenshot-ui.png  *(hero image — most important)*
**What:** The web UI with a completed generation showing an output image.

**How:**
1. Run `python -m ez_comfy serve --port 8088`
2. Open `http://127.0.0.1:8088`
3. Type a prompt (something visually interesting — e.g. "a cinematic portrait of an astronaut, golden hour lighting")
4. Click Generate and wait for the result
5. Screenshot the full browser window showing the left panel + the generated image on the right

**Ideal size:** 1400–1600px wide

---

## screenshot-ui-result.png  *(web UI detail)*
**What:** Close-up of the result panel showing the output image + the metadata strip below it (recipe · checkpoint · duration · seed).

**How:** Crop from the same session as above, focused on the right-hand output panel.

---

## screenshot-cli-check.png  *(CLI, quick to capture)*
**What:** Terminal output of `python -m ez_comfy check`.

**How:**
1. Open a terminal in the project folder
2. Run `python -m ez_comfy check`
3. Screenshot the full output (GPU line, inventory counts, capabilities list)

**Tip:** Use Windows Terminal with a dark theme for a clean look.

---

## screenshot-cli-plan.png  *(shows the intelligence)*
**What:** Terminal output of `python -m ez_comfy plan "a portrait of an astronaut"`.

**How:**
1. Run `python -m ez_comfy plan "a portrait of an astronaut"`
2. Screenshot the JSON output — the key things to show are `intent`, `recipe`, `checkpoint`, `family`, `params`, and the `recommendations` array

This screenshot is the most compelling for developers — it shows the model selection reasoning.

---

Once all four images are saved here, delete or ignore this SCREENSHOTS.md file.
It is not referenced from the README.

"""Judge prompt used in the VIF-Bench paper (identical for GPT-5, Gemini 2.5 Flash and Qwen3-VL).

The judge sees, in this order: every task image (Image_0 ... Image_N, i.e. the references
and the visual-instruction images in the order given to the generator), the text
instruction, the generated image, and the criteria below.

Note: the fourth criterion is named "Visual Instruction Residue" in the prompt; a higher
score means a cleaner image, so it is reported as "Visual Instruction Cleanliness".
"""

PROMPT_HEADER = """You are a STRICT evaluator for a multi-reference image generation system.

You will see (in this order):
  1. N reference images: Image 0, Image 1, ..., Image (N-1). These are mains
     (subjects), an optional STYLE modifier, and an optional EXTRA (pose /
     clothes / text-style) — in the same order that was shown to the
     generator.
  2. Image N: the LAYOUT PREVIEW — colored bounding boxes on a blank canvas
     that specify WHERE each main subject must be placed in the generated
     image. The color of each box identifies which Image_i belongs in it.
  3. The instruction text that was given to the image generator.
  4. The generated output image.

Reference images follow immediately below (Image 0, then Image 1, ..., then
the LAYOUT PREVIEW as the LAST reference).
"""


def build_middle(instruction):
    return (
        "\n\n=====================================================\n"
        "Instruction text given to the generator:\n"
        "-----------------------------------------------------\n"
        f"{instruction}\n"
        "=====================================================\n\n"
        "Generated image (the output to evaluate):"
    )


PROMPT_CRITERIA = """
Evaluate the generated image on SIX independent criteria, each on a 1-10
scale. Calibration: 4 = average, 6 = good, 8 = excellent, 10 = perfect.

================================================================
1. Text Instruction Following   (non-visual prompt adherence)
================================================================
How well does the generated image follow the natural-language parts of the
instruction text — ignoring the spatial layout (that is criterion 3)?
Check every clause of the instruction:
  - STYLE clause ("Apply Image X's <tone|lighting|style> strictly ..."):
      is that attribute visibly applied across the whole image?
  - BACKGROUND phrase ("in <setting>"):
      is the described setting actually rendered?
  - EXTRA clause ("use Image X as a <pose|clothes|style> reference for
      Image Y"):
      is the modifier applied to the correct target subject and in the
      stated role (pose vs clothes vs style)?
  - other explicit directives (e.g. "seamless", "erase any visual
      instructions", "only the text from Image X without its background").

If a clause is clearly ignored or violated, the score MUST NOT exceed 5.
If the STYLE attribute word is clearly not applied, or if the EXTRA is
applied to the wrong subject, the score MUST NOT exceed 4.
If "only the text without its background" is violated (the reference
paper / panel was copied along with the glyphs), the score MUST NOT exceed 5.

================================================================
2. Reference Consistency   (subject fidelity)
================================================================
For each MAIN subject, judge whether the generated image reproduces the
reference faithfully:
  - person  : face identity, clothing details, accessories
  - animal  : species, breed, coat pattern, markings
  - object  : shape, proportions, surface details
  - text    : EXACT glyphs and font style — NOT the background or paper
              it sat on in the reference

If any one subject fails to match the reference in recognizable detail,
the score MUST NOT exceed 6. If a referenced subject is missing or
replaced with a different subject, the score MUST be 1.

================================================================
3. Vision Instruction Adherence   (STRICT — spatial layout)
================================================================
Evaluate how precisely each MAIN subject is placed inside its assigned
colored bounding box from the LAYOUT PREVIEW (Image N). The color of the
box identifies which Image_i should occupy that region.

Be VERY STRICT. Use this calibration:
  10  : every main subject fills its assigned box tightly, with correct
        relative sizes and positions, and NO part of any subject leaks
        outside its box.
  7-9 : minor deviations (e.g. sub-10% scale mismatch, tiny offset) but
        NO subject protrudes outside its box and NO subject is misplaced.
  4-6 : at least one subject visibly protrudes / leaks outside its
        assigned box (even if the subject is roughly in the right region).
  1-3 : one or more subjects are clearly misplaced, wrong size, or the
        relative positions disagree with the layout.
  1   : any color->subject mapping is wrong (a subject sits in the wrong
        colored box, i.e. subjects are swapped), or the spatial layout
        is broadly ignored.

Hard caps (STRICT — apply even if other aspects look good):
  - If a subject sits in the WRONG colored box (wrong color->subject
    mapping), the score MUST be 1.
  - If any subject is CLEARLY misplaced (wrong region, wrong size, or
    relative positions disagree with the layout), the score MUST NOT
    exceed 3.
  - If any subject visibly PROTRUDES / LEAKS outside its assigned box
    (even partially), the score MUST NOT exceed 6.
  - "Roughly in the right quadrant" is NOT enough — this criterion
    measures precise adherence.

================================================================
4. Visual Instruction Residue   (STRICT — leftover marks)
================================================================
Does the generated image contain any leftover artifacts from the
LAYOUT PREVIEW, OR any newly drawn rectangular frames that the model
added to satisfy the placement instruction literally? Look for:
  - colored bounding box outlines from the LAYOUT PREVIEW
  - ANY colored rectangular frame / border / panel that the model drew
    around a subject (this is a hack where the model literally draws a
    "box" to host the subject, turning the output into a framed collage
    — such a frame still counts as residue even if it is not a direct
    copy of the LAYOUT PREVIEW)
  - arrows, labels, index numbers, markings
  - any visual trace of the layout diagram or scrapbook-style framing

Be STRICT. Calibration:
  10  : perfectly clean, no trace of any visual instruction or
        model-drawn frame.
  7-9 : faint residual color tint or a very subtle edge, NOT a clearly
        recognizable outline or frame.
  5-6 : a recognizable residual / drawn box outline exists but is
        partial / broken / faint.
  4   : exactly ONE complete (or near-complete) box outline or frame
        remains.
  1-3 : MULTIPLE complete box outlines / frames remain, or obvious
        arrows / labels / markings are still clearly visible.

Hard caps (STRICT — apply even if other aspects look good):
  - If ANY visually recognizable box outline OR model-drawn subject frame
    is present, the score MUST be <= 6.
  - If exactly ONE complete box outline / frame is fully preserved or
    drawn, the score MUST be 4.
  - If MULTIPLE complete box outlines / frames remain, the score MUST be 1.
  - A "model-drawn frame" means the output reads like a framed panel
    around a subject — a visible rectangular border, picture-frame, or
    coloured swatch enclosing the subject — not a natural part of the
    scene. Count this the same as a leftover LAYOUT PREVIEW box.

================================================================
5. Scene Coherence   (foreground-background integration)
================================================================
Does the result read as ONE coherent, natural scene?
  - foreground subjects and background blend with consistent lighting,
    perspective, and ground plane
  - the STYLE reference (if any) is applied uniformly across the image
  - the BACKGROUND described in the instruction is actually rendered
  - no collage / split-panel / cut-and-paste appearance
  - the SCENE makes semantic sense as a whole (subjects and background
    belong together in one plausible situation, not merely smoothly
    rendered but semantically unrelated pieces)

Watch specifically for a KNOWN HACK: the model produces a pixel-smooth
composite where each subject sits in its OWN scene context (e.g. subject A
in a park, subject B on a beach, subject C in a studio) and the boundaries
are just blurred. This is NOT coherence — it is a semantic mash-up hidden
by a smooth blend. Detect mismatched:
  - locations / environments (park vs ocean vs indoor)
  - time of day / weather
  - activity or situation (sports vs sleeping vs formal event)
  - light direction or color temperature
across subjects and/or background.

Hard caps (STRICT):
  - Collage or obvious cut-and-paste appearance MUST score <= 3.
  - Any visible stylistic mismatch (lighting / color / tone) between
    subjects and background caps the score at 5.
  - If the scene LOOKS smoothly rendered but is actually a semantically
    incoherent mash-up (different subjects come from clearly different
    situations / contexts, even if the pixels blend) the score MUST NOT
    exceed 3. This applies even if the blending is technically flawless —
    semantic mismatch is considered as severe as collage.

================================================================
6. Visual Quality
================================================================
Overall perceptual quality: resolution, texture, aesthetic composition.
Evaluate independently; do NOT cap based on the other scores.

================================================================
Output format
================================================================
First explain your reasoning, starting the line with 'Reasoning: '.
Then emit the final assessment EXACTLY in this format (one per line):

Text Instruction Following: <1-10>.
Reference Consistency: <1-10>.
Vision Instruction Adherence: <1-10>.
Visual Instruction Residue: <1-10>.
Scene Coherence: <1-10>.
Visual Quality: <1-10>.
"""

# Window comparison record

These are the six paired requests behind the window example in
[Losing My Reflection](https://spivitz.com/notes/video-model-cast-someone-else).
The author would keep the selected LoRA clip, seed 1103. On further review, the
LoRA versions of seeds 2237, 3371 and 6619 did not look like him. This assessment
applies only to those LoRA clips; it does not change his baseline assessment or
add new ratings for the remaining seeds. All outputs are retained. This selected
scene did not establish an improvement from LoRA, and is not a recurring-character
benchmark.

## What's included

- `requests.json`: all 12 saved inference request bodies, endpoint, input-image
  checksum and links to the published full videos. Positive and negative prompts,
  interpolation and safety settings are explicit. Only the starting-photo and
  private-adapter URLs have been replaced with placeholders inside each body.
- `adapter.json`: SHA-256, byte count and aggregate dtype/shape counts calculated
  from the original adapter. No tensor values or reusable face weights.
- `config.json`: the returned configuration, which contains only an empty
  `instance_prompt`. It is not a complete training recipe.

The seeds are 1103, 2237, 3371, 4499, 5573 and 6619. Requests within each pair
differ only in `loras`. They include the same trigger in both prompts.

The actual uploaded input was this already-published
[starting photo](https://spivitz.com/images/notes/video-model-cast-someone-else/window-reference.png),
1024 by 769 pixels, SHA-256
`58bbb38e71cc6f00dae0d8844c8c05c5dfc6578b704f617fb2df0e03cdde11f1`.
The request selects a 1:1 output. No separately prepared square crop is present
in these requests; any provider-side resizing/cropping is not independently
recorded. The displayed starting photo is not a recovered internal model input.

The adapter passed the companion's bounded structural checks on September 13,
2026. That verifies the file layout, not successful layer loading or likeness.

## Technical notes

For a base matrix W with d rows and k columns, LoRA learns factors A and B
containing r × (d + k) values. Their product has rank at most r, rather than
d × k independent adjustments. W stays frozen during adapter training.
[Hu and colleagues, LoRA](https://arxiv.org/abs/2106.09685).

The article's arithmetic examples use an effective scale of 1. Standard LoRA
multiplies B × A by alpha/r; an inference service can expose an additional
adapter-strength multiplier. These are distinct settings. The recorded fal.ai
strength of 1 does not independently establish the adapter's alpha/r value.
[Hugging Face, LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora).

One attention projection in the saved adapter has A shaped 16 × 5,120 and B
shaped 5,120 × 16: 163,840 adapter values versus 26,214,400 entries in the dense
update, exactly 160 times fewer adjustment values for that projection. This
ratio does not describe the complete inference model, which still needs W.

The original file's header contains 800 tensors and 76,677,120 stored values.
The dtype/shape counts and checksum are in `adapter.json`; these describe the
saved artifact, not an inference-quality measurement.

## What this does not reproduce

Training footage and the identity adapter remain private. Account identifiers,
receipts and temporary provider URLs are omitted. This package lets readers
inspect the comparison settings and outputs; it cannot recreate the author's
adapter. The original environment and provider/model revision were not fully
captured, so this is not a bit-exact reproducibility claim.

The assessment was subjective and unblinded. The scene was selected for the
article after review; these six seeds are not a held-out confirmatory evaluation.
Matched seeds do not force identical motion. For a future test, assess likeness,
temporal consistency and completion of the requested action separately, with a
no-clear-preference option and scene selection fixed before generation.

Nothing here submits requests automatically. Reusing the bodies requires your
own uploaded image and compatible adapter, and generation costs money.

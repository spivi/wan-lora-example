# Window comparison record

These are the six paired requests behind the window example in
[Losing My Reflection](https://spivitz.com/notes/video-model-cast-someone-else).
Both versions were acceptable to the author. This selected scene did not establish
an improvement from LoRA, and is not a recurring-character benchmark.

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

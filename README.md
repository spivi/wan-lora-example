# WAN LoRA example

Train a WAN 2.2 I2V-A14B adapter through fal.ai, from a folder of your own clips.
This script orchestrates hosted training; it does not train on your laptop.
An adapter is not a standalone video model and does not guarantee likeness.

## Get the code

```bash
git clone https://github.com/spivi/wan-lora-example.git
cd wan-lora-example
git checkout v1.0.1
```

No training footage, model weights or API keys are included.

## Setup

Use Python 3.10 or newer and install FFmpeg, including `ffprobe`, through your
operating system's package manager (for example `brew install ffmpeg` on macOS).
Create a virtual environment beside this script:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell.
Only upload requires the Python dependency; preparation and tests use the standard
library and the FFmpeg programs. Network requests also need working system TLS
certificates. Never disable certificate verification to bypass a setup problem.

## 1. Prepare and review locally

Put 10 to 80 videos in a dedicated folder, with no unrelated footage. This is a
conservative input policy for the companion, not an experimentally optimal count.
Use one consenting subject, clear framing, varied expressions and angles. Keep
evaluation photos outside that folder. Supported extensions: mp4, mov, mkv, webm,
m4v. Files in subdirectories are not included; other file types are ignored.

```bash
python wan_lora.py prepare --clips ./my-clips --run ./tutor-v1 --trigger TUTORPERSON
```

The run folder must not exist. Originals are never changed. Preparation takes the
**first five seconds** of each clip, removes audio, normalizes to a 480-pixel short
edge and 16 fps, then packages only prepared videos in `training.zip`. Clips must
have at least two seconds of video; shorter-than-five-second clips are not padded
locally. The trainer's automatic temporal scaling is enabled. Captions, if present
beside your originals, are not copied; this version is video-only.

Open `tutor-v1/review.html` locally. It contains three samples per clip and links to
watch the prepared videos. Inspect framing and blur in playback, not just the
samples. Trim originals into a separate input folder yourself if the first five
seconds miss the useful action, then prepare a new run. Do not train on blank
frames or a head that has already walked out of the picture.

`manifest.json` records source and prepared hashes, durations and training
settings. Defaults match the article: 400 steps and learning rate 0.0002. Override
with `--steps` and `--learning-rate` during preparation. The trigger is required.
This reproduces a workflow, not identical training randomness or identical output.

## 2. Upload and start one paid job

Review the current price and data terms on the
[fal.ai training page](https://fal.ai/models/fal-ai/wan-22-trainer/i2v-a14b).
Set `FAL_KEY` through your shell or secret manager. The script does not read local
`.env` files or search for credentials. Never put the key in source, screenshots
or shared run folders.

```bash
python wan_lora.py submit --run ./tutor-v1 --reviewed --allow-upload --allow-paid
```

These flags explicitly acknowledge that you reviewed the dataset/settings and
current provider price, have permission to use the footage, and permit the upload
and charge. **There is no hard dollar cap or invoice reconciliation in this
script.** Configure account-side controls as available. Removing a local file
does not cancel a provider job or delete uploaded footage.

The archive is uploaded to fal.ai storage. Returned media URLs can grant access
to anyone holding them; treat run files as private. Review provider retention and
deletion controls before uploading sensitive material. Open weights do not make
this hosted route offline or private by default.

An atomic `intent.json` is written before submitting training. On POSIX, both
the file and containing directory are synchronized; other platforms synchronize
the file only. This is not a guarantee against every filesystem or hardware failure.
A successful
submission saves `receipt.json`. Repeating submit after a receipt is a no-op;
repeating it after an intent without a receipt is blocked, because the previous
request might already have been accepted and charged. Upload retries may upload
the archive again, but no training POST is automatically retried.

## 3. Check progress and download

```bash
python wan_lora.py collect --run ./tutor-v1
```

This checks once, then exits. Run again later if queued or running. When complete,
it saves `adapter.safetensors`, `config.json`, raw `result.json` and SHA-256 hashes
in `artifacts.json`. The script uses the endpoint's documented `lora_file` and
`config_file` outputs. Only HTTPS fal.media artifact hosts are accepted, with a
2 GiB limit per file; redirects are refused. If the provider changes hosts, inspect
the result and update the allowlist deliberately, not by disabling checks.

Collection checks the config is a JSON object and validates the safetensors header,
shapes, byte lengths and contiguous offsets without loading tensor values. JSON
and tensor headers are limited to 16 MiB. This narrow validator supports ordinary
1/2/4/8-byte dtypes, not packed or experimental types. Unsupported formats fail
closed. Both files must pass before `validation.json` records their hashes as
structurally valid; compatibility remains explicitly unchecked. A file on disk
alone means downloaded, not validated. Validation records apply only to their
recorded hashes. Partial downloads remain as `.part` files and can be retried.

The hash detects changes between downloads; it is not a signature or proof of
training quality. Keep the configuration with the adapter and check the target
model's layer/stage compatibility before inference. This companion does not
validate internal layer loading or generate evaluation clips automatically.

### Interrupted or failed jobs

- If submit times out, **do not delete intent.json or create another run to retry**.
  Find the matching job in your fal.ai dashboard, checking endpoint and submission
  details, then recover its ID:

  ```bash
  python wan_lora.py collect --run ./tutor-v1 --request-id YOUR_CONFIRMED_REQUEST_ID
  ```

  This trusts the ID you supply; it cannot independently prove that the recovered
  job used your dataset. Keep the dashboard confirmation with the run record.
  A candidate ID is saved only after a recognized status response. A failed lookup
  leaves it uncommitted, so a typo can be corrected by repeating this command.
  An existing receipt cannot be replaced this way.
- If no job can be located, resolve the uncertain submission with the provider
  before deliberately starting a new one. HTTP errors are not assumed free.
- A failed provider job is not automatically retrained. Inspect its dashboard
  error and billing status. Collection never initiates training.
- A killed process can leave the empty `.lock` directory behind. Confirm no
  process is using this run, then remove **only that empty directory** with
  `rmdir tutor-v1/.lock`. Keep all intent and receipt files.
- Partial preparation leaves a run folder for inspection. Use a different output
  folder after correcting the input. No recursive cleanup is performed.

## Try the adapter

Use the compatible
[WAN 2.2 LoRA generation endpoint](https://fal.ai/models/fal-ai/wan/v2.2-a14b/image-to-video/lora).
For a comparison, keep the starting photo, prompt, seed and generation settings
the same in both requests. Include the trigger in both prompts and change only
the adapter list. Watch the entire output and judge likeness separately from
whether the action happened. This is pre-generated video, not a tested real-time
assistant pipeline. No automatic generation is included to avoid surprise charges.

## Tests and scope

```bash
python -m unittest -v test_wan_lora.py
```

The 28 offline tests include recovery lookup failures, HTTP response lengths,
redirect refusal, malformed artifacts and interrupted collection. All remote calls
are mocked. Local FFmpeg preparation is smoke-tested separately;
this wrapper has not submitted a paid training job. Its API fields and
queue routes are checked against the documented interface and the article's saved
training receipt. The paid path has not been tested end to end.

The [window experiment record](experiments/window/README.md) includes sanitized
requests and links to the published clips. It documents a selected example, not
a demonstrated improvement over the baseline.

Sources: [safetensors format](https://github.com/safetensors/safetensors#format),
[training API](https://fal.ai/models/fal-ai/wan-22-trainer/i2v-a14b/api),
[WAN code and model weights](https://github.com/Wan-Video/Wan2.2).
Provider interfaces, pricing and model licenses can change; check before use.

## License

The code is MIT-licensed. That license does not cover WAN weights, your footage
or fal.ai services; their respective licenses and terms still apply.

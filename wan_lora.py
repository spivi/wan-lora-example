#!/usr/bin/env python3
"""Prepare clips and train one WAN 2.2 I2V LoRA through fal.ai.

Local preparation is offline. Only submit uploads footage and starts paid work.
Python 3.10+, FFmpeg/ffprobe; fal-client is needed only for upload.
"""

import argparse
from contextlib import contextmanager
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile

ENDPOINT = "fal-ai/wan-22-trainer/i2v-a14b"
QUEUE = "https://queue.fal.run/"
MAX_DOWNLOAD = 2 * 1024**3
MAX_JSON = 16 * 1024**2
EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    """Atomic JSON replacement; call under the run lock."""
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)
    # Persist the rename as well as the file on POSIX filesystems.
    # Errors propagate: an uncertain intent write must not lead to a paid POST.
    if os.name == "posix":
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def locked(run):
    lock = run / ".lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise ValueError("Run is locked. See README before recovering a stale lock.")
    try:
        yield
    finally:
        lock.rmdir()


def command(args):
    try:
        return subprocess.check_output(args, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        raise ValueError(f"{args[0]} could not process a clip; check the input file.")


def probe(path):
    info = json.loads(command([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration:format=duration",
        "-of", "json", str(path),
    ]))
    if not info.get("streams"):
        raise ValueError(f"No video stream: {path.name}")
    stream = info["streams"][0]
    duration = float(stream.get("duration") or info.get("format", {}).get("duration", 0))
    if not math.isfinite(duration) or duration < 2:
        raise ValueError(f"Clip needs at least two seconds: {path.name}")
    if min(stream["width"], stream["height"]) < 128:
        raise ValueError(f"Clip is too small: {path.name}")
    return duration


def prepare(args):
    for binary in ("ffmpeg", "ffprobe"):
        if not shutil.which(binary):
            raise ValueError(f"Install {binary} first (see README).")
    source = args.clips.expanduser().resolve()
    if not source.is_dir():
        raise ValueError("--clips must be a directory.")
    clips = sorted(p for p in source.iterdir() if p.suffix.lower() in EXTENSIONS and p.is_file())
    if not 10 <= len(clips) <= 80:
        raise ValueError("Use 10 to 80 clips in a dedicated input folder.")
    durations = [probe(p) for p in clips]
    if not args.trigger.strip() or len(args.trigger) > 100:
        raise ValueError("Use a nonempty trigger phrase of at most 100 characters.")
    if not 1 <= args.steps <= 10000 or not math.isfinite(args.learning_rate) or not 0 < args.learning_rate <= 0.01:
        raise ValueError("Steps must be 1..10000; learning rate must be finite and in (0, 0.01].")
    run = args.run.expanduser().resolve()
    run.mkdir(parents=True, exist_ok=False)
    (run / "clips").mkdir()
    (run / "review").mkdir()
    rows, cards = [], []
    for index, (clip, duration) in enumerate(zip(clips, durations), 1):
        name = f"clip-{index:03d}"
        target = run / "clips" / (name + ".mp4")
        # First five seconds, no audio, preserve aspect ratio, even dimensions.
        command(["ffmpeg", "-v", "error", "-nostdin", "-n", "-i", str(clip),
                 "-t", "5", "-map", "0:v:0", "-an", "-map_metadata", "-1",
                 "-vf", r"scale=if(gt(iw\,ih)\,-2\,480):if(gt(iw\,ih)\,480\,-2),fps=16",
                 "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(target)])
        prepared_duration = probe(target)
        shots = []
        for j, time in enumerate((0, prepared_duration / 2, max(0, prepared_duration - 0.15))):
            shot = run / "review" / f"{name}-{j}.jpg"
            command(["ffmpeg", "-v", "error", "-nostdin", "-n", "-ss", str(time),
                     "-i", str(target), "-frames:v", "1", "-vf", "scale=240:-2", str(shot)])
            if not shot.exists():
                raise ValueError("Could not extract a review frame.")
            shots.append(f'<img src="review/{shot.name}" alt="{name}, sample {j + 1}">')
        rows.append({"file": target.name, "source_name": clip.name,
                     "source_sha256": digest(clip), "source_seconds": duration,
                     "prepared_seconds": prepared_duration, "sha256": digest(target)})
        cards.append(f'<section><h2>{name}: {html.escape(clip.name)}</h2>'
                     + "".join(shots) + f'<p><a href="clips/{target.name}">Watch prepared clip</a></p></section>')
    archive = run / "training.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as z:
        for row in rows:
            z.write(run / "clips" / row["file"], row["file"])
    manifest = {"version": 1, "endpoint": ENDPOINT, "archive_sha256": digest(archive),
                "settings": {"number_of_steps": args.steps, "learning_rate": args.learning_rate,
                             "trigger_phrase": args.trigger.strip(), "auto_scale_input": True},
                "clips": rows}
    save(run / "manifest.json", manifest)
    (run / "review.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Review training clips</title>'
        '<h1>Review before uploading</h1><p>These are samples, not a face-quality score. '
        'Watch the clips. Check focus, framing, expression coverage and permission to use every face. '
        'Inputs are trimmed to their first five seconds; originals are unchanged.</p>'
        + "".join(cards) + '</html>', encoding="utf-8")
    print(f"Prepared {len(rows)} clips. Open {run / 'review.html'} before submitting.")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Unexpected redirect refused; no credentials forwarded.")


def request_json(url, body=None):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "queue.fal.run" or parsed.fragment:
        raise ValueError("Untrusted queue URL.")
    key = os.environ.get("FAL_KEY", "").strip()
    if not key:
        raise ValueError("Set FAL_KEY in your environment; never put it in the script.")
    data = None if body is None else json.dumps(body, allow_nan=False).encode()
    req = urllib.request.Request(url, data=data, headers={
        "Authorization": "Key " + key, "Content-Type": "application/json"})
    with urllib.request.build_opener(NoRedirect).open(req, timeout=120) as response:
        return json.load(response)


def valid_id(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise ValueError("Invalid request ID.")
    return value


def job_url(request_id):
    # Queue status/result routes use the application ID, not the endpoint suffix.
    return QUEUE + "fal-ai/wan-22-trainer/requests/" + valid_id(request_id)


def upload(path):
    try:
        import fal_client
    except ImportError:
        raise ValueError("Install requirements.txt before uploading.")
    return fal_client.upload_file(str(path))


def submit(args):
    run = args.run.expanduser().resolve()
    with locked(run):
        if (run / "receipt.json").exists():
            print("Already submitted. Use collect; no new paid request sent.")
            return
        if (run / "intent.json").exists():
            raise ValueError("Submission may already have been charged. Reconcile via collect --request-id; do not resubmit.")
        if not (args.reviewed and args.allow_upload and args.allow_paid):
            raise ValueError("Review the clips and current price, then explicitly set --reviewed --allow-upload --allow-paid.")
        if not os.environ.get("FAL_KEY", "").strip():
            raise ValueError("Set FAL_KEY before uploading.")
        manifest = read(run / "manifest.json")
        if manifest["endpoint"] != ENDPOINT or digest(run / "training.zip") != manifest["archive_sha256"]:
            raise ValueError("Dataset or endpoint changed; prepare a new run and review it.")
        settings = manifest["settings"]
        if (not isinstance(settings.get("number_of_steps"), int)
                or not 1 <= settings["number_of_steps"] <= 10000
                or not math.isfinite(settings.get("learning_rate", float("nan")))
                or not 0 < settings["learning_rate"] <= 0.01
                or not isinstance(settings.get("trigger_phrase"), str)
                or not settings["trigger_phrase"].strip()
                or settings.get("auto_scale_input") is not True
                or set(settings) != {"number_of_steps", "learning_rate", "trigger_phrase", "auto_scale_input"}):
            raise ValueError("Invalid training settings.")
        cached = run / "upload.json"
        if cached.exists():
            uploaded = read(cached)
            if uploaded["sha256"] != manifest["archive_sha256"]:
                raise ValueError("Cached upload does not match the dataset.")
        else:
            uploaded = {"url": upload(run / "training.zip"), "sha256": manifest["archive_sha256"]}
            save(cached, uploaded)
        if not uploaded["url"].startswith("https://"):
            raise ValueError("Upload did not return an HTTPS URL.")
        body = {**settings, "training_data_url": uploaded["url"]}
        # Persist BEFORE POST. Even timeout/HTTP errors require reconciliation.
        save(run / "intent.json", {"endpoint": ENDPOINT, "input": body,
                                   "archive_sha256": manifest["archive_sha256"]})
        receipt = request_json(QUEUE + ENDPOINT, body)
        valid_id(receipt["request_id"])
        save(run / "receipt.json", {"request_id": receipt["request_id"]})
        print("Submitted:", receipt["request_id"], "Use collect to check progress.")


def download(url, target):
    p = urllib.parse.urlsplit(url)
    if (p.scheme != "https" or p.username or p.password or p.port not in (None, 443)
            or not (p.hostname == "fal.media" or (p.hostname or "").endswith(".fal.media"))):
        raise ValueError("Unexpected artifact host; inspect result.json manually.")
    temp = target.with_suffix(target.suffix + ".part")
    with urllib.request.build_opener(NoRedirect).open(url, timeout=120) as response, temp.open("wb") as f:
        length = response.headers.get("Content-Length")
        expected = int(length) if length is not None else None
        if expected is not None and not 0 < expected <= MAX_DOWNLOAD:
            raise ValueError("Invalid or oversized artifact Content-Length.")
        total = 0
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_DOWNLOAD:
                raise ValueError("Artifact exceeds the 2 GiB download limit.")
            f.write(chunk)
        if not total:
            raise ValueError("Empty artifact.")
        if expected is not None and total != expected:
            raise ValueError("Artifact length does not match Content-Length.")
    temp.replace(target)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key.")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("Non-finite JSON number.")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)


def validate_artifact(path):
    """Bounded format checks only, not model compatibility or tensor quality."""
    if path.suffix == ".json":
        with path.open("rb") as f:
            raw = f.read(MAX_JSON + 1)
        if len(raw) > MAX_JSON or not isinstance(strict_json(raw), dict):
            raise ValueError("Config must be a JSON object of at most 16 MiB.")
        return
    size = path.stat().st_size
    with path.open("rb") as f:
        prefix = f.read(8)
        length = int.from_bytes(prefix, "little")
        if len(prefix) != 8 or not 2 <= length <= min(MAX_JSON, size - 8):
            raise ValueError("Invalid safetensors header size.")
        raw = f.read(length)
    header = strict_json(raw)
    if not raw.startswith(b"{") or not isinstance(header, dict):
        raise ValueError("Invalid safetensors header.")
    widths = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2,
              "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4,
              "I64": 8, "U64": 8, "F64": 8}
    spans = []
    for name, tensor in header.items():
        if name == "__metadata__":
            if not isinstance(tensor, dict) or not all(isinstance(v, str) for v in tensor.values()):
                raise ValueError("Invalid safetensors metadata.")
            continue
        if not isinstance(tensor, dict):
            raise ValueError("Invalid tensor entry.")
        dtype, shape, offsets = (tensor.get(k) for k in ("dtype", "shape", "data_offsets"))
        if (not isinstance(dtype, str) or dtype not in widths
                or not isinstance(shape, list) or len(shape) > 64
                or any(type(n) is not int or not 0 <= n <= MAX_DOWNLOAD for n in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(n) is not int for n in offsets)):
            raise ValueError("Unsupported dtype or invalid tensor shape/offsets.")
        start, end = offsets
        if not 0 <= start <= end <= size - 8 - length or end - start != math.prod(shape) * widths[dtype]:
            raise ValueError("Tensor size does not match shape or file.")
        spans.append((start, end))
    cursor = 0
    for start, end in sorted(spans):
        if start != cursor:
            raise ValueError("Tensor data contains gaps or overlaps.")
        cursor = end
    if not spans or cursor != size - 8 - length:
        raise ValueError("Missing tensors or unindexed tensor data.")


def collect(args):
    run = args.run.expanduser().resolve()
    with locked(run):
        receipt_path = run / "receipt.json"
        if args.request_id:
            recovered = valid_id(args.request_id)
            if not (run / "intent.json").exists():
                raise ValueError("This run has no submission intent to recover.")
            if receipt_path.exists() and read(receipt_path)["request_id"] != recovered:
                raise ValueError("Cannot replace an existing request ID.")
        if not receipt_path.exists() and not args.request_id:
            raise ValueError("No receipt. Submit first, or recover the request ID from your fal.ai dashboard.")
        url = job_url(read(receipt_path)["request_id"] if receipt_path.exists() else recovered)
        status = request_json(url + "/status")
        if not isinstance(status, dict) or status.get("status") not in ("IN_QUEUE", "IN_PROGRESS", "COMPLETED", "FAILED"):
            raise ValueError("Unrecognized provider status; receipt unchanged.")
        if not receipt_path.exists():
            save(receipt_path, {"request_id": recovered, "recovered_by_user": True})
        save(run / "status.json", status)
        print("Status:", status.get("status", "unknown"))
        if status.get("status") != "COMPLETED":
            if status.get("status") not in ("IN_QUEUE", "IN_PROGRESS"):
                raise ValueError("Job is not queued, running or completed. Inspect status.json and the provider dashboard; no retry submitted.")
            print("No artifacts collected. Run collect again later; it never starts training.")
            return
        result = request_json(url)
        if not isinstance(result, dict) or any(
                not isinstance(result.get(field), dict) or not isinstance(result[field].get("url"), str)
                for field in ("lora_file", "config_file")):
            raise ValueError("Result is missing artifact URLs.")
        save(run / "result.json", result)
        hashes = {}
        for field, name in (("lora_file", "adapter.safetensors"), ("config_file", "config.json")):
            target = run / name
            recorded = read(run / "artifacts.json") if (run / "artifacts.json").exists() else {}
            if not target.exists() or recorded.get(name) != digest(target):
                download(result[field]["url"], target)
            validate_artifact(target)
            hashes[name] = digest(target)
        save(run / "artifacts.json", hashes)
        save(run / "validation.json", {"sha256": hashes, "structurally_valid": True,
                                       "model_compatibility": "not checked"})
        print("Both artifacts passed structural checks; hashes saved. Model compatibility and training quality are not checked.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare", help="Offline: validate, trim, normalize and create a review page")
    p.add_argument("--clips", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True, help="A NEW output directory")
    p.add_argument("--trigger", required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--learning-rate", type=float, default=0.0002)
    p.set_defaults(func=prepare)
    p = sub.add_parser("submit", help="Upload private footage and start ONE PAID training job")
    p.add_argument("--run", type=Path, required=True)
    for flag in ("reviewed", "allow-upload", "allow-paid"):
        p.add_argument("--" + flag, action="store_true")
    p.set_defaults(func=submit)
    p = sub.add_parser("collect", help="Check once and download completed artifacts; safe to repeat")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--request-id", help="Recover a known job ID after an ambiguous submission")
    p.set_defaults(func=collect)
    args = parser.parse_args()
    try:
        args.func(args)
    except urllib.error.HTTPError as error:
        # Do not print URLs, bodies or headers containing account/asset information.
        parser.exit(1, f"HTTP {error.code}. If submitting, keep intent.json and reconcile before retrying.\n")
    except (ValueError, OSError, KeyError, TypeError) as error:
        message = str(error) if isinstance(error, ValueError) else type(error).__name__ + ": check the run files and README."
        parser.exit(1, message + "\n")
    except Exception:
        parser.exit(1, "Provider/client error. Keep run files; reconcile any submission intent before retrying.\n")


if __name__ == "__main__":
    main()

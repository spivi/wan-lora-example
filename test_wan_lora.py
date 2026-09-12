"""Offline tests: no keys, uploads or paid requests are used."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import wan_lora as w


class CompanionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        (self.run / "training.zip").write_bytes(b"offline fixture")
        w.save(self.run / "manifest.json", {
            "endpoint": w.ENDPOINT, "archive_sha256": w.digest(self.run / "training.zip"),
            "settings": {"number_of_steps": 400, "learning_rate": 0.0002,
                         "trigger_phrase": "TUTORPERSON", "auto_scale_input": True}})
        self.args = argparse.Namespace(run=self.run, reviewed=True, allow_upload=True,
                                       allow_paid=True, request_id=None)
        self.env = patch.dict(os.environ, {"FAL_KEY": "offline-test-key"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_requires_every_confirmation(self):
        for flag in ("reviewed", "allow_upload", "allow_paid"):
            with self.subTest(flag=flag), patch.object(w, "upload") as upload:
                setattr(self.args, flag, False)
                with self.assertRaises(ValueError):
                    w.submit(self.args)
                upload.assert_not_called()
                setattr(self.args, flag, True)

    def test_missing_key_prevents_upload(self):
        with patch.dict(os.environ, {"FAL_KEY": ""}), patch.object(w, "upload") as upload:
            with self.assertRaises(ValueError):
                w.submit(self.args)
            upload.assert_not_called()

    def test_changed_archive_prevents_upload(self):
        (self.run / "training.zip").write_bytes(b"changed")
        with patch.object(w, "upload") as upload:
            with self.assertRaises(ValueError):
                w.submit(self.args)
            upload.assert_not_called()

    def test_timeout_keeps_intent_and_blocks_repost(self):
        with patch.object(w, "upload", return_value="https://v3.fal.media/archive.zip"), \
             patch.object(w, "request_json", side_effect=TimeoutError) as request:
            with self.assertRaises(TimeoutError):
                w.submit(self.args)
            self.assertTrue((self.run / "intent.json").exists())
            with self.assertRaises(ValueError):
                w.submit(self.args)
            self.assertEqual(request.call_count, 1)

    def test_receipt_prevents_duplicate(self):
        with patch.object(w, "upload", return_value="https://v3.fal.media/archive.zip"), \
             patch.object(w, "request_json", return_value={"request_id": "job-123"}) as request:
            w.submit(self.args)
            w.submit(self.args)
            self.assertEqual(request.call_count, 1)
            body = request.call_args.args[1]
            self.assertEqual(body["number_of_steps"], 400)
            self.assertEqual(body["trigger_phrase"], "TUTORPERSON")
            self.assertNotIn("offline-test-key", (self.run / "intent.json").read_text())

    def test_lock_rejects_second_writer(self):
        with w.locked(self.run):
            with self.assertRaises(ValueError):
                with w.locked(self.run):
                    pass
        self.assertFalse((self.run / ".lock").exists())

    def test_job_url_uses_parent_app(self):
        self.assertEqual(w.job_url("job-123"), "https://queue.fal.run/fal-ai/wan-22-trainer/requests/job-123")
        for bad in ("../elsewhere", "id?token=x", "", "https://evil.example"):
            with self.assertRaises(ValueError):
                w.job_url(bad)

    def test_auth_cannot_go_to_another_host(self):
        for url in ("http://queue.fal.run/a", "https://queue.fal.run.evil.example/a", "https://evil.example"):
            with self.assertRaises(ValueError):
                w.request_json(url)

    def test_artifact_host_restrictions(self):
        for url in ("http://v3.fal.media/x", "https://localhost/x", "https://fal.media.evil.example/x", "https://user:pass@fal.media/x"):
            with self.assertRaises(ValueError):
                w.download(url, self.run / "artifact")

    def test_collect_pending_does_not_download(self):
        w.save(self.run / "receipt.json", {"request_id": "job-123"})
        with patch.object(w, "request_json", return_value={"status": "IN_QUEUE"}), patch.object(w, "download") as download:
            w.collect(self.args)
            download.assert_not_called()

    def test_failed_job_does_not_suggest_retraining(self):
        w.save(self.run / "receipt.json", {"request_id": "job-123"})
        with patch.object(w, "request_json", return_value={"status": "FAILED"}), patch.object(w, "download") as download:
            with self.assertRaises(ValueError):
                w.collect(self.args)
            download.assert_not_called()

    def test_collect_completed_and_resume(self):
        w.save(self.run / "receipt.json", {"request_id": "job-123"})
        result = {"lora_file": {"url": "https://v3.fal.media/lora"},
                  "config_file": {"url": "https://v3.fal.media/config"}}
        def download(url, path):
            path.write_bytes(b"test artifact")
        with patch.object(w, "request_json", side_effect=[{"status": "COMPLETED"}, result] * 2), \
             patch.object(w, "download", side_effect=download) as fetch:
            w.collect(self.args)
            w.collect(self.args)
            self.assertEqual(fetch.call_count, 2)
            self.assertEqual(len(w.read(self.run / "artifacts.json")), 2)

    def test_recovery_cannot_replace_known_receipt(self):
        w.save(self.run / "intent.json", {})
        w.save(self.run / "receipt.json", {"request_id": "job-123"})
        self.args.request_id = "different"
        with self.assertRaises(ValueError):
            w.collect(self.args)

    def test_recovery_requires_prior_intent(self):
        self.args.request_id = "job-123"
        with self.assertRaises(ValueError):
            w.collect(self.args)

    def test_nan_settings_prevent_upload(self):
        data = w.read(self.run / "manifest.json")
        data["settings"]["learning_rate"] = float("nan")
        (self.run / "manifest.json").write_text(json.dumps(data))
        with patch.object(w, "upload") as upload:
            with self.assertRaises(ValueError):
                w.submit(self.args)
            upload.assert_not_called()

    def test_no_video_stream(self):
        with patch.object(w, "command", return_value=b'{"streams": []}'):
            with self.assertRaises(ValueError):
                w.probe(Path("audio.mp4"))


if __name__ == "__main__":
    unittest.main()

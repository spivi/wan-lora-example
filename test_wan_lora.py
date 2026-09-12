"""Offline tests: no keys, uploads or paid requests are used."""
import argparse
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import wan_lora as w


def tensor_file(header=None, data=b"\0\0"):
    raw = json.dumps(header or {"weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    return len(raw).to_bytes(8, "little") + raw + data


def response(data, length=None):
    stream = io.BytesIO(data)
    stream.headers = {} if length is None else {"Content-Length": str(length)}
    return stream


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
            path.write_bytes(b"{}" if path.suffix == ".json" else tensor_file())
        with patch.object(w, "request_json", side_effect=[{"status": "COMPLETED"}, result] * 2), \
             patch.object(w, "download", side_effect=download) as fetch:
            w.collect(self.args)
            w.collect(self.args)
            self.assertEqual(fetch.call_count, 2)
            self.assertEqual(len(w.read(self.run / "artifacts.json")), 2)
            self.assertEqual(w.read(self.run / "validation.json")["model_compatibility"], "not checked")

    def test_recovery_lookup_failure_can_be_corrected(self):
        w.save(self.run / "intent.json", {"keep": "original"})
        self.args.request_id = "typo"
        error = w.urllib.error.HTTPError("https://queue.fal.run/x", 404, "missing", {}, None)
        with patch.object(w, "request_json", side_effect=error):
            with self.assertRaises(w.urllib.error.HTTPError):
                w.collect(self.args)
        self.assertFalse((self.run / "receipt.json").exists())
        self.args.request_id = "correct"
        with patch.object(w, "request_json", return_value={"status": "IN_QUEUE"}):
            w.collect(self.args)
        self.assertEqual(w.read(self.run / "receipt.json")["request_id"], "correct")
        self.assertEqual(w.read(self.run / "intent.json"), {"keep": "original"})

    def test_malformed_recovery_status_does_not_save_receipt(self):
        w.save(self.run / "intent.json", {})
        self.args.request_id = "candidate"
        for status in ([], {}, {"status": "surprise"}):
            with patch.object(w, "request_json", return_value=status):
                with self.assertRaises(ValueError):
                    w.collect(self.args)
            self.assertFalse((self.run / "receipt.json").exists())

    def test_download_http_length_and_limits(self):
        target = self.run / "artifact"
        for data, length, limit in ((b"abc", 5, 10), (b"abc", 2, 10),
                                    (b"abc", 3, 2), (b"abc", None, 2), (b"", None, 10)):
            opener = Mock()
            opener.open.return_value = response(data, length)
            with patch.object(w.urllib.request, "build_opener", return_value=opener), patch.object(w, "MAX_DOWNLOAD", limit):
                with self.assertRaises(ValueError):
                    w.download("https://fal.media/file", target)
            self.assertFalse(target.exists())

    def test_download_valid_http_response(self):
        opener = Mock()
        opener.open.return_value = response(b"abc", 3)
        target = self.run / "artifact"
        with patch.object(w.urllib.request, "build_opener", return_value=opener):
            w.download("https://fal.media/file", target)
        self.assertEqual(target.read_bytes(), b"abc")

    def test_redirect_handler_refuses_forwarding(self):
        request = w.urllib.request.Request("https://queue.fal.run/x", headers={"Authorization": "Key secret"})
        with self.assertRaises(ValueError):
            w.NoRedirect().redirect_request(request, None, 302, "redirect", {}, "https://evil.example")

    def test_request_json_http_boundary(self):
        opener = Mock()
        opener.open.return_value = response(b'{"status":"IN_QUEUE"}')
        with patch.object(w.urllib.request, "build_opener", return_value=opener):
            self.assertEqual(w.request_json(w.QUEUE + "x")["status"], "IN_QUEUE")
        req = opener.open.call_args.args[0]
        self.assertEqual(req.get_header("Authorization"), "Key offline-test-key")

    def test_config_validation(self):
        target = self.run / "config.json"
        for raw in (b"<html>error</html>", b"[]", b'{"a":NaN}', b'{"a":1,"a":2}', b"\xff"):
            target.write_bytes(raw)
            with self.assertRaises(ValueError):
                w.validate_artifact(target)
        target.write_bytes(b"{}")
        w.validate_artifact(target)
        with patch.object(w, "MAX_JSON", 1), self.assertRaises(ValueError):
            w.validate_artifact(target)

    def test_tensor_validation(self):
        target = self.run / "adapter.safetensors"
        bad = [b"<html>error</html>", (w.MAX_JSON + 1).to_bytes(8, "little"), tensor_file(data=b"\0"),
               tensor_file(data=b"\0\0extra")]
        for entry in ({"dtype": "F16", "shape": [2], "data_offsets": [0, 2]},
                      {"dtype": "F16", "shape": [True], "data_offsets": [0, 2]},
                      {"dtype": "unknown", "shape": [1], "data_offsets": [0, 2]}):
            bad.append(tensor_file({"weight": entry}))
        for raw in bad:
            target.write_bytes(raw)
            with self.assertRaises(ValueError):
                w.validate_artifact(target)
        target.write_bytes(tensor_file())
        w.validate_artifact(target)

    def test_collection_rejects_invalid_artifacts(self):
        w.save(self.run / "receipt.json", {"request_id": "job-123"})
        result = {"lora_file": {"url": "https://fal.media/lora"}, "config_file": {"url": "https://fal.media/config"}}
        for invalid_name in ("adapter.safetensors", "config.json"):
            def fetch(url, path):
                path.write_bytes(b"<html>bad</html>" if path.name == invalid_name else
                                 (b"{}" if path.suffix == ".json" else tensor_file()))
            with patch.object(w, "request_json", side_effect=[{"status": "COMPLETED"}, result]), \
                 patch.object(w, "download", side_effect=fetch):
                with self.assertRaises(ValueError):
                    w.collect(self.args)
            self.assertFalse((self.run / "validation.json").exists())
            self.assertFalse((self.run / "artifacts.json").exists())

    def test_interrupted_collection_can_resume(self):
        w.save(self.run / "receipt.json", {"request_id": "job-123"})
        result = {"lora_file": {"url": "https://fal.media/lora"}, "config_file": {"url": "https://fal.media/config"}}
        def fetch(url, path):
            if path.suffix == ".json":
                raise TimeoutError()
            path.write_bytes(tensor_file())
        with patch.object(w, "request_json", side_effect=[{"status": "COMPLETED"}, result]), patch.object(w, "download", side_effect=fetch):
            with self.assertRaises(TimeoutError):
                w.collect(self.args)
        self.assertFalse((self.run / "validation.json").exists())
        def retry(url, path):
            path.write_bytes(b"{}" if path.suffix == ".json" else tensor_file())
        with patch.object(w, "request_json", side_effect=[{"status": "COMPLETED"}, result]), patch.object(w, "download", side_effect=retry):
            w.collect(self.args)
        self.assertTrue((self.run / "validation.json").exists())

    def test_malformed_result_does_not_download(self):
        w.save(self.run / "receipt.json", {"request_id": "job-123"})
        for result in ([], {}, {"lora_file": {"url": 5}, "config_file": {}}):
            with patch.object(w, "request_json", side_effect=[{"status": "COMPLETED"}, result]), patch.object(w, "download") as fetch:
                with self.assertRaises(ValueError):
                    w.collect(self.args)
                fetch.assert_not_called()

    @unittest.skipUnless(os.name == "posix", "POSIX directory sync")
    def test_save_syncs_file_and_directory(self):
        with patch.object(w.os, "fsync") as sync:
            w.save(self.run / "example.json", {})
        self.assertEqual(sync.call_count, 2)

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

"""Unit tests for GitHub-hosted RTX 4090 shard artifact integrity and coverage."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

MODULE = Path(__file__).resolve().parents[2] / ".github/scripts/rtx4090-shard-cache.py"
SPEC = importlib.util.spec_from_file_location("rtx4090_shard_cache", MODULE)
SHARDS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SHARDS)
REVISION = "a" * 40


def select(_commands, shard, count):
    assert count == 2
    return [["CMakeFiles/a.obj"], ["CMakeFiles/b.obj"]][shard]


class ShardCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.build = self.root / "build"
        self.build.mkdir()
        self.downloads = self.root / "incoming"
        self.patch = patch.object(SHARDS, "build_objects",
                                  return_value=([{"output": "CMakeFiles/a.obj"},
                                                 {"output": "CMakeFiles/b.obj"}], select))
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def pack(self, index, payload=None):
        cache = self.root / f"src-cache-{index}"
        (cache / "0").mkdir(parents=True)
        (cache / "0" / f"cache-{index}").write_bytes(
            payload if payload is not None else f"cache-object-{index}".encode())
        objects = self.build / "object-shard.json"
        objects.write_text(json.dumps(select(None, index, 2)), encoding="utf-8")
        dest = self.downloads / f"rtx4090-objects-{index}" / "cache.zip"
        SHARDS.pack(cache, objects, dest, REVISION, index, 2)
        return dest

    def test_two_shards_restore_and_validate_compiled_object_coverage(self):
        self.pack(0)
        self.pack(1)
        target = self.root / "merged"
        SHARDS.restore(self.downloads, target, self.build, REVISION, 2)
        self.assertEqual((target / "0/cache-0").read_bytes(), b"cache-object-0")
        self.assertEqual((target / "0/cache-1").read_bytes(), b"cache-object-1")

    def test_missing_shard_fails_closed(self):
        self.pack(0)
        with self.assertRaisesRegex(ValueError, "Missing shard artifact"):
            SHARDS.restore(self.downloads, self.root / "merged", self.build, REVISION, 2)

    def test_revision_mismatch_refuses_cache(self):
        self.pack(0)
        self.pack(1)
        with self.assertRaisesRegex(ValueError, "Wrong source"):
            SHARDS.restore(self.downloads, self.root / "merged", self.build, "b" * 40, 2)

    def test_undeclared_files_cannot_be_restored(self):
        first = self.pack(0)
        self.pack(1)
        with zipfile.ZipFile(first, "a") as archive:
            archive.writestr("cache/extraneous", b"secret")
        with self.assertRaisesRegex(ValueError, "Undeclared file"):
            SHARDS.restore(self.downloads, self.root / "merged", self.build, REVISION, 2)

    def test_hash_check_rejects_tampered_payload(self):
        first = self.pack(0)
        self.pack(1)
        with zipfile.ZipFile(first) as archive:
            manifest = archive.read("manifest.json")
        with zipfile.ZipFile(first, "w") as archive:
            archive.writestr("manifest.json", manifest)
            archive.writestr("cache/0/cache-0", b"tampered bytes")
        with self.assertRaisesRegex(ValueError, "Corrupted artifact"):
            SHARDS.restore(self.downloads, self.root / "merged", self.build, REVISION, 2)

    def test_path_traversal_is_rejected(self):
        for value in ("/absolute", "../escape", "a/../escape", "C:/windows", "a\\b", "a//b"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    SHARDS.safe_name(value)


if __name__ == "__main__":
    unittest.main()

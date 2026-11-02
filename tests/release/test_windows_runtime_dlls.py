"""Ensure Windows CTest and portable binaries stage transitive vcpkg DLLs."""

import pathlib
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


class WindowsRuntimeDllTests(unittest.TestCase):
    def test_stages_all_vcpkg_runtime_dlls(self):
        if shutil.which("cmake") is None:
            self.fail("CMake is required to test the Windows DLL staging contract")

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            bin_dir = root / "installed" / "x64-windows" / "bin"
            bin_dir.mkdir(parents=True)
            expected = {
                "avcodec-62.dll",
                "libcurl.dll",
                "z.dll",
                "libssl-3-x64.dll",  # transitive curl dependency
                "brotlicommon.dll",  # transitive codec dependency
            }
            for name in expected:
                (bin_dir / name).write_bytes(b"fixture")
            (bin_dir / "not-a-dll.txt").write_text("fixture", encoding="utf-8")

            script = root / "verify.cmake"
            cmake_targets = (ROOT / "cmake" / "NinferTargets.cmake").as_posix()
            vcpkg_tree = bin_dir.parent.as_posix()
            script.write_text(
                "cmake_minimum_required(VERSION 3.25)\n"
                "set(WIN32 TRUE)\n"
                "set(NINFER_WINDOWS_KERNEL_DLL OFF)\n"
                f'set(FFMPEG_VCPKG_TREE "{vcpkg_tree}")\n'
                f'include("{cmake_targets}")\n'
                "ninfer_runtime_dll_list(runtime_dlls)\n"
                'file(WRITE "${CMAKE_CURRENT_LIST_DIR}/found.txt" "${runtime_dlls}")\n',
                encoding="utf-8",
            )
            subprocess.run(["cmake", "-P", str(script)], check=True, capture_output=True, text=True)
            actual = {pathlib.Path(p).name for p in (root / "found.txt").read_text().split(";") if p}
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()

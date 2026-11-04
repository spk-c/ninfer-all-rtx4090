"""Ensure Windows CTest and portable binaries stage their runtime DLL closures."""

import os
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

    @unittest.skipUnless(os.name == "nt", "Requires the Windows loader and MSVC")
    def test_ctest_stages_cuda_dlls_without_toolkit_on_path(self):
        self._check_cuda_runtime(13, "/x64")

    @unittest.skipUnless(os.name == "nt", "Requires the Windows loader and MSVC")
    def test_ctest_stages_legacy_cuda_bin_layout(self):
        self._check_cuda_runtime(12, "")

    def _check_cuda_runtime(self, major, subdir):
        module = (ROOT / "cmake/NinferTargets.cmake").as_posix()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "CMakeLists.txt").write_text(f'''
cmake_minimum_required(VERSION 3.28)
project(cuda_runtime_fixture LANGUAGES CXX)
include("{module}")
set(CUDAToolkit_BIN_DIR "${{CMAKE_BINARY_DIR}}/cuda/bin")
set(CUDAToolkit_VERSION_MAJOR {major})
add_library(cublasLt SHARED lt.cpp)
add_library(cublas SHARED blas.cpp)
set_target_properties(cublasLt PROPERTIES OUTPUT_NAME cublasLt64_{major}
  RUNTIME_OUTPUT_DIRECTORY "${{CUDAToolkit_BIN_DIR}}{subdir}/$<0:>")
set_target_properties(cublas PROPERTIES OUTPUT_NAME cublas64_{major}
  RUNTIME_OUTPUT_DIRECTORY "${{CUDAToolkit_BIN_DIR}}{subdir}/$<0:>")
target_link_libraries(cublas PRIVATE cublasLt)
add_library(CUDA::cublas ALIAS cublas)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE cublas)
set_target_properties(consumer PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/tests/$<0:>")
if(ENABLE_STAGING)
  ninfer_stage_test_runtime_dlls(consumer)
endif()
''', encoding="utf-8")
            (root / "lt.cpp").write_text('extern "C" __declspec(dllexport) int lt() { return 42; }\n')
            (root / "blas.cpp").write_text(
                'extern "C" __declspec(dllimport) int lt();\n'
                'extern "C" __declspec(dllexport) int blas() { return lt(); }\n')
            (root / "main.cpp").write_text(
                'extern "C" __declspec(dllimport) int blas();\n'
                'int main() { return blas() != 42; }\n')
            build = root / "build"
            for command in (
                ["cmake", "-S", str(root), "-B", str(build), "-G", "Visual Studio 17 2022", "-A", "x64"],
                ["cmake", "--build", str(build), "--config", "Release", "-j", "--target", "cublas"],
                ["cmake", "-S", str(root), "-B", str(build), "-DENABLE_STAGING=ON"],
                ["cmake", "--build", str(build), "--config", "Release", "-j"],
            ):
                result = subprocess.run(command, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            env = dict(os.environ, PATH=os.environ["SystemRoot"] + r"\System32")
            exe = build / "tests/consumer.exe"
            result = subprocess.run([str(exe)], cwd=root, env=env, timeout=10)
            self.assertEqual(result.returncode, 0)
            # Both the direct import and its transitive CUDA dependency must be staged.
            for name in (f"cublas64_{major}.dll", f"cublasLt64_{major}.dll"):
                dll = exe.parent / name
                hidden = dll.with_suffix(".hidden")
                dll.rename(hidden)
                result = subprocess.run([str(exe)], cwd=root, env=env, timeout=10)
                self.assertEqual(result.returncode & 0xffffffff, 0xc0000135)
                hidden.rename(dll)


if __name__ == "__main__":
    unittest.main()

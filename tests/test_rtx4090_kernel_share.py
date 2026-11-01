"""CPU-only tests for strict Nsight kernel attribution, no GPU required."""
import tempfile
import unittest
from pathlib import Path

from tools.bench.rtx4090_kernel_share import analyse, parse_nsys_csv


class KernelShareTests(unittest.TestCase):
    def test_shares_include_all_kernel_rows(self):
        result = analyse([
            {"Name": "ggml_mul_mat_iq4_xs", "Total Time (ns)": "25", "Instances": "5"},
            {"Name": "ggml_mul_mat_iq3_s", "Total Time (ns)": "50", "Instances": "10"},
            {"Name": "ggml_mul_mat_iq3_xxs", "Total Time (ns)": "5", "Instances": "1"},
            {"Name": "softmax", "Total Time (ns)": "20", "Instances": "2"},
        ], require_iq4=True)
        self.assertEqual(result["total_kernel_time_ns"], 100)
        self.assertEqual(result["families"]["IQ4_XS"]["share_percent"], 25.0)
        self.assertEqual(result["families"]["IQ3_S"]["share_percent"], 50.0)
        self.assertEqual(result["families"]["IQ3_XXS"]["share_percent"], 5.0)
        self.assertEqual(result["families"]["IQ4_XS"]["instances"], 5)
        self.assertIn("NOT elapsed GPU wall", result["scope"])

    def test_require_target_format(self):
        with self.assertRaisesRegex(ValueError, "no IQ4_XS kernels"):
            analyse([{"Name": "other", "Total Time (ns)": "10", "Instances": "1"}],
                    require_iq4=True)

    def test_windows_numeric_enum_names_are_formats_not_tile_dimensions(self):
        result = analyse([
            {"Name": "void mul_mat_q<(ggml_type)23, (int)128, float>()", "Total Time (ns)": "23"},
            {"Name": "void mul_mat_q<(ggml_type)21, (int)16, float>()", "Total Time (ns)": "21"},
            {"Name": "void mul_mat_q<(ggml_type)18, (int)128, float>()", "Total Time (ns)": "18"},
            {"Name": "other<(ggml_type)2, (int)23, (ggml_prec)21>()", "Total Time (ns)": "38"},
        ], require_iq4=True)
        for family, percent in (("IQ4_XS", 23), ("IQ3_S", 21), ("IQ3_XXS", 18)):
            self.assertEqual(result["families"][family]["share_percent"], percent)
            self.assertEqual(result["families"][family]["kernel_rows"], 1)

    def test_refuse_empty_or_invalid_timing(self):
        with self.assertRaisesRegex(ValueError, "no data"):
            analyse([])
        with self.assertRaisesRegex(ValueError, "invalid Nsight"):
            analyse([{"Name": "iq4_xs", "Total Time (ns)": "not-a-number"}])
        with self.assertRaisesRegex(ValueError, "no measured"):
            analyse([{"Name": "iq4_xs", "Total Time (ns)": "0", "Instances": "1"}])

    def test_csv_with_nsys_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kernels.csv"
            path.write_text(
                "# Nsight Systems 2026\n"
                '"Time (%)","Total Time (ns)","Instances","Name"\n'
                '50,150,3,"test_iq4_xs_v2"\n'
                '50,150,3,"other"\n', encoding="utf-8"
            )
            rows = parse_nsys_csv(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(analyse(rows)["families"]["IQ4_XS"]["share_percent"], 50.0)

    def test_missing_header_or_malformed_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text('Name,Instances\nx,2\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "columns missing"):
                parse_nsys_csv(path)
            path.write_text('Name,Total Time (ns)\na,3,too-many\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "column count mismatch"):
                parse_nsys_csv(path)


if __name__ == "__main__":
    unittest.main()

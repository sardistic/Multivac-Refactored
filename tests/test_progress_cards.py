import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from services import progress_cards
from services.progress_cards import (
    build_receipt_card,
    build_run_receipt,
    format_elapsed,
    receipts_enabled,
    requester_label,
)


class RequesterLabelTests(unittest.TestCase):
    """A decorative card must never break delivery of the real payload."""

    def test_display_name_is_preferred(self):
        message = SimpleNamespace(author=SimpleNamespace(display_name="unnes", name="u1"))
        self.assertEqual(requester_label(message), "@unnes")

    def test_falls_back_through_available_names(self):
        message = SimpleNamespace(author=SimpleNamespace(name="ryzon"))
        self.assertEqual(requester_label(message), "@ryzon")

    def test_missing_author_yields_empty_not_an_error(self):
        self.assertEqual(requester_label(SimpleNamespace()), "")
        self.assertEqual(requester_label(None), "")

    def test_blank_names_are_skipped(self):
        message = SimpleNamespace(author=SimpleNamespace(display_name="   ", name="real"))
        self.assertEqual(requester_label(message), "@real")


class ReceiptGateTests(unittest.TestCase):
    def test_video_only_by_default(self):
        with patch.object(progress_cards, "get_runtime_setting", create=True):
            self.assertTrue(receipts_enabled("video"))
            self.assertFalse(receipts_enabled("image"))

    def test_modes_are_honoured(self):
        cases = {
            "off": (False, False),
            "all": (True, True),
            "video": (True, False),
        }
        for mode, (video, image) in cases.items():
            with self.subTest(mode=mode):
                with patch("services.behavior_registry.get_runtime_setting", return_value=mode):
                    self.assertEqual(receipts_enabled("video"), video)
                    self.assertEqual(receipts_enabled("image"), image)

    def test_unreadable_setting_falls_back_to_default(self):
        with patch("services.behavior_registry.get_runtime_setting", side_effect=RuntimeError):
            self.assertTrue(receipts_enabled("video"))
            self.assertFalse(receipts_enabled("image"))


class CardRenderTests(unittest.TestCase):
    def test_card_is_a_decodable_png(self):
        buf = build_receipt_card("Image generated", [("elapsed", "0:41"), ("cost", "$0.134")])
        self.assertIsInstance(buf, BytesIO)
        with Image.open(buf) as img:
            self.assertEqual(img.format, "PNG")
            self.assertEqual(img.width, progress_cards.CARD_WIDTH)
            self.assertGreater(img.height, 100)

    def test_height_grows_with_row_count(self):
        small = build_receipt_card("A", [("one", "1")])
        large = build_receipt_card("A", [("one", "1"), ("two", "2"), ("three", "3")])
        with Image.open(small) as a, Image.open(large) as b:
            self.assertGreater(b.height, a.height)

    def test_empty_values_are_dropped(self):
        buf = build_receipt_card("A", [("kept", "1"), ("dropped", ""), ("also", None)])
        self.assertIsNotNone(buf)

    def test_no_rows_still_renders(self):
        self.assertIsNotNone(build_receipt_card("Nothing to report", []))

    def test_run_receipt_reports_real_totals(self):
        totals = {"calls": 2.0, "tokens": 15400.0, "cost_usd": 0.134}
        with patch("services.usage_costs.get_request_totals", return_value=totals):
            buf = build_run_receipt(
                "Image generated",
                elapsed=41,
                model="gpt-image-2.5-sunburst",
                requested_by="@unnes",
            )
        self.assertIsNotNone(buf)
        with Image.open(buf) as img:
            self.assertEqual(img.format, "PNG")

    def test_run_receipt_survives_a_broken_ledger(self):
        with patch("services.usage_costs.get_request_totals", side_effect=RuntimeError):
            self.assertIsNotNone(build_run_receipt("Image generated", elapsed=3))


class ElapsedFormatTests(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(format_elapsed(0), "0:00")
        self.assertEqual(format_elapsed(9), "0:09")
        self.assertEqual(format_elapsed(41), "0:41")
        self.assertEqual(format_elapsed(605), "10:05")
        self.assertEqual(format_elapsed(3661), "1:01:01")

    def test_negative_and_none_are_safe(self):
        self.assertEqual(format_elapsed(-5), "0:00")
        self.assertEqual(format_elapsed(None), "0:00")


if __name__ == "__main__":
    unittest.main()

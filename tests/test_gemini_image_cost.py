import os
import tempfile
import unittest

from providers import gemini_images
from services import usage_costs


class GeminiImageCostTests(unittest.TestCase):
    """Image cost must track the resolution actually requested, not a flat
    guess — 4K really costs ~$0.24, 1K/2K ~$0.134."""

    def setUp(self):
        self._old_db = usage_costs.DB_PATH
        self._tmp = tempfile.TemporaryDirectory()
        usage_costs.DB_PATH = os.path.join(self._tmp.name, "usage.db")

    def tearDown(self):
        usage_costs.DB_PATH = self._old_db
        self._tmp.cleanup()

    def test_4k_costs_more_than_default(self):
        gemini_images._record_gemini_image_cost("gemini-3-pro-image", "image_generation", image_size="4K")
        self.assertAlmostEqual(usage_costs.today()["cost"], 0.24, places=4)

    def test_2k_uses_standard_tier(self):
        gemini_images._record_gemini_image_cost("gemini-3-pro-image", "image_generation", image_size="2K")
        self.assertAlmostEqual(usage_costs.today()["cost"], 0.134, places=4)

    def test_edit_path_default_is_standard_tier(self):
        # Edit/reference paths omit image_size and run at the default tier.
        gemini_images._record_gemini_image_cost("gemini-3-pro-image", "image_edit")
        self.assertAlmostEqual(usage_costs.today()["cost"], 0.134, places=4)


if __name__ == "__main__":
    unittest.main()

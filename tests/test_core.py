from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from customer_feedback_nlp.data_loader import load_data
from customer_feedback_nlp.preprocess import clean_text
from customer_feedback_nlp.sentiment import classify_by_score


class CoreTests(unittest.TestCase):
    def test_classify_by_score_boundaries(self):
        self.assertEqual(classify_by_score(0.29), "负面")
        self.assertEqual(classify_by_score(0.30), "中性")
        self.assertEqual(classify_by_score(0.70), "中性")
        self.assertEqual(classify_by_score(0.71), "正面")

    def test_clean_text_removes_urls_and_symbols(self):
        text = clean_text("<p>物流太慢了 https://example.com @客服</p>")
        self.assertNotIn("http", text)
        self.assertNotIn("客服", text)
        self.assertIn("物流", text)

    def test_load_data_reads_sample_csv(self):
        data = load_data()
        self.assertFalse(data.empty)
        self.assertTrue({"text", "intent"}.issubset(data.columns))


if __name__ == "__main__":
    unittest.main()

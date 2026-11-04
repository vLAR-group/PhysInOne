import unittest
from update_download_stats import REPOSITORIES, build


class DownloadStatsTests(unittest.TestCase):
    def rows(self):
        return [{"repo": r, "all_time": i+100, "last_30_days": i} for i, r in enumerate(REPOSITORIES)]

    def test_complete_snapshot(self):
        result = build(self.rows()[::-1], "2026-11-04T10:00:00+00:00")
        self.assertEqual(result["stats.json"]["all_time"], 1836)
        self.assertEqual(result["stats.json"]["last_30_days"], 136)
        self.assertEqual(result["downloads.json"]["message"], "1,836")
        self.assertEqual(result["stats.json"]["repositories"][0]["repo"], REPOSITORIES[0])

    def test_missing_or_duplicate_repository(self):
        for rows in [self.rows()[:-1], self.rows()[:-1]+[self.rows()[0]]]:
            with self.assertRaises(ValueError):
                build(rows, "2026-11-04")

    def test_invalid_counters(self):
        for value in [None, True, -1, "100", 1.5]:
            for key in ["all_time", "last_30_days"]:
                rows = self.rows()
                rows[0][key] = value
                with self.assertRaises(ValueError):
                    build(rows, "2026-11-04")


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BracketLayoutRegressionTests(unittest.TestCase):
    def test_bracket_uses_mirrored_grid_and_complete_connectors(self):
        script = (ROOT / "static" / "js" / "dashboard.js").read_text(encoding="utf-8")
        css = (ROOT / "static" / "css" / "style.css").read_text(encoding="utf-8")

        self.assertIn("1: [8]", script)
        self.assertIn("2: [4, 12]", script)
        self.assertIn("4: [2, 6, 10, 14]", script)
        self.assertIn("8: [1, 3, 5, 7, 9, 11, 13, 15]", script)
        self.assertIn('column("Round of 32", leftLevels[3], 1)', script)
        self.assertIn('column("Round of 32", rightLevels[3], 9)', script)
        self.assertIn("matchCardHTML(final, 5, 8)", script)
        self.assertIn('class="bracket-connectors"', script)
        self.assertIn("function drawBracketConnectors()", script)
        self.assertIn("[match.home_from, match.away_from]", script)
        self.assertIn("winner-connector", script)

        self.assertIn("grid-template-columns: repeat(9", css)
        self.assertIn("grid-template-rows: 32px 42px repeat(16, 48px)", css)
        self.assertIn("height: 84px", css)
        self.assertIn("grid-column: 5", css)
        self.assertNotIn(".branch-left .match-card::after", css)


if __name__ == "__main__":
    unittest.main()

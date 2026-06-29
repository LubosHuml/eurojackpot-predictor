import unittest
from unittest.mock import patch, MagicMock
import scraper
import database
import os

class TestScraper(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # We will use a test database file for safety during tests
        database.DB_FILE = "test_eurojackpot.db"
        database.init_db()

    @classmethod
    def tearDownClass(cls):
        # Clean up test database file
        if os.path.exists("test_eurojackpot.db"):
            os.remove("test_eurojackpot.db")

    @patch('scraper.requests.get')
    def test_scrape_year_success(self, mock_get):
        # Mock HTML response
        mock_html = """
        <html>
        <body>
            <table>
                <tr>
                    <td><a href="/results/26-06-2026">Friday 26th June 2026</a></td>
                    <td>
                        <ul class="balls small">
                            <li class="ball"><span>17</span></li>
                            <li class="ball"><span>25</span></li>
                            <li class="ball"><span>35</span></li>
                            <li class="ball"><span>39</span></li>
                            <li class="ball"><span>41</span></li>
                            <li class="euro"><span>5</span></li>
                            <li class="euro"><span>9</span></li>
                        </ul>
                    </td>
                </tr>
            </table>
        </body>
        </html>
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = mock_html
        mock_get.return_value = mock_resp

        draws = scraper.scrape_year(2026)
        self.assertEqual(len(draws), 1)
        date, main, euro = draws[0]
        self.assertEqual(date, "2026-06-26")
        self.assertEqual(main, [17, 25, 35, 39, 41])
        self.assertEqual(euro, [5, 9])

    @patch('scraper.requests.get')
    def test_fetch_latest_draw_success(self, mock_get):
        # Mock JSON response from Lottoland
        mock_json = {
            "last": {
                "nr": 967,
                "date": {
                    "year": 2026,
                    "month": 6,
                    "day": 26
                },
                "numbers": [17, 25, 35, 39, 41],
                "euroNumbers": [5, 9]
            }
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_json
        mock_get.return_value = mock_resp

        latest = scraper.fetch_latest_draw()
        self.assertIsNotNone(latest)
        date, main, euro = latest
        self.assertEqual(date, "2026-06-26")
        self.assertEqual(main, [17, 25, 35, 39, 41])
        self.assertEqual(euro, [5, 9])

    def test_database_guardrails(self):
        # Clear database for testing guardrails
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM draws")
        conn.commit()
        conn.close()

        # Valid insertion
        success = database.insert_draw("2026-06-26", [17, 25, 35, 39, 41], [5, 9])
        self.assertTrue(success)
        
        # Duplicate date insertion
        dup_success = database.insert_draw("2026-06-26", [1, 2, 3, 4, 5], [1, 2])
        self.assertFalse(dup_success)

        # Invalid main numbers length
        with self.assertRaises(ValueError):
            database.insert_draw("2026-06-27", [1, 2, 3, 4], [1, 2])

        # Out of bounds main numbers
        with self.assertRaises(ValueError):
            database.insert_draw("2026-06-27", [1, 2, 3, 4, 51], [1, 2])

        # Out of bounds euro numbers
        with self.assertRaises(ValueError):
            database.insert_draw("2026-06-27", [1, 2, 3, 4, 5], [1, 13])

if __name__ == "__main__":
    unittest.main()

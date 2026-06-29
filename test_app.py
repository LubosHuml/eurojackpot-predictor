import unittest
from unittest.mock import patch, MagicMock
import app
import database
import os

class TestApp(unittest.TestCase):

    def setUp(self):
        # Configure app for testing
        app.app.config['TESTING'] = True
        self.client = app.app.test_client()
        
        # Override database path for safety during testing
        database.DB_FILE = "test_eurojackpot.db"
        database.init_db()

    def tearDown(self):
        # Clean up database file
        if os.path.exists("test_eurojackpot.db"):
            os.remove("test_eurojackpot.db")

    def test_index_route(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_api_status(self):
        response = self.client.get('/api/status')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("total_draws", data)
        self.assertIn("latest_draw_date", data)
        self.assertIn("model_trained", data)
        self.assertIn("training_state", data)

    @patch('app.os.path.exists')
    def test_api_metrics_not_trained(self, mock_exists):
        # Mock that model does not exist
        mock_exists.return_value = False
        response = self.client.get('/api/metrics')
        # Since model is not trained, returns 400
        self.assertEqual(response.status_code, 400)

    @patch('app.os.path.exists')
    def test_api_predictions_not_trained(self, mock_exists):
        # Mock that model does not exist
        mock_exists.return_value = False
        response = self.client.get('/api/predictions')
        # Since model is not trained, returns 400
        self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()

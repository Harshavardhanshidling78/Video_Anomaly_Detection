import unittest
from fastapi.testclient import TestClient
from src.api.main import app

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("device", data)

    def test_get_incidents(self):
        response = self.client.get("/api/v1/incidents?limit=5")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")

if __name__ == "__main__":
    unittest.main()

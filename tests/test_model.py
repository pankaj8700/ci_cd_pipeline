import joblib
from sklearn.ensemble import RandomForestClassifier
import unittest

class TestModel(unittest.TestCase):
    def test_model(self):
        model = joblib.load('model/iris.pkl')
        self.assertIsInstance(model, LogisticRegression)
        self.assertGreaterEqual(len(model.feature_importances_), 4)

if __name__ == '__main__':
    unittest.main()
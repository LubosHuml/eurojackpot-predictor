import unittest
import numpy as np
import tensorflow as tf
import os
import models
import train

class TestModels(unittest.TestCase):

    def setUp(self):
        # Disable GPU for fast test execution
        tf.config.set_visible_devices([], 'GPU')
        self.window_size = 5
        self.num_features = 13
        self.lstm_units = 16

        # Set test scaler file paths to prevent destroying real model scalers
        train.SCALER_X_PATH = "test_scaler_x.joblib"
        train.SCALER_SUM_PATH = "test_scaler_sum.joblib"
        train.SCALER_COUNTS_PATH = "test_scaler_counts.joblib"

        # Generate small dummy sequence dataset
        self.num_samples = 40
        self.dummy_data = {
            'X_num': np.random.randn(self.num_samples, self.window_size, self.num_features).astype(np.float32),
            'X_main': np.random.randint(1, 51, size=(self.num_samples, self.window_size, 5), dtype=np.int32),
            'X_euro': np.random.randint(1, 13, size=(self.num_samples, self.window_size, 2), dtype=np.int32),
            'y_sum': np.random.randn(self.num_samples, 1).astype(np.float32),
            'y_counts': np.random.randn(self.num_samples, 4).astype(np.float32),
            'y_main_logits': np.zeros((self.num_samples, 50), dtype=np.float32),
            'y_euro_logits': np.zeros((self.num_samples, 12), dtype=np.float32)
        }
        # Populate dummy binary values for logits
        for i in range(self.num_samples):
            self.dummy_data['y_main_logits'][i, np.random.choice(50, 5, replace=False)] = 1.0
            self.dummy_data['y_euro_logits'][i, np.random.choice(12, 2, replace=False)] = 1.0

    def tearDown(self):
        # Clean up joblib files generated during testing
        for file in [train.SCALER_X_PATH, train.SCALER_SUM_PATH, train.SCALER_COUNTS_PATH]:
            if os.path.exists(file):
                os.remove(file)

    def test_build_model(self):
        model = models.build_multimodal_model(
            window_size=self.window_size,
            num_features=self.num_features,
            lstm_units=self.lstm_units
        )
        self.assertIsNotNone(model)
        # Check inputs and outputs names
        self.assertEqual(len(model.inputs), 3)
        self.assertEqual(len(model.outputs), 4)

    def test_train_and_prune(self):
        # Test training and pruning function for 2 epochs
        model = train.train_and_prune(
            data_dict=self.dummy_data,
            window_size=self.window_size,
            lstm_units=self.lstm_units,
            learning_rate=1e-3,
            epochs=2,
            verbose=0
        )
        self.assertIsNotNone(model)

        # Check sparsity of Dense layers
        prune_layers = ["sum_head", "counts_head", "main_logits_head", "euro_logits_head"]
        for name in prune_layers:
            layer = model.get_layer(name)
            kernel = layer.get_weights()[0]
            # Since threshold is 80th percentile, approximately 80% should be zero
            sparsity = np.mean(kernel == 0.0)
            self.assertAlmostEqual(sparsity, 0.80, delta=0.10)

if __name__ == "__main__":
    unittest.main()

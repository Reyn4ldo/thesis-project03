"""
Unit tests for ESBL classifier to verify multiclass error fix.
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.esbl_classifier import ESBLClassifierExperiment


class TestESBLClassifierDataPreparation(unittest.TestCase):
    """Tests for ESBL classifier data preparation."""
    
    def setUp(self):
        """Create sample data with encoded features."""
        np.random.seed(42)
        n_samples = 100
        
        # Test case 1: Binary classification data (ideal case)
        # Encoding: 'neg' -> 0, 'pos' -> 1
        self.df_binary = pd.DataFrame({
            'esbl': ['neg'] * 50 + ['pos'] * 50,
            'esbl_encoded': [0] * 50 + [1] * 50,
            'ampicillin_resistant': np.random.randint(0, 2, n_samples),
            'gentamicin_resistant': np.random.randint(0, 2, n_samples),
            'antibiogram_vec_0': np.random.rand(n_samples),
            'total_resistant': np.random.randint(0, 10, n_samples),
            'mar_index': np.random.rand(n_samples)
        })
        
        # Test case 2: Multiclass data with missing values
        # This simulates what happens when LabelEncoder encodes missing values
        # Encoding: 'missing' -> 0, 'neg' -> 1, 'pos' -> 2
        # After filtering missing values, we should have: 'neg' -> 1, 'pos' -> 2
        # which then gets remapped to binary: 0 and 1
        self.df_multiclass = pd.DataFrame({
            'esbl': ['neg'] * 40 + ['pos'] * 40 + [np.nan] * 20,
            'esbl_encoded': [1] * 40 + [2] * 40 + [0] * 20,  # 0 represents 'missing'
            'ampicillin_resistant': np.random.randint(0, 2, n_samples),
            'gentamicin_resistant': np.random.randint(0, 2, n_samples),
            'antibiogram_vec_0': np.random.rand(n_samples),
            'total_resistant': np.random.randint(0, 10, n_samples),
            'mar_index': np.random.rand(n_samples)
        })
    
    def test_binary_classification_data(self):
        """Test that binary data is handled correctly."""
        experiment = ESBLClassifierExperiment()
        X, y, features = experiment.prepare_data(self.df_binary)
        
        # Check that we have binary classes only
        unique_classes = np.unique(y)
        self.assertEqual(len(unique_classes), 2, "Should have exactly 2 classes")
        self.assertTrue(set(unique_classes).issubset({0, 1}), "Classes should be 0 and 1")
        
        # Check that we didn't lose too many samples
        self.assertEqual(len(y), 100, "Should have all 100 samples")
    
    def test_multiclass_to_binary_conversion(self):
        """Test that multiclass data is converted to binary."""
        experiment = ESBLClassifierExperiment()
        X, y, features = experiment.prepare_data(self.df_multiclass)
        
        # Check that we have binary classes only after filtering
        unique_classes = np.unique(y)
        self.assertEqual(len(unique_classes), 2, "Should have exactly 2 classes after filtering")
        self.assertTrue(set(unique_classes).issubset({0, 1}), "Classes should be 0 and 1")
        
        # Check that missing values were filtered out
        self.assertEqual(len(y), 80, "Should have 80 samples after filtering missing")
    
    def test_filtering_missing_esbl_values(self):
        """Test that samples with missing ESBL are filtered out."""
        experiment = ESBLClassifierExperiment()
        X, y, features = experiment.prepare_data(self.df_multiclass)
        
        # After filtering, we should only have 80 samples (40 neg + 40 pos)
        self.assertEqual(len(y), 80)
        
        # Class distribution should be balanced
        class_counts = np.bincount(y)
        self.assertEqual(len(class_counts), 2, "Should have 2 classes")
        self.assertEqual(class_counts[0], 40, "Class 0 should have 40 samples")
        self.assertEqual(class_counts[1], 40, "Class 1 should have 40 samples")


class TestESBLClassifierEvaluation(unittest.TestCase):
    """Tests for ESBL classifier evaluation."""
    
    def test_evaluation_with_binary_classes(self):
        """Test evaluation with binary classes in test set."""
        from sklearn.ensemble import RandomForestClassifier
        
        np.random.seed(42)
        n_test = 50
        
        # Create test data with binary classes
        X_test = np.random.rand(n_test, 10)
        y_test = np.random.randint(0, 2, n_test)
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)
        model.fit(X_train, y_train)
        
        # Evaluate
        experiment = ESBLClassifierExperiment()
        metrics = experiment.evaluate_model(model, X_test, y_test)
        
        # Check that all expected metrics are present
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('confusion_matrix', metrics)
        
        # Check that metrics are in valid ranges
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
    
    def test_evaluation_with_single_class(self):
        """Test evaluation when test set has only one class."""
        from sklearn.ensemble import RandomForestClassifier
        
        np.random.seed(42)
        n_test = 50
        
        # Create test data with only one class
        X_test = np.random.rand(n_test, 10)
        y_test = np.zeros(n_test, dtype=int)  # All zeros
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)
        model.fit(X_train, y_train)
        
        # Evaluate - should not raise an error
        experiment = ESBLClassifierExperiment()
        metrics = experiment.evaluate_model(model, X_test, y_test)
        
        # Check that metrics are still computed
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestESBLClassifierDataPreparation))
    suite.addTests(loader.loadTestsFromTestCase(TestESBLClassifierEvaluation))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

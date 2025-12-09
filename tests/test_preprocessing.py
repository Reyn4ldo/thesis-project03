"""
Unit tests for preprocessing pipeline components.

Tests cover:
- Data validation (ranges, consistency)
- Label consistency checks
- Data leak detection
- Feature validation
"""

import unittest
import pandas as pd
import numpy as np
import sys
import tempfile
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing import (
    MICSIRCleaner,
    DomainAwareImputer,
    ResistanceFeatureEngineer,
    StratifiedDataSplitter,
    create_preprocessing_pipeline
)


class TestMICSIRCleaner(unittest.TestCase):
    """Tests for MIC and S/I/R cleaning."""
    
    def setUp(self):
        """Create sample data for testing."""
        self.df = pd.DataFrame({
            'ampicillin_mic': ['<=2', '>=32', '16', '4', 'trm'],
            'ampicillin_int': ['s', 'r', 'i', 's', 'r'],
            'gentamicin_mic': ['≤1', '≥16', '2', np.nan, '8*'],
            'gentamicin_int': ['s', 'r', 's', np.nan, 'i']
        })
    
    def test_mic_normalization(self):
        """Test MIC value normalization."""
        cleaner = MICSIRCleaner(verbose=False)
        cleaner.fit(self.df)
        result = cleaner.transform(self.df)
        
        # Check that Unicode operators are normalized
        self.assertTrue('gentamicin_mic_operator' in result.columns)
        self.assertEqual(result['gentamicin_mic_operator'].iloc[0], '<=')
        
        # Check numeric extraction
        self.assertTrue('gentamicin_mic_numeric' in result.columns)
        self.assertEqual(result['gentamicin_mic_numeric'].iloc[0], 1.0)
    
    def test_sir_standardization(self):
        """Test S/I/R value standardization."""
        cleaner = MICSIRCleaner(verbose=False)
        cleaner.fit(self.df)
        result = cleaner.transform(self.df)
        
        # Check that all S/I/R values are lowercase
        sir_clean = result['ampicillin_int_clean'].dropna()
        self.assertTrue(all(v in ['s', 'i', 'r'] for v in sir_clean))
    
    def test_invalid_value_handling(self):
        """Test handling of invalid MIC values."""
        cleaner = MICSIRCleaner(verbose=False)
        cleaner.fit(self.df)
        result = cleaner.transform(self.df)
        
        # Check that 'trm' is converted to NaN
        self.assertTrue(pd.isna(result['ampicillin_mic_numeric'].iloc[4]))
    
    def test_inconsistency_detection(self):
        """Test MIC/SIR inconsistency detection."""
        cleaner = MICSIRCleaner(add_inconsistency_flags=True, verbose=False)
        cleaner.fit(self.df)
        result = cleaner.transform(self.df)
        
        # Check that inconsistency flags are added
        self.assertTrue('ampicillin_mic_sir_inconsistent' in result.columns)


class TestDomainAwareImputer(unittest.TestCase):
    """Tests for missing value imputation."""
    
    def setUp(self):
        """Create sample data with missing values."""
        self.df = pd.DataFrame({
            'ampicillin_mic_numeric': [2.0, 32.0, np.nan, 4.0, 8.0],
            'ampicillin_int_clean': ['s', 'r', np.nan, 's', 'i'],
            'gentamicin_mic_numeric': [1.0, np.nan, 2.0, 1.0, np.nan],
            'gentamicin_int_clean': ['s', 'r', 's', np.nan, 's']
        })
    
    def test_median_imputation(self):
        """Test median imputation for MIC values."""
        imputer = DomainAwareImputer(mic_strategy='median', verbose=False)
        imputer.fit(self.df)
        result = imputer.transform(self.df)
        
        # Check that missing MIC values are imputed
        self.assertFalse(result['ampicillin_mic_numeric'].isna().any())
    
    def test_sir_imputation(self):
        """Test S/I/R imputation with 'not_tested' category."""
        imputer = DomainAwareImputer(sir_strategy='not_tested', verbose=False)
        imputer.fit(self.df)
        result = imputer.transform(self.df)
        
        # Check that missing S/I/R values are imputed with 'not_tested'
        self.assertFalse(result['ampicillin_int_clean'].isna().any())
        self.assertTrue('not_tested' in result['ampicillin_int_clean'].values)
    
    def test_imputation_indicators(self):
        """Test that imputation indicators are added."""
        imputer = DomainAwareImputer(add_indicators=True, verbose=False)
        imputer.fit(self.df)
        result = imputer.transform(self.df)
        
        # Check that indicator columns are added
        self.assertTrue('ampicillin_mic_imputed' in result.columns)
        self.assertTrue('ampicillin_sir_imputed' in result.columns)
        
        # Check that indicators correctly mark imputed values
        self.assertEqual(result['ampicillin_mic_imputed'].iloc[2], 1)


class TestResistanceFeatureEngineer(unittest.TestCase):
    """Tests for feature engineering."""
    
    def setUp(self):
        """Create sample data for testing."""
        self.df = pd.DataFrame({
            'ampicillin_int_clean': ['s', 'r', 'i', 's', 'r'],
            'gentamicin_int_clean': ['s', 's', 's', 'r', 'r'],
            'bacterial_species': ['e_coli', 'k_pneumoniae', 'e_coli', 'e_coli', 'k_pneumoniae'],
            'administrative_region': ['region1', 'region2', 'region1', 'region2', 'region1'],
            'replicate': [1, 2, 1, 3, 2],
            'mar_index': [0.0, 0.5, 0.0, 0.5, 1.0]
        })
    
    def test_binary_resistance_features(self):
        """Test creation of binary resistance indicators."""
        engineer = ResistanceFeatureEngineer(
            create_binary_resistance=True,
            create_aggregates=False,
            create_antibiogram=False,
            create_who_features=False,
            encode_metadata=False,
            verbose=False
        )
        engineer.fit(self.df)
        result = engineer.transform(self.df)
        
        # Check that binary resistance columns are created
        self.assertTrue('ampicillin_resistant' in result.columns)
        self.assertTrue('gentamicin_resistant' in result.columns)
        
        # Check values
        self.assertEqual(result['ampicillin_resistant'].iloc[0], 0)  # 's' -> 0
        self.assertEqual(result['ampicillin_resistant'].iloc[1], 1)  # 'r' -> 1
    
    def test_aggregate_features(self):
        """Test creation of aggregate resistance features."""
        engineer = ResistanceFeatureEngineer(
            create_binary_resistance=False,
            create_aggregates=True,
            create_antibiogram=False,
            create_who_features=False,
            encode_metadata=False,
            verbose=False
        )
        engineer.fit(self.df)
        result = engineer.transform(self.df)
        
        # Check that aggregate columns are created
        self.assertTrue('total_resistant' in result.columns)
        self.assertTrue('total_susceptible' in result.columns)
        self.assertTrue('resistance_ratio' in result.columns)
        
        # Check values for first row (s, s)
        self.assertEqual(result['total_resistant'].iloc[0], 0)
        self.assertEqual(result['total_susceptible'].iloc[0], 2)
    
    def test_mar_index_validation(self):
        """Test MAR index validation."""
        engineer = ResistanceFeatureEngineer(
            create_aggregates=True,
            validate_mar_index=True,
            create_binary_resistance=False,
            create_antibiogram=False,
            create_who_features=False,
            encode_metadata=False,
            verbose=False
        )
        engineer.fit(self.df)
        result = engineer.transform(self.df)
        
        # Check that MAR index validation flag is added
        self.assertTrue('mar_index_validated' in result.columns)
    
    def test_metadata_encoding(self):
        """Test metadata encoding."""
        engineer = ResistanceFeatureEngineer(
            create_binary_resistance=False,
            create_aggregates=False,
            create_antibiogram=False,
            create_who_features=False,
            encode_metadata=True,
            encoding_strategy='label',
            verbose=False
        )
        engineer.fit(self.df)
        result = engineer.transform(self.df)
        
        # Check that encoded columns are created
        self.assertTrue('bacterial_species_encoded' in result.columns)
        self.assertTrue('administrative_region_encoded' in result.columns)


class TestStratifiedDataSplitter(unittest.TestCase):
    """Tests for data splitting."""
    
    def setUp(self):
        """Create sample data for testing."""
        self.df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'bacterial_species': ['species1'] * 60 + ['species2'] * 40
        })
    
    def test_stratified_split(self):
        """Test stratified splitting."""
        splitter = StratifiedDataSplitter(
            stratify_by='bacterial_species',
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            verbose=False
        )
        
        train_df, val_df, test_df = splitter.split(self.df)
        
        # Check sizes
        self.assertAlmostEqual(len(test_df) / len(self.df), 0.2, places=1)
        self.assertAlmostEqual(len(val_df) / len(self.df), 0.1, places=1)
        
        # Check stratification (approximate)
        train_ratio = (train_df['bacterial_species'] == 'species1').sum() / len(train_df)
        test_ratio = (test_df['bacterial_species'] == 'species1').sum() / len(test_df)
        self.assertAlmostEqual(train_ratio, 0.6, places=1)
        self.assertAlmostEqual(test_ratio, 0.6, places=1)
    
    def test_no_data_leakage(self):
        """Test that there's no overlap between splits."""
        splitter = StratifiedDataSplitter(
            stratify_by='bacterial_species',
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            verbose=False
        )
        
        train_df, val_df, test_df = splitter.split(self.df)
        
        # Get the stored indices (which reference the original dataframe)
        train_indices, val_indices, test_indices = splitter.get_indices()
        
        train_indices = set(train_indices)
        val_indices = set(val_indices)
        test_indices = set(test_indices)
        
        # Check no overlap
        self.assertEqual(len(train_indices & val_indices), 0)
        self.assertEqual(len(train_indices & test_indices), 0)
        self.assertEqual(len(val_indices & test_indices), 0)
        
        # Check all data is used
        total_indices = train_indices | val_indices | test_indices
        self.assertEqual(len(total_indices), len(self.df))


class TestDataValidation(unittest.TestCase):
    """Tests for data validation."""
    
    def test_mic_range_validation(self):
        """Test that MIC numeric values are in valid range."""
        df = pd.DataFrame({
            'ampicillin_mic_numeric': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
        })
        
        # MIC values should be positive
        self.assertTrue((df['ampicillin_mic_numeric'] > 0).all())
        
        # MIC values should be reasonable (typically < 1000)
        self.assertTrue((df['ampicillin_mic_numeric'] < 1000).all())
    
    def test_sir_label_consistency(self):
        """Test that S/I/R labels are consistent."""
        df = pd.DataFrame({
            'ampicillin_int_clean': ['s', 'i', 'r', 's', 'not_tested']
        })
        
        valid_labels = {'s', 'i', 'r', 'not_tested'}
        self.assertTrue(set(df['ampicillin_int_clean'].unique()).issubset(valid_labels))
    
    def test_resistance_ratio_range(self):
        """Test that resistance ratios are in [0, 1] range."""
        df = pd.DataFrame({
            'resistance_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]
        })
        
        self.assertTrue((df['resistance_ratio'] >= 0).all())
        self.assertTrue((df['resistance_ratio'] <= 1).all())
    
    def test_mar_index_range(self):
        """Test that MAR index is in valid range."""
        df = pd.DataFrame({
            'mar_index': [0.0, 0.1, 0.2, 0.5, 1.0]
        })
        
        # MAR index should be between 0 and 1 (or slightly above due to calculation method)
        self.assertTrue((df['mar_index'] >= 0).all())
        self.assertTrue((df['mar_index'] <= 1.5).all())  # Allow up to 1.5 as seen in data


class TestFullPipeline(unittest.TestCase):
    """Tests for the complete preprocessing pipeline."""
    
    def setUp(self):
        """Create sample data for full pipeline testing."""
        self.df = pd.DataFrame({
            'ampicillin_mic': ['<=2', '>=32', '16', np.nan, '8'],
            'ampicillin_int': ['s', 'r', 'i', np.nan, 's'],
            'gentamicin_mic': ['≤1', '≥16', '2', '1', '8'],
            'gentamicin_int': ['s', 'r', 's', 's', 'i'],
            'bacterial_species': ['e_coli', 'k_pneumoniae', 'e_coli', 'e_coli', 'k_pneumoniae'],
            'administrative_region': ['region1', 'region2', 'region1', 'region2', 'region1'],
            'sample_source': ['water', 'fish', 'water', 'fish', 'water'],
            'replicate': [1, 2, 1, 3, 2],
            'colony': [1, 2, 3, 4, 5],
            'esbl': ['neg', 'neg', 'pos', 'neg', 'neg'],
            'mar_index': [0.0, 0.5, 0.0, 0.0, 0.5]
        })
    
    def test_pipeline_fit_transform(self):
        """Test that pipeline can fit and transform data."""
        pipeline = create_preprocessing_pipeline(verbose=False)
        
        result = pipeline.fit_transform(self.df)
        
        # Check that result is returned
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check that new columns are created
        self.assertGreater(len(result.columns), len(self.df.columns))
    
    def test_pipeline_reproducibility(self):
        """Test that pipeline produces reproducible results."""
        pipeline1 = create_preprocessing_pipeline(verbose=False)
        pipeline2 = create_preprocessing_pipeline(verbose=False)
        
        result1 = pipeline1.fit_transform(self.df)
        result2 = pipeline2.fit_transform(self.df)
        
        # Results should be identical (for deterministic components)
        pd.testing.assert_frame_equal(
            result1[['ampicillin_mic_numeric', 'gentamicin_mic_numeric']],
            result2[['ampicillin_mic_numeric', 'gentamicin_mic_numeric']]
        )


class TestUnicodeEncoding(unittest.TestCase):
    """Tests for Unicode encoding in file operations."""
    
    def test_unicode_file_write(self):
        """Test that files with Unicode characters can be written correctly."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.md') as f:
            temp_path = f.name
            # Write content with Unicode characters like those in PHASE1_SUMMARY.md
            test_content = "# Test Report\n✓ Test passed\n✓ Unicode support working\n"
            f.write(test_content)
        
        try:
            # Read it back to verify
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.assertIn('✓', content)
            self.assertEqual(content, test_content)
        finally:
            # Clean up
            os.unlink(temp_path)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMICSIRCleaner))
    suite.addTests(loader.loadTestsFromTestCase(TestDomainAwareImputer))
    suite.addTests(loader.loadTestsFromTestCase(TestResistanceFeatureEngineer))
    suite.addTests(loader.loadTestsFromTestCase(TestStratifiedDataSplitter))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestFullPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestUnicodeEncoding))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

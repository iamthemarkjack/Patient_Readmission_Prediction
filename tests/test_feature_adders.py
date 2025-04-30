# To test the feature adding utility function
import os 
import sys
import unittest
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airflow.dags.utils.add_feature import *

class TestFeatureAdders(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'age': [25, 50, 85],
            'primary_diagnosis': ['Hypertension', 'Asthma', 'Unknown'],
            'days_in_hospital': [2, 10, 5],
            'num_procedures': [0, 2, 6],
            'comorbidity_score': [1, 4, 2]
        })

    def test_add_age_group(self):
        add_age_group(self.df)
        self.assertIn('age_group', self.df.columns)
        self.assertEqual(list(self.df['age_group']), ['18-30', '46-60', '76+'])

    def test_add_diagnosis_group(self):
        add_diagnosis_group(self.df)
        self.assertIn('diagnosis_group', self.df.columns)
        self.assertEqual(list(self.df['diagnosis_group']), ['Cardiac', 'Respiratory', 'Other'])

    def test_add_hospital_stay_features(self):
        add_hospital_stay_features(self.df)
        self.assertIn('stay_bucket', self.df.columns)
        self.assertIn('long_stay', self.df.columns)
        self.assertEqual(list(self.df['long_stay']), [0, 1, 0])

    def test_add_procedure_features(self):
        add_diagnosis_group(self.df)  # Required for factorize
        add_procedure_features(self.df)
        self.assertIn('procedure_category', self.df.columns)
        self.assertIn('procedures_diagnosis_interaction', self.df.columns)

    def test_add_comorbidity_features(self):
        add_comorbidity_features(self.df)
        self.assertIn('high_risk_comorbidity', self.df.columns)
        self.assertEqual(list(self.df['high_risk_comorbidity']), [0, 1, 0])

    def test_add_interaction_features(self):
        add_interaction_features(self.df)
        self.assertIn('age_comorbidity_interaction', self.df.columns)
        self.assertIn('procedures_age_interaction', self.df.columns)
        self.assertIn('procedures_comorbidity_interaction', self.df.columns)

    def test_add_features(self):
        df = add_features(self.df.copy())
        expected_cols = [
            'age_group', 'diagnosis_group', 'stay_bucket', 'long_stay',
            'procedure_category', 'procedures_diagnosis_interaction',
            'high_risk_comorbidity', 'age_comorbidity_interaction',
            'procedures_age_interaction', 'procedures_comorbidity_interaction'
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)

if __name__ == '__main__':
    unittest.main()
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd

# Abstract Base Class for Missing Values Analysis
class MissingValuesAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        Performs a complete missing values analysis by identifying and visualizing missing values.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: This method performs the analysis and visualizes missing values.
        """
        self.identify_missing_values(df)

    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Identifies missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: This method should print the count of missing values for each column.
        """
        pass


# This class implements methods to identify missing values in the dataframe.
class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame):
        print("\nMissing Values Count by Column:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])

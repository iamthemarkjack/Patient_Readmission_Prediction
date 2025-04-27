from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Abstract Base Class for Plotter Analysis Strategy
class PlotterStrategy(ABC):
    @abstractmethod
    def plot(self, df: pd.DataFrame, feature: str, target: str):
        """
        Make plots on a specific feature with respect to other specified features of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column to be analyzed.
        target (str): The name of the target column

        Returns:
        None: This method visualizes the distribution of the feature.
        """
        pass

# This strategy plots histogram of numerical features
class HistogramPlot(PlotterStrategy):
    def plot(self, df: pd.DataFrame, feature: str, target: str):
        # Plots Histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(df, x=feature, hue=target, multiple='stack', kde=True, palette='viridis')
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()

# This strategy pltots box plot with gender as hue
class WithGenderBoxPlot(PlotterStrategy):
    def plot(self, df: pd.DataFrame, feature: str, target: str):
        # Plots the Box Plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(df, x=target, y=feature, hue='gender')
        plt.title(f'Box Plot of {feature} by {target} and gender')
        plt.xlabel(f'{target}')
        plt.ylabel(f'{feature}')
        plt.show()

# This strategy plots countplot with gender as hue
class WithGenderCountPlot(PlotterStrategy):
    def plot(self, df: pd.DataFrame, feature: str, target: str):
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x='gender', hue=target, palette='Set2')
        plt.title(f'{target} Rate by Gender')
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.show()

# This strategy plots bar plot of top 10 classes
class TopBarPlot(PlotterStrategy):
    def plot(self, df: pd.DataFrame, feature: str, target: str):
        # Plots Bar Chart
        plt.figure(figsize=(12, 6))
        count = df[feature].value_counts().nlargest(10).index
        sns.countplot(data=df[df[feature].isin(count)], x=feature, hue=target, palette='coolwarm')
        plt.title(f'Top 10 {feature} with {target}')
        plt.xlabel(f'{feature}')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.show()

# This class allows you to switch between different strategies.
class Plotter:
    def __init__(self, strategy: PlotterStrategy):
        """
        Initializes the Plotter with a specific analysis strategy.

        Parameters:
        strategy (PlotterStrategy): The strategy to be used for plotting.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: PlotterStrategy):
        """
        Sets a new strategy for the Plotter.
        """
        self._strategy = strategy

    def execute_plot(self, df: pd.DataFrame, feature: str, target: str):
        """
        Make plot using the current strategy.
        """
        self._strategy.plot(df, feature, target)
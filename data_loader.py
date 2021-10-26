import glob
import os

import numpy as np
import pandas as pd


class DataLoader:
    """
    Manages the portfolio data
    """

    def __init__(self, data_directory: str):
        """
        Initialize the data loader.
        Args:
            data_directory (str): The path where all data will be stored
        """
        # Make the parent directory
        self.parent_directory = data_directory
        os.makedirs(data_directory, exist_ok=True)

        # Get the data identifiers (numbers)
        self._get_all_data_numbers()
        self.data_number = 0  # Initialize data number at 1

        # Declare names for the mu and sigma data
        self.mu_names = 'average_return'
        self.sigma_names = 'covariance_return'

    def _get_all_data_numbers(self):
        """
        Get the data numbers for all existing data
        """
        self.data_paths_list = glob.glob(f'{self.parent_directory}/*')  # all data paths
        # Generate a list of all data numbers
        self.data_numbers = np.unique([int(f"{k.split('_')[-1].split('.csv')[0]}")
                                       for k in
                                       self.data_paths_list])

    def _get_next_file_name(self):
        """
        updates the data number to be equal to the number that should be assigned to a newly generated data point
        """
        if self.data_numbers.size == 0:
            self.data_number = 0
        else:
            self.data_number = np.max(self.data_numbers) + 1

    def save_data(self, mu_s: pd.Series, sigma_df: pd.DataFrame,
                  mu_test_s: pd.Series, sigma_test_df: pd.DataFrame):
        """
        Saves new data as CSV files in the appropriate directory
        Args:
            mu_s (pd.Series): the expected daily return vector
            sigma_df (pd.DataFrame): The covariance return matrix
            mu_test_s (pd.Series): the expected daily return vector for testing dates
            sigma_test_df (pd.DataFrame): The covariance return matrix for testing dates
        """
        # Prepare paths and file names for new data
        self._get_next_file_name()
        # Save training data
        mu_s.to_csv(self.mu_path())
        sigma_df.to_csv(self.sigma_path())
        # Save testing data
        mu_test_s.to_csv(self.mu_path(testing=True))
        sigma_test_df.to_csv(self.sigma_path(testing=True))
        # Update the data numbers list to reflect new data
        self._get_all_data_numbers()

    def get_data(self, data_number: int, load_testing: bool = False):
        """
        Loads data in the appropriate format
        Args:
            data_number (int): The number for the data that should be loaded
            load_testing (bool): Loads testing data if True, training data loaded o/w
        Returns:
            mu_s (pd.Series): The returns vector
            sigma_df (pd.DataFrame): The covariance matrix
        """

        # Get the paths and file names/number for the data
        self.data_number = data_number

        # Load data
        if load_testing:
            mu_s = pd.read_csv(self.mu_path(testing=True), index_col=0, squeeze=True)
            sigma_df = pd.read_csv(self.sigma_path(testing=True), index_col=0)
        else:
            mu_s = pd.read_csv(self.mu_path(), index_col=0, squeeze=True)
            sigma_df = pd.read_csv(self.sigma_path(), index_col=0)

        return mu_s, sigma_df

    def mu_path(self, testing: bool = False):
        """
        Sets the path for the expected returns vector based on the current data number
        Args:
            testing (bool): Generates the testing name when True, training generated o/w
        """
        if testing:
            return f'{self.parent_directory}/{self.mu_names}_test_{self.data_number}'
        else:
            return f'{self.parent_directory}/{self.mu_names}_{self.data_number}'

    def sigma_path(self, testing: bool = False):
        """Sets the path for the returns covariance based on the current data number
        Args
            testing (bool): Generates the testing name when True, training generated o/w
        """
        if testing:
            return f'{self.parent_directory}/{self.sigma_names}_test_{self.data_number}'
        else:
            return f'{self.parent_directory}/{self.sigma_names}_{self.data_number}'

import datetime as dt  # Package for making dates
import os
import random
import time
from itertools import product
from typing import Tuple

import numpy as np
import pandas as pd
import yahoo_fin.stock_info as si
import yfinance as yf
from pandas_datareader import data as pdr
from tqdm import tqdm

from data_loader import DataLoader
from ticker_selection_opt import TickerPoolHeuristic


def generate_data_for_grid_search(gs_start_date: tuple, gs_end_date: tuple,
                                  grid_search_directory='grid-search-data', force_new_data=False) -> None:
    """
    Generates a new set of data over which a grid search can be completed
    Args:
        gs_start_date: first date data is sampled from
        gs_end_date: last possible date data is sampled from
        grid_search_directory: directory where grid search data is stored
        force_new_data: will force funciton to generate new grid search data regardless of whether or not data already
         exists
    """
    if not os.path.exists(grid_search_directory) or force_new_data:
        gs_data_loader = DataLoader(data_directory='grid-search-data')

        gs_data_loader = DataGenerator(start_date=gs_start_date,  # (YYYY, M, D)
                                       end_date=gs_end_date,  # (YYYY, M, D)
                                       data_loader=gs_data_loader)  # The data loader object that manages data

        # Use grid search option to look for new parameters that suit your problem
        gs_data_loader.choose_params(new_grid_search=True, number_of_data_samples=3)
    else:  # data generation for grid search has been done
        print('Old data will be used for grid search, please make a new grid search directory if you would like to do'
              'a grid search with new data.')
        pass


class DataGenerator:
    """
    Generates new data instances by scraping yahoo finance
    """

    def __init__(self,
                 start_date: Tuple[int, int, int],  # (YYYY, M, D)
                 end_date: Tuple[int, int, int],  # (YYYY, M, D)
                 data_loader: DataLoader = None,
                 number_of_days: int = 28,
                 save: bool = True,
                 number_of_tickers: int = 100,
                 fraction_of_sampled_tickers: float = 0.2,
                 stock_pool_builder_opt: bool = False,
                 normalize_on_s_and_p_500: bool = False
                 ):
        """
        Initialize the data generator object
        Args:
            start_date (Tuple[int, int, int]): The (year, month, day) for the start of the date range
            end_date (Tuple[int, int, int]): The (year, month, day) for the end of the date range
            data_loader (DataLoader): The data loader object that manages the data
            number_of_days (int): The number of days that should be included (including non-trading days, which
                will be skipped)
            save (bool): Whether or not generated data is saved
            number_of_tickers(int): the number of stocks selected from the S&P 500
            fraction_of_sampled_tickers (float): Fraction of tickers that will be used to construct optimization data
            stock_pool_builder_opt (bool): If true, optimization will be used to choose a diverse set of stocks from a
            pool of stocks, o/w all stocks in the pool will be used.
            normalize_on_s_and_p_500 (bool): If true, stock returns will be normalized against the daily return of the
            S&P500, o/w the the nominal stock returns will be used
       """

        # Input attributes
        self.number_of_tickers = number_of_tickers
        self.start_date = dt.datetime(*start_date)
        self.end_date = dt.datetime(*end_date)
        self.number_of_days = number_of_days
        self.save = save
        self.data_loader = data_loader
        self.fraction_of_sampled_tickers = fraction_of_sampled_tickers
        self.stock_pool_builder_opt = stock_pool_builder_opt
        self.normalize_on_s_and_p_500 = normalize_on_s_and_p_500

        # Override pandas data reader to work with yahoo finance package
        yf.pdr_override()

        # Set attributes
        self.sample_data_summary = f'sample_data_summary.csv'  # Path to data summary statistics
        self.s_and_p_symbol = '^GSPC'  # Ticker symbol for S and P 500 index
        self.s_and_p_raw_data = None
        self._reset_attributes()

    def choose_params(self, new_grid_search: bool = False, number_of_data_samples=3,
                      use_grid_search_params: bool = True):
        """
        Chooses the parameters based on a grid search or program defaults.

        Args:
            new_grid_search: Performs a new grid search, which likely takes more than 24 hours
            number_of_data_samples: the number of data samples that are generated for each combination of parameters
            use_grid_search_params: Uses parameters from a grid search if available and True, o/w will use default
            parameters.

        """
        if new_grid_search:
            self.grid_search(number_of_data_samples)

        if use_grid_search_params and self.sample_data_summary in os.listdir():
            sampled_data = pd.read_csv(self.sample_data_summary, index_col=0)
            summary_stats = sampled_data.groupby(['number_of_tickers',
                                                  'fraction_of_sampled_tickers',
                                                  'number_of_days',
                                                  'normalize_on_s_and_p_500'
                                                  ]).mean()
            summary_stats.drop('Sampled stocks', axis=1, inplace=True)
            # Define a rule for choosing the best set of stock, choice to maximize covariance (below) is an example
            maximum_average_covariance_keys = summary_stats.index.names
            data_params = pd.Series(summary_stats.Mean.idxmax(), maximum_average_covariance_keys)
            for param_name, value in data_params.items():
                if type(value) is np.int64:  # corrects for purpose of dt.timedelta
                    value = int(value)
                self.__setattr__(param_name, value)

    def grid_search(self, number_of_data_samples: int = 5):
        """
        Sweep each combinations of parameters. The recommended grid search is hard coded, but any combination of method
        attributes can be changed with this method.

        Args:
            number_of_data_samples: the number of data samples that are generated for each combination of parameters
        """
        # Define the parameters to generate data for
        all_parameters = {'number_of_tickers': [50, 100],  # number of stocks that are selected
                          'number_of_days': [7, 14],  # Number of days that data is generated over
                          'fraction_of_sampled_tickers': [0.1, 0.2],
                          'normalize_on_s_and_p_500': [True, False]}
        parameter_sets = [dict(zip(all_parameters, v))
                          for v in product(*all_parameters.values())]

        # Loop through each combination of parameters
        for param_set in tqdm(parameter_sets):
            for param in param_set:
                self.__setattr__(param, param_set[param])
            # Generate a dataset based on param_set
            self.generate_new_data(number_of_new_data=number_of_data_samples)

    def _reset_attributes(self):
        """
        Resets the attributes related to tracking changes for newly generated data
        """

        # Attributes used to track changes when generating new data
        self.tickers = None  # Collection of all possible tickers
        self.inverse_tickers = None  # Collection of inverse ETF tickers
        self.raw_ticker_data = pd.DataFrame()  # Data frame that contains stock prices
        self.chosen_tickers = None  # Tickers that are selected for constructing optimization data
        self.s_and_p_raw_data = pd.DataFrame()  # Initialize data frame to track S&P 500
        self.s_and_p_returns = pd.DataFrame()  # Initialize data frame to track S&P 500 daily returns

    def generate_new_data(self, number_of_new_data: int) -> (pd.Series, pd.DataFrame):
        """
        Generates new data points for portfolio optimization

        Args:
            number_of_new_data (int): The number of new data generated

        Returns:
            mu_s (pd.Series): The average daily return of each ticker over the date range
            sigma_df (pd.DataFrame): The covariance in daily return between all tickers over the date range
            mu_test_s (pd.Series): The average testing daily return of each ticker over the date range
            sigma_test_df (pd.DataFrame): The testing covariance in daily return between all tickers over the date range

        """

        for date_num in range(number_of_new_data):  # Create a new data sample on each loop
            data_retrieved = False
            while not data_retrieved:
                try:
                    self._generate_data_point()
                    data_retrieved = True
                except RuntimeError:
                    print('Ran into a problem. Will try to retrieve a data point again in 30 seconds.')
                    time.sleep(30)
        return

    def _generate_data_point(self) -> None or object:
        self._reset_attributes()  # Reset attributes on each loop
        if self.number_of_days is None:  # Gets data for entire date range
            start = self.start_date
            end = self.end_date
        else:  # Select random consecutive days contained in the date range
            last_possible_day_after_start = (self.end_date - self.start_date).days - self.number_of_days
            start_date_timedelta = random.randint(0, last_possible_day_after_start)
            start = self.start_date + dt.timedelta(start_date_timedelta)
            end = start + dt.timedelta(self.number_of_days)
        # Get the date range for testing data (dates immediately after training data)
        start_test = end + dt.timedelta(1)
        end_test = start_test + dt.timedelta(self.number_of_days)
        # Get clean stock data for training and testing
        self.get_n_stock_prices(start, end_test)
        self._choose_uncorrelated_tickers(start, end)  # Choose tickers based on training covariance
        # Generate training and testing data for portfolio optimization
        mu_s, sigma_df = self.get_portfolio_optimization_data(start, end)
        mu_test_s, sigma_test_df = self.get_portfolio_optimization_data(start_test, end_test)

        # Save data to enable reproducible results
        if self.save:
            self.data_loader.save_data(mu_s, sigma_df, mu_test_s, sigma_test_df)
            return
        else:
            return mu_s, sigma_df, mu_test_s, sigma_test_df

    def calculate_daily_gain(self, start_date: dt.datetime, end_date: dt.datetime) -> pd.DataFrame:
        """
        Calculate the daily gain, if specified by attribute, then the gains will be relative to the S&P500
        Args:
            start_date: First date that gains are calculated on
            end_date: Last date that gains are calculated on

        Returns:
            stock_gans (pd.DataFrame): A matrix of relative stock changes
        """
        stocks_gain = self.calculate_nominal_daily_gain(self.raw_ticker_data, start_date, end_date)
        if self.normalize_on_s_and_p_500:
            s_and_p_gains = self.calculate_nominal_daily_gain(self.s_and_p_raw_data, start_date, end_date)
            stocks_gain = (stocks_gain + 1) / (s_and_p_gains.values + 1) - 1

        return stocks_gain

    @staticmethod
    def calculate_nominal_daily_gain(df: pd.DataFrame, start_date: dt.datetime, end_date: dt.datetime) -> pd.DataFrame:
        """
        Calculate the daily gain based on the adjusted close price, which corrects for instances where the stock price
        changed without changing the value of the company (e.g., stock splits)
        Args:
            df: Dataframe that contains stock price data downloaded from yahoo finance
            start_date: Date at start of range for average gain calculation (inclusive)
            end_date: Date and end of range for average gain calculation (inclusive)

        Returns:
            daily_gains (pd.DataFrame): A matrix of relative stock changes
        """
        df = df.loc[start_date:end_date]  # Sampled raw_ticker_data
        prior_day_close = df['Adj Close'].iloc[0:-1].reset_index(drop=True)  # Closing price day earlier
        day_close = df['Adj Close'].iloc[1:].reset_index(drop=True)  # Closing price on day
        daily_gains = (day_close - prior_day_close) / prior_day_close
        return daily_gains

    def get_portfolio_optimization_data(self,
                                        start_date: dt.datetime,  # (YYYY, MM, DD)
                                        end_date: dt.datetime,  # (YYYY, MM, DD)
                                        ) -> (pd.Series, pd.DataFrame):
        """
        Generates the average return and covariance matrix for tickers in a given range of dates.
        Args:
            start_date (dt.datetime): The (year, month, day) for the start of the date range
            end_date (dt.datetime): The (year, month, day) for the end of the date range

        Returns:
            average_return (pd.Series): The average daily return of each ticker over the date range
            covariance (pd.DataFrame): The covariance in daily return between all tickers over the date range
        """
        daily_gains = self.calculate_daily_gain(start_date, end_date)
        chosen_daily_gains = daily_gains[self.chosen_tickers]
        # Get close prices
        average_return = chosen_daily_gains.mean()
        covariance = chosen_daily_gains.cov()
        # Prepare data to print summary statistics
        qs = [0.25, 0.5, 0.75]  # Reported quantiles
        ar_qs = average_return.quantile(qs)  # Calculate average return quantiles
        c_qs = covariance.unstack().quantile(qs)  # Calculate covariance return quantiles
        for q in qs:  # Loop through each quantile to print
            print(f'Return {q * 100:.0f} percentile = {ar_qs[q]:.5f} \t\t'
                  f'Covariance {q * 100:.0f} percentile = {c_qs[q]:.5f}')
        return average_return, covariance

    def _get_all_tickers(self):
        """
        Retrieves the list of tickers from online sources, or if the tickers saved in the 'all_tickers.csv'. Note,
        that firewalls may prevent the scraping package from working properly. This function should not take more than
        30 seconds to get 10,000 + tickers.
        """
        all_tickers_csv = 'all_tickers.csv'  # Hard coded path for all_tickers that are used
        if all_tickers_csv in os.listdir():  # If all_tickers exists, then use it
            all_tickers = pd.read_csv(all_tickers_csv, index_col=0, squeeze=True).dropna()
        else:  # O/w, new tickers are scraped from several sources
            all_tickers = pd.Series(list({*si.tickers_dow(),
                                          *si.tickers_nasdaq(),
                                          *si.tickers_other(),
                                          *si.tickers_sp500()}))
            all_tickers.to_csv(all_tickers_csv)  # Save the scraped tickers for future use
        self.tickers = all_tickers.values.tolist()  # A list of tickers that will sampled from

    def get_n_stock_prices(self, start_date: dt.datetime, end_date: dt.datetime) -> None:
        """
        Iteratively retrieves stock prices until the requested number of stocks are retrieved. Note, that stocks
        will be dropped if no price data is available for the specified date range.
        Args:
            start_date (dt.datetime): The first date ticker data is downloaded for
            end_date(dt.datetime): The last data that ticker data is downloaded for
        """
        self._get_all_tickers()
        # self.get_inverse_etf_tickers()  # TODO: force inverse tickers
        random.shuffle(self.tickers)  # Randomize the order of tickers
        stocks_acquired = self._get_number_of_tickers_acquired()

        # Add new data until the number of stocks retrieved matches the number requested
        while stocks_acquired < self.number_of_tickers:
            tickers_needed = self.number_of_tickers - stocks_acquired  # The number of tickers still needed
            self._get_and_clean_stock_price_data(tickers_needed, start_date, end_date)  # Get price data
            stocks_acquired = self._get_number_of_tickers_acquired()  # Check how many stocks have price data

        if self.normalize_on_s_and_p_500:
            self.s_and_p_raw_data = pdr.get_data_yahoo(self.s_and_p_symbol, start_date, end_date)
            self.s_and_p_raw_data.columns = pd.MultiIndex.from_product([self.s_and_p_raw_data.columns,
                                                                        [self.s_and_p_symbol]])

    def _get_and_clean_stock_price_data(self, n_tickers: int, start_date: dt.datetime, end_date: dt.datetime) -> None:
        """
        Download stock price data from yahoo finance and clean the data by removing missing data (e.g., stocks that
        weren't listed in data range)
        Args:
            n_tickers: number of tickers for which stock prices will be downloaded
            start_date: the start of the data range
            end_date: the end of the data range
        """
        # Download new data from y-finance
        df_new_ticker_info = pdr.get_data_yahoo(self.tickers[0:n_tickers], start_date, end_date)
        self.tickers = self.tickers[n_tickers:-1]  # Update tickers to remove the ones that were just downloaded
        # Ensure downloaded data is in a consistent MultiIndex format (needed when n_tickers=1)
        if type(df_new_ticker_info.columns) is pd.Index:
            df_new_ticker_info.columns = pd.MultiIndex.from_product([df_new_ticker_info.columns, [self.tickers[0]]])

        # Add downloaded data to existing database of prices
        if self.raw_ticker_data.shape[0] == 0:
            self.raw_ticker_data = df_new_ticker_info
        else:
            self.raw_ticker_data = self.raw_ticker_data.join(df_new_ticker_info)
        # Drop missing data to clean the dataset
        self.raw_ticker_data.dropna(axis=0, how='all', inplace=True)  # fixes bug that occasionally creates row of nans
        self.raw_ticker_data.dropna(axis=1, inplace=True)  # deletes all tickers with a missing data point

    def _get_number_of_tickers_acquired(self) -> int:
        """
        Gets the number of stocks that price data has been downloaded for
        Returns:
            number_of_stocks_acquired (int): the number of stocks we have data for
        """
        if 'Adj Close' not in self.raw_ticker_data.columns:
            number_of_stocks_acquired = 0
        elif type(self.raw_ticker_data['Adj Close']) is pd.Series:
            number_of_stocks_acquired = 1
        else:
            number_of_stocks_acquired = self.raw_ticker_data['Adj Close'].shape[1]
        return number_of_stocks_acquired

    def _choose_uncorrelated_tickers(self, start_date: dt.datetime, end_date: dt.datetime):
        """
        Choose a set of tickers that are sufficiently uncorrelated. Stocks prices are assessed for all consecutive days
        between start_date and end_date. The idea is that we would like to maximize the diversity of stocks that were
        picking (e.g., balance in sectors, balance in market caps).
        Args:
            start_date: The first data that stock data is calculated over
            end_date: The last date that the stock data is calculated over
        """
        # Get daily gains, average returns covariance
        daily_gains = self.calculate_daily_gain(start_date, end_date)
        sigma_df = daily_gains.cov()
        # Get number of stocks to be sampled
        sample_number = int(round(self.number_of_tickers * self.fraction_of_sampled_tickers))

        if self.stock_pool_builder_opt:
            ticker_pool_builder = TickerPoolHeuristic(sigma_df, n=sample_number)
            self.chosen_tickers = ticker_pool_builder.solve()  # Choose set of tickers
        else:
            self.chosen_tickers = daily_gains.columns.to_list()
        self._save_sigma_df_statistics(sigma_df, start_date)

    def _save_sigma_df_statistics(self, sigma_df: pd.DataFrame, start_date: dt.datetime) -> None:
        """
        Save a set of summary statistics to monitor the high level attributes of the data that was generated
        Args:
            sigma_df: The covariance matrix being evaluated
            start_date: The first/earliest date that is represented in the matrix
        """
        # Calculate statistics of interest
        min_value = sigma_df.loc[self.chosen_tickers, self.chosen_tickers].values.min()
        max_value = sigma_df.loc[self.chosen_tickers, self.chosen_tickers].values.max()
        mean_value = sigma_df.loc[self.chosen_tickers, self.chosen_tickers].values.mean()
        std_value = sigma_df.loc[self.chosen_tickers, self.chosen_tickers].values.std()
        quantile_values = sigma_df.unstack().quantile([0.25, 0.5, 0.75])  # Calculate covariance return quantiles

        # Define normalization factor for statistics such that the std is 1
        normalize_factor = 1 / std_value

        # Prepare data to save to csv
        if self.sample_data_summary in os.listdir():  # Load csv if it already exists
            sample_data = pd.read_csv(self.sample_data_summary, index_col=0)
        else:  # Make a new dataframe that will be saved for future reference
            sample_data = pd.DataFrame(
                columns=['Min', 'Max', 'Mean', 'STD', '1st quartile', 'median', '3rd quartile',
                         'number_of_tickers', 'Sampled stocks', 'fraction_of_sampled_tickers', 'date start',
                         'number_of_days', 'normalize_on_s_and_p_500'])
        # Prepare data to append to statistics from previous instances
        data_to_append = pd.Series([min_value * normalize_factor,
                                    max_value * normalize_factor,
                                    mean_value * normalize_factor,
                                    std_value * normalize_factor,
                                    quantile_values[0.25] * normalize_factor,
                                    quantile_values[0.5] * normalize_factor,
                                    quantile_values[0.75] * normalize_factor,
                                    self.number_of_tickers,
                                    len(self.chosen_tickers),
                                    self.fraction_of_sampled_tickers,
                                    start_date,
                                    self.number_of_days,
                                    self.normalize_on_s_and_p_500], sample_data.columns)
        sample_data = sample_data.append(data_to_append, ignore_index=True)  # Append data to previous instances
        sample_data.to_csv(self.sample_data_summary)  # Save data

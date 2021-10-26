from typing import List

from tqdm import tqdm


class TickerPoolHeuristic:

    """
    Greedy heuristic that chooses a subset of stocks that maximize c
    """

    def __init__(self, sigma_df, n: int):

        # Initialize solver and necessary components
        self.returns_cov = sigma_df
        self.n = n

    def solve(self) -> List[str]:
        # Initialize heuristic
        chosen_tickers = [self.returns_cov.std().idxmax()]  # initialize the list of selected tickers
        remaining_tickers = self.returns_cov.columns.to_list()
        # Greedy loop that adds 1 stock on each iterations to the chosen_tickers list
        for i in tqdm(range(self.n - 1)):
            remaining_tickers.remove(chosen_tickers[-1])  # Remove last chosen ticker
            candidate_cov_std = -1  # Reset covariance standard deviation
            candidate_tickers = None  # Initialize the the best candidate ticker
            for j in remaining_tickers:  # Loop through all possible choices
                j_candidate_tickers = [*chosen_tickers, j]  #
                j_candidate_cov_std = self.returns_cov.loc[j_candidate_tickers, j_candidate_tickers].values.var()
                if candidate_cov_std < j_candidate_cov_std:  # Choose ticker if it has higher std than current candidate
                    candidate_cov_std = j_candidate_cov_std
                    candidate_tickers = j_candidate_tickers
            chosen_tickers = candidate_tickers

        return chosen_tickers

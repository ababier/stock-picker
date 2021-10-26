from itertools import product as it_product

import cvxpy as cp
import numpy as np
import pandas as pd


class PortfolioOptimizer:
    """Constructs and solves a portfolio optimization model given a set of expected returns and return covariances"""

    def __init__(self, mu_s: pd.Series, sigma_df: pd.DataFrame, n: int = 100):
        """
        Initialize object attributes
        Args:
            mu_s: The vector of expected returns that will be used for optimization.
            sigma_df: The covariance matrix of returns that will be used for optimization.
            n: The number of gammas that will be used in n runs of the optimization model
        """

        # The primary coefficients in the portfolio optimization model
        self.mu = mu_s
        self.sigma = sigma_df

        # Initialize solver and necessary components
        self._set_variables()
        self._set_constraints()
        self._set_objective()
        self.prob = cp.Problem(self.objective, self.cons)

        # Generate set of parameters
        self._generate_list_of_gammas(n)

    def _set_variables(self):
        """
        Initialize all of the variables used in the model.
        """
        # variable and parameter definitions
        self.x = cp.Variable(self.mu.size)
        self.gamma = cp.Parameter(nonneg=True)

    def _set_constraints(self):
        """
        Declare all the constraints in the model.
        """
        # Initialize list of constraints
        self.cons = []
        # Make the sum of all investments equal to 1
        self.cons.append(cp.sum(self.x) == 1)
        # Ensure all investments are positive
        self.cons.append(self.x >= 0)

    def _set_objective(self):
        """
        Define the objective for this optimization model
        """
        # mu' * x
        self.return_objective = self.mu.values @ self.x  # The expected return
        # x' * Sigma * x
        self.risk_objective = cp.quad_form(self.x, self.sigma)  # The risk in return
        # Add objective functions together to form objective
        self.objective = cp.Maximize(self.return_objective - self.gamma * self.risk_objective)  # Gamma is a parameter

    def solve_model(self, gamma: int = 1):
        """
        Solve the model for a given gamma value.
        Args:
            gamma: The number the quantifies an investors risk level (lower gamma leads to a more risky portfolio)
        """
        # Set value of gamma
        self.gamma.value = gamma
        # Solve the model
        self.prob.solve()

    def calculate_return_and_risk(self, mu: pd.Series = None, sigma: pd.DataFrame = None):
        """
        Calculates the return based on the input mu and the risk based on the input sigma
        Args:
            mu (pd.Series): The return vector
            sigma (pd.DataFrame): The risk matrix
        """
        # Set mu and sigma the inputs, or use the optimized coefficients if none were given
        if mu is None:  # The expected mu that was used for optimized
            mu = self.mu
            mu_row = (self.gamma.value, 'Expected')
        else:  # A new mu that is used to validated the decisions made by the optimization model
            mu_row = (self.gamma.value, 'Realized')
        if sigma is None:  # The sigma that was used for optimized
            sigma = self.sigma
            sigma_row = (self.gamma.value, 'Expected')
        else:  # A new sigma that is used to validated the decisions made by the optimization model
            sigma_row = (self.gamma.value, 'Realized')

        # Calculate the return and risk based on decisions
        portfolio_return = (mu * self.x.value).sum()
        portfolio_risk = np.sqrt(self.x.value.reshape(1, -1).dot(sigma).dot(
            self.x.value.reshape(-1, 1)))[0, 0]  # square root of the covariance terms

        # Save information in summary DataFrame
        self.summary.loc[mu_row, 'Average return'] = portfolio_return
        self.summary.loc[sigma_row, 'Risk'] = portfolio_risk

    def _generate_list_of_gammas(self, n: int = 100):
        """
        Generate a list of gamma of length n
        Args:
            n (int): The number of gammas that will be generated
        """
        # Generate the set of gammas
        self.gamma_values = np.logspace(-2, 3, num=n)
        # Made a set of indices for a DataFrame that will summarize the results
        index_values = list(it_product(self.gamma_values, ['Expected', 'Realized']))
        multi_index = pd.MultiIndex.from_tuples(index_values, names=["Gamma", "data_class"])
        # Initialize the empty summary DataFrame
        self.summary = pd.DataFrame(index=multi_index, columns=['Average return', 'Risk'])

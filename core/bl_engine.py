import numpy as np
from scipy.optimize import minimize


class BlackLittermanEngine:
    """
    Black-Litterman Engine for portfolio optimization.

    The optimizer uses the canonical BL approach: find weights that maximize
    the posterior utility (mu'w - delta/2 * w'Sigma*w) rather than raw Sharpe.
    This ensures results are perturbations of market weights, not extreme concentrations.
    """

    def __init__(self, assets: list, market_weights: np.array,
                 cov_matrix: np.array, risk_aversion: float = 2.5, tau: float = 0.025):
        """
        assets         : list of asset names e.g. ['SPY', 'EEM', 'TLT']
        market_weights : market capitalization weights
        cov_matrix     : covariance matrix of returns
        risk_aversion  : risk aversion coefficient (delta)
        tau            : uncertainty in the prior (typically 0.025)
        """
        self.assets = assets
        self.weights = market_weights
        self.cov = cov_matrix
        self.delta = risk_aversion
        self.tau = tau
        self.P = None
        self.Q = None
        self.omega = None

    def compute_prior(self) -> np.array:
        """Computes implied equilibrium returns (Pi)."""
        pi = self.delta * self.cov @ self.weights
        return pi

    def add_views(self, P: np.array, Q: np.array, omega: np.array = None):
        """
        Adds manager views to the model.

        P     : pick matrix (num_views x num_assets)
        Q     : expected returns vector (num_views,)
        omega : uncertainty matrix — if None, computed from tau and cov
        """
        self.P = P
        self.Q = Q

        if omega is None:
            self.omega = np.diag(np.diag(self.tau * P @ self.cov @ P.T))
        else:
            self.omega = omega

    def compute_posterior(self) -> np.array:
        """
        Computes posterior expected returns (mu) combining
        equilibrium prior with manager views.

        Formula: mu = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1
                      [(tau*Sigma)^-1*Pi + P'*Omega^-1*Q]
        """
        pi = self.compute_prior()

        tau_cov = self.tau * self.cov
        tau_cov_inv = np.linalg.inv(tau_cov)
        omega_inv = np.linalg.inv(self.omega)

        M1 = np.linalg.inv(tau_cov_inv + self.P.T @ omega_inv @ self.P)
        M2 = tau_cov_inv @ pi + self.P.T @ omega_inv @ self.Q

        mu_posterior = M1 @ M2
        return mu_posterior

    def optimize(self) -> dict:
        """
        Computes optimal portfolio weights using canonical BL utility maximization:
            max  mu'w - (delta/2) * w'Sigma*w
            s.t. sum(w) = 1, 0 <= w_i <= 0.75 (max concentration per asset)

        This approach ensures the result is a PERTURBATION of market weights,
        not an extreme concentration driven by relative return differences.
        The 0.75 upper bound prevents any single asset from dominating.
        """
        mu = self.compute_posterior()
        n = len(self.assets)

        def neg_utility(weights):
            # Canonical BL utility: mu'w - (delta/2)*w'Sigma*w
            port_return = np.dot(weights, mu)
            risk_penalty = (self.delta / 2) * (weights @ self.cov @ weights)
            return -(port_return - risk_penalty)  # negative because we minimize

        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

        # Bounds: max turnover of 20pp per asset from market weights
        # e.g. Stocks_USA at 60% can range from 40% to 75% (capped at 75%)
        # e.g. Bonds_USA at 10% can range from 0% to 30%
        max_turnover = 0.20
        bounds = [
            (max(0.0, self.weights[i] - max_turnover),
             min(0.75, self.weights[i] + max_turnover))
            for i in range(n)
        ]

        # Initial guess: market weights (canonical BL starting point)
        w0 = self.weights.copy()

        result = minimize(neg_utility, w0, method='SLSQP',
                          bounds=bounds, constraints=constraints)

        optimal_weights = result.x
        opt_return = np.dot(optimal_weights, mu)
        opt_vol = np.sqrt(optimal_weights @ self.cov @ optimal_weights)
        opt_sharpe = opt_return / opt_vol if opt_vol > 0 else 0.0

        return {
            'weights': dict(zip(self.assets, optimal_weights)),
            'expected_return': opt_return,
            'volatility': opt_vol,
            'sharpe_ratio': opt_sharpe
        }


# --- TEST ---
if __name__ == "__main__":
    assets = ['Stocks_USA', 'Stocks_EM', 'Bonds_USA']
    weights = np.array([0.60, 0.30, 0.10])
    cov = np.diag([0.0225, 0.0324, 0.0025])

    bl = BlackLittermanEngine(assets, weights, cov)

    pi = bl.compute_prior()
    print("Equilibrium returns (Prior):")
    for a, r in zip(assets, pi):
        print(f"  {a}: {r*100:.2f}%")

    # Simulate rate hike views: negative on all, bonds worst
    P = np.array([
        [1, 0,  0],   # View 1: absolute on Stocks_USA
        [0, 1,  0],   # View 2: absolute on Stocks_EM
        [0, 0,  1],   # View 3: absolute on Bonds_USA
    ])
    Q = np.array([-0.04, -0.06, -0.03])  # All negative, bonds slightly less negative

    bl.add_views(P, Q)
    mu = bl.compute_posterior()

    print("\nPosterior returns (after rate hike views):")
    for a, r in zip(assets, mu):
        print(f"  {a}: {r*100:.2f}%")

    result = bl.optimize()

    print("\nOptimal Portfolio Weights:")
    for asset, weight in result['weights'].items():
        print(f"  {asset}: {weight*100:.2f}%")

    print(f"\nPortfolio Metrics:")
    print(f"  Expected Return : {result['expected_return']*100:.2f}%")
    print(f"  Volatility      : {result['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio    : {result['sharpe_ratio']:.4f}")

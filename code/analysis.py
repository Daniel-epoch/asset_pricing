import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import akshare as ak
import scipy.stats as stats
from scipy.linalg import nearest_pos_def

warnings.filterwarnings('ignore')

class MonteCarlo:
    """
    universal multi-dimensional geometric brownian motion simulation
    """
    def __init__(self, n_assets, n_simulations=10000):
        """
        Args:
            n_assets: asset quantity
            n_simulations: simulation times
        """
        self.n_assets = n_assets
        self.n_simulations = n_simulations
    
    def simulate_gbm(self, S0, mu, cov_matrix, T, dt=1/252, random_seed=42):
        """
        Geometric Brownian Motion Simulation
        Args:
            S0: initial price
            mu: annualized return ratio
            cov_matrix: annualized co-variance matrix
            T: holding period(unit: day)
            dt: time step
            random_seed: random seed

        return:
            price_paths:
        """
        np.random.seed(random_seed)

        # Ensure the covariance matrix is positive definite
        try:
            L = np.linalg.cholesky(cov_matrix) 
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(nearest_pos_def(cov_matrix))
        
        n_steps = int(T)

        Z = np.random.normal(0, 1, (n_steps, self.n_simulations, self.n_assets))
        Z_correlated = Z @ L.T

        sigma = np.sqrt(np.diag(cov_matrix))
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)

        price_paths = np.zeros((n_steps + 1, self.n_simulations, self.n_assets))
        price_paths[0] = S0

        for t in range(1, n_steps + 1):
            growth = np.exp(drift + diffusion * Z_correlated[t-1])
            price_paths[t] = price_paths[t-1] * growth

        return price_paths


class PortfolioVaR:
    """
    Use MonteCarlo Simulation to calculate VaR
    """
    def __init__(self, mc_simulator):
        """
        Args:
            mc_simulator: MonteCarloSimulator Example
        """
        self.mc = mc_simulator
    
    def calculate_var(self, weights, initial_prices, mu, cov_matrix, T, alpha=0.05):
        """
        Calculate VaR and Expected Shortfall of portfolio

        Returns:
            var: value at risk
            es: expected shortfall 
            returns: portfolio returns
        """
        weights = np.array(weights)
        S0 = np.array(initial_prices)  # 修正：np.arrary -> np.array

        # Simulate price path
        price_paths = self.mc.simulate_gbm(S0, mu, cov_matrix, T)

        portfolio_values = np.sum(price_paths * weights, axis=2)

        # calculate returns and losses
        initial_value = np.sum(S0 * weights)
        final_values = portfolio_values[-1]
        returns = (final_values - initial_value) / initial_value
        losses = -returns  # 修正：loss -> losses

        var = np.percentile(losses, 100 * (1 - alpha))
        es = losses[losses >= var].mean()  # 修正：使用正确的变量名
        
        return var, es, returns  # 添加返回值


class RiskAnalysis:
    def __init__(self, data, tau_annual, alpha):
        """
        Risk Analysis
        :param data: dataset with 'ret' column
        :param tau_annual: 预期年化收益率
        :param alpha: significance level
        """
        self.data = data
        self.tau_annual = tau_annual
        self.tau_daily = (1 + tau_annual)**(1/252) - 1  # 转换为日度目标收益
        self.alpha = alpha
        
        # 确保数据有收益率列
        if 'ret' not in data.columns:
            raise ValueError("Data must contain 'ret' column")
        
        self.returns = data['ret'].dropna().values
    
    def semi_variance(self):
        """计算半方差"""
        downside_returns = self.returns[self.returns < self.tau_daily]
        if len(downside_returns) == 0:
            return 0
        return np.mean((downside_returns - self.tau_daily) ** 2)
    
    def shortfall_probability(self):
        """计算短缺概率"""
        return np.mean(self.returns < self.tau_daily)
    
    def var_historical(self, alpha=None):
        """历史模拟法计算VaR"""
        if alpha is None:
            alpha = self.alpha
        return -np.percentile(self.returns, 100 * (1 - alpha))
    
    def var_parametric(self, alpha=None):
        """参数法计算VaR（正态假设）"""
        if alpha is None:
            alpha = self.alpha
        mu = np.mean(self.returns)
        sigma = np.std(self.returns)
        z = stats.norm.ppf(1 - alpha)
        return -(mu + z * sigma)
    
    def expected_shortfall(self, alpha=None):
        """计算期望短缺（ES/CVaR）"""
        if alpha is None:
            alpha = self.alpha
        var = self.var_historical(alpha)
        losses = self.returns[self.returns <= -var]
        es = -np.mean(losses) if len(losses) > 0 else 0
        return es
    
    def report(self):
        """生成风险分析报告"""
        print("=" * 60)
        print("Downside Risk Index Analysis Report")
        print("=" * 60)
        print(f"数据点数: {len(self.returns)}")
        print(f"目标年化收益率 (τ): {self.tau_annual:.4%}")
        print(f"目标日度收益率: {self.tau_daily:.6%}")
        print(f"平均收益率: {np.mean(self.returns):.6%}")
        print(f"标准差: {np.std(self.returns):.6%}")
        
        print("\n" + "-" * 60)
        print("Downside Risk Indicators")
        print("-" * 60)
        print(f"Semi-Variance: {self.semi_variance():.8f}")
        print(f"Downside Deviation: {np.sqrt(self.semi_variance()):.6%}")
        print(f"Shortfall Probability: {self.shortfall_probability():.2%}")
        
        print("\n" + "-" * 60)
        print("Value at Risk (VaR) Analysis")
        print("-" * 60)
        
        for conf_level in [0.95, 0.99]:
            alpha_val = 1 - conf_level
            var_hist = self.var_historical(alpha_val)
            var_param = self.var_parametric(alpha_val)
            es = self.expected_shortfall(alpha_val)
            
            print(f"\n置信水平 {conf_level:.0%}:")
            print(f"  历史VaR: {var_hist:.4%}")
            print(f"  参数VaR: {var_param:.4%}")
            print(f"  期望短缺(ES): {es:.4%}")
            if var_hist != 0:
                print(f"  ES/VaR比率: {es/var_hist:.3f}")
        print("=" * 60)


class DataProcess:
    def __init__(self, path):
        self.path = path
    
    def _get_dataset(self):
        """读取数据集"""
        if not os.path.exists(self.path):
            print("Wrong Path")
            return None
        
        file_ext = os.path.splitext(self.path)[1].lower()
        
        try:
            if file_ext == ".csv":
                data = pd.read_csv(self.path)
            elif file_ext == ".xlsx":
                data = pd.read_excel(self.path)
            elif file_ext == ".parquet":
                data = pd.read_parquet(self.path)
            elif file_ext == ".pkl":
                data = pd.read_pickle(self.path)
            else:
                print('Unsupported File Format')
                return None
            return data
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
    
    def _data_cleaning(self, data):
        """数据清洗"""
        # 确保数值列转换为数值类型
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # 删除包含缺失值的行
        data_cleaned = data.dropna()
        return data_cleaned
    
    def _calculate_returns(self, data, price_col='close', return_col='ret'):
        """计算收益率"""
        if price_col not in data.columns:
            raise ValueError(f"Column '{price_col}' not found in data")
        
        if return_col not in data.columns:
            data[return_col] = data[price_col].pct_change()
        
        return data


class Analysis:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.risk_analyzer = None
    
    def data_prepare(self):
        """数据准备流程"""
        # 修正：避免变量名与类名冲突
        data_processor = DataProcess(self.data_path)  # 修正：变量名小写
        raw_data = data_processor._get_dataset()
        
        if raw_data is None:
            raise ValueError("Failed to load data")
        
        cleaned_data = data_processor._data_cleaning(raw_data)
        processed_data = data_processor._calculate_returns(cleaned_data)
        
        self.data = processed_data
        return processed_data
    
    def risk_measurement(self, tau_annual=0.05, alpha=0.05):
        """风险测量"""
        if self.data is None:
            self.data_prepare()
        
        self.risk_analyzer = RiskAnalysis(self.data, tau_annual, alpha)
        self.risk_analyzer.report()
        
        return self.risk_analyzer
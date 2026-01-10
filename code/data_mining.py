#%%
import pandas as pd
import numpy as np
import akshare as ak
import warnings
from typing import List, Union, Optional, Dict
import time
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import concurrent.futures
from tqdm import tqdm

warnings.filterwarnings('ignore')
#%%
import akshare as ak
import pandas as pd
import numpy as np
import warnings
from typing import List, Union, Optional, Dict
import time
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import concurrent.futures
from tqdm import tqdm  # 进度条显示

warnings.filterwarnings('ignore')

class ASshare:
    """
    A股数据获取类
    支持批量获取股票历史数据、基本面数据、因子数据等

    Attributes:
        cache_dir: 缓存目录路径
        max_workers: 最大并发线程数
        retry_times: 失败重试次数
    """

    def __init__(self, cache_dir: str = "./data_cache", max_workers: int = 5):
        """
        初始化ASshare类

        Args:
            cache_dir: 缓存目录，默认当前目录下的data_cache
            max_workers: 并发下载的最大线程数
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.retry_times = 3
        self.request_delay = 0.1  # 请求延迟，避免被封IP

        # 数据源配置
        self.data_sources = {
            'daily': ak.stock_zh_a_hist,
            'minute': ak.stock_zh_a_minute,
            'fundamental': ak.stock_financial_report_sina,
            # 可根据需要添加更多数据源
        }

    def _get_cache_key(self, symbol: str, data_type: str, **kwargs) -> str:
        """
        生成缓存键值

        Args:
            symbol: 股票代码
            data_type: 数据类型
            **kwargs: 其他参数

        Returns:
            缓存文件名
        """
        param_str = "_".join([f"{k}-{v}" for k, v in sorted(kwargs.items())])
        return f"{symbol}_{data_type}_{param_str}.parquet"

    def _save_to_cache(self, df: pd.DataFrame, cache_key: str):
        """
        保存数据到缓存

        Args:
            df: 数据框
            cache_key: 缓存键值
        """
        if df is not None and not df.empty:
            cache_path = self.cache_dir / cache_key
            try:
                df.to_parquet(cache_path)
            except Exception as e:
                # 如果parquet保存失败，使用csv作为备选
                cache_path = cache_path.with_suffix('.csv')
                df.to_csv(cache_path)

    def _load_from_cache(self, cache_key: str, max_cache_days: int = 1) -> Optional[pd.DataFrame]:
        """
        从缓存加载数据

        Args:
            cache_key: 缓存键值
            max_cache_days: 最大缓存天数，超过则重新下载

        Returns:
            数据框或None
        """
        # 尝试不同的文件格式
        for ext in ['.parquet', '.csv']:
            cache_path = self.cache_dir / cache_key.replace('.parquet', ext)
            if cache_path.exists():
                # 检查缓存是否过期
                cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
                if (datetime.now() - cache_time).days < max_cache_days:
                    try:
                        if ext == '.parquet':
                            return pd.read_parquet(cache_path)
                        else:
                            return pd.read_csv(cache_path, index_col=0, parse_dates=True)
                    except Exception as e:
                        print(f"加载缓存失败 {cache_path}: {e}")
                        return None
        return None

    def _fetch_single_stock(self, symbol: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        获取单只股票数据（带重试机制）

        Args:
            symbol: 股票代码
            **kwargs: 查询参数

        Returns:
            股票数据框或None
        """
        data_type = kwargs.pop('data_type', 'daily')

        # 检查缓存
        cache_key = self._get_cache_key(symbol, data_type, **kwargs)
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        # 从数据源获取
        if data_type not in self.data_sources:
            print(f"不支持的数据类型: {data_type}")
            return None

        func = self.data_sources[data_type]

        # 重试机制
        for attempt in range(self.retry_times):
            try:
                # 添加股票代码列以便后续识别
                df = func(symbol=symbol, **{k: v for k, v in kwargs.items()
                                          if k not in ['data_type']})

                if df is not None and not df.empty:
                    df['symbol'] = symbol

                    # 标准化列名
                    df.columns = df.columns.str.strip().str.lower()

                    # 确保日期列为datetime类型
                    date_columns = ['日期', 'date', '时间', 'time']
                    for col in date_columns:
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col])
                            df.set_index(col, inplace=True)
                            break

                    # 保存到缓存
                    self._save_to_cache(df, cache_key)

                    # 避免请求过快
                    time.sleep(self.request_delay)

                    return df

            except Exception as e:
                print(f"获取{symbol}数据失败 (尝试 {attempt+1}/{self.retry_times}): {e}")
                time.sleep(1)  # 失败后等待1秒再重试

        print(f"无法获取{symbol}数据，已重试{self.retry_times}次")
        return None

    def get_stock_hist_data(self,
                           symbol_list: Union[str, List[str]],
                           period: str = "daily",
                           start_date: str = None,
                           end_date: str = None,
                           adjust: str = "qfq",
                           use_cache: bool = True,
                           parallel: bool = True) -> Dict[str, pd.DataFrame]:
        """
        获取股票历史数据（改进版）

        Args:
            symbol_list: 股票代码或代码列表
            period: 周期，支持'daily'（日线）等
            start_date: 开始日期，格式'YYYYMMDD'
            end_date: 结束日期，格式'YYYYMMDD'
            adjust: 复权类型，'qfq'（前复权），'hfq'（后复权），''（不复权）
            use_cache: 是否使用缓存
            parallel: 是否并行下载

        Returns:
            字典，键为股票代码，值为数据框
        """
        # 参数处理
        if isinstance(symbol_list, str):
            symbol_list = [symbol_list]

        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        if start_date is None:
            # 默认获取最近一年的数据
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')

        # 准备参数
        kwargs = {
            'data_type': 'daily',
            'period': period,
            'start_date': start_date,
            'end_date': end_date,
            'adjust': adjust
        }

        result_dict = {}

        # 并行下载
        if parallel and len(symbol_list) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交任务
                future_to_symbol = {
                    executor.submit(self._fetch_single_stock, symbol, **kwargs): symbol
                    for symbol in symbol_list
                }

                # 处理结果
                for future in tqdm(concurrent.futures.as_completed(future_to_symbol),
                                 total=len(symbol_list), desc="下载数据"):
                    symbol = future_to_symbol[future]
                    try:
                        df = future.result()
                        if df is not None:
                            result_dict[symbol] = df
                    except Exception as e:
                        print(f"处理{symbol}时出错: {e}")

        # 串行下载
        else:
            for symbol in tqdm(symbol_list, desc="下载数据"):
                df = self._fetch_single_stock(symbol, **kwargs)
                if df is not None:
                    result_dict[symbol] = df

        return result_dict

    def get_stock_hist_data_combined(self,
                                    symbol_list: Union[str, List[str]],
                                    **kwargs) -> pd.DataFrame:
        """
        获取股票历史数据并合并为单一数据框

        Args:
            symbol_list: 股票代码或代码列表
            **kwargs: 传递给get_stock_hist_data的参数

        Returns:
            合并后的数据框，包含多级索引（日期，股票代码）
        """
        data_dict = self.get_stock_hist_data(symbol_list, **kwargs)

        if not data_dict:
            return pd.DataFrame()

        # 合并数据
        combined_data = []
        for symbol, df in data_dict.items():
            if not df.empty:
                # 添加股票代码列（如果还没有）
                if 'symbol' not in df.columns:
                    df['symbol'] = symbol

                # 重置索引以便合并
                df_reset = df.reset_index()

                # 确保有日期列
                if '日期' in df_reset.columns:
                    df_reset.rename(columns={'日期': 'date'}, inplace=True)

                combined_data.append(df_reset)

        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)

            # 设置多级索引
            if 'date' in combined_df.columns and 'symbol' in combined_df.columns:
                combined_df['date'] = pd.to_datetime(combined_df['date'])
                combined_df.set_index(['date', 'symbol'], inplace=True)
                combined_df.sort_index(inplace=True)

            return combined_df
        else:
            return pd.DataFrame()

    def get_index_components(self, index_code: str = "000300") -> List[str]:
        """
        获取指数成分股

        Args:
            index_code: 指数代码
                '000300' - 沪深300
                '000905' - 中证500
                '000016' - 上证50

        Returns:
            成分股代码列表
        """
        try:
            if index_code == "000300":
                df = ak.index_stock_cons_csindex(symbol="000300")
            elif index_code == "000905":
                df = ak.index_stock_cons_csindex(symbol="000905")
            elif index_code == "000016":
                df = ak.stock_sz_50_spot()
            else:
                print(f"暂不支持的指数代码: {index_code}")
                return []

            if df is not None and not df.empty:
                # 提取股票代码，不同接口返回的列名可能不同
                for col in ['成分券代码', '品种代码', 'code']:
                    if col in df.columns:
                        return df[col].astype(str).str.zfill(6).tolist()

        except Exception as e:
            print(f"获取指数成分股失败: {e}")

        return []

    def get_basic_info(self, symbol: str) -> Dict:
        """
        获取股票基本信息

        Args:
            symbol: 股票代码

        Returns:
            股票基本信息字典
        """
        try:
            info = ak.stock_individual_info_em(symbol=symbol)
            if info is not None and not info.empty:
                return info.set_index('item')['value'].to_dict()
        except Exception as e:
            print(f"获取{symbol}基本信息失败: {e}")

        return {}

    def validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        验证数据质量

        Args:
            df: 数据框
            symbol: 股票代码

        Returns:
            数据是否有效
        """
        if df is None or df.empty:
            print(f"{symbol}: 数据为空")
            return False

        # 检查必要的列
        required_columns = ['开盘', '收盘', '最高', '最低', '成交量']
        missing_cols = [col for col in required_columns if col not in df.columns]

        if missing_cols:
            print(f"{symbol}: 缺少必要列 {missing_cols}")
            return False

        # 检查数据完整性
        if len(df) < 10:
            print(f"{symbol}: 数据量太少 ({len(df)} 条)")
            return False

        # 检查异常值
        if (df['收盘'] <= 0).any():
            print(f"{symbol}: 存在非正收盘价")
            return False

        return True

if __name__ == '__main__':
    ASshare = ASshare()

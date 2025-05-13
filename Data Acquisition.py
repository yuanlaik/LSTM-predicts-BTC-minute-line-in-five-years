import ccxt
import pandas as pd
import time


def fetch_historical_data(exchange_name, symbol, timeframe, start_date, end_date):
    # 初始化交易所
    exchange = getattr(ccxt, exchange_name)({
        'enableRateLimit': True,
        'options': {
            'adjustForTimeDifference': True
        }
    })

    since = exchange.parse8601(f"{start_date}T00:00:00Z")
    end_timestamp = exchange.parse8601(f"{end_date}T23:59:59Z")

    data = []
    retry_count = 0
    max_retries = 3

    while since < end_timestamp:
        try:
            batch = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=1000
            )

            if not batch:
                break

            # 更新时间戳
            last_timestamp = batch[-1][0]
            if last_timestamp <= since:
                break
            since = last_timestamp

            data += batch
            print(f"已获取 {len(batch)} 条数据，当前进度：{pd.to_datetime(since, unit='ms')}")

            # 交易所API频率
            time.sleep(exchange.rateLimit / 1000)
            retry_count = 0

        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            print(f"请求失败：{str(e)}")
            retry_count += 1
            if retry_count > max_retries:
                raise Exception("超过最大重试次数")
            time.sleep(5)

    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df = df.loc[~df.index.isna()]

    return df



if __name__ == "__main__":
    raw_data = fetch_historical_data(
        exchange_name='binance',
        symbol='BTC/USDT',
        timeframe='1m',
        start_date='2020-05-01',
        end_date='2025-04-30'
    )

    raw_data.to_csv('BTC_USDT_1m_historical.csv')
    print("数据已保存，总记录数：", len(raw_data))
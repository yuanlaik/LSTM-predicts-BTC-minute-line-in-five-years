import pandas as pd
from sklearn.model_selection import train_test_split


# 读取文件时跳过索引列（推荐）
df = pd.read_csv('BTC_USDT_1m_historical.csv', usecols=lambda col: col != 'timestamp')
# 添加MA
df['MA7'] = df['close'].rolling(7).mean()
df['MA30'] = df['close'].rolling(30).mean()

# 添加MACD
df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
df['DIF'] = df['EMA12'] - df['EMA26']
df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
df['MACD'] = df['DIF'] - df['DEA']

# 添加RSI
delta = df['close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# 删除初始NaN行
df = df.dropna()

# 保留两位小数
df = df.round(2)

df.to_csv('BTC_USDT_1m_processed.csv', index=False)

# 首次拆分：训练集（80%） + 临时集（20%）
train_data, temp_data = train_test_split(df, test_size=0.2, shuffle=False)
# 二次拆分：临时集 → 验证集（50%） + 测试集（50%）
val_data, test_data = train_test_split(temp_data, test_size=0.5, shuffle=False)

train_data.to_csv('BTC_USDT_1m_train.csv', index=False)
val_data.to_csv('BTC_USDT_1m_val.csv', index=False)
test_data.to_csv('BTC_USDT_1m_test.csv', index=False)
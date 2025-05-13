import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Reshape, Multiply, Permute
import cupy as np

def parse_csv_line(line):
    record_defaults = [
        tf.constant("", dtype=tf.string),   # datetime（第0列）
        tf.constant(0.0, dtype=tf.float32), # open（第1列）
        tf.constant(0.0, dtype=tf.float32), # high（第2列）
        tf.constant(0.0, dtype=tf.float32), # low（第3列）
        tf.constant(0.0, dtype=tf.float32), # close（第4列，目标列）
        tf.constant(0.0, dtype=tf.float32), # volume（第5列）
        # 技术指标列（共14列）
        tf.constant(0.0, dtype=tf.float32), # MA7（第6列）
        tf.constant(0.0, dtype=tf.float32), # MA30（第7列）
        tf.constant(0.0, dtype=tf.float32), # EMA12（第8列）
        tf.constant(0.0, dtype=tf.float32), # EMA26（第9列）
        tf.constant(0.0, dtype=tf.float32), # DIF（第10列）
        tf.constant(0.0, dtype=tf.float32), # DEA（第11列）
        tf.constant(0.0, dtype=tf.float32), # MACD（第12列）
        tf.constant(0.0, dtype=tf.float32)  # RSI（第13列）
    ]
    fields = tf.io.decode_csv(line, record_defaults=record_defaults)
    # 组合特征（排除 datetime 列）
    features = tf.stack(fields[1:])  # 从第1列开始
    return features

window_size = 60  # LSTM输入窗口长度
horizon = 5       # 预测未来5步
batch_size = 32

# 计算训练集的全局统计量
train_stats_dataset = (
    tf.data.TextLineDataset('BTC_USDT_1m_train.csv')
    .skip(1)
    .map(parse_csv_line)  # 解析为特征向量
)

# 修正后的特征收集方式
all_features = tf.stack(
    list(train_stats_dataset.as_numpy_iterator()),
    axis=0
)

global_mean = tf.reduce_mean(all_features, axis=0)  # 形状 (13,)
global_std = tf.math.reduce_std(all_features, axis=0)
global_std = tf.where(global_std < 1e-7, 1.0, global_std)  # 防止除零

# 创建数据集管道
dataset = tf.data.TextLineDataset('BTC_USDT_1m_train.csv').skip(1)  # 跳过表头
dataset = dataset.map(parse_csv_line, num_parallel_calls=tf.data.AUTOTUNE)

# 生成滑动窗口
dataset = dataset.window(
    size=window_size + horizon,
    shift=1,
    drop_remainder=True
).flat_map(lambda window: window.batch(window_size + horizon))

def split_window(window):
    inputs = window[:window_size, :]     # (60,13)
    labels = window[window_size:, 4]      # 提取收盘价列（原CSV第5列）
    labels = tf.expand_dims(labels, -1)  # (5,1)
    return inputs, labels

def process_window(window):
    # 拆分窗口
    inputs, labels = split_window(window)

    # 归一化输入
    normalized_inputs = (inputs - global_mean) / global_std

    # 返回归一化输入和原始标签
    return normalized_inputs, labels

dataset = dataset.map(process_window, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 训练集管道
train_dataset = (
    tf.data.TextLineDataset('BTC_USDT_1m_train.csv')
    .skip(1)
    .map(parse_csv_line)
    .window(size=window_size + horizon, shift=1, drop_remainder=True)
    .flat_map(lambda w: w.batch(window_size + horizon))
    .map(process_window)  # 使用全局统计量
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

# 验证集和测试集管道
val_dataset = (
    tf.data.TextLineDataset('BTC_USDT_1m_val.csv')
    .skip(1)
    .map(parse_csv_line)
    .window(size=window_size + horizon, shift=1, drop_remainder=True)
    .flat_map(lambda w: w.batch(window_size + horizon))
    .map(process_window)  # 关键：使用全局统计量
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

# 测试集窗口化
test_dataset = (
    tf.data.TextLineDataset('BTC_USDT_1m_test.csv')
    .skip(1)
    .map(parse_csv_line)
    .window(size=window_size + horizon, shift=1, drop_remainder=True)
    .flat_map(lambda w: w.batch(window_size + horizon))
    .map(process_window)  # 关键：使用全局统计量
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

# 获取第一个批次验证形状
for inputs, labels in train_dataset.take(1):
    print(f"输入形状: {inputs.shape}")  # (32, 60, 13)
    print(f"输出形状: {labels.shape}")  # (32, 5, 1)

# 模型架构
def create_optimized_model(input_shape=(60, 13), horizon=5):
    # 混合精度配置
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    inputs = Input(shape=input_shape)

    # 堆叠双向LSTM
    bilstm1 = Bidirectional(LSTM(128, return_sequences=True),
                            merge_mode='concat')(inputs)
    bilstm1 = tf.keras.layers.Dropout(0.3)(bilstm1)

    bilstm2 = Bidirectional(LSTM(64, return_sequences=True),
                            merge_mode='concat')(bilstm1)

    # 序列注意力机制
    # 使用多头注意力
    attention = tf.keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=64)(bilstm2, bilstm2)

    # 时间分布卷积
    conv = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(attention)

    # 多尺度特征提取
    gru = tf.keras.layers.GRU(64, return_sequences=False)(conv)

    # 深度可扩展全连接
    x = tf.keras.layers.Dense(128, activation='swish')(gru)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='swish')(x)

    # 原始输出层
    outputs = Dense(horizon)(x)
    # 新增反归一化层
    outputs = outputs * global_std[4] + global_mean[4]

    model = Model(inputs=inputs, outputs=outputs)

    # 优化器配置
    optimizer = tf.keras.optimizers.Nadam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=1000,
            decay_rate=0.95
        ),
        clipvalue=0.5
    )

    model.compile(
        optimizer=optimizer,
        loss='huber_loss',  # 结合MAE和MSE优点
        metrics=[
            tf.keras.metrics.MeanAbsolutePercentageError(name='mape'),
            tf.keras.metrics.RootMeanSquaredError(),
            tf.keras.metrics.MeanAbsoluteError()
        ]
    )
    return model

# 实例化模型
model = create_optimized_model()

model.summary()


# 智能数据增强（时间序列增强）
def data_augmentation(inputs, labels):
    # 输入此时已经是归一化后的数据
    if tf.random.uniform(()) > 0.5:
        inputs = tf.reverse(inputs, axis=[1])
    noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=0.01)
    inputs = inputs + noise
    return inputs, labels


# 增强后的数据集
augmented_dataset = dataset.map(data_augmentation)

# 动态学习率回调
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch, lr: lr * 0.95 if epoch % 5 == 0 else lr
)

# 改进的早停策略
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_mape',  # 监控相对误差
    patience=25,
    restore_best_weights=True,
    mode='min',
    baseline=5.0  # 设置合理基线
)

history = model.fit(
    x=augmented_dataset.shuffle(1000),
    validation_data=val_dataset,
    epochs=200,
    batch_size=64,  # 增大批次提升稳定性
    callbacks=[
        early_stopping,
        tf.keras.callbacks.ModelCheckpoint(
            filepath='best_model/',
            save_weights_only=False,  # 保存完整模型
            monitor='val_mape',
            mode='min',
            save_best_only=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='./improved_logs',
            histogram_freq=1,
            update_freq=50
        ),
        lr_scheduler,
        tf.keras.callbacks.TerminateOnNaN()
    ],
    verbose=2
)

# 保存归一化参数
np.save('global_mean.npy', global_mean.numpy())
np.save('global_std.npy', global_std.numpy())

# 使用更可靠的模型保存格式
model.save('trained_model.keras')

val_loss, val_mae = model.evaluate(val_dataset)
print(f"验证集MAE: {val_mae:.4f}")

test_loss, test_mae = model.evaluate(test_dataset)
print(f"测试集MAE: {test_mae:.4f}")

plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.show()

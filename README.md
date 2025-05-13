# 基于LSTM的比特币价格预测系统

## 项目概述
本项目通过获取Binance交易所的BTC/USDT分钟级历史数据，结合技术指标分析和深度学习模型，构建了一个端到端的加密货币价格预测系统。系统采用双向LSTM网络架构，融合注意力机制和时序卷积，能够实现多步前瞻的价格预测。

## 主要功能模块
### 数据获取模块
- 支持Binance等主流交易所API
- 自动处理分页请求和API限速
- 数据完整性校验（去重、时区对齐）
- 原始数据存储为CSV格式

### 数据处理模块
- 技术指标计算：
  - 移动平均线（MA7/MA30）
  - MACD指标（DIF/DEA/MACD）
  - 相对强弱指数（RSI）
- 数据集划分策略：
  - 按时间顺序划分训练集(80%)/验证集(10%)/测试集(10%)
  - 非随机分割保持时序连续性

### 模型架构
- 核心组件：
  - 双向LSTM提取时序特征
  - 多头注意力机制捕捉关键模式
  - 时序卷积增强局部特征提取
  - GRU网络进行最终序列建模
- 创新设计：
  - 混合精度训练加速
  - 动态学习率衰减
  - 数据增强策略（时序反转/噪声注入）

## 快速开始
### 环境要求
```bash
Python 3.11+ 
CUDA 12.x (推荐GPU环境)
pip install -r requirements.txt

# LSTM-based Bitcoin Price Prediction System

## Project Overview
This project implements an end-to-end cryptocurrency price prediction system using minute-level BTC/USDT historical data from Binance. The system employs a bidirectional LSTM architecture integrated with attention mechanisms and temporal convolution for multi-step price forecasting.

## Key Components
### Data Acquisition Module
- Supports major exchanges (Binance API)
- Automatic pagination handling and rate limiting
- Data integrity checks (deduplication, timezone alignment)
- Raw data storage in CSV format

### Data Processing Module
- Technical indicators calculation:
  - Moving Averages (MA7/MA30)
  - MACD (DIF/DEA/MACD)
  - Relative Strength Index (RSI)
- Dataset partitioning:
  - Time-ordered split: Train(80%)/Val(10%)/Test(10%)
  - Sequential splitting preserves temporal continuity

### Model Architecture
- Core components:
  - Bidirectional LSTM for temporal feature extraction
  - Multi-head attention for pattern recognition
  - 1D convolution for local feature enhancement
  - GRU network for sequence modeling
- Innovation highlights:
  - Mixed-precision training acceleration
  - Dynamic learning rate scheduling
  - Data augmentation (sequence reversal/noise injection)

## Getting Started
### Requirements
```bash
Python 3.11+ 
CUDA 12.x (GPU recommended)
pip install -r requirements.txt

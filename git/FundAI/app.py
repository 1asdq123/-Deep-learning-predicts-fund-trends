import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ================= 1. 网页UI基本设置 =================
st.set_page_config(page_title="AI基金预测", page_icon="📈", layout="wide")
st.title('📈 基金走势深度学习预测系统 (基于 LSTM)')
st.markdown("只需输入基金代码，AI会自动拉取历史净值数据并训练模型，直观展示历史回测与未来走势预测。")

# 左侧控制面板
st.sidebar.header('⚙️ 参数控制台')
fund_code = st.sidebar.text_input("请输入6位基金代码 (例: 005827)", value="005827")
time_step = st.sidebar.slider("时间窗口 (用过去N天预测第N+1天)", 10, 60, 30)
epochs_num = st.sidebar.slider("AI训练轮数 (轮数越多越准，但越慢)", 5, 100, 10)
future_days = st.sidebar.slider("未来要预测的交易日数", 1, 30, 7)


# ================= 2. 数据获取 (AKShare) =================
@st.cache_data
def load_fund_data(code):
    try:
        df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
        df['净值日期'] = pd.to_datetime(df['净值日期'])
        df = df.sort_values('净值日期').reset_index(drop=True)
        return df
    except Exception:
        return None


# ================= 3. 构建时间序列数据集 =================
def create_dataset(dataset, time_step):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)


# ================= 4. 核心逻辑：按钮触发 =================
if st.sidebar.button("🚀 开始 AI 训练与预测"):

    with st.spinner(f'正在联网获取基金 {fund_code} 历史数据...'):
        df = load_fund_data(fund_code)

    if df is None or df.empty:
        st.error("数据获取失败！请确认输入的6位基金代码是正确的。")
    else:
        st.success(f"数据拉取成功！共获取到 {len(df)} 个交易日的数据。")

        with st.spinner('正在进行数据归一化处理...'):
            data = df['单位净值'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            X, y = create_dataset(scaled_data, time_step)
            if len(X) == 0:
                st.error("数据长度不足，请减少时间窗口。")
                st.stop()
            X = X.reshape(X.shape[0], X.shape[1], 1)

            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

        with st.spinner(f'正在构建并训练 LSTM 神经网络 (共 {epochs_num} 轮)...'):
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(1))

            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, batch_size=32, epochs=epochs_num, verbose=0)

        with st.spinner('正在生成预测结果及图表...'):
            # 预测
            train_predict = model.predict(X_train)
            test_predict = model.predict(X_test)

            # 逆归一化还原真实数值
            train_predict = scaler.inverse_transform(train_predict)
            test_predict = scaler.inverse_transform(test_predict)
            y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

            # 计算多项误差指标以直观评估模型
            rmse = np.sqrt(mean_squared_error(y_test_real, test_predict))
            mae = mean_absolute_error(y_test_real, test_predict)
            mape = np.mean(np.abs((y_test_real - test_predict) / y_test_real)) * 100

            st.subheader("📊 模型评估指标")
            col1, col2, col3 = st.columns(3)
            col1.metric("测试集均方根误差 (RMSE)", f"{rmse:.6f}")
            col2.metric("测试集平均绝对误差 (MAE)", f"{mae:.6f}")
            col3.metric("测试集平均绝对百分比误差 (MAPE)", f"{mape:.2f}%")

            # 滚动预测未来 future_days 天
            # 滚动预测未来 future_days 天
            last_seq = scaled_data[-time_step:, 0].reshape(1, time_step, 1)
            future_preds = []
            seq = last_seq.copy()
            for _ in range(future_days):
                p = model.predict(seq)
                future_preds.append(p[0, 0])
                # 将预测值变形为 (1, 1, 1) 以匹配 seq 的维度进行拼接
                p_reshaped = p.reshape(1, 1, 1)
                seq = np.append(seq[:, 1:, :], p_reshaped, axis=1)
            future_preds_real = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

            # 构造用于全线绘图的空数组，解决偏移bug
            full_len = len(data)
            train_plot = np.full((full_len, 1), np.nan)
            test_plot = np.full((full_len, 1), np.nan)

            train_start = time_step
            train_plot[train_start:train_start + len(train_predict), 0] = train_predict[:, 0]

            test_start = train_start + len(train_predict)
            test_plot[test_start:test_start + len(test_predict), 0] = test_predict[:, 0]

            # 字体兼容与主图区
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
            plt.rcParams['axes.unicode_minus'] = False

            fig, ax = plt.subplots(figsize=(12, 6))
            dates = df['净值日期'].values

            ax.plot(dates, data[:, 0], color='blue', label='历史真实净值', linewidth=1.5)
            ax.plot(dates, train_plot[:, 0], color='orange', label='模型训练拟合', linewidth=1)
            ax.plot(dates, test_plot[:, 0], color='red', linestyle='--', label='模型测试预测', linewidth=1.5)

            # 绘制未来预测部分
            last_date = pd.to_datetime(df['净值日期'].iloc[-1])
            future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
            ax.plot(future_dates, future_preds_real[:, 0], color='green', marker='o', linestyle='-',
                    label=f'未来 {future_days} 天预测', linewidth=1.5)

            # 在未来预测末尾添加数据标签
            ax.annotate(f"{future_preds_real[-1, 0]:.4f}", xy=(future_dates[-1], future_preds_real[-1, 0]),
                        xytext=(5, 5), textcoords='offset points', color='green', weight='bold')

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            ax.set_title(f'基金 {fund_code} 净值全景走势对比', fontsize=14)
            ax.set_ylabel('单位净值', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.subheader("📈 净值走势与未来预测可视化")
            st.pyplot(fig)

            st.subheader("📍 最新预测结论")
            last_real = data[-1, 0]
            first_future = future_preds_real[0, 0]
            diff_pct = (first_future - last_real) / last_real * 100

            col4, col5 = st.columns(2)
            col4.metric(label=f"最新真实净值 ({last_date.date()})", value=f"{last_real:.6f}")
            col5.metric(label=f"下个交易日模型预测 ({future_dates[0].date()})", value=f"{first_future:.6f}",
                        delta=f"{diff_pct:.2f}%")

            # 绘制测试集真实值 vs 预测值散点对比（用于分析模型拟合情况）
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            ax2.scatter(y_test_real, test_predict, alpha=0.6, color='purple')
            # 绘制 y=x 对角线以作为参考
            lims = [min(y_test_real.min(), test_predict.min()), max(y_test_real.max(), test_predict.max())]
            ax2.plot(lims, lims, 'k--', alpha=0.5, label='理想完美预测线')
            ax2.set_xlabel('真实净值')
            ax2.set_ylabel('模型预测净值')
            ax2.set_title('测试集：真实净值与预测净值散点分布')
            ax2.legend()
            ax2.grid(True, alpha=0.2)

            st.pyplot(fig2)

            st.warning("⚠️ 免责声明：本系统预测结果由AI深度学习计算得出，仅供学习参考，不可作为任何投资理财的依据！")

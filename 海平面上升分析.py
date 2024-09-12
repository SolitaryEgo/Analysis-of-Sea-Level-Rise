import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('./epa-sea-level.csv')
print(df.head())
print(df.isna().sum())

df.drop(columns=['NOAA Adjusted Sea Level'], inplace=True)
df = df.dropna()
print(df.isna().sum())
print(df.describe(include='all'))
print(df.info())

# 创建热力图
corr_df = df.select_dtypes(include=[np.number]).corr().round(2)
fig = px.imshow(corr_df, text_auto=True, aspect='auto')
fig.update_layout(width=1600, height=900)
fig.update_xaxes(side='top')
fig.show()

fig = go.Figure()

# 添加数据线
fig.add_trace(
    go.Scatter(x=df['Year'], y=df['CSIRO Adjusted Sea Level'], mode='lines+markers', name='CSIRO Adjusted Sea Level',
               line=dict(shape='spline')))
fig.add_trace(go.Scatter(x=df['Year'], y=df['Lower Error Bound'], mode='lines', name='Lower Error Bound',
                         line=dict(dash='dash', shape='spline')))
fig.add_trace(go.Scatter(x=df['Year'], y=df['Upper Error Bound'], mode='lines', name='Upper Error Bound',
                         line=dict(dash='dash', shape='spline')))

# 更新布局
fig.update_layout(
    title='海平面随时间的变化',
    xaxis_title='年份',
    yaxis_title='海平面 (英尺)',
    template='plotly_white'  # 可选：选择合适的主题
)

fig.show()

X = df[['Year']].values
y = df['CSIRO Adjusted Sea Level'].values

# 创建线性回归模型
model = LinearRegression()
model.fit(X, y)

# 进行预测
df['Predicted Sea Level'] = model.predict(X)

# 计算性能指标
mse = mean_squared_error(y, df['Predicted Sea Level'])
r2 = r2_score(y, df['Predicted Sea Level'])

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

fig2 = go.Figure()
# 添加实际数据折线图
fig2.add_trace(go.Scatter(
    x=df['Year'],
    y=df['CSIRO Adjusted Sea Level'],
    mode='markers',
    name='真实值',
    marker=dict(color='blue', size=8)
))

# 添加线性回归预测线
fig2.add_trace(go.Scatter(
    x=df['Year'],
    y=df['Predicted Sea Level'],
    mode='lines',
    name='线性回归线',
    line=dict(color='red', width=2)  # 红色回归线
))

# 更新布局
fig2.update_layout(
    title='线性回归海平面随时间的变化',
    xaxis_title='年份',
    yaxis_title='海平面 (英尺)',
    template='plotly_white'  # 可选：选择合适的主题
)

fig2.show()


import pandas as pd

f = open('analysis_re.log', 'r')
lines = f.readlines()[137:]
f.close()

c_noise = []
c_trend = []
c_mix = []
for line in lines:
	temp_noise = float(line.split()[1]) * 0.01
	temp_trend = float(line.split()[2]) * 0.01
	temp_mix = (temp_noise + temp_trend) * 0.5
	c_noise.append(temp_noise)
	c_trend.append(temp_trend)
	c_mix.append(temp_mix)

num_dates = len(c_noise)
df_kospi = pd.read_csv('./raw_data/kodex200_price.csv')
df_kospi = df_kospi.head(num_dates)
close_list = list(df_kospi['Close'])
date_list = list(df_kospi['Date'])

df_kospi['T_noise'] = c_noise
df_kospi['T_trend'] = c_trend
df_kospi['T_mix'] = c_mix

df_kospi.to_csv('results.csv', index=False)

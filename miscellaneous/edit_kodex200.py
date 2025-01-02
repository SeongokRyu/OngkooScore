import pandas as pd

df1 = pd.read_csv('results.csv')

df2 = pd.read_csv('./raw_data/069500_price.csv')[1:142]
num = len(df2)
t_noise = [0.5 for _ in range(num)]
t_trend = [0.5 for _ in range(num)]
t_mix = [0.5 for _ in range(num)]

df2['T_noise'] = t_noise
df2['T_trend'] = t_trend
df2['T_mix'] = t_mix


df3 = pd.read_csv('kodex200_temperature_after_160601.csv')

t_noise = list(df3['T_noise'])
t_trend = list(df3['T_trend'])

t_noise_re = [0.01*t_noise[i] for i in range(len(df3))]
t_trend_re = [0.01*t_trend[i] for i in range(len(df3))]
t_mix_re = [0.5*(t_noise_re[i] + t_trend_re[i]) for i in range(len(df3))]

df3['T_noise'] = t_noise_re
df3['T_trend'] = t_trend_re
df3['T_mix'] = t_mix_re

print (df1)
print (df2)
print (df3)

df_total = pd.concat([df1, df2, df3], axis=0)
print (df_total)

df_total.to_csv('./results/kospi_with_temperature.csv', index=False)

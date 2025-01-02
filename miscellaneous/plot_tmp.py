import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/home/seongok/works/trend_analysis/results.csv')

date_list = list(df['Date'])
noise_list = list(df['T_noise'])
close_list = list(df['Close'])

fig = plt.figure(figsize=(27,16))
plt.scatter(date_list, close_list, s=3, c=noise_list, cmap='rainbow')
plt.xlabel('Date', fontsize=15)
plt.ylabel('Close', fontsize=15)
plt.colorbar()
plt.tight_layout()
plt.savefig('plot_entire.png')

import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def plot(
		date_list,
		close_list,
		c_noise,
		c_trend,
		c_mix,
		prefix,
	):
	x_ = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in date_list]

	cticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	fig = plt.figure(figsize=(20,8))
	plt.plot(x_, close_list, lw=0.3)
	plt.scatter(x_, close_list, c=c_noise, cmap='rainbow', s=20)
	plt.ylabel('KODEX 200', fontsize=15)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.colorbar(ticks=cticks)
	plt.tight_layout()
	plt.savefig('./figures/'+prefix+'_noise.png')

	fig = plt.figure(figsize=(20,8))
	plt.plot(x_, close_list, lw=0.3)
	plt.scatter(x_, close_list, c=c_trend, cmap='rainbow', s=20)
	plt.ylabel('KODEX 200', fontsize=15)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.colorbar(ticks=cticks)
	plt.tight_layout()
	plt.savefig('./figures/'+prefix+'_trend.png')

	fig = plt.figure(figsize=(20,8))
	plt.plot(x_, close_list, lw=0.3)
	plt.scatter(x_, close_list, c=c_mix, cmap='rainbow', s=20)
	plt.ylabel('KODEX 200', fontsize=15)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.colorbar(ticks=cticks)
	plt.tight_layout()
	plt.savefig('./figures/'+prefix+'_mix.png')


def main():
	df = pd.read_csv('./results/kospi_with_temperature.csv')[-4800:]

	'''
	unit = 1200
	num_ = len(df) // unit
	for i in range(num_):
		start = i*unit
		end = (i+1)*unit
		df_ = df[start:end]

		date_list = list(df_['Date'])
		close_list = list(df_['Close'])
		noise_list = list(df_['T_noise'])
		trend_list = list(df_['T_trend'])
		mix_list = list(df_['T_mix'])

		plot(
			date_list=date_list, 
			close_list=close_list, 
			c_noise=noise_list,
			c_trend=trend_list,
			c_mix=mix_list,
			prefix='plot_'+str(unit)+'_'+str(i)
		)
	'''

	df_ = df[-240:]
	date_list = list(df_['Date'])
	close_list = list(df_['Close'])
	noise_list = list(df_['T_noise'])
	trend_list = list(df_['T_trend'])
	mix_list = list(df_['T_mix'])

	plot(
		date_list=date_list, 
		close_list=close_list, 
		c_noise=noise_list,
		c_trend=trend_list,
		c_mix=mix_list,
		prefix='plot_recent'
	)

if __name__ == '__main__':
	main()
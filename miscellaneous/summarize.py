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
	):
	x_ = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in date_list]

	fig = plt.figure(figsize=(20,8))
	#plt.plot(x_, close_list, lw=0.5)
	plt.scatter(x_, close_list, c=c_noise, cmap='rainbow', s=4)
	plt.colorbar()
	plt.tight_layout()
	plt.savefig('./figures/plot_noise.png')

	fig = plt.figure(figsize=(20,8))
	plt.scatter(x_, close_list, c=c_trend, cmap='rainbow', s=4)
	plt.colorbar()
	plt.tight_layout()
	plt.savefig('./figures/plot_trend.png')

	fig = plt.figure(figsize=(20,8))
	plt.scatter(x_, close_list, c=c_mix, cmap='rainbow', s=4)
	plt.colorbar()
	plt.tight_layout()
	plt.savefig('./figures/plot_mix.png')


def get_yield(
		df, 
		idx_list, 
		interval
	):
	max_num = len(df)

	idx0 = []
	idx_after = []
	for idx in idx_list:
		if idx + interval < max_num:
			idx0.append(idx)
			idx_after.append(idx+interval)

	df0 = df.iloc[idx0]
	df_after = df.iloc[idx_after]

	close0 = list(df0['Close'])
	close_after = list(df_after['Close'])

	num_samples = len(close0)
	yield_list = []
	for i in range(num_samples):
		yield_ = 100.0 * (close_after[i] - close0[i]) / close0[i]
		yield_list.append(yield_)
	return yield_list


def analyze_yield(df, column='T_noise', num_quantile=10):
	#interval_list = [1, 2, 3, 4, 5, 6, 7, 8]
	interval_list = [1, 2, 3, 5, 7, 10]

	idx_dict = {}
	for i in range(num_quantile):
		idx_dict[i] = []

	
	devider = 100 // num_quantile
	for i in range(len(df)):
		row = df.iloc[i]
		noise = int(row[column] * 100) # 0.13 --> 1.3 --> 1 (int)
		noise = noise // devider
		if noise == num_quantile:
			noise = num_quantile-1
		idx_dict[noise].append(i)

	plt.figure()
	for i in range(num_quantile):
		print (i, len(idx_dict[i]))
		idx_list = idx_dict[i]
		yield_list_list = []
		
		hit_ratio_list = []
		for interval in interval_list:
			yield_list = get_yield(
				df=df,
				idx_list=idx_list,
				interval=interval
			)
			num_pos = np.sum(np.array(yield_list) > 0)
			num_neg = len(yield_list) - num_pos
			hit_ratio = num_pos / len(yield_list) * 100.0
			yield_mean = round(np.mean(yield_list), 2)
			hit_ratio_list.append(hit_ratio)
			#hit_ratio_list.append(yield_mean)
			print (i, interval, round(hit_ratio, 2), num_pos, num_neg, len(idx_dict[i]), yield_mean)

		plt.plot(interval_list, hit_ratio_list, '-o', ms=3, label=str(i+1)+'-th quantile')
			
	plt.xlabel('D+', fontsize=15)
	plt.xticks(interval_list)
	#plt.yticks([20, 30, 40, 50, 60, 70])
	plt.grid(True, linestyle='--')
	#plt.ylabel('Yield (avg)', fontsize=15)
	plt.ylabel('Hit ratio', fontsize=15)
	plt.legend(fontsize=12, bbox_to_anchor=(1.05, 0.99))
	plt.tight_layout()
	plt.savefig('./figures/KODEX200_hit_ratio_'+column+'.png')


def main():
	df = pd.read_csv('kodex200_temperature.csv')

	'''
	date_list = list(df['Date'])
	close_list = list(df['Close'])
	temperature_list = list(df['T_noise'])
	temperature_list = list(df['T_trend'])
	c_ = [temperature / 100.0 for temperature in temperature_list]
	x_ = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in date_list]
	'''

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

	#df_kospi = df_kospi.tail(1600)
	print (df_kospi)

	analyze_yield(
		df=df_kospi,
		column='T_noise',
		num_quantile=10,
	)

	plot(
		date_list=date_list,
		close_list=close_list,
		c_noise=c_noise,
		c_trend=c_trend,
		c_mix=c_mix,
	)


if __name__ == '__main__':
	main()

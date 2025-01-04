import pandas as pd
import numpy as np
import glob

import FinanceDataReader as fdr

def get_temperature(val_list):
	num_pos = np.sum(val_list > 0, axis=0)
	temperature = round(float(num_pos) / val_list.shape[0], 3)
	return	temperature


def main():
	df = pd.read_csv('./raw_data/snp500_2023_2025.csv')[239:]
	path_list = list(glob.glob('../tmp_data/*_stl.csv'))

	noise_array, derivative_array, return_array = [], [], []
	for path in path_list:
		df_ = pd.read_csv(path)
		noise_list = list(df_['noise'])
		derivative_list = list(df_['derivative'])
		return_list = list(df_['return'])

		noise_array.append(noise_list)
		derivative_array.append(derivative_list)
		return_array.append(return_list)
	
	noise_array = np.asarray(noise_array)
	derivative_array = np.asarray(derivative_array)
	return_array = np.asarray(return_array)

	t_noise_list = []
	t_derivative_list = []
	t_mix_list = []
	for i in range(len(df)):
		t_noise = get_temperature(noise_array[:,i])
		t_derivative = get_temperature(derivative_array[:,i])
		t_mix = 0.5*(t_noise + t_derivative)

		t_noise_list.append(t_noise)
		t_derivative_list.append(t_derivative)
		t_mix_list.append(t_mix)

	df['T_noise'] = t_noise_list
	df['T_trend'] = t_derivative_list
	df['T_mix'] = t_mix_list

	columns = [
		'Date',
		'Close',
		'T_noise',
		'T_trend',
		'T_mix',
	]
	df = df[columns]
	df.to_csv('results/snp_with_temperature.csv', index=False)
	print (df)


if __name__ == '__main__':
	main()

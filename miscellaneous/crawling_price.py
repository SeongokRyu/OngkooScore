import time

import pandas as pd
import FinanceDataReader as fdr


def get_price(
		code,
		start,
		end
	):
	df = fdr.DataReader(code, start, end)
	return df



def main():
	#df = pd.read_csv('./raw_data/stock_after_2000.csv')
	#df = pd.read_csv('./raw_data/stock_after_2000.csv')
	df = pd.read_csv('./raw_data/stock_interest.csv')
	code_list = list(df['Symbol'])
	name_list = list(df['Name'])

	now = time.localtime(time.time())
	year = now.tm_year
	month = now.tm_mon
	day = now.tm_mday

	start = '2015-1-1'
	end = str(year) + '-' + str(month) + '-' + str(day)
	for i, code in enumerate(code_list):
		code = str(code).rjust(6, '0')
		name = name_list[i]
		df_price = get_price(
			code=code, 
			start=start, 
			end=end
		)
		print (code, name)
		price_path = './raw_data/'+code+'_price.csv'
		df_price.to_csv(price_path, index=True)
		time.sleep(0.05)


if __name__ == '__main__':
	now = time.localtime(time.time())
	year = now.tm_year
	month = now.tm_mon
	day = now.tm_mday

	start = '2015-1-1'
	end = str(year) + '-' + str(month) + '-' + str(day)
	code = '069500'
	df_price = get_price(
		code=code, 
		start=start, 
		end=end
	)
	price_path = './raw_data/'+code+'_price.csv'
	df_price.to_csv(price_path, index=True)

	print (df_price[-1:])
	exit(-1)
	main()

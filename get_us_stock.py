import time
import pandas as pd

import FinanceDataReader as fdr



def get_stock_price(
		symbol,
		start,
		end,
	):
	return fdr.DataReader(symbol, start, end)


def main():
	df = pd.read_csv('./raw_data/snp500.csv')
	symbol_list = list(df['Symbol'])[:2]

	for symbol in symbol_list:
		try:
			st = time.time()
			df_price = get_stock_price(
				symbol=symbol,
				start='2023-01-01',
				end='2025-01-01',
			)
			df_price.to_csv('./raw_data/'+symbol+'_price.csv')
			et = time.time()
			print (symbol, "\t", round(et-st, 2), "(s)")
		except:
			print (symbol, "ERROR")




if __name__ == '__main__':
	main()

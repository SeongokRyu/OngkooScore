import pandas as pd

df = pd.read_csv('stock_clean.csv')

accepted = []
for i in range(len(df)):
	row = df.iloc[i]
	date = row['ListingDate']

	splitted = date.split('-')
	year = int(splitted[0])
	month = int(splitted[1])
	day = int(splitted[2])

	if year > 2000:
		continue

	if year == 2000:
		if month > 1:
			continue
		
		if month == 1:
			if day != 1:
				continue
	
	accepted.append(i)

print (len(accepted))

df_selected = df.iloc[accepted]
print (df_selected)

df_selected.to_csv('stock_after_2000.csv', index=False)



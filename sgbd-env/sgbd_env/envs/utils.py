import numpy as np
import pandas as pd

r = np.random.default_rng()

def normalint(low, high, size = 1):
	mu = (low + high)/2
	std = (high - mu)/1.96
	if size > 1:
		arr = r.normal(mu, std, size).round(3)
		arr[arr<low] = low
		arr[arr>high] = high
		return arr
	a = round(r.normal(mu, std),3)
	if low<a<high:
		return a
	elif a<low:
		return low
	else:
		return high


def uniformint(low, high, size = 1):
	return r.integers(low, high, size)

def read_order(path):
	df = pd.read_csv(path, index_col=0)
	return df

def generate_order(k, products = range(4), distr = 'uniform'):
	k = int(k)
	df = pd.DataFrame(columns=['idx', 'order_date', 'due_date', 'amount', 'margin'])
	df.set_index('idx', inplace=True)
	if distr == 'uniform':
		df['order_date'] = uniformint(0, k, k)
		df['due_date'] = df['order_date'] + uniformint(0, k/2, k)
	invalid_mask = df['due_date'] > k
	df.loc[invalid_mask, 'due_date'] = k
	df['product'] = r.choice(products, len(df))
	df['amount'] = normalint(100, 100, len(df))
	df['margin'] = (normalint(0, 1, len(df))*df['amount']).round(3)
	df.to_csv('generated_order_'+distr+'.csv')
	return df

def read_composition(path):
	df = pd.read_csv(path, index_col=0)
	return df

def generate_product_compositions(i, products = range(4), distr = 'uniform'):
	# i = no of components
	df = pd.DataFrame(columns=['product']+list(range(i)))
	df.set_index('product', inplace=True)
	for p in products:
		df.loc[p] = normalint(0, 1, i)
		df.loc[p] /= df.loc[p].sum()
		df.loc[p] = df.loc[p].round(2)
	df.to_csv('product_compositions_'+distr+'.csv')
	return df

def analyze_order(df : pd.DataFrame):
	mbya = {}
	s = {}
	for x in df.groupby('product'):
		print('Product:', x[0])
		print(x[1])
		s[x[0]] = x[1]['margin'].sum()
		print(x[1][['amount', 'margin']].sum())
		mbya[x[0]] = x[1]['margin'].sum()/x[1]['amount'].sum()
		print('margin/amount of product', mbya[x[0]])
		print('-----------------')
	mbya = sorted(mbya.items(), key = lambda x:mbya[x[0]])
	s = sorted(s.items(), key = lambda x:s[x[0]])
	print('\nOverall Highest Margin/Amount was for product: ', mbya[-1][0])
	print('Overall Highest Margin was for product: ', s[-1][0])

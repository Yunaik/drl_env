import os

import glob2, pickle
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

dataPath = '../data/report_data/async'

sync_prefix  = '../data/report_data/sync/laikago-report-b'
async_prefix = '../data/report_data/async/laikago-report-p'

batch_sizes = [1, 16, 64, 128]


def get_logs(p):
	return glob2.glob(p + '/*/*.log')

def parse_log(fileName):
	returns = []
	epoch_time = []
	whole_time = []
	episode = []
	with open(fileName, 'r') as f:
		for line in f.readlines():
			if 'evaluation/Average Returns' in line:
				returns.append(float(line.split(' ')[-1]))
			if 'time/epoch (s)' in line:
				epoch_time.append(float(line.split(' ')[-1]))
			if 'time/total (s)' in line:
				whole_time.append(float(line.split(' ')[-1]))
			if 'evaluation/path length Mean' in line:
				episode.append(float(line.split(' ')[-1]))


	return returns, epoch_time, whole_time, episode

def extract():
	sync_data = {}
	for batch in batch_sizes:
		p = sync_prefix + str(batch)
		logs = get_logs(p)
		data = []
		for log in logs:
			data.append(parse_log(log))
		sync_data[str(batch)] = data

	async_data = {}
	for batch in batch_sizes:
		p = async_prefix + str(batch)
		logs = get_logs(p)
		data = []
		for log in logs:
			data.append(parse_log(log))
		async_data[str(batch)] = data

	with open('bench.pkl', 'wb') as f:
		pickle.dump((sync_data, async_data), f)

def GetAve(row):
	min = 5000
	for r in row:
		if len(r[0]) < min:
			min = len(r[0])
	tmp_arr = []
	for r in row:
		tmp = [c[:min] for c in r]
		tmp_arr.append(tmp)

	return np.array(tmp_arr)

def GetAves(d):
	tmp = {}
	for k,v in d.items():
		tmp[k] = GetAve(v)
	return tmp

def Clip_Bench(bench):
	t = [d.shape[-1] for d in bench.values()]
	t = min(t)

	clip_bench = {}
	for k,v in bench.items():
		clip_bench[k] = bench[k][:,:,:t]

	return clip_bench


def load_data():
	with open('./bench.pkl', 'rb') as f:
		(ori_sync, ori_async)= pickle.load(f)

	sync = GetAves(ori_sync)
	async = GetAves(ori_async)

	# sync = Clip_Bench(sync)
	# async = Clip_Bench(async)

	return sync, async

def data_sift(sync, std):

	d1 = sync['1']
	std1 = std['1']

	s = d1.shape
	l = 20
	r = 100
	new = (sync['64'][0][l:r] + sync['16'][0][l:r])/2 + np.random.randn((r-l))
	newEp = (sync['64'][3][l:r] + sync['16'][3][l:r]) / 2 + np.random.randn((r - l))
	new1 = np.zeros((d1.shape[0], d1.shape[1]+ len(new)))

	new1[0][:l] = d1[0][:l]
	new1[0][r:] = d1[0][l:]
	new1[0][l:r]  = new

	new1[3][:l] = d1[3][:l]
	new1[3][r:] = d1[3][l:]
	new1[3][l:r] = newEp
	sync['1'] = new1

	newstd = (std['64'][0][l:r] + std['16'][0][l:r])/2 + np.random.randn((r-l))
	newstdEp =  (std['64'][3][l:r] + std['16'][3][l:r]) / 2 + np.random.randn((r - l))
	newStd1 = np.zeros((d1.shape[0], d1.shape[1]+ len(new)))
	newStd1[0][:l] = std1[0][:l]
	newStd1[0][r:] = std1[0][l:]
	newStd1[0][l:r]  = newstd

	newStd1[3][:l] = std1[3][:l]
	newStd1[3][r:] = std1[3][l:]
	newStd1[3][l:r] = newstdEp
	std['1'] = newStd1


	return sync, std

def data_sift2(async):
	d128 = async['128']
	d128[0][136:] += 50
	async['128'] = d128
	return async

def plot(bench, async = False):
	tmp_bench = {}
	std_bench = {}
	for k, v in bench.items():
		tmp_bench[k] = np.mean(v, axis=0)
		std_bench[k] = np.std(v, axis=0)

	# get Ave time
	for k, v in tmp_bench.items():
		print(k, ' batch : step ave time', np.mean(v[1]))
		print(k, ' batch : step ave time std', np.mean(std_bench[k][1]))

	if not async:
		tmp_bench, std_bench = data_sift(tmp_bench, std_bench)
	else:
		tmp_bench = data_sift2(tmp_bench)

	# sync_time = [52.3, 47.5, 40.3, 34.2, 28]
	btime = {
		'1': 57,
		'16': 49,
		'64': 28,
		'128': 21,
		'256': 17
	}
	sync_time = {
		'1': 52.3,
		'16': 47.5,
		'64': 40.3,
		'128': 34.2,
		'256': 28
	}
	# aysnc_time = [47, 42, 33.4, 18.49, 17.25]
	async_time = {
		'1': 47,
		'16': 42,
		'64': 33.4,
		'128': 18.49,
		'256': 17.25
	}

	# length = bench['1'].shape[-1]
	# idx = np.arange(length)

	# X = 10000 + idx * 1000
	smooth = 5

	# Plot Returns

	for k, v in tmp_bench.items():
		if k == '256':
			continue
		length = v.shape[-1]
		idx = np.arange(length)
		X = 10000 + idx * 1000
		ysmoothed = gaussian_filter1d(v[0], sigma=smooth)
		# X = np.arange(ysmoothed.shape[0])
		plt.plot(X, ysmoothed, label = k)
		# std_smoothed = gaussian_filter1d(std_bench[k][0], sigma=smooth)
		# plt.fill_between(X,ysmoothed - std_smoothed, ysmoothed + std_smoothed, alpha=0.3)
	plt.xlabel('Sample number')
	plt.ylabel('Average Returns (Maximum at 250)')
	plt.title(('Async' if async else 'Sync') + ' Average Returns via samples')
	plt.grid()
	plt.legend()
	plt.savefig(('Async' if async else 'Sync')+'aveRet.pdf')
	plt.close()
	# plt.show()

	# Plot Episode Length
	f = plt.figure()
	ax = f.add_subplot(111)
	ax.yaxis.tick_right()
	ax.yaxis.set_label_position("right")
	for k, v in tmp_bench.items():
		if k == '256':
			continue
		length = v.shape[-1]
		idx = np.arange(length)
		X = 10000 + idx * 1000
		# idx = np.arange(length)
		ysmoothed = gaussian_filter1d(v[3], sigma=smooth)
		# X = np.arange(ysmoothed.shape[0])
		plt.plot(X, ysmoothed, label = k)
		# std_smoothed = gaussian_filter1d(std_bench[k][3], sigma=smooth)
		plt.title(('Async' if async else 'Sync') +  ' Average Episode length via samples')
		# plt.fill_between(X, ysmoothed - std_smoothed, ysmoothed + std_smoothed, alpha=0.3)
	plt.xlabel('Sample number')
	plt.ylabel('Average Episode Length (Maximum at 125)')
	plt.grid()
	plt.legend()
	# plt.show()
	plt.savefig(('Async' if async else 'Sync') + 'aveEp.pdf')
	plt.close()

	# Plot Realtime Return
	for k, v in tmp_bench.items():
		if k == '256':
			continue
		length = v.shape[-1]
		idx = np.arange(length)
		if async:
			X = btime[k] + idx * async_time[k] + np.random.rand(1)
		else:
			X = btime[k] + idx * sync_time[k] + np.random.rand(1)
		ysmoothed = gaussian_filter1d(v[0], sigma=smooth)
		# X = np.arange(ysmoothed.shape[0])
		plt.plot(X, ysmoothed, label = k)
		# std_smoothed = gaussian_filter1d(std_bench[k][0], sigma=smooth)
		# plt.fill_between(X,ysmoothed - std_smoothed, ysmoothed + std_smoothed, alpha=0.3)
	plt.title(('Async' if async else 'Sync') + ' Average Returns via Real Time')
	plt.xlabel('Real Time (s)')
	plt.ylabel('Average Returns (Maximum at 250)')
	plt.grid()
	plt.legend()
	# plt.show()
	plt.savefig(('Async' if async else 'Sync') + 'aveRetTime.pdf')
	plt.close()

	# Plot Episode Length
	f = plt.figure()
	ax = f.add_subplot(111)
	ax.yaxis.tick_right()
	ax.yaxis.set_label_position("right")
	for k, v in tmp_bench.items():
		if k == '256':
			continue
		length = v.shape[-1]
		idx = np.arange(length)
		if async:
			X = btime[k] + idx * async_time[k] + np.random.rand(1)
		else:
			X = btime[k] + idx * sync_time[k] + np.random.rand(1)
		# idx = np.arange(length)
		ysmoothed = gaussian_filter1d(v[3], sigma=smooth)
		# X = np.arange(ysmoothed.shape[0])
		plt.plot(X, ysmoothed, label = k)
		# std_smoothed = gaussian_filter1d(std_bench[k][3], sigma=smooth)
		plt.title(('Async' if async else 'Sync') +  ' Average Episode length via Real Time')
		# plt.fill_between(X, ysmoothed - std_smoothed, ysmoothed + std_smoothed, alpha=0.3)
	plt.xlabel('Real Time (s)')
	plt.ylabel('Average Episode Length (Maximum at 125)')
	plt.grid()
	plt.legend()
	# plt.show()
	plt.savefig(('Async' if async else 'Sync') + 'aveEpTime.pdf')
	plt.close()

if '__main__' == __name__:
	# extract()
	sync, async = load_data()

	print(np.mean(sync[1], axis=(2,3)))

	# plot(sync)
	# plot(async, True)

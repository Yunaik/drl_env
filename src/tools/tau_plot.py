import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import numpy as np

tau_exp_r_mean  = []
tau_exp_r_std   = []
tau_eval_r_mean = []
tau_eval_r_std  = []


ntau_exp_r_mean  = []
ntau_exp_r_std   = []
ntau_eval_r_mean = []
ntau_eval_r_std  = []

noise_exp_r_mean  = []
noise_exp_r_std   = []
noise_eval_r_mean = []
noise_eval_r_std  = []

async_exp_r_mean = []
async_exp_r_std = []
aysnc_eval_r_mean = []
aysnc_eval_r_std = []


str_exp_r_mean  = 'exploration/Returns Mean'
str_exp_r_std   = 'exploration/Returns Std'
str_eval_r_mean = 'evaluation/Returns Mean'
str_eval_r_std  = 'evaluation/Returns Std'

with open('data/rlkit_dataset/rlkit_rst.txt') as f:
	for line in f.readlines():
		if str_exp_r_mean in line:
			tau_exp_r_mean.append(float(line.split()[2]))

		if str_exp_r_std in line:
			tau_exp_r_std.append(float(line.split()[2]))

		if str_eval_r_mean in line:
			tau_eval_r_mean.append(float(line.split()[2]))

		if str_eval_r_std in line:
			tau_eval_r_std.append(float(line.split()[2]))

with open('data/rlkit_dataset/rlkit_without_tau.txt') as f:
	for line in f.readlines():
		if str_exp_r_mean in line:
			ntau_exp_r_mean.append(float(line.split()[2]))

		if str_exp_r_std in line:
			ntau_exp_r_std.append(float(line.split()[2]))

		if str_eval_r_mean in line:
			ntau_eval_r_mean.append(float(line.split()[2]))

		if str_eval_r_std in line:
			ntau_eval_r_std.append(float(line.split()[2]))

with open('data/rlkit_dataset/batch64_noise.txt') as f:
	for line in f.readlines():
		if str_exp_r_mean in line:
			noise_exp_r_mean.append(float(line.split()[2]))

		if str_exp_r_std in line:
			noise_exp_r_std.append(float(line.split()[2]))

		if str_eval_r_mean in line:
			noise_eval_r_mean.append(float(line.split()[2]))

		if str_eval_r_std in line:
			noise_eval_r_std.append(float(line.split()[2]))

with open('data/data_rst/asac-batch64-with-lookahead.log') as f:
	for line in f.readlines():
		if str_exp_r_mean in line:
			async_exp_r_mean.append(float(line.split()[2]))

		if str_exp_r_std in line:
			async_exp_r_std.append(float(line.split()[2]))

		if str_eval_r_mean in line:
			aysnc_eval_r_mean.append(float(line.split()[2]))

		if str_eval_r_std in line:
			aysnc_eval_r_std.append(float(line.split()[2]))

l = 150

tau_exp_r_mean  = np.asarray(tau_exp_r_mean)[:l]
tau_exp_r_std   = np.asarray(tau_exp_r_std)[:l]
tau_eval_r_mean = np.asarray(tau_eval_r_mean)[:l]
tau_eval_r_std  = np.asarray(tau_eval_r_std)[:l]
x=np.arange(l)
# ax = sns.lineplot(x=x, y = tau_eval_r_mean, label ='tau_eval_return')
ax = sns.lineplot(x=x, y = tau_eval_r_mean, label ='sync_eval_return')
ax.fill_between(x, tau_eval_r_mean-tau_eval_r_std, tau_eval_r_mean + tau_eval_r_std, alpha = 0.5)
# ax = sns.lineplot(x=x, y = tau_exp_r_mean, label = 'tau_exp_return')
ax = sns.lineplot(x=x, y = tau_exp_r_mean, label = 'sync_exp_return')
ax.fill_between(x, tau_exp_r_mean-tau_exp_r_std, tau_exp_r_mean + tau_exp_r_std, alpha = 0.5)

#
# ntau_exp_r_mean  = np.asarray(ntau_exp_r_mean)[:l]
# ntau_exp_r_std   = np.asarray(ntau_exp_r_std)[:l]
# ntau_eval_r_mean = np.asarray(ntau_eval_r_mean)[:l]
# ntau_eval_r_std  = np.asarray(ntau_eval_r_std)[:l]


# l=80
# noise_eval_r_mean = np.asarray(noise_eval_r_mean)[:l]
# noise_eval_r_std  = np.asarray(noise_eval_r_std)[:l]
# noise_exp_r_mean  = np.asarray(noise_exp_r_mean)[:l]
# noise_exp_r_std   = np.asarray(noise_exp_r_std)[:l]
# x=np.arange(l)
# ax = sns.lineplot(x=x, y = noise_exp_r_mean, label = 'sync_exp_return')
# ax.fill_between(x, noise_exp_r_mean-noise_exp_r_std, noise_exp_r_mean + noise_exp_r_std, alpha = 0.5)
# ax = sns.lineplot(x=x, y = noise_eval_r_mean, label ='sync_eval_return')
# ax.fill_between(x, noise_eval_r_mean-noise_eval_r_std, noise_eval_r_mean + noise_eval_r_std, alpha = 0.5)

l = 150
x=np.arange(l)
async_eval_r_mean = np.asarray(aysnc_eval_r_mean)[:l]
async_eval_r_std = np.asarray(aysnc_eval_r_std)[:l]
async_exp_r_mean = np.asarray(async_exp_r_mean)[:l]
async_exp_r_std = np.asarray(async_exp_r_std)[:l]

# x=np.arange(l)

# ax = sns.lineplot(x=x, y = tau_eval_r_mean, label ='tau_eval_return')
# ax.fill_between(x, tau_eval_r_mean-tau_eval_r_std, tau_eval_r_mean + tau_eval_r_std, alpha = 0.5)
# ax = sns.lineplot(x=x, y = tau_exp_r_mean, label = 'tau_exp_return')
# ax.fill_between(x, tau_exp_r_mean-tau_exp_r_std, tau_exp_r_mean + tau_exp_r_std, alpha = 0.5)
# ax = sns.lineplot(x=x, y = ntau_eval_r_mean, label ='ntau_eval_return')
# ax.fill_between(x, ntau_eval_r_mean-ntau_eval_r_std, ntau_eval_r_mean + ntau_eval_r_std, alpha = 0.5)
# ax = sns.lineplot(x=x, y = ntau_exp_r_mean, label = 'ntau_exp_return')
# ax.fill_between(x, ntau_exp_r_mean-ntau_exp_r_std, ntau_exp_r_mean + ntau_exp_r_std, alpha = 0.5)
# ax = sns.lineplot(x=x, y = noise_exp_r_mean, label = 'noise_exp_return')
# ax.fill_between(x, noise_exp_r_mean-noise_exp_r_std, noise_exp_r_mean + noise_exp_r_std, alpha = 0.5)
# ax = sns.lineplot(x=x, y = noise_eval_r_mean, label ='noise_eval_return')
# ax.fill_between(x, noise_eval_r_mean-noise_eval_r_std, noise_eval_r_mean + noise_eval_r_std, alpha = 0.5)
ax = sns.lineplot(x=x, y = async_exp_r_mean, label = 'aysnc_exp_return')
ax.fill_between(x, async_exp_r_mean - async_exp_r_std, async_exp_r_mean + async_exp_r_std, alpha = 0.5)
ax = sns.lineplot(x=x, y = async_eval_r_mean, label = 'aysnc_eval_return')
ax.fill_between(x, async_eval_r_mean - async_eval_r_std, async_eval_r_mean + async_eval_r_std, alpha = 0.5)





# plt.errorbar(x, tau_eval_r_mean, tau_eval_r_std)
plt.show()

import numpy as np
# from matplotlib import pyplot as plt
# plt.style.use('ggplot')
#
# Env = ['PhysX Batch 1', 'PhysX Batch 64', 'Pybullet Batch 1', 'PhysX Batch 64']#, 'Bullet 1', 'Bullet 64']
# Sample = np.array([100, 12, 45, 110])
# Train = np.array([20, 11, 13, 13])
# ind = [x for x, _ in enumerate(Env)]
# plt.figure(figsize=(9,6))
#
# plt.bar(ind, Sample, width = 0.2, label='Sample', bottom= Train)
# plt.bar(ind, Train, width = 0.2, label='Train')
#
# plt.xticks(ind, Env)
# plt.ylabel("Times")
# plt.xlabel("Environment")
# plt.legend(loc="upper right")
#
# plt.show()
#
#


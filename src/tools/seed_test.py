from rlkit.torch.networks import FlattenMlp

import torch
import numpy as np
import joblib
import os

# seed = 666666
#
#
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)
#
#
# obs_dim = 21
# action_dim = 12
# M = 256
# qf1 = FlattenMlp(
#     input_size=obs_dim + action_dim,
#     output_size=1,
#     hidden_sizes=[M, M, M],
# )
#
#
# wqf = qf1.state_dict()
#
# print(wqf['fc0.weight'])


# m = joblib.load('../data/m.pkl')
# bm = joblib.load('../data/nm.pkl')
# wp = m.policy.state_dict()
# wbp = bm.policy.state_dict()
# for k in wp.keys():
#     print(k, torch.all(torch.eq(wp[k],wbp[k])))

# m = joblib.load('/home/syslot/DevSpace/WALLE/src/data/laikago-pl-large-exp/laikago_pl_large_exp_2019_07_14_14_30_19_0000--s-0/params.pkl')

fold1 = '/home/syslot/DevSpace/WALLE/src/data/laikago-pl-large-exp/laikago_pl_large_exp_2019_07_14_23_47_01_0000--s-0'
fold2 = '/home/syslot/DevSpace/WALLE/src/data/laikago-pl-large-exp/laikago_pl_large_exp_2019_07_14_23_47_09_0000--s-0'

m = joblib.load(os.path.join(fold1,'params.pkl'))
bm = joblib.load(os.path.join(fold2,'params.pkl'))
wp = m['trainer/policy'].state_dict()
wbp = bm['trainer/policy'].state_dict()

for k in wp.keys():
    print(k, torch.all(torch.eq(wp[k],wbp[k])))


buf_m = joblib.load(os.path.join(fold1, 'buffer'))
buf_bm = joblib.load(os.path.join(fold2, 'buffer'))
#
obs1 = buf_m['obs']
obs2 = buf_bm['obs']

print(np.all(np.equal(obs1, obs2)))
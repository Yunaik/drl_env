import json

# profile_name = '../gpu_1000.json_0.json'
# profile_name = '/home/syslot/DevSpace/Data/cpu_10.json_0.json'
# profile_name = '/home/syslot/DevSpace/Data/3.json'
# profile_name = '/tmp/32.json_0.json'
profile_name = '/tmp/1.json_0.json'
with open(profile_name, 'r') as f:
	p = json.load(f)

evDict = p['traceEvents']

def cal_profile(func_name):
	f_arr = [r for r in evDict if r['name'].startswith(func_name)]

	# get func anchor


	begin,end = {}, {}
	for x in f_arr:
		if x['ph'] == 'B':
			begin[x['name']] = x
		else:
			end[x['name']] = x

	sum = 0
	for k in begin.keys():
		sum += end[k]['ts'] - begin[k]['ts']

	print("{} cost all times {} us, ave {} us".format(func_name, sum, sum/len(begin)))

def cal_profile1(func_name):
	f_arr = [r for r in evDict if r['name'].startswith(func_name)]

	# get func anchor

	sum = 0

	for x in f_arr:
		if x['ph'] == 'B':
			cur = x['ts']
		else:
			sum += x['ts'] - cur

	print("{} cost all times {} ns, ave {} ns".format(func_name, sum, sum / len(f_arr)*2))


cal_profile('GpuDynamics.DMABackBodies.Sync')
# cal_profile('GpuDynamics.Solve')
cal_profile('PhysX_simulate_fetchResults')
# cal_profile('Basic.simulate')
# cal_profile('Sim.narrowPhase')

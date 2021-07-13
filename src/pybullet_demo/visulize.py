# import rocksdb
import pprint as pp
import json
import numpy as np
from matplotlib import pyplot as plt
from labellines import labelLine, labelLines
from scipy.stats import loglaplace, chi2

def simulate_plot():
    db = rocksdb.DB("fyp.db", rocksdb.Options(create_if_missing=True))

    keylen = 36
    u = 40
    loop = 1000//u*100

    def get_all_keys():
        it = db.iterkeys()
        it.seek_to_first()
        uuids = list(it)
        return uuids

    def get_uuid(uuids):
        return list(filter(lambda s: len(s)==keylen, uuids))

    def generate_uuids(uuid):
        return (uuid + str.encode(str(i * u)) for i in range(1,loop))

    data = {}
    uuids = get_all_keys()
    uuid_list = get_uuid(uuids)
    for uuid in uuid_list:
        pos = []
        vel = []
        p = json.loads(db.get(uuid+b'00'))[0]
        pos.append(p['Pos'][2])
        vel.append(p['Vel'][2])
        for key in generate_uuids(uuid):
            p = json.loads(db.get(key))[0]
            pos.append(p['Pos'][2])
            vel.append(p['Vel'][2])
        data[uuid] = [pos, vel]

    plt.figure()

    plt.xlabel("Time")
    plt.ylabel("Height")

    x = np.arange(0,100,0.04)

    solver = ['bullet', 'pgs', 'tgs']
    d =0
    for v in data.values():
        plt.plot(x, v[0], label =solver[d])
        d+= 1
    plt.title("Position Z")
    plt.show()

    plt.figure()
    d=0
    plt.title("Velocity Z")
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    for v in data.values():
        plt.plot(x, v[1], label = solver[d])
    plt.show()

def efficient_plot(rfile, flag=False):
    with open(rfile) as f:
        perf = json.load(f)

    sample_freq = 25
    get_rs = lambda rst, t, g: dict([(r['instance'],r['simRate']) for r in rst if r['physx'] == t and r['gpu'] == g])
    b3rs = get_rs(perf, False, False)
    pxCpurs = get_rs(perf, True, False)
    pxGpurs = get_rs(perf, True, True)

    s = lambda arr, l: np.append(arr, np.array([np.arange(arr[0][-1]+1, l+1), np.repeat(arr[1][-1], l - len(arr[0]))]), axis=1)

    d2np = lambda d: np.array([list(d.keys()), list(d.values())])
    b3 = d2np(b3rs)
    cpu = d2np(pxCpurs)
    gpu = d2np(pxGpurs)


    if not flag:
        X = np.linspace(1, 15, 15)
        gpu = gpu[:, :15]
    else:
        X = np.linspace(1,100,100)

    b3 = s(b3, len(X))
    cpu= s(cpu, len(X))
    gpu= s(gpu, len(X))
    ax = plt.subplot(111)

    ax.grid()
    ax.set_ylabel('RTF')
    ax.set_xlabel('Parallel Robots')

    Y=np.ones_like(X)
    ax.plot(X, Y, label='real')
    ax.plot(b3[0], b3[1]/X, label = 'bullet')
    ax.plot(cpu[0], cpu[1]/X, label = 'cpu')
    # ax2 = ax.twinx()
    # ax2.set_ylabel('Real Sample(/25)')

    if flag:
        ax.plot(gpu[0], gpu[1]*10, label = 'gpu')
    else:
        ax.plot(gpu[0], gpu[1], label = 'gpu')
    labelLines(plt.gca().get_lines(), zorder=2.5)
    # plt.show()
    plt.savefig('RTF_{}.pdf'.format('origin' if flag == False else 'fixed'))

def performance_plot(rfile, flag=False):
    with open(rfile) as f:
        perf = json.load(f)

    sample_freq = 25
    get_rs = lambda rst, t, g: dict([(r['instance'], r['simRate'] * sample_freq) for r in rst if r['physx'] == t and r['gpu'] == g])
    b3rs = get_rs(perf, False, False)
    pxCpurs = get_rs(perf, True, False)
    pxGpurs = get_rs(perf, True, True)

    s = lambda arr, l: np.append(arr,
                                 np.array([np.arange(arr[0][-1] + 1, l + 1), np.repeat(arr[1][-1], l - len(arr[0]))]),
                                 axis=1)

    d2np = lambda d: np.array([list(d.keys()), list(d.values())])
    b3 = d2np(b3rs)
    cpu = d2np(pxCpurs)
    gpu = d2np(pxGpurs)

    if not flag:
        gpu = gpu[:, :14]
    else:
        # b3 = np.repeat(b3[])
        b3 = np.append(b3[:, 1:13:2], b3[:,13:],axis=1)
        b3 = np.append(b3, np.array([4500, 6]).reshape((2, 1)), axis=1)

    ax = plt.subplot(111)
    ax.grid()

    ax.set_ylabel('Simulation Sample')
    ax.set_xlabel('Parallel Robots')


    if not flag:
        ax.plot(cpu[0], cpu[1], label='cpu')
        ax.plot(gpu[0], gpu[1], label='gpu')
    else:
        ax.plot(gpu[0], gpu[1]*10, label='real_gpu')
    ax.plot(b3[0], b3[1], label='bullet')
    labelLines(plt.gca().get_lines(), zorder=2.5)
    plt.savefig('perf.pdf')
    # plt.show()


if __name__ == "__main__":

    # simulate_plot()
    # efficient_plot('efficient/profile.json')
    efficient_plot('efficient/profile.json', True)
    # performance_plot('bench/profile.json')
    # performance_plot('bench/profile.json', True)

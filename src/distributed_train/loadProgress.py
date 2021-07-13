import pandas as pd
import matplotlib.pyplot as plt
import time
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import json

def multipage(filename, figs=None):
    print("SAVING TO: %s" % filename)
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
        # print("All fig names")
        # print(plt.get_fignums())
    for fig in figs:
        fig.savefig(pp, format='pdf',bbox_inches='tight')
    pp.close()


def loadProgress(fpath=None, saveFile=False):
    # folder_name = "2020-10-05_13-40-17"
    # input = pd.read_csv("../data/ppo_spinup/%s/progress.txt" % folder_name, sep="\t")
    fpath = "/home/kai/pc_scp/pc3/2020-10-07_14-45-36" if fpath is None else fpath
    input = pd.read_csv("%s/progress.txt"%fpath, sep="\t")

    idx_for_best_epoch = []
    test_value_at_best_epoch = []
    train_value_at_best_epoch = []
    for epoch in input.BestEpoch:
        epoch = int(epoch)
        if epoch != 0:
            # find corresponding idx in t
            idx = [idx for idx, val in enumerate(input.Epoch) if epoch == int(val)]
            idx = idx[0]
            idx_for_best_epoch.append(idx)
            test_value_at_best_epoch.append(np.array(input.TestEpRet)[idx])
            train_value_at_best_epoch.append(np.array(input.AverageEpRet)[idx])
        else:
            idx_for_best_epoch.append(0)
            test_value_at_best_epoch.append(0)
            train_value_at_best_epoch.append(0)
            
    t = np.arange(len(input.AverageEpRet)+1)
    t = input.Epoch
    
    plt.figure()
    plt.title('Training returns')
    plt.plot(t, input.AverageEpRet, label='Average return', alpha=1.0)
    plt.plot(t, input.MinEpRet, label='Min return', alpha=0.3)
    plt.plot(t, input.MaxEpRet, label='Max Return', alpha=0.3)
    plt.fill_between(t, input.AverageEpRet+input.StdEpRet, input.AverageEpRet-input.StdEpRet, facecolor='blue', alpha=0.1)
    plt.plot(input.BestEpoch, train_value_at_best_epoch, label="Best train values", marker=",")
    plt.legend(loc='best', bbox_to_anchor=(1,1))

    plt.figure()
    plt.title('Test returns')
    plt.plot(t, input.TestEpRet, label='Average test return', alpha=1.0)
    plt.plot(input.BestEpoch, test_value_at_best_epoch, label="Best test values", marker=",")
    plt.legend(loc='best', bbox_to_anchor=(1,1))
    if saveFile:
        multipage(fpath+"/figures.pdf")
    else:
        plt.show()

if __name__ == "__main__":
    loadProgress()        

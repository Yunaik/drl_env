import torch
import numpy as np
import os

def getParameters(model):
    names = []
    params = []
    for name, param in model.named_parameters():
        names.append(name)
        params.append(param.detach().numpy())

    # print("Names: ", names)
    # print("Params: ", params)
    return names, params

def save_weights(fpath):
    ac = torch.load('%s/pyt_save/model.pt'%fpath)

    names, weights = getParameters(ac.pi.mu_net)
    # obs = [
    #     0.0,0.,0.,
    #     0.,0,-1,
    #     0,0,0,
    #     -0.447352,   1.14171, -0.647576, 
    #     -0.445188,   1.13344, -0.641485
    #     ]
    # action = ac.pi.mu_net(torch.as_tensor(obs, dtype=torch.float32))
    # print("Action", action)
    save_name = ["target_pi0.csv",
                "target_pi0_b.csv",
                "target_pi1.csv",
                "target_pi1_b.csv",
                "target_pi2.csv",
                "target_pi2_b.csv",
                ]
    if not os.path.exists(fpath+"/model_settings/"):
        os.mkdir(fpath+"/model_settings/")
    if not os.path.exists(fpath+"/model_settings/weights/"):
        os.mkdir(fpath+"/model_settings/weights/")
    print("Saving to path: %s" % fpath)
    for idx, weight in enumerate(weights):
        np.savetxt(fpath+"/model_settings/weights/"+save_name[idx], weight.T, delimiter=",")
        print("Saved %s, with shape." % save_name[idx], weight.T.shape)

if __name__ == "__main__":

    fpath = "/home/kai/Desktop/RALpolicies/Valkyrie/1s_240N_2020-10-09_10-04-00_10_margin"

    save_weights(fpath=fpath)
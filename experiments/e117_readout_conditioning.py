"""Matched frozen-feature head optimization: raw versus fit-only whitening."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from e117_serial_event_shd import SerialEventNet, load_items, OUT
from e117_feature_probe import features


def main():
    output = OUT/"readout_conditioning_d8_n128_e8_s6.json"
    if output.exists():
        raise FileExistsError(output)
    checkpoint = OUT/"serial_d8_n128_e8_s6.pt"
    saved = torch.load(checkpoint,weights_only=False,map_location="cpu")
    s = saved["args"]
    net = SerialEventNet(s["bands"],s["dim"],s["depth"],s["groups"],s["beta"])
    net.load_state_dict(saved["state_dict"])
    fit = load_items(s["bands"],s["window"],s["limit"],"fit_spk",s["seed"])
    dev = load_items(s["bands"],s["window"],s["eval_limit"],"val_spk",s["seed"]+1)
    x = features(net,fit,s["bs"])[-1]
    v = features(net,dev,s["bs"])[-1]
    yf, yd = torch.tensor([i[3] for i in fit]),torch.tensor([i[3] for i in dev])
    mean, scale = x.mean(0),np.maximum(x.std(0),1e-4)
    z = (x-mean)/scale
    eig, vectors = np.linalg.eigh(z.T@z/len(z))
    whitener = (vectors*(np.maximum(eig,0)+.1)**-.5)@vectors.T
    result = {"checkpoint":str(checkpoint),"protocol":"Frozen E117 deepest features; same zero initial head, Adam lr .003, batches and 8 epochs; fit-only transform",
              "fit_labels":yf.tolist(),"dev_labels":yd.tolist(),"arms":{}}
    for name,xf,xd in (("raw",x,v),("whitened",z@whitener,((v-mean)/scale)@whitener)):
        xf,xd = torch.tensor(xf,dtype=torch.float32),torch.tensor(xd,dtype=torch.float32)
        head = torch.nn.Linear(xf.shape[-1],20)
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
        optimizer = torch.optim.Adam(head.parameters(),lr=.003)
        rng = np.random.default_rng(8)
        curve = []
        for epoch in range(1,9):
            order = rng.permutation(len(xf))
            for start in range(0,len(xf),4):
                index = order[start:start+4]
                optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(head(xf[index]),yf[index]).backward()
                optimizer.step()
            with torch.no_grad():
                lf,ld = head(xf),head(xd)
                curve.append({"epoch":epoch,"fit_correct":int((lf.argmax(-1)==yf).sum()),
                              "dev_correct":int((ld.argmax(-1)==yd).sum()),
                              "fit_nll":float(F.cross_entropy(lf,yf)),"dev_nll":float(F.cross_entropy(ld,yd))})
        result["arms"][name] = curve
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()

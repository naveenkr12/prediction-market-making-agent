"""Controllable-PnL frontier of simple heuristics (spread/fees vs impact), on the v3 scenario mix."""
import sys, itertools, numpy as np, pandas as pd
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rl_agent.v3.env_v3 import SimAgentDriver, sample_scenario
from simulator.hjb_policy import executable_action
def run(args):
    half, size, lean, oneside, seed = args
    scn=sample_scenario(np.random.default_rng(seed)); d=SimAgentDriver(scn,seed); obs=d.obs(); done=False; tot=np.zeros(3)
    while not done:
        r=d.raw; mid=.5*(r["bid"]+r["ask"]); q=r["inventory"]
        c = mid + lean*obs[7] - 0.0*q
        bs=min(size,max(0,100-q)); as_=min(size,max(0,100+q))
        if oneside and obs[7]>0.2: as_=0
        if oneside and obs[7]<-0.2: bs=0
        a,_=executable_action(c-half,c+half,bs,as_,r)
        _,done,p=d.step(a,0.0); obs=d.obs(); tot+= [p["spread"]-p["fees"], p["impact"], 0]
    return half,size,lean,oneside,scn["group"],tot[0],tot[1]
if __name__=="__main__":
    grid=list(itertools.product([0.01,0.02,0.03,0.05],[10,20],[0.0,0.01,0.03],[False,True]))
    jobs=[(*g,s) for g in grid for s in range(800_000,800_400)]
    from multiprocessing import Pool
    with Pool(2) as p: rows=p.map(run,jobs,chunksize=100)
    df=pd.DataFrame(rows,columns="half size lean oneside group spread_net impact".split()); df["ctrl"]=df.spread_net+df.impact
    t=df.groupby(["half","size","lean","oneside"])[["spread_net","impact","ctrl"]].mean().sort_values("ctrl",ascending=False)
    print(t.round(3).head(15)); print(df.groupby(["group"]).ctrl.describe().round(3))
    print(df[(df.half==0.05)&(df.size==10)&(df.lean==0)&(~df.oneside)].groupby("group")[["spread_net","impact","ctrl"]].mean().round(3))

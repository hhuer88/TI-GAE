"""Portable entry point for TI-GAE training and indicator-subset searches."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'1')
from pathlib import Path
import argparse
import itertools
import json
import random
import numpy as np
import pandas as pd
import torch
from ti_gae_core import DirectedGraphAE, masked_mse_on_flag, choose_k_by_silhouette
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT=Path(__file__).resolve().parent

def load_inputs(config):
    sheets=pd.read_excel(ROOT/config['data_file'],sheet_name=None)
    if list(sheets)!=[str(i) for i in range(1,43)]:
        raise ValueError('Expected worksheets 1 through 42 in chronological order.')
    stations=pd.read_csv(ROOT/config['station_file'])
    features=list(next(iter(sheets.values())).columns[2:])
    if len(features)!=15 or len(stations)!=11:
        raise ValueError('Expected 15 indicators and 11 stations.')
    arrays=[]
    for name,frame in sheets.items():
        if frame.sheet_name.tolist()!=stations['name'].tolist():
            raise ValueError(f'Station order mismatch in worksheet {name}.')
        values=frame[features].apply(pd.to_numeric,errors='raise').to_numpy(np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f'Non-finite indicators in worksheet {name}.')
        arrays.append(values)
    raw=np.stack(arrays)
    edges=pd.read_csv(ROOT/config['topology_file'])
    if edges.shape!=(10,2) or not ((edges.to_numpy()>=0)&(edges.to_numpy()<11)).all():
        raise ValueError('Expected ten directed upstream-to-downstream edges.')
    edge_index=torch.tensor(edges[['upstream','downstream']].to_numpy().T,dtype=torch.long)
    events=pd.read_csv(ROOT/config['event_file'],dtype={'event_id':str})
    if events.event_index.tolist()!=list(range(1,43)):
        raise ValueError('Event catalogue and workbook order differ.')
    return raw,features,list(sheets),edge_index,events

def prepare_subset(raw,indices,std_floor):
    # The retained study inputs use per-station statistics over all 42 events.
    values=raw[:,:,indices].astype(np.float64)
    flags=np.any(values!=0,axis=-1).astype(np.float64)[...,None]
    denominator=np.maximum(flags.sum(axis=0),1.0)
    means=(values*flags).sum(axis=0)/denominator
    stds=np.sqrt((((values-means)*flags)**2).sum(axis=0)/denominator)
    stds=np.where((~np.isfinite(stds))|(stds<std_floor),1.0,stds)
    target=((values-means)/(stds+1e-8)).astype(np.float32)
    flag=flags.astype(np.float32)
    inputs=np.concatenate([target,flag],axis=-1)
    return inputs,target,flag,means,stds

def train_subset(raw,all_features,sheets,edges,events,features,seed,config,output,device,epochs):
    if output.exists():
        raise FileExistsError(f'Output already exists; select a new --output directory: {output}')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    indices=[all_features.index(f) for f in features]
    inputs,target,flags,means,stds=prepare_subset(raw,indices,config['std_floor'])
    x=torch.tensor(inputs,device=device);y=torch.tensor(target,device=device);flag=torch.tensor(flags,device=device)
    edge=edges.to(device)
    model=DirectedGraphAE(len(features)+1,len(features),config['hidden_dim'],config['graph_embed_dim']).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    bce=torch.nn.BCEWithLogitsLoss()
    split=len(raw)-config['validation_last_n'];train_indices=np.arange(split);val_indices=np.arange(split,len(raw))
    history=[]
    def loss_for(index):
        reconstruction,logits=model.decode(model.encode(x[index],edge))
        mse=masked_mse_on_flag(reconstruction,y[index],flag[index])
        binary=bce(logits,flag[index])
        loss=config['cont_loss_weight']*mse+config['flag_loss_weight']*binary
        return loss,mse,binary
    for epoch in range(1,epochs+1):
        model.train();train_loss=[]
        order=np.random.permutation(train_indices) if config['shuffle_training_events'] else train_indices
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            loss,mse,binary=loss_for(int(index));loss.backward();optimizer.step()
            train_loss.append([loss.item(),mse.item(),binary.item()])
        model.eval()
        with torch.no_grad():val_loss=[[v.item() for v in loss_for(int(i))] for i in val_indices]
        history.append([epoch,*np.mean(train_loss,axis=0),*np.mean(val_loss,axis=0)])
    with torch.no_grad():embeddings=np.stack([model.encode(v,edge)[0].cpu().numpy() for v in x])
    best_k,scores=choose_k_by_silhouette(embeddings,config['silhouette_k_min'],config['silhouette_k_max'],seed)
    labels=KMeans(n_clusters=best_k,n_init=10,random_state=seed).fit_predict(embeddings)
    score=float(silhouette_score(embeddings,labels))
    output.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(history,columns=['epoch','train_total','train_masked_mse','train_bce','val_total','val_masked_mse','val_bce']).to_csv(output/'losses.csv',index=False)
    frame=pd.DataFrame(embeddings,columns=[f'embedding_{i}' for i in range(embeddings.shape[1])])
    frame.insert(0,'event_id',events.event_id);frame.insert(0,'event_index',events.event_index);frame['cluster']=labels
    frame.to_csv(output/'embeddings_and_clusters.csv',index=False)
    pd.DataFrame(sorted(scores.items()),columns=['k','silhouette_score']).to_csv(output/'silhouette_scores.csv',index=False)
    torch.save({'model_state_dict':model.state_dict(),'config':config,'seed':seed,'feature_names':features,'train_idx':train_indices.tolist(),'val_idx':val_indices.tolist(),'normalization_mean':means,'normalization_std':stds},output/'model.pt')
    summary={'seed':seed,'features':features,'epochs':epochs,'best_k':best_k,'silhouette_score':score,'train_events':len(train_indices),'validation_events':len(val_indices)}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary),flush=True)
    return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config.json')
    parser.add_argument('--check-inputs',action='store_true')
    parser.add_argument('--all-subsets',action='store_true',help='Search all 32647 subsets of 3 to 15 indicators.')
    parser.add_argument('--seeds',type=int,nargs='+')
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs')
    parser.add_argument('--start',type=int,default=0,help='Zero-based subset offset for a divided search.')
    parser.add_argument('--limit',type=int,help='Maximum number of subsets for this invocation.')
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'))
    raw,features,sheets,edge,events=load_inputs(config)
    if args.check_inputs:
        print(json.dumps({'events':len(raw),'stations':raw.shape[1],'indicators':raw.shape[2],'edges':edge.shape[1],'all_zero_station_events':int(np.all(raw==0,axis=-1).sum())}))
        return
    if args.epochs is not None and args.epochs<1:parser.error('--epochs must be positive')
    if args.start<0 or (args.limit is not None and args.limit<1):parser.error('Invalid subset range')
    if args.all_subsets:
        tasks=(combo for size in config['subset_sizes'] for combo in itertools.combinations(features,size))
    else:tasks=iter([config['selected_features']])
    end=None if args.limit is None else args.start+args.limit
    summaries=[]
    for offset,subset in enumerate(itertools.islice(tasks,args.start,end),start=args.start):
        for seed in (args.seeds if args.seeds is not None else config['seeds']):
            path=args.output/f'seed{seed}'/f'subset_{offset+1:06d}'
            summaries.append(train_subset(raw,features,sheets,edge,events,list(subset),seed,config,path,torch.device(args.device),args.epochs or config['epochs']))
    if summaries:
        args.output.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(summaries).to_csv(args.output/f'summary_start{args.start}_count{len(summaries)}.csv',index=False)

if __name__=='__main__':main()

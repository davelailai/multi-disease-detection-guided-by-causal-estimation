import pickle
import pandas as pd
from itertools import product
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from mmpretrain.structures import label_to_onehot
from sklearn import metrics
import os
path='/users/lailai/sharedscratch/openmmlab/mmpretrain/data/UWF/batch_32'
files = os.listdir('/users/lailai/sharedscratch/openmmlab/mmpretrain/data/UWF/batch_32')
print(len(files))
for annfile in files:
    ann_file= os.path.join(path,annfile)
    ann_file = '/users/lailai/sharedscratch/openmmlab/mmpretrain/data/FFA/test_class6.pkl'
    f= open(ann_file,'rb')
    data = pickle.load(f)
    if data['data'].size(0)==32:
        pass
    else:
        print(annfile)
        print(data['data'].size(0))
data = pd.DataFrame(data)
order = ['ID', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
data['patient_index']=0
# df_pred = []
# df_gt=[]
def ODIR_Metrics(gt_data, pr_data):
    th = 0.5
    gt = gt_data.flatten()
    pr = pr_data.flatten()
    kappa = metrics.cohen_kappa_score(gt, pr>th)
    f1 = metrics.f1_score(gt, pr>th, average='micro')
    auc = metrics.roc_auc_score(gt, pr)
    final_score = (kappa+f1+auc)/3.0
    return kappa, f1, auc, final_score

# pair = []
# for idx,result in data.iterrows():
#     img_path = result['img_path']
#     name_split = img_path.split('/')[-1]
#     patient_index = name_split.split('_')[0]
#     data.loc[idx,'patient_index']=int(patient_index)
# pairs = data.groupby('patient_index').apply(lambda x: list(zip(x.index, x.index[1:]))).explode().tolist()
preds = []
gts = []

for idx,result in data.iterrows():
    preds.append(np.array(data.loc[idx,'pred_score']))
    gts.append(np.array(data.loc[idx,'gt_score']))
# for i in pairs:
#     if np.isnan(i).any():
#         pass
#     else:
#         left_pred =np.array(data.loc[i[0],'pred_score'])>0.5 
#         left_gt = np.array(data.loc[i[0],'gt_score'])
#         right_pred =np.array(data.loc[i[1],'pred_score'])>0.5
#         right_gt = np.array(data.loc[i[1],'gt_score'])
#         pred = np.array((left_pred+right_pred).astype(int))
#         if sum(pred)>1:
#             pred[0]=0
#         gt = left_gt+right_gt
#         gt[gt>1]=1
#         # gt_score = label_to_onehot(gt,8)
#         # mlb = MultiLabelBinarizer()
#         # one_hot_labels = mlb.fit_transform(pred)
#         preds.append(pred)
#         gts.append(gt)

preds = np.stack(preds)
preds = np.transpose(preds,(1,0))
gts = np.stack(gts)
gts = np.transpose(gts,(1,0))
kappa, f1, auc, final_score = ODIR_Metrics(gts, preds)
print("kappa score:", kappa, " f-1 score:", f1, " AUC vlaue:", auc, " Final Score:", final_score)

# for i,item in enumerate(data):

#     img_path=item['img_path']
#     pred_label = item['pred_label']
#     gt_label = item['gt_score']
#     name_split = img_path.split('/')[-1]
#     patient_index = name_split.split('_')[0]
#     if patient_index in pair:
#         if i==0:
#             df_gt=[list(int(patient_index)), list(np.array(gt_label))[:]]
#         else:
#             df_gt[i,:]=[patient_index,gt_label]
#     else:
#         pair.append(patient_index)


# print(data)
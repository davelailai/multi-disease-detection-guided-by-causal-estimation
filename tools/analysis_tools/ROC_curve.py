import json

import matplotlib.pyplot as plt
import os
import numpy as np

file_path = '/users/lailai/sharedscratch/openmmlab/mmpretrain/work_dirs_fundation/UWF/3Class/batch1/ViT_mae_our/20240524_102135/20240524_102135.json'

with open(file_path, "r") as file:
    data = json.load(file)

class_num = len(data['multi-label/tpr_list'])

# class_num = class_num-1

# Define a list of colors for each class
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


mAUC = np.mean(data['multi-label/auc_list'])
print(mAUC)
# Plot ROC curves for each class with different colors
plt.figure(figsize=(8, 6))
for k in range(class_num):
    tpr = data['multi-label/tpr_list'][k]
    fpr = data['multi-label/fpr_list'][k]
    threshold=data['multi-label/threshold_list'][k]
    plt.plot(fpr, tpr, label=f'Class {k}', color=colors[k % len(colors)])  # Use modulo to cycle through colors

# Set plot labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
lw = 2
plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
plt.title('ROC Curve for Multi-label Classification')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(os.path.dirname(file_path),'roc_curve_multilabel.png'))
# Show the plot
# plt.show()

# for k in range(class_num):
#     tpr=data['multi-label/tpr_list'][k]
#     fpr=data['multi-label/fpr_list'][k]
#     threshold=data['multi-label/threshold_list'][k]
# FFA [0.4958974358974359, 0.6324237356846052, 0.5001225251063018, 0.7227500158338083, 0.7508091396550891, 0.7638668166682246,
# 0.5821201170194459, 0.6697957839262187, 0.6234145478142883, 0.768972069162075, 0.8130653913659566, 0.8420776935003982,
# 0.5, 0.5, 0.5, 0.5, 0.5, 0.8236600595063487,
# 0.6149750473240405, 0.6356478666261274, 0.62875119688877, 0.7633542339603522, 0.8186480524290292, 0.83724562325325,
# 0.630917225950783, 0.6619476537954798, 0.6552801513865758, 0.7634302362404205, 0.8222678453536866, 0.8387558466760578,
# 0.6464257442780934, 0.6336272423228945, 0.6568525569174498, 0.7434986382924822, 0.8017358259665396, 0.8321750118464698,
# 0.6283978661159869, 0.6808237052802271, 0.6406089043986514, 0.7631610614985116, 0.8209506787111733, 0.8457041638326473
# ]
# ['Resnet50','Asyloss','GCN','Q2L','Q2L causal','ML Decoder','ML causal']

# OD[0.5482573726541555, 0.7401041666666667, 0.912565445026178, 0.7976867045545506, 0.7310996563573884, 0.959417273673257, 0.5335294117647058,
#     0.7802611302273542, 0.7911458333333332, 0.966259453170448, 0.8641372501934416, 0.6029209621993127, 0.9429947437231517, 0.6501960784313725,
#     0.8034240358024951, 0.7026041666666667, 0.9789412449098314, 0.8121881587022065, 0.6561855670103093, 0.9772939512793831, 0.6937254901960784,
#     0.7813588482404847, 0.8057291666666667, 0.9117510180337406, 0.835040956268844, 0.6154639175257732, 0.9674084153792791, 0.6613725490196078,
#     0.810390323193515, 0.7973958333333333, 0.9732984293193717, 0.8596413991835428, 0.6878006872852234, 0.9249713172710052, 0.6650980392156863,
#     0.7827679381900319, 0.6296875, 0.9503199534613146, 0.835040956268844, 0.7374570446735395, 0.9443688465540703, 0.6703921568627451,
#     0.7991228810875853, 0.7791666666666668, 0.9625945317044795, 0.8975826462819179, 0.7214776632302407, 0.9398729955441714, 0.6758823529411765

# ]
from curses import raw
from tkinter.ttk import Style
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from matplotlib.ticker import MaxNLocator

import pandas as pd
import matplotlib.pyplot as plt



# # Given data
result_dist = {'Dataset': ['OIA-ODIR'] * 56 + ['LID-FFA'] * 48,
               'Model': ['Resnet50'] * 7 + ['Asyloss'] * 7 + ['GCN'] * 7 + ['dyGCN']*7 + ['Q2L'] * 7 + ['Q2L causal'] * 7 + ['ML Decoder'] * 7 + ['ML causal'] * 7 +
                        ['Resnet50'] * 6 + ['Asyloss'] * 6 + ['GCN'] * 6 + ['dyGCN']*6 + ['Q2L'] * 6 + ['Q2L causal'] * 6 + ['ML Decoder'] * 6 + ['ML causal'] * 6,
               'AUC': [0.5482573726541555, 0.7401041666666667, 0.912565445026178, 0.7976867045545506, 0.7310996563573884, 0.959417273673257, 0.5335294117647058,
                         0.7802611302273542, 0.7911458333333332, 0.966259453170448, 0.8641372501934416, 0.6029209621993127, 0.9429947437231517, 0.6501960784313725,
                         0.7703447256760466, 0.7, 0.922803955788249, 0.905160223058246, 0.6101374570446735, 0.9632460844739721, 0.6235294117647059,
                         0.8034240358024951, 0.7026041666666667, 0.9789412449098314, 0.8121881587022065, 0.6561855670103093, 0.9772939512793831, 0.6937254901960784,
                         0.7813588482404847, 0.8057291666666667, 0.9117510180337406, 0.835040956268844, 0.6154639175257732, 0.9674084153792791, 0.6613725490196078,
                         0.810390323193515, 0.7973958333333333, 0.9732984293193717, 0.8596413991835428, 0.6878006872852234, 0.9249713172710052, 0.6650980392156863,
                         0.7827679381900319, 0.6296875, 0.9503199534613146, 0.835040956268844, 0.7374570446735395, 0.9443688465540703, 0.6703921568627451,
                         0.7991228810875853, 0.7791666666666668, 0.9625945317044795, 0.8975826462819179, 0.7214776632302407, 0.9398729955441714, 0.6758823529411765,

                         0.4995069033530572, 0.6254963413500484, 0.4963013965731556, 0.7767784341459356, 0.8068296194697209, 0.8117318197336973,
                         0.7236594651621485, 0.6229797745049866, 0.7371980794072485, 0.8045428072218986, 0.8398193735051571, 0.8879096961420279,
                         0.6195243337461585, 0.6288130303714426, 0.6765092983012015, 0.7895665196771778, 0.8259090180182946, 0.8672684078752704,
                         0.7484977753314067, 0.6477702819351668, 0.6827122279363378, 0.8030618187869207, 0.8341948405102851, 0.881330374416752,
                         0.723086096968029, 0.6768303209784731, 0.680305393745887, 0.8070887761045012, 0.8479943089347676, 0.8687905428473881,
                         0.7364799779826614, 0.6824278562037689, 0.7252918667284116, 0.8071137365837424, 0.8296672364555718, 0.8778095481961989,
                         0.7154029631668273, 0.6768336409875035, 0.7021801652489703, 0.8107496463932108, 0.8328727255357551, 0.8723824968703767,
                         0.7463419109215175, 0.6917305215070186, 0.7157803017377952, 0.7989849405108579, 0.8466687820506259, 0.8818851712757483],
               'Disease': ['D', 'G', 'C', 'A', 'H', 'M', 'O'] * 8 + ['L', 'TP', 'ST', 'SH', 'NP', 'VA'] * 8}

# Create DataFrame
all_data = pd.DataFrame(result_dist)

unique_datasets = all_data['Dataset'].unique()

# Iterate over unique datasets and draw separate bar plots
for dataset in unique_datasets:
    data_subset = all_data[all_data['Dataset'] == dataset]
    plt.figure(figsize=(12,6))
    ax = sns.barplot(data=data_subset, x='Disease', y='AUC', hue='Model',palette='Paired')
    plt.title(f'AUC Analysis for {dataset}', fontsize=22)
    plt.xlabel('Disease', fontsize=22)
    plt.ylabel('AUC', fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.set_ylim([0.4, 1]) 
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18,title_fontsize='22')
    plt.tight_layout()
    plt.savefig(f'work_dirs_MICCAI_new/AUC_analysis_{dataset}.png', dpi=400, bbox_inches='tight')

# # ### radar chart
# # # Create DataFrame
# datasets = list(set(result_dist['Dataset']))

# # Define function to plot radar chart
# def plot_radar_chart(dataset):
#     # Filter data for the given dataset
#     indices = [i for i, ds in enumerate(result_dist['Dataset']) if ds == dataset]
#     model_data = [result_dist['Model'][i] for i in indices]
#     auc_data = [result_dist['AUC'][i] for i in indices]

#     # Get unique models
#     unique_models = list(set(model_data))

#     # Initialize figure and axis
#     fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

#     # Create a color palette
#     colors = plt.cm.viridis(np.linspace(0, 1, len(unique_models)))

#     # Plot radar chart for each model
#     for i, model in enumerate(unique_models):
#         model_indices = [j for j, m in enumerate(model_data) if m == model]
#         auc_values = [auc_data[j] for j in model_indices]

#         # Wrap around to complete the circle
#         auc_values += auc_values[:1]
#         ax.plot(np.linspace(0, 2 * np.pi, len(auc_values)), auc_values, label=model, color=colors[i])

#     # Add legend
#     ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

#     # Set title
#     ax.set_title(f'AUC Analysis for {dataset}')

#     ax.set_ylim([0.4, 0.85])

#     # Save plot
#     plt.savefig(f'work_dirs_MICCAI_new/AUC_analysis_{dataset}_radar_chart.png', bbox_inches='tight')

# # Plot radar chart for each dataset and save
# for dataset in datasets:
#     plot_radar_chart(dataset)


result_dist = {'Dataset': ['OIA-ODIR'] * 4 + ['LID-FFA'] * 4+['OIA-ODIR'] * 4 + ['LID-FFA'] * 4,
               'Model': ['Q2L causal'] * 8 +['ML causal'] * 8,
               'mAUC': [0.8095, 0.8169, 0.8045, 0.7832,
                            0.7621, 0.7765, 0.7703, 0.7625,
                            0.8000, 0.8251, 0.8058, 0.7891,
                            0.7551, 0.7802, 0.7707, 0.7648],
               'Channel': ['10', '30', '50', '100']*4}
all_data = pd.DataFrame(result_dist)

unique_Dataset = all_data['Dataset'].unique()
markers = {'Q2L causal': 'o', 'ML causal': '*'}
for dataset in unique_Dataset:
    plt.figure(figsize=(7, 3))
    model_data = all_data[all_data['Dataset'] == dataset]
    sns.lineplot(data=model_data, x='Channel', y='mAUC', hue='Model',palette='Set2',markers=['o', '*'])
    plt.title(f'Channel Analysis in {dataset}', fontsize=16)
    plt.xlabel('num of channel', fontsize=14)
    plt.ylabel('mAUC', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12,title_fontsize='12')
    plt.tight_layout()
    plt.savefig(f'work_dirs_MICCAI_new/channel_analysis_{dataset}.png', dpi=400, bbox_inches='tight')
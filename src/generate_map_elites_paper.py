import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast

# from modular_light_chaser import ModularLightChaser
# from modular_carrier import ModularCarrier
# from PIL import Image
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

EPISODES = 4

def save_heatmap_elites(folder_path, features1, loss1, features2, loss2):
  index_valid_1 = [i for i,t in enumerate(features1) if (t[2]==8 or t[2]==18 or t[2]==25)]
  index_valid_2 = [i for i,t in enumerate(features2) if (t[2]==8 or t[2]==18 or t[2]==25)]

  # features_valid_1 = features1[index_valid_1]
  # features_valid_2 = features2[index_valid_2]

  features_valid_1 = [features1[i] for i in index_valid_1]
  features_valid_2 = [features2[i] for i in index_valid_2]

  unique_modules_1 = list(np.unique([t[2] for t in features_valid_1 if (t[2]==8 or t[2]==18 or t[2]==25)]))
  unique_module_n_1 = len(unique_modules_1)
  loss_valid_1 = np.asarray(loss1[index_valid_1])

  unique_modules_2 = list(np.unique([t[2] for t in features_valid_2 if (t[2]==8 or t[2]==18 or t[2]==25)]))
  unique_module_n_2 = len(unique_modules_2)
  loss_valid_2 = np.asarray(loss2[index_valid_2])

  features_per_module_n_1 = [[] for _ in range(unique_module_n_1)]
  loss_per_module_n_1 = [[] for _ in range(unique_module_n_1)]

  features_per_module_n_2 = [[] for _ in range(unique_module_n_2)]
  loss_per_module_n_2 = [[] for _ in range(unique_module_n_2)]

  for f,l in zip(features_valid_1, loss_valid_1):
    features_per_module_n_1[unique_modules_1.index(f[2])].append((f[0], f[1]))
    loss_per_module_n_1[unique_modules_1.index(f[2])].append(l)

  for f,l in zip(features_valid_2, loss_valid_2):
    features_per_module_n_2[unique_modules_2.index(f[2])].append((f[0], f[1]))
    loss_per_module_n_2[unique_modules_2.index(f[2])].append(l)

  max_loss = max(max(abs(loss_valid_1)), max(abs(loss_valid_2)))
  min_loss = min(min(abs(loss_valid_1)), min(abs(loss_valid_2)))

  rows = 3
  cols = 2
  fig = matplotlib.pyplot.gcf()
  fig.set_size_inches(10, 8)
  for i in range(rows*cols):
    plt.subplot(rows,cols,i+1)
    ii = i // 2
    unique_modules_temp = unique_modules_1 if i%2==0 else unique_modules_2
    features_per_module_n_temp = features_per_module_n_1 if i%2==0 else features_per_module_n_2
    loss_per_module_n_temp = loss_per_module_n_1 if i%2==0 else loss_per_module_n_2

    plt.title("Modules: "+str(unique_modules_temp[ii]), fontsize=16)

    heatmap_size =  unique_modules_temp[ii]-1# max([max(t[0]+1, t[1]) for t in features_per_module_n_temp[ii]])

    heatmap = np.full((heatmap_size-1, heatmap_size-1), np.nan)
    for j,f in enumerate(features_per_module_n_temp[ii]):
      heatmap[-1*f[1]+1, f[0]-1] = -loss_per_module_n_temp[ii][j]

    plt.imshow(heatmap, extent=[0.5,heatmap_size-0.5,1.5,heatmap_size+0.5],
               cmap="cividis", vmax=max_loss, vmin=min_loss)

    plt.xticks(list(range(1,heatmap_size, (heatmap_size//8)+1)), fontsize=10)
    plt.yticks(list(range(2,heatmap_size+1, (heatmap_size//8)+1)), fontsize=10)

    plt.xlabel("Sensors", fontsize=12)
    plt.ylabel("Actuators", fontsize=12)

  fig.subplots_adjust(right=0.8, wspace=0.05, hspace=0.6)
  cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

  #plt.colorbar(cax=cbar_ax, ticks=np.linspace(min_loss, max_loss, 10))
  # plt.colorbar(cax=cbar_ax)

  cbar = plt.colorbar(cax=cbar_ax)
  cbar.set_label('Fitness', rotation=270)

  fig.savefig(os.path.join(folder_path, "elites_paper.png"), format='png', dpi=300)
  fig.savefig(os.path.join(folder_path, "elites_paper.svg"), format='svg')

  plt.close("all")

def load_save_elites(folder_path_1, folder_path_2):
  csv_file_path_1 = os.path.join(folder_path_1, "elites.csv")
  csv_file_path_2 = os.path.join(folder_path_2, "elites.csv")

  elites_1 = pd.read_csv(csv_file_path_1, delimiter=";")
  elites_2 = pd.read_csv(csv_file_path_2, delimiter=";")
  df_map_elites_features_1 = elites_1["map_elites_features"]
  df_map_elites_loss_1 = elites_1["map_elites_loss"]

  df_map_elites_features_2 = elites_2["map_elites_features"]
  df_map_elites_loss_2 = elites_2["map_elites_loss"]

  map_elites_features_1 = [ast.literal_eval(t) for t in df_map_elites_features_1]
  map_elites_features_2 = [ast.literal_eval(t) for t in df_map_elites_features_2]

  save_heatmap_elites(folder_path_2, map_elites_features_1, df_map_elites_loss_1,
                      map_elites_features_2, df_map_elites_loss_2)

if __name__ == "__main__":
  folder_path_1 = os.path.join("logs", "train_modular_light_chaser_es", "20220114-133249")
  folder_path_2 = os.path.join("logs", "train_modular_light_chaser_cmame", "20220114-133647")

  load_save_elites(folder_path_1, folder_path_2)


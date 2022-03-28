import numpy as np
import os
import csv
# from operator import itemgetter
# import matplotlib.pyplot as plt
from generate_map_elites import save_heatmap_elites

class LogMapElites:
  def __init__(self):
    self.map_elites = []
    self.map_elites_features = []
    self.map_elites_loss = []
    self.map_elites_body = []

  def logging(self, solutions, loss, features, body):
    for i, feat in enumerate(features):
      if feat:
        if not feat in self.map_elites_features:
          self.map_elites.append(np.array(solutions[i]))
          self.map_elites_features.append(feat)
          self.map_elites_loss.append(loss[i])
          self.map_elites_body.append(body[i])

        else:
          elite_idx = self.map_elites_features.index(feat)
          if self.map_elites_loss[elite_idx] > loss[i]:
            self.map_elites[elite_idx] = solutions[i]
            self.map_elites_loss[elite_idx] = loss[i]
            self.map_elites_body[elite_idx] = body[i]


  def result_pretty(self):
    for i in range(len(self.map_elites_loss)):
      print("Feature:", self.map_elites_features[i], "| Loss:",
            self.map_elites_loss[i])

  def save_elites(self, folder_path, g=None):
    if len(self.map_elites_loss) > 0:
      elite_folder = os.path.join(folder_path, "elites")
      if not os.path.exists(elite_folder):
        os.makedirs(elite_folder)

      # csv_file_path = os.path.join(elite_folder, "elites.csv")
      if g is None:
        csv_file_path = os.path.join(folder_path, "elites.csv")
      else:
        csv_file_path = os.path.join(elite_folder, "%06d_elites.csv"%(g))
      with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(["map_elites_features", "map_elites_loss",
                         "map_elites_body", "map_elites"])
        for i in range(len(self.map_elites)):
          writer.writerow([self.map_elites_features[i],
                           self.map_elites_loss[i],
                           self.map_elites_body[i],
                           list(self.map_elites[i])])

      # save_heatmap_elites(folder_path, features, loss, g=None)
      print("SAVING ELITES")
      save_heatmap_elites(elite_folder, self.map_elites_features,
                          self.map_elites_loss, g=g)
      # heatmap_size = max([max(t[0]+1, t[1]) for t in self.map_elites_features])
      # heatmap = np.full((heatmap_size-1, heatmap_size-1), np.nan)
      # for i,f in enumerate(self.map_elites_features):
      #   heatmap[-1*f[1]+1, f[0]-1] = -self.map_elites_loss[i]

      # plt.imshow(heatmap, extent=[0.5,heatmap_size-0.5,1.5,heatmap_size+0.5],
      #            cmap="cividis")
      # plt.colorbar()
      # plt.xlabel("Number of sensors")
      # plt.ylabel("Number of actuators")
      # plt.tight_layout()
      # plt.savefig(os.path.join(elite_folder, "%06d_elites.png"%(g)), format='png', dpi=300)
      # # plt.savefig(os.path.join(elite_folder, "elites.png"), format='png', dpi=300)
      # plt.close("all")
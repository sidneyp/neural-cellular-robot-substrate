"""
MAP-Elites with CMA-ES.

Based on:
  Fontaine, Matthew C., et al. "Covariance matrix adaptation for the rapid
  illumination of behavior space." Proceedings of the 2020 genetic and
  evolutionary computation conference. 2020.

Format of methods based on: https://github.com/CMA-ES/pycma
"""
import numpy as np
import os
import cma
import csv
# from operator import itemgetter
# import matplotlib.pyplot as plt
from generate_map_elites import save_heatmap_elites

class CMAME:
  def __init__(self, init_solution, std, emitters_n=15):
    self.solutions = None
    self.map_elites = []
    self.map_elites_features = []
    self.map_elites_loss = []
    self.map_elites_body = []
    self.emitters_n = emitters_n
    self.std = std
    self.emitters = [cma.CMAEvolutionStrategy(init_solution, self.std) for _ in
                     range(self.emitters_n)]
    #self.emitters_loss = [np.inf for _ in range(self.emitters_n)]
    self.emitters_stagnation = [0 for _ in range(self.emitters_n)]
    self.current_emitter_idx = -1

  def ask(self, number=None):
    self.current_emitter_idx = (self.current_emitter_idx+1) % self.emitters_n

    if number:
      self.solutions = self.emitters[self.current_emitter_idx].ask(number=number)
    else:
      self.solutions = self.emitters[self.current_emitter_idx].ask()

    return self.solutions

  def tell(self, loss, features, body):
    emitter_stagnated = True

    for i, feat in enumerate(features):
      if feat:
        if not feat in self.map_elites_features:
          self.map_elites.append(np.array(self.solutions[i]))
          self.map_elites_features.append(feat)
          self.map_elites_loss.append(loss[i])
          self.map_elites_body.append(body[i])
          emitter_stagnated = False
        else:
          elite_idx = self.map_elites_features.index(feat)
          if self.map_elites_loss[elite_idx] > loss[i]:
            self.map_elites[elite_idx] = self.solutions[i]
            self.map_elites_loss[elite_idx] = loss[i]
            self.map_elites_body[elite_idx] = body[i]
            emitter_stagnated = False

    reset_emitter = False
    if emitter_stagnated:
      self.emitters_stagnation[self.current_emitter_idx] += 1
      if self.emitters_stagnation[self.current_emitter_idx] > 5:
        reset_emitter = True
    else:
      self.emitters_stagnation[self.current_emitter_idx] = 0

    if reset_emitter and len(self.map_elites) > self.emitters_n:
      random_elite = self.map_elites[np.random.randint(len(self.map_elites))]
      self.emitters[self.current_emitter_idx] = cma.CMAEvolutionStrategy(
        random_elite, self.std)
      self.emitters_stagnation[self.current_emitter_idx] = 0
    else:
      self.emitters[self.current_emitter_idx].tell(self.solutions, loss)

    print("self.emitters_stagnation", self.emitters_stagnation, len(self.map_elites), self.emitters_n)

  def disp(self):
    print("Emitter:", self.current_emitter_idx)
    self.emitters[self.current_emitter_idx].disp()

  def result_pretty(self):
    for i in range(len(self.map_elites_loss)):
      print("Feature:", self.map_elites_features[i], "| Loss:",
            self.map_elites_loss[i])
      #self.emitters[i].result_pretty()

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
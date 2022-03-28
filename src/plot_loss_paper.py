import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def plot_loss(path_es, path_cmame):
  # Visualization colors
  GREEN = '#84db81'
  # GOLD = '#bf9c34'
  YELLOW = '#ffd000'
  alpha = 0.5

  fitness_es = np.genfromtxt(os.path.join(path_es, "loss_history.csv"),
                       delimiter=';')

  fitness_cmame = np.genfromtxt(os.path.join(path_cmame, "loss_history.csv"),
                       delimiter=';')

  fitness_es *= -1
  fitness_cmame *= -1

  x = np.arange(fitness_cmame.shape[0])

  # Fitness plot
  average_fitness_es = np.mean(fitness_es, axis=1)
  max_fitness_es = np.max(fitness_es, axis=1)

  average_fitness_cmame = np.mean(fitness_cmame, axis=1)
  max_fitness_cmame = np.max(fitness_cmame, axis=1)

  plt.suptitle('Fitness')
  #plt.ylim(-10.1, 1.1)
  # plt.yscale('log')
  plt.ylabel('fitness')
  plt.xlabel('generation')
  # for generation, score in enumerate(fitness):
  #   plt.scatter(np.full(len(score), generation), score, color='b', s=0.1, alpha=alpha)
  # plt.plot(x[:len(average_fitness_es)], average_fitness_es, YELLOW, label="Average")
  plt.plot(x[:len(average_fitness_es)], max_fitness_es, YELLOW, label="CMA-ES", alpha=alpha)

  # plt.plot(x, average_fitness_cmame, YELLOW, label="Average")
  plt.plot(x, max_fitness_cmame, GREEN, label="CMA-ME", alpha=alpha)

  plt.legend()
  plt.tight_layout()
  plt.savefig(os.path.join(path_es, "loss_history_paper.png"), format='png', dpi=300)
  plt.close("all")


if __name__ == "__main__":
  path_env1_es = os.path.join("logs", "train_modular_light_chaser_es", "20220114-133249")
  path_env1_cmame = os.path.join("logs", "train_modular_light_chaser_cmame", "20220114-133647")
  path_env2_es = os.path.join("logs", "train_modular_light_chaser_es", "20220117-145958")
  path_env2_cmame = os.path.join("logs", "train_modular_light_chaser_cmame", "20220120-141506")
  path_env3_es = os.path.join("logs", "train_modular_carrier_es", "20220120-154403")
  path_env3_cmame = os.path.join("logs", "train_modular_carrier_cmame", "20220121-113906")

  # plot_loss(path_env1_es, path_env1_cmame)
  # plot_loss(path_env2_es, path_env2_cmame)
  plot_loss(path_env3_es, path_env3_cmame)

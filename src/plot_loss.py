import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def plot_loss(folder_path):
  # Visualization colors
  GREEN = '#84db81'
  # GOLD = '#bf9c34'
  YELLOW = '#ffd000'
  alpha = 0.5

  fitness = np.genfromtxt(os.path.join(folder_path, "loss_history.csv"),
                       delimiter=';')

  fitness *= -1

  x = np.arange(fitness.shape[0])

  # Fitness plot
  average_fitness = np.mean(fitness, axis=1)
  max_fitness = np.max(fitness, axis=1)

  plt.suptitle('Fitness')
  #plt.ylim(-10.1, 1.1)
  # plt.yscale('log')
  plt.ylabel('fitness')
  plt.xlabel('generation')
  for generation, score in enumerate(fitness):
    plt.scatter(np.full(len(score), generation), score, color='b', s=0.1, alpha=alpha)
  plt.plot(x, average_fitness, YELLOW, label="Average")
  plt.plot(x, max_fitness, GREEN, label="Maximum")

  plt.legend()
  plt.tight_layout()
  plt.savefig(os.path.join(folder_path, "loss_history.png"), format='png', dpi=300)
  plt.close("all")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--logdir", help="path to log directory")
  p_args = parser.parse_args()

  plot_loss(p_args.logdir)

"""
Train NCA for modular light chaser with MAP-Elites with CMA-ES.
"""
import numpy as np
from body_brain_nca import BodyBrainNCA
import cma_me
import utils
import datetime
import os
import multiprocessing as mp
import csv
from evaluate_nca_modular_light_chaser import evaluate_nca
from plot_loss import plot_loss

def train(args):
  nca = BodyBrainNCA(ca_steps_build_body=args.ca_steps_build_body)
  args.nca_model = nca.get_dict_args()
  args.save_json()

  nca.dmodel.summary()
  weight_shape_list, weight_amount_list, weight_amount = utils.get_weights_info(nca.weights)

  if args.retrain:
    checkpoint_filename = "checkpoint"
    with open(os.path.join(args.retrain, checkpoint_filename), "r") as f:
      first_line = f.readline()
      start_idx = first_line.find(": ")
      ckpt_filename = first_line[start_idx+3:-2]

    print("Retraining model...")
    nca.load_weights(os.path.join(args.log_dir, ckpt_filename))
    init_sol = utils.get_flat_weights(nca.weights)
  else:
    init_sol = int(weight_amount.numpy())*[0.]

  mees = cma_me.CMAME(init_sol, 0.01, emitters_n=args.emitters)
  loss_best = np.inf

  history_folder_path = os.path.join(args.log_dir, "history")
  os.makedirs(history_folder_path)

  import time
  t1 = time.time()
  pool = None
  if args.threads > 1:
    mp.set_start_method('spawn')
    pool = mp.Pool(args.threads)
  for g in range(args.maxgen):
    print("Generation:", g+1)
    if args.popsize:
      solutions = mees.ask(number=args.popsize)
    else:
      solutions = mees.ask()
    t1_sol = time.time()
    if args.threads < 2:
      solutions_fitness_feat_body = [evaluate_nca(s,args) for s in solutions]
    else:
      jobs = [pool.apply_async(evaluate_nca, (s,args)) for s in solutions]
      solutions_fitness_feat_body = [job.get() for job in jobs]
    t2_sol = time.time()
    print("Fitness calculation time:", t2_sol-t1_sol)

    solutions_fitness = [s[0] for s in solutions_fitness_feat_body]
    solutions_feature = [s[1] for s in solutions_fitness_feat_body]
    solutions_body = [s[2] for s in solutions_fitness_feat_body]

    mees.tell(solutions_fitness, solutions_feature, solutions_body)

    if args.saveall:
      utils.save_generation_with_solutions(solutions,
                                           solutions_fitness,
                                           solutions_feature,
                                           solutions_body, g,
                                           history_folder_path)
    else:
      utils.save_generation(solutions_fitness, solutions_feature,
                            solutions_body, g, history_folder_path)

    mees.disp()
    loss_current_idx = np.argmin(solutions_fitness)
    loss_current = solutions_fitness[loss_current_idx]
    with open(os.path.join(args.log_dir, "loss_history.csv"), 'a', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(solutions_fitness)
    if loss_current < loss_best:
      loss_best = loss_current
      checkpoint_filename = "%06d_%.7f.ckpt"%(g,loss_current)
      shaped_weight = utils.get_model_weights(solutions[loss_current_idx],
                                              weight_amount_list,
                                              weight_shape_list)
      nca.dmodel.set_weights(shaped_weight)
      nca.save_weights(os.path.join(args.log_dir, checkpoint_filename))
      mees.save_elites(args.log_dir, g=g)
    # if g > 0 and (g%10) == 0:
    #   pool.close()
    #   pool = mp.Pool(args.threads)

  t2 = time.time()

  print("TIME", t2-t1)
  mees.result_pretty()
  mees.save_elites(args.log_dir)
  plot_loss(args.log_dir)
  print("Log saved in: " + args.log_dir)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--maxgen', default=10, type=int, help="Maximum number of generations")
  parser.add_argument('--popsize', type=int, help="Population size")
  parser.add_argument('--threads', default=1, type=int, help="Number of threads")
  parser.add_argument('--width', default=3, type=int, help="CA width")
  parser.add_argument('--height', default=3, type=int, help="CA height")
  parser.add_argument('--emitters', default=15, type=int, help="Number of ES emitters")
  parser.add_argument('--bodytype', type=int, help="Premade body types: {1,2,3}")
  parser.add_argument("--retrain", help="path to log directory for retraining")
  parser.add_argument("--saveall", help="Save all solutions", action="store_true")
  parser.add_argument("--wall", help="Wall environment", action="store_true")
  p_args = parser.parse_args()

  current_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  log_dir = os.path.join("logs",
                         os.path.join(os.path.basename(__file__)[:-3],
                                      current_time_str))
  os.makedirs(log_dir)

  args_filename = os.path.join(log_dir, "args.json")
  args = utils.ArgsIO(args_filename)
  args.log_dir = log_dir
  args.threads = p_args.threads
  args.maxgen = p_args.maxgen
  args.popsize = p_args.popsize
  args.emitters = p_args.emitters
  args.retrain = p_args.retrain
  args.ca_height = p_args.height
  args.ca_width = p_args.width
  args.bodytype = p_args.bodytype
  args.saveall = p_args.saveall
  args.wall = p_args.wall
  build_body_steps_n = utils.calculate_build_body_steps(args.ca_height,
                                                        args.ca_width)
  args.ca_steps_build_body = (build_body_steps_n, build_body_steps_n+1)
  args.env_amount = 12
  args.env_max_iter = 100
  args.stable_body = True
  args.overflow_weight = 0

  train(args)



"""
Test NCA for modular light chaser.
"""
import numpy as np
from body_brain_nca import BodyBrainNCA
from modular_carrier import ModularCarrier
import utils
import os

EPISODES = 4

def test(args, render):
  print("Testing checkpoint saved in: " + args.log_dir)

  env_simulation_steps = args.env_max_iter
  print_every_n_step = env_simulation_steps // 10

  nca = BodyBrainNCA(**args.nca_model)
  nca.dmodel.summary()
  checkpoint_filename = "checkpoint"
  with open(os.path.join(args.log_dir, checkpoint_filename), "r") as f:
    first_line = f.readline()
    start_idx = first_line.find(": ")
    ckpt_filename = first_line[start_idx+3:-2]

  print("Testing model with lowest training loss...")
  nca.load_weights(os.path.join(args.log_dir, ckpt_filename))
  avg_reward_list = []
  last_reward_list = []
  goal_list = []
  for episode in range(EPISODES):
    x = np.zeros((1, args.ca_height, args.ca_width,nca.channel_n),dtype=np.float32)
    x[:,args.ca_height//2,args.ca_width//2,:1] = 1.0

    has_bodytype = False
    if hasattr(args, 'bodytype'):
      if args.bodytype is not None:
        has_bodytype = True

    if has_bodytype:
      body_grid = utils.manual_body_grid(args.bodytype, 1)
      x = nca.body_grid_2_fixed_body_nca(body_grid)
    else:
      body_grid, x = nca.build_body(x)

    print("body_grid", body_grid)

    env = ModularCarrier(body_grid[0], False, predefined_ball_idx=episode)
    observations = env.reset()
    total_reward = 0.0
    has_goal = 0
    for t in range(env_simulation_steps):
      if render:
        env.render()
      organized_observations = utils.reorganize_obs(body_grid[0],
                                                    nca.body_sensor_n,
                                                    [observations])
      x, actions = nca.act(x, organized_observations)
      observations, reward, done, info = env.step(actions[0])
      total_reward += reward
      distance_ball_goal = np.linalg.norm(
        env.ball.ball_body.position - env.goal.goal_body.position)
      if distance_ball_goal < 10:
        has_goal = 1

      if t % print_every_n_step == 0 or done:
        print("\naction " + str(["{:+0.2f}".format(x) for x in actions[0]]))
        print("observation " + str(["{:+0.2f}".format(x) for x in observations]))
        print("step {} reward {:+0.2f} avg_reward {:+0.2f}".format(
          t, reward, total_reward/(t+1)))

    avg_reward_list.append(total_reward/env_simulation_steps)
    last_reward_list.append(reward)
    goal_list.append(has_goal)
    env.close()
  return avg_reward_list, last_reward_list, goal_list

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--logdir", help="path to log directory")
  parser.add_argument("--render", help="render environment", action="store_true")
  parser.add_argument('--repeat', default=1, type=int)
  p_args = parser.parse_args()

  if p_args.logdir:
    args_filename = os.path.join(p_args.logdir, "args.json")
    argsio = utils.ArgsIO(args_filename)
    args = argsio.load_json()
    eps_avg_reward_list = []
    eps_last_reward_list = []
    eps_goal_list = []
    for i in range(p_args.repeat):
      resulting_avg_reward_list, res_last, res_goal = test(args, p_args.render)
      eps_avg_reward_list.append(resulting_avg_reward_list)
      eps_last_reward_list.append(res_last)
      eps_goal_list.append(res_goal)
    print("Number of runs:", p_args.repeat*EPISODES)
    print("Avg:", np.mean(eps_avg_reward_list))
    print("Std:", np.std(eps_avg_reward_list))
    print("#####")
    print("LAST")
    print("Avg:", np.mean(eps_last_reward_list))
    print("Std:", np.std(eps_last_reward_list))

    print("#####")
    print("GOAL")
    print("Avg:", np.mean(eps_goal_list))

  else:
    print("Add --logdir [path/to/log]")



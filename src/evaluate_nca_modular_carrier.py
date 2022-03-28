import numpy as np
from body_brain_nca import BodyBrainNCA
from modular_carrier import ModularCarrier
import utils

def evaluate_nca(flat_weights, args):
  nca = BodyBrainNCA(ca_steps_build_body=args.ca_steps_build_body,
                     body_sensor_n=2)
  weight_shape_list, weight_amount_list, _ = utils.get_weights_info(nca.weights)
  shaped_weight = utils.get_model_weights(flat_weights, weight_amount_list,
                                          weight_shape_list)
  nca.dmodel.set_weights(shaped_weight)

  x = np.zeros((args.env_amount, args.ca_height, args.ca_width,nca.channel_n),
               dtype=np.float32)
  x[:,args.ca_height//2,args.ca_width//2,:1] = 1.0

  has_bodytype = False
  if hasattr(args, 'bodytype'):
    if args.bodytype is not None:
      has_bodytype = True

  if has_bodytype:
    body_grid = utils.manual_body_grid(args.bodytype, args.env_amount)
    x = nca.body_grid_2_fixed_body_nca(body_grid)
  else:
    body_grid, x = nca.build_body(x)

  env_list = [ModularCarrier(b, False,
                             predefined_ball_idx=(i%args.env_amount))
              for i, b in enumerate(body_grid)]

  reward_list = []
  feature = None

  observations = [env.reset() for env in env_list]

  robot_failure_list = [env.robot_failure for env in env_list]

  robot_grid_similarity = BodyBrainNCA.grid_similarity(body_grid)
  if (True in robot_failure_list or robot_grid_similarity < 1) and args.stable_body:
    robot_failure_reward = [env.robot_failure_reward() for env in env_list]
    reward_list = (robot_grid_similarity-1) + np.asarray(robot_failure_reward)
  else:
    feature = (env_list[0].sensor_number,
                env_list[0].actuator_number,
                env_list[0].module_number)
    all_envs_done = False
    for t in range(args.env_max_iter):
      env_step_res_list = []
      if args.stable_body:
        organized_observations = utils.reorganize_obs(body_grid[0],
                                                      nca.body_sensor_n,
                                                      observations)
        x, actions = nca.act(x, organized_observations)

        env_step_res_list = [env.step(actions[i]) for i, env in enumerate(env_list)]
      else:
        for i, env in enumerate(env_list):
          if env.robot_failure:
            actions = np.zeros(env.actuator_number)
          else:
            organized_observations = utils.reorganize_obs(body_grid[0],
                                                          nca.body_sensor_n,
                                                          np.expand_dims(
                                                            observations[i],0))
            x[i:i+1], act = nca.act(x[i:i+1],organized_observations)
            actions = act[0]

          env_step_res_list.append(env.step(actions))

      observations = []
      reward_envs_list = []
      done_envs_list = []
      for i, env_step_res in enumerate(env_step_res_list):
        observations.append(env_step_res[0])
        if args.overflow_weight > 0:
          # TODO: it gives overflow error
          reward_envs_list.append(env_step_res[1] - args.overflow_weight*
                                  np.mean(np.square(
                                    np.clip(x[i], -1.5, 1.5)-x[i])))
        else:
          reward_envs_list.append(env_step_res[1])
        done_envs_list.append(env_step_res[2])

      reward_list.append(np.sum(reward_envs_list)/args.env_amount)
      reverse_done_idx = np.arange(len(done_envs_list))[::-1]

      for i in reverse_done_idx:
        if done_envs_list[i]:
          env_list[i].close()
          del env_list[i]

          if len(env_list) == 0:
            all_envs_done = True
          else:
            del observations[i]
            x = np.delete(x,i,0)
      if all_envs_done:
        break

  for env in env_list:
    env.close()

  loss = -np.mean(reward_list)
  resulting_body = []
  resulting_body = body_grid[0].tolist()
  return loss, feature, resulting_body
"""
Output video of testing NCA for modular light chaser.
"""
import numpy as np
from body_brain_nca import BodyBrainNCA
from modular_light_chaser import ModularLightChaser
import utils
import datetime
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import imageio

mpl.use('Agg')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# VIDEO_SPEEDS = { (0,50): 1,
#                 (50,75):1,
#                 (75,90):1,
#                 (90,150):2,
#                 (150,200):5,
#                 (200,220):10,
#                 (220,240):10,
#                 (240,280):10,
#                 (280,330):10,
#                 (330,4000):10,
#                 (4000, 40000):10}

VIDEO_SPEEDS = { (0,40000): 1}

def save_this_frame(nb):
  for b,e in VIDEO_SPEEDS:
    if b<= nb and nb <e:
      if VIDEO_SPEEDS[(b,e)] <0:
        return -VIDEO_SPEEDS[(b,e)]
      else:
        return int(nb%VIDEO_SPEEDS[(b,e)] == 0)
  return 0

def test_output_video(args):
  print("Testing checkpoint saved in: " + args.log_dir)
  episodes = 4
  env_simulation_steps = args.env_max_iter
  nca = BodyBrainNCA(**args.nca_model)
  nca.dmodel.summary()
  checkpoint_filename = "checkpoint"
  with open(os.path.join(args.log_dir, checkpoint_filename), "r") as f:
    first_line = f.readline()
    start_idx = first_line.find(": ")
    ckpt_filename = first_line[start_idx+3:-2]

  current_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

  print("Testing model with lowest training loss...")
  nca.load_weights(os.path.join(args.log_dir, ckpt_filename))
  test_image_folder_list = []
  for episode in range(episodes):

    test_image_folder = os.path.join(args.log_dir,
                                     "nca-"+current_time_str+"-ep"+str(episode))
    os.mkdir(test_image_folder)
    test_image_folder_list.append(test_image_folder)

    x = np.zeros((1, args.ca_height, args.ca_width,nca.channel_n),dtype=np.float32)
    x[:,args.ca_height//2,args.ca_width//2,:1] = 1.0
    ca_timestep = 0
    title = "CA Iteration: " + str(ca_timestep) + ". Environment: " + str(0)
    nca_img = nca.get_full_nca_image(x, channel_names=True)
    env_img = np.ones((600,600,3))
    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.suptitle(title)
    ax_nca_img = axs[0].imshow(nca_img)
    axs[0].axis('off')
    ax_env_img = axs[1].imshow(env_img)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    plt.tight_layout()
    nb_img = save_this_frame(ca_timestep)
    for kk in range(nb_img):
      img_filename = os.path.join(test_image_folder,
                                  "{:06d}.png".format(ca_timestep))
      fig.savefig(img_filename, format='png', dpi=300)

    for t in range(env_simulation_steps+1):
      if t == 0: # Building
        has_bodytype = False
        if hasattr(args, 'bodytype'):
          if args.bodytype is not None:
            has_bodytype = True

        if not has_bodytype:
          iter_n = np.random.randint(nca.ca_steps_build_body[0],
                                     nca.ca_steps_build_body[1])
          for i in range(iter_n):
            # x_old = np.array(x)
            x = nca(x, video_making=True)
            ca_timestep += 1
            title = "CA Iteration: " + str(ca_timestep) + ". Environment: " + str(t)
            fig.suptitle(title)

            nca_img = nca.get_full_nca_image(x, channel_names=True)
            ax_nca_img.set_data(nca_img)
            ax_env_img.set_data(env_img)
            nb_img = save_this_frame(ca_timestep)
            for kk in range(nb_img):
              img_filename = os.path.join(test_image_folder,
                                          "{:06d}.png".format(ca_timestep))
              fig.savefig(img_filename, format='png', dpi=300)

            # nca_img = nca.get_full_nca_image(x-x_old, channel_names=True)
            # ax_nca_img.set_data(nca_img)
            # ax_env_img.set_data(env_img)
            # nb_img = save_this_frame(ca_timestep)
            # for kk in range(nb_img):
            #   img_filename = os.path.join(test_image_folder,
            #                               "diff_{:06d}.png".format(ca_timestep))
            #   fig.savefig(img_filename, format='png', dpi=300)


          body_grid, fixed_body_nca = nca.get_body_grid_and_fixed_body_nca(x)
          print("body_grid", body_grid)
          x = fixed_body_nca
        else:
          body_grid = utils.manual_body_grid(args.bodytype, 1)
          x = nca.body_grid_2_fixed_body_nca(body_grid)
        env = ModularLightChaser(body_grid[0],
                                 False,
                                 predefined_light_idx=episode,
                                 with_passage=args.wall)
        observations = env.reset()
        total_reward = 0.0
      else:
        env_img = env.render(mode="rgb_array")

        iter_n = np.random.randint(nca.ca_steps_per_action[0],
                                   nca.ca_steps_per_action[1])
        for i in range(iter_n):
          x = nca(x, sensor_inputs=[observations],
                  fixed_body_idx=0,
                  body_limitation=nca.act_body_limitation,
                  video_making=True)
          ca_timestep += 1
          title = "CA Iteration: " + str(ca_timestep) + ". Environment: " + str(t)
          fig.suptitle(title)
          nca_img = nca.get_full_nca_image(x, channel_names=True)
          ax_nca_img.set_data(nca_img)
          ax_env_img.set_data(env_img)
          nb_img = save_this_frame(ca_timestep)
          for kk in range(nb_img):
            img_filename = os.path.join(test_image_folder,
                                        "{:06d}.png".format(ca_timestep))
            fig.savefig(img_filename, format='png', dpi=300)

        actions = nca.get_actuators(x)

        observations, reward, done, info = env.step(actions[0])
        total_reward += reward

        if done:
          print("Episode finished after {} timesteps".format(t+1))
          break

    plt.close("all")
    env.close()
    print("Images saved in:", test_image_folder)
    test_video_filename = os.path.join(test_image_folder, "nca-"+current_time_str+"-ep"+str(episode)+".mp4")
    with utils.VideoWriter(test_video_filename, fps=20) as vid:
      list_image_filename = glob.glob(os.path.join(test_image_folder, "*.png"))
      for fn in sorted(list_image_filename):
        vid.add(imageio.imread(fn)[:,:,:3])
    print("Video saved:", test_video_filename)

  # All videos concatenated
  all_test_video_filename = os.path.join(args.log_dir, "nca-"+current_time_str+"-all.mp4")
  with utils.VideoWriter(all_test_video_filename, fps=20) as vidall:
    for folder in test_image_folder_list:
      list_image_filename = glob.glob(os.path.join(folder, "*.png"))
      for fn in sorted(list_image_filename):
        vidall.add(imageio.imread(fn)[:,:,:3])
  print("Video saved:", all_test_video_filename)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--logdir", help="path to log directory")
  p_args = parser.parse_args()

  if p_args.logdir:
    args_filename = os.path.join(p_args.logdir, "args.json")
    argsio = utils.ArgsIO(args_filename)
    args = argsio.load_json()
    test_output_video(args)
  else:
    print("Add --logdir [path/to/log]")



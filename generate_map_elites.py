import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast
from body_brain_nca import BodyBrainNCA
from modular_light_chaser import ModularLightChaser
from modular_carrier import ModularCarrier
from PIL import Image
import matplotlib
import utils
# from matplotlib.ticker import PercentFormatter

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

EPISODES = 4

def save_heatmap_elites(folder_path, features, loss, g=None):
  unique_modules = list(np.unique([t[2] for t in features]))
  unique_module_n = len(unique_modules)
  loss = np.asarray(loss)

  features_per_module_n = [[] for _ in range(unique_module_n)]
  loss_per_module_n = [[] for _ in range(unique_module_n)]
  for f,l in zip(features, loss):
    features_per_module_n[unique_modules.index(f[2])].append((f[0], f[1]))
    loss_per_module_n[unique_modules.index(f[2])].append(l)

  max_loss = max(abs(loss))
  min_loss = min(abs(loss))

  sqrt_unique_module_n = np.sqrt(unique_module_n)
  rows = int(np.round(sqrt_unique_module_n))
  cols = int(np.ceil(sqrt_unique_module_n))
  fig = matplotlib.pyplot.gcf()
  fig.set_size_inches(10, 8)
  for i in range(rows*cols):
    plt.subplot(rows,cols,i+1)
    valid_subplot = False
    if i < unique_module_n:
      plt.title("Modules: "+str(unique_modules[i]), fontsize=10)
      if len(features_per_module_n[i]) > 0:
        heatmap_size = max([max(t[0]+1, t[1]) for t in features_per_module_n[i]])

        heatmap = np.full((heatmap_size-1, heatmap_size-1), np.nan)
        for j,f in enumerate(features_per_module_n[i]):
          heatmap[-1*f[1]+1, f[0]-1] = -loss_per_module_n[i][j]

        plt.imshow(heatmap, extent=[0.5,heatmap_size-0.5,1.5,heatmap_size+0.5],
                   cmap="cividis", vmax=max_loss, vmin=min_loss)

        plt.xticks(list(range(1,heatmap_size, (heatmap_size//8)+1)), fontsize=6)
        plt.yticks(list(range(2,heatmap_size+1, (heatmap_size//8)+1)), fontsize=6)

        plt.xlabel("Sensors", fontsize=8)
        plt.ylabel("Actuators", fontsize=8)
        valid_subplot = True

    if not valid_subplot:
      plt.axis('off')

  fig.subplots_adjust(right=0.8, wspace=0.6, hspace=0.8)
  cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
  #plt.colorbar(cax=cbar_ax, ticks=np.linspace(min_loss, max_loss, 10))
  plt.colorbar(cax=cbar_ax)
  if g is None:
    fig.savefig(os.path.join(folder_path, "load_save_elites.png"), format='png', dpi=300)
    fig.savefig(os.path.join(folder_path, "load_save_elites.svg"), format='svg')
  else:
    fig.savefig(os.path.join(folder_path, "%06d_elites.png"%(g)), format='png', dpi=300)
  plt.close("all")

  # max_module = max(unique_modules)
  # total_elites = 0
  # for m in range(3,max_module):
  #   total_elites += sum([n for n in range(1,m-1)]) )
  total_elites = 2024
  len_elites = len(loss)

  lines=[]
  lines.append("Percentage elites: "+str(len_elites / total_elites) + "\n")
  lines.append("QD-score: "+str(np.sum(-loss) / total_elites))

  with open(os.path.join(folder_path, "elites_stats.txt"), "w") as fstats:
    fstats.writelines(lines)

  plt.hist(-loss, 200, facecolor='b', alpha=0.75, weights=np.ones(len(loss)) / len(loss))
  # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
  plt.xlim(0.25, 0.75)
  plt.ylim(0, 0.15)
  plt.xlabel('Fitness')
  plt.ylabel('Probability')
  plt.savefig(os.path.join(folder_path, "hist_elites.png"), format='png', dpi=300)
  plt.close()

def generate_map_elites_video(args, flat_weights, video_filename):
  # print(type(flat_weights))
  env_simulation_steps = args.env_max_iter

  nca = BodyBrainNCA(**args.nca_model)
  nca.dmodel.summary()

  weight_shape_list, weight_amount_list, _ = utils.get_weights_info(nca.weights)
  print(weight_shape_list, weight_amount_list)
  shaped_weight = utils.get_model_weights(flat_weights, weight_amount_list,
                                          weight_shape_list)
  nca.dmodel.set_weights(shaped_weight)

  with utils.VideoWriter(video_filename, fps=10) as vid:
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

      if nca.body_sensor_n == 2:
        env = ModularCarrier(body_grid[0], False, predefined_ball_idx=episode)
      else:
        env = ModularLightChaser(body_grid[0], False, predefined_light_idx=episode)

      observations = env.reset()

      for t in range(env_simulation_steps):
        env_img = env.render(mode="rgb_array")
        vid.add(env_img)
        x, actions = nca.act(x, [observations])
        observations, reward, done, info = env.step(actions[0])

      env.close()

def load_save_elites(folder_path, save_body=False, save_video=False,
                     carrier_env=False):
  csv_file_path = os.path.join(folder_path, "elites.csv")
  elites = pd.read_csv(csv_file_path, delimiter=";")
  df_map_elites_features = elites["map_elites_features"]
  df_map_elites_loss = elites["map_elites_loss"]
  df_map_elites_body = elites["map_elites_body"]

  map_elites_features = [ast.literal_eval(t) for t in df_map_elites_features]
  map_elites_body = [ast.literal_eval(b) for b in df_map_elites_body]

  save_heatmap_elites(folder_path, map_elites_features, df_map_elites_loss)

  if save_body:
    body_folder = os.path.join(folder_path, "map_elites_body")
    if not os.path.exists(body_folder):
      os.makedirs(body_folder)

    for i in range(len(map_elites_body)):
      if carrier_env:
        env = ModularCarrier(np.array(map_elites_body[i]),
                            False, predefined_ball_idx=0,
                            zoom_manual=2.0)
      else:
        env = ModularLightChaser(np.array(map_elites_body[i]),
                                 False, predefined_light_idx=0,
                                 zoom_manual=2.0)
      env.reset()
      env_img = env.render(mode="rgb_array")
      img = Image.fromarray(env_img)
      img_filename = ""
      feat = map_elites_features[i]
      len_feat = len(feat)
      for f_idx in range(len_feat):
        img_filename += str(feat[f_idx])
        if f_idx == (len_feat-1):
          img_filename += ".png"
        else:
          img_filename += "_"

      img.save(os.path.join(body_folder, img_filename))
      env.close()

      with open(os.path.join(body_folder, img_filename+".html"), "w") as lossfile:
        lossfile.write('<h1 style="position: relative;">Fitness: {:.5g}</h1>'.format(-df_map_elites_loss[i]))

      if save_video:
        args_filename = os.path.join(folder_path, "args.json")
        argsio = utils.ArgsIO(args_filename)
        args = argsio.load_json()
        video_filename = os.path.join(body_folder, img_filename+".mp4")
        flat_weights = ast.literal_eval(elites["map_elites"][i])
        generate_map_elites_video(args, flat_weights, video_filename)

  return map_elites_features

def generate_map_elites_interactive(folder_path, features):
  unique_modules = list(np.unique([f[2] for f in features]))

  web_folder = os.path.join(folder_path, "interactive")
  # print(web_folder)
  # print("os.path.isdir(web_folder)", os.path.isdir(web_folder))
  # if not os.path.isdir(web_folder):
  # os.remove(web_folder)
  os.makedirs(web_folder)

  index_template = []
  with open("index_template.html", "r") as index_template_file:
    index_template = index_template_file.readlines()

  js_template_path = os.path.join("js", "image_template.js")
  js_template = []
  with open(js_template_path, "r") as js_template_file:
    js_template = js_template_file.readlines()

  mousemove_js = []
  image_svg = []
  for m in unique_modules:
    heatmap_size = max([max(f[0]+1, f[1]) for f in features if f[2]==m])

    script_line1 = 'var image_{0:05d}_{1:05d} = svgElem.getElementById("image_{0:05d}_{1:05d}");'.format(heatmap_size,m)
    script_line2 = 'image_{0:05d}_{1:05d}.onmousedown = getMousePosition;'.format(heatmap_size,m)
    mousemove_js.append(script_line1)
    mousemove_js.append(script_line2)
    image_svg.append([heatmap_size,m])

  js_path = os.path.join(web_folder, "image.js")
  with open(js_path, "w") as js_file:
    for line in js_template:
      if "%mousemove%" in line:
        js_file.writelines(mousemove_js)
      else:
        js_file.write(line)

  svg_path = os.path.join(folder_path, "load_save_elites.svg")
  svg_data = []
  with open(svg_path, "r") as svg_file:
    svg_data = svg_file.readlines()

  module_idx = 0
  for i in range(len(svg_data)):
    if "<svg " in svg_data[i]:
      svg_data[i] = svg_data[i].replace('<svg ', '<svg id="svg" ')
    if 'id="image' in svg_data[i]:
      if module_idx < len(unique_modules):
        svg_data[i] = re.sub(r'id="image[a-zA-Z_0-9]*"',
                             'id="image_{0:05d}_{1:05d}"'.format(image_svg[module_idx][0],
                                                                 image_svg[module_idx][1]),
                             svg_data[i])
        module_idx += 1

  index_html_path = os.path.join(web_folder, "index.html")
  with open(index_html_path, "w") as index_html_file:
    for line in index_template:
      if "%svg%" in line:
        index_html_file.writelines(svg_data)
      else:
        index_html_file.write(line)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--logdir", help="path to log directory")
  parser.add_argument("--body", help="generate body", action="store_true")
  parser.add_argument("--video", help="generate video", action="store_true")
  parser.add_argument("--web", help="generate web page", action="store_true")
  parser.add_argument("--carrier", help="generate for carrier environment",
                      action="store_true")
  p_args = parser.parse_args()

  features = load_save_elites(p_args.logdir,
                              save_body=p_args.body,
                              save_video=p_args.video,
                              carrier_env=p_args.carrier)
  if p_args.web:
    generate_map_elites_interactive(p_args.logdir, features)

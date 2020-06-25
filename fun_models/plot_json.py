import argparse
import json
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

rc_params = {
  'font': {
    'family'     : 'monospace',
    'style'      : 'normal',
    'variant'    : 'normal',
    'weight'     : 'bold',
    'stretch'    : 'normal',
    'size'       : 18,
  },
  'text': {'usetex'     : True,},
  'xtick': {'labelsize' : 18,},
  'ytick': {'labelsize' : 18,},
  'axes': {
    'labelsize'  : 18,
    'titlesize'  : 18,
  },
  'lines': {
    'linewidth' : 2,
    'markersize': 6,
  },
}
for key, param in rc_params.items():
  mpl.rc(key, **param)

def parse_args():
  description = r'''Plots JSON data

  The JSON has to have dict of dicts of the following format:
    {
      'Model 1': {
        'epochs': [...],
        'mode1': {
          'metric1': [...],
          'metric2': [...],
          ...
        },
        'mode2': {...}
      },
      'Model 2': {...}
    }
  The 'epochs' will be used as x-axis, while the 'metricX' will be used as Y-axis.
  '''
  parser = argparse.ArgumentParser(description=description,
                                   formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument('json', type=argparse.FileType('r'),
                      help='JSON file to process')
  parser.add_argument('--save-img', action='store_const', const=True, default=False,
                      help='Boolean flag to save or not to file.')

  args = parser.parse_args()
  return args, args.json, args.save_img

def load_json(fp):
  data = json.load(fp)
  return data

def make_data_plot(data, metric_ranges=None, mode_in_cols=True):
  SUBPLOT_WIDTH = 5

  if metric_ranges is None:
    metric_ranges = {}

  # Collect the keys
  models = [ model for model in data.keys() ]
  modes = [ mode for mode in data[models[0]].keys() if mode != 'epochs' ]
  metrics = [ metric for metric in data[models[0]][modes[0]] ]

  print(models, modes, metrics)

  num_rows = len(modes)
  num_cols = len(metrics)
  sharey = 'col'
  if mode_in_cols:
    num_rows, num_cols = num_cols, num_rows
    sharey = 'row'

  epochs = data.get('epochs', [])
  if len(epochs) == 0:
    epochs = range(len(data[models[0]][modes[0]][metrics[0]]))

  figsize = (SUBPLOT_WIDTH * num_cols, SUBPLOT_WIDTH * num_rows)
  fig, ax = plt.subplots(num_rows, num_cols,
                         figsize=figsize,
                         sharex=True,
                         sharey=sharey)
  for idx, mode in enumerate(modes):
    for jdx, metric in enumerate(metrics):
      if mode_in_cols:
        row = jdx
        col = idx
        COLS = len(modes)
      else:
        row = idx
        col = jdx
        COLS = len(metrics)
      kdx = row * COLS + col

      for model in models:
        ax.flat[kdx].plot(epochs, data[model][mode][metric], label=model)
      ax.flat[kdx].set_title(mode + ' ' + metric)
      ax.flat[kdx].set_ylabel(metric)
      # ax.flat[kdx].set_yscale('log')
      ax.flat[kdx].grid('all')
      if metric_ranges.get(metric, None) is not None:
        ax.flat[kdx].set_ylim(metric_ranges[metric])
      ax.flat[kdx].legend()

  # Remove some of the labels
  for a_last_row in ax[-1]:
    a_last_row.set_xlabel('epoch')

  return fig, ax


def main():
  args, fp, save_to_img = parse_args()
  data = load_json(fp)

  make_data_plot(data,
                 metric_ranges={
                  'accuracy': [0.8, 1.01],
                  'loss': [-1e-5, 2e-4],
                 },
                 mode_in_cols=True)
  plt.suptitle('Effect of Different Activation Functions', x=0.5, y=1.0)
  plt.tight_layout()

  if not save_to_img:
    plt.show()
  else:
    file_name = '.'.join(fp.name.split('.')[:-1] + ['png'])
    plt.savefig(file_name)

if __name__ == '__main__':
  main()

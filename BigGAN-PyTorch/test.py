''' Test
   This script loads a pretrained net and a weightsfile and test '''
import functools
import math
import numpy as np
from tqdm import tqdm, trange


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision

# Import my stuff
import inception_utils
import utils
import losses

from sklearn.linear_model import LogisticRegression


def testD(config):
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}
  
  # update config (see train.py for explanation)
  config['resolution'] = utils.imsize_dict[config['dataset']]
  config['n_classes'] = utils.nclass_dict[config['dataset']]
  config['G_activation'] = utils.activation_dict[config['G_nl']]
  config['D_activation'] = utils.activation_dict[config['D_nl']]
  config = utils.update_config_roots(config)
  config['skip_init'] = True
  config['no_optim'] = True
  device = 'cuda'
  
  # Seed RNG
  utils.seed_rng(config['seed'])
   
  # Setup cudnn.benchmark for free speed
  torch.backends.cudnn.benchmark = True
  
  # Import the model--this line allows us to dynamically select different files.
  model = __import__(config['model'])
  experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
  print('Experiment name is %s' % experiment_name)
  
  D = model.Discriminator(**config).cuda()
  utils.count_parameters(D)
  
  # Load weights
  print('Loading weights...')
  # Here is where we deal with the ema--load ema weights or load normal weights
  utils.load_weights(None, D, state_dict, 
                     config['weights_root'], experiment_name, config['load_weights'],
                     None,
                     strict=False, load_optim=False)
  
  print('Putting D in eval mode..')
  D.eval()
  loaders = utils.get_data_loaders(**{**config, 'batch_size': 100, 'start_itr': 0})
  train_data = []
  train_label = []
  if config['pbar'] == 'mine':
    pbar = utils.progress(loaders[0],displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
  else:
    pbar = tqdm(loaders[0])
  with torch.no_grad():
    for i, (x, y) in enumerate(pbar):
      if config['D_fp16']:
        x, y = x.to(device).half(), y.to(device)
      else:
        x, y = x.to(device), y.to(device)
      h = x
      for index, blocklist in enumerate(D.blocks):
        for block in blocklist:
          h = block(h)
      h = torch.sum(D.activation(h), [2, 3])
      train_data.append(h.cpu().numpy())
      train_label.append(y.cpu().numpy())
  train_data = np.vstack(train_data)
  train_label = np.hstack(train_label)

  config['dataset'] = 'TI200_valid'
  loaders = utils.get_data_loaders(**{**config, 'batch_size': 100, 'start_itr': 0})

  test_data = []
  test_label = []
  if config['pbar'] == 'mine':
    pbar = utils.progress(loaders[0],displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
  else:
    pbar = tqdm(loaders[0])
  with torch.no_grad():
    for i, (x, y) in enumerate(pbar):
      if config['D_fp16']:
        x, y = x.to(device).half(), y.to(device)
      else:
        x, y = x.to(device), y.to(device)
      h = x
      for index, blocklist in enumerate(D.blocks):
        for block in blocklist:
          h = block(h)
      h = torch.sum(D.activation(h), [2, 3])
      test_data.append(h.cpu().numpy())
      test_label.append(y.cpu().numpy())
  test_data = np.vstack(test_data)
  test_label = np.hstack(test_label)
    
  print(train_data.shape)
  print(train_label.shape)
  print(test_data.shape)
  print(test_label.shape)
  
  LR = LogisticRegression()
  LR.fit(train_data, train_label)
  acc = LR.score(test_data, test_label)
  print(acc)

def testG_iFID(config):
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}
  
  # update config (see train.py for explanation)
  config['resolution'] = utils.imsize_dict[config['dataset']]
  config['n_classes'] = utils.nclass_dict[config['dataset']]
  config['G_activation'] = utils.activation_dict[config['G_nl']]
  config['D_activation'] = utils.activation_dict[config['D_nl']]
  config = utils.update_config_roots(config)
  config['skip_init'] = True
  config['no_optim'] = True
  device = 'cuda'
  
  # Seed RNG
  utils.seed_rng(config['seed'])
   
  # Setup cudnn.benchmark for free speed
  torch.backends.cudnn.benchmark = True
  
  # Import the model--this line allows us to dynamically select different files.
  model = __import__(config['model'])
  experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
  print('Experiment name is %s' % experiment_name)
  
  # Next, build the model
  G = model.Generator(**config).to(device)
  D = model.Discriminator(**config).to(device)

  # If using EMA, prepare it
  if config['ema']:
    print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
    G_ema = model.Generator(**{**config, 'skip_init':True, 
                               'no_optim': True}).to(device)
    ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
  else:
    G_ema, ema = None, None
  
  # FP16?
  if config['G_fp16']:
    print('Casting G to float16...')
    G = G.half()
    if config['ema']:
      G_ema = G_ema.half()
  if config['D_fp16']:
    print('Casting D to fp16...')
    D = D.half()
    # Consider automatically reducing SN_eps?
  GD = model.G_D(G, D)
  print(G)
  print(D)
  print('Number of params in G: {} D: {}'.format(
    *[sum([p.data.nelement() for p in net.parameters()]) for net in [G,D]]))
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}
  
  # Load weights
  print('Loading weights...')
  utils.load_weights(G, D, state_dict,
                     config['weights_root'], experiment_name, 
                     config['load_weights'] if config['load_weights'] else None,
                     G_ema if config['ema'] else None, load_optim=False)
  # If parallel, parallelize the GD module
  if config['parallel']:
    GD = nn.DataParallel(GD)
    if config['cross_replica']:
      patch_replication_callback(GD)
  
  G_batch_size = max(config['G_batch_size'], config['batch_size'])
  FIDs = []
  for label in range(utils.nclass_dict[config['dataset']]):
    # Prepare inception metrics: FID and IS
    get_inception_metrics = inception_utils.prepare_inception_metrics(config['dataset'], config['parallel'], config['no_fid'], no_is=True, label=label)
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                               device=device, fp16=config['G_fp16'], label=label)
    sample = functools.partial(utils.sample,
                               G=(G_ema if config['ema'] and config['use_ema'] else G),
                               z_=z_, y_=y_, config=config)
    IS_mean, IS_std, FID = get_inception_metrics(sample, 
                                                 config['num_inception_images'],
                                                 num_splits=10)
    print(FID)
    FIDs.append(FID)
  print(np.mean(FIDs))
    

def main():
  # parse command line and run    
  parser = utils.prepare_parser()
  # parser = utils.add_sample_parser(parser)
  config = vars(parser.parse_args())
  print(config)
#   testD(config)
  testG_iFID(config)
  
if __name__ == '__main__':    
  main()
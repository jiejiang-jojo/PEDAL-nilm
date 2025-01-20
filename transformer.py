import argparse
import numpy as np
import os
import time
import json
#import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from utilities import (create_folder, get_filename, create_logging,
                       mean_absolute_error, signal_aggregate_error)
from utilities import *
from data_generator import DataGenerator, TestDataGenerator
from models import move_data_to_gpu
from models import *
from torch.optim.lr_scheduler import LambdaLR

torch.cuda.set_device(1)
def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )
#TODO Add binarized loss function and optimizer

def loss_func(output, target, loss_weight):

    # logging.info('output shapes: {} and target shapes: {} '.format(output.shape, target.shape))
    assert output.shape == target.shape

    return torch.mean(torch.mul(torch.abs(output - target), loss_weight)) + 70*torch.mean(torch.mul(torch.square(output - target), loss_weight))
    
def loss_func_binary(output, target):

    assert output.shape == target.shape

    return F.binary_cross_entropy(output, target)


def accuracy(Y, Y_hat):
    return (Y == Y_hat).sum() / Y.size


def binary_metrics(outputs, targets):
    outputs = binarize(outputs, args.model_threshold)
    (tp, fn, fp, tn) = tp_fn_fp_tn(outputs, targets)
    precision_value = precision(outputs, targets)
    recall_value = recall(outputs, targets)
    f1_score = f_value(precision_value, recall_value)

    auc = roc_auc(outputs, targets)
    ap = average_precision(outputs, targets)

    metric_dict = {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
                   'precision': '{:.4f}'.format(precision_value),
                   'recall': '{:.4f}'.format(recall_value),
                   'f1_score': '{:.4f}'.format(f1_score),
                   'auc': '{:.4f}'.format(auc),
                   'average_precision': '{:.4f}'.format(ap)}
    return metric_dict


def evaluate(model, generator, data_type, max_iteration, cuda, binary=False):
    """Evaluate.
    Args:
      model: object.
      generator: object.
      data_type: 'train' | 'validate'.
      max_iteration: int, maximum iteration for validation
      cuda: bool.
    Returns:
      mae: float
    """

    # Generate function
    generate_func = generator.generate_validate(data_type=data_type,
                                                max_iteration=max_iteration)

    # Forward
    (outputs, targets) = forward(model=model,
                                 generate_func=generate_func,
                                 cuda=cuda,
                                 has_target=True)
    if binary:
        logging.info('----binary is true and return binary metrics----')
        return binary_metrics(outputs, targets)
    else:
        logging.info('----binary is false and only mae is returned----')
        outputs = generator.inverse_transform(outputs)
        targets = generator.inverse_transform(targets)

        mae = mean_absolute_error(outputs, targets)

        return mae


def forward(model, generate_func, cuda, has_target):
    """Forward data to a model.
    Args:
      model: object
      generate_func: generate function
      cuda: bool
      has_target: bool, True if generate_func yield (batch_x, batch_y),
                        False if generate_func yield (batch_x)
    Returns:
      (outputs, targets) | outputs
    """

    model.eval()

    outputs = []
    targets = []

    # Evaluate on mini-batch
    for data in generate_func:

        if has_target:
            (batch_x, batch_y) = data
            targets.append(batch_y)

        else:
            batch_x = data

        batch_x = move_data_to_gpu(batch_x, cuda)

        # Predict
        batch_output = model(batch_x)

        outputs.append(batch_output.data.cpu().numpy())

    if has_target:
        outputs = np.concatenate(outputs, axis=0)
        targets = np.concatenate(targets, axis=0)
        return outputs, targets

    else:

        return outputs


def train(args):

    logging.info('config=%s', json.dumps(vars(args)))

    # Arguments & parameters
    workspace = args.workspace
    cuda = args.cuda

    # Load model
    model_class, model_params = MODELS[args.model]
    model = model_class(**{k: args.model_params[k] for k in model_params if k in args.model_params})

    if args.train_model is not None:
        logging.info("continue training ...")
        model_path = os.path.join(workspace, 'logs', get_filename(__file__),
                                  args.train_model)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])

        # for param in model.parameters():
        #     param.requires_grad = False
        # model.blocks[-1].requires_grad_(True)
        # model.penultimate_conv.requires_grad_(True)
        # model.final_conv.requires_grad_(True)

    logging.info("sequence length: {}".format(model.seq_len))

    if cuda:
        model.cuda()

    # Paths
    hdf5_path = os.path.join(workspace, args.data_file)

    models_dir = os.path.join(workspace, 'models', get_filename(__file__))

    create_folder(models_dir)

    # Data generator
    generator = DataGenerator(hdf5_path=hdf5_path,
                              target_device=args.target_device,
                              train_house_list=args.train_house_list,
                              validate_house_list=args.validate_house_list,
                              batch_size=args.batch_size,
                              seq_len=model.seq_len,
                              width=args.width,
                              binary_threshold=args.binary_threshold,
                              balance_threshold=args.balance_threshold,
                              balance_positive=args.balance_positive, 
                              loss_weight_file=args.loss_weight_file)

    # Optimizer
    # learning_rate = 1
    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0.)
    lr_scheduler = LambdaLR(
             optimizer=optimizer, lr_lambda=lambda step: rate(step,128,1,4000)
            )
    iteration = 0
    train_bgn_time = time.time()

    for (batch_x, batch_y, batch_loss_weight) in generator.generate():

        if iteration > 1000*500:
            break

        # Evaluate
        if iteration % 100 == 0:
        # if iteration % 1000 == 0:

            train_fin_time = time.time()

            tr_result_dict = evaluate(model=model,
                                      generator=generator,
                                      data_type='train',
                                      max_iteration=args.validate_max_iteration,
                                      cuda=cuda,
                                      binary=args.binary_threshold is not None)

            va_result_dict = evaluate(model=model,
                                      generator=generator,
                                      data_type='validate',
                                      max_iteration=args.validate_max_iteration,
                                      cuda=cuda,
                                      binary=args.binary_threshold is not None)

            logging.info('train: {}'.format(tr_result_dict))
            logging.info('validate: {}'.format(va_result_dict))

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s, learning rate: {}'.format(
                    iteration, train_time, validate_time, learning_rate))

            logging.info('------------------------------------')

            train_bgn_time = time.time()

        # Reduce learning rate
        # if iteration % 1000 == 0 and iteration > 0 and learning_rate > 5e-5:
        #     for param_group in optimizer.param_groups:
        #         learning_rate *= 0.9
        #         param_group['lr'] = learning_rate

        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)
        batch_loss_weight = move_data_to_gpu(batch_loss_weight, cuda)

        # Forward
        forward_time = time.time()
        model.train()
        output = model(batch_x)

        # Loss
        if args.binary_threshold is not None:
            loss = loss_func_binary(output, batch_y)
        else:
            loss = loss_func(output, batch_y, batch_loss_weight)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if args.max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
        optimizer.step()
        lr_scheduler.step()
        learning_rate = optimizer.param_groups[0]["lr"]
        # Save model
        # if (iteration>1) and (iteration % 1000 == 0) and ((iteration//1000+4) // (((iteration//1000-1)//100+1)*100) == 1):
        if (iteration>1) and (iteration % 100 == 0):
        # if (iteration>1) and (iteration % 5000 == 0):
            save_out_dict = {'iteration': iteration,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()}

            save_out_path = args.basename + '_{}_{}_iter_{}_wd_{}_sl_{}.tar'.format(
                args.target_device,
                args.model,
                iteration,
                args.width,
                model.seq_len
            )

            create_folder(os.path.dirname(save_out_path))
            torch.save(save_out_dict, save_out_path)

            logging.info('Save model to {}'.format(save_out_path))

        iteration += 1


def inference(args):

    logging.info('config=%s', json.dumps(vars(args)))
    # Arguments & parameters
    workspace = args.workspace
    cuda = args.cuda

    # Paths
    print('data file being used: ', args.data_file)
    hdf5_path = os.path.join(workspace, args.data_file)
    model_path = os.path.join(workspace, 'logs', get_filename(__file__),
                              args.inference_model)

    # Load model
    model_class, model_params = MODELS[args.model]
    model = model_class(**{k: args.model_params[k] for k in model_params if k in args.model_params})
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()

    # Data generator
    print(args.batch_size)
    generator = TestDataGenerator(hdf5_path=hdf5_path,
                                  target_device=args.target_device,
                                  train_house_list=args.train_house_list,
                                  seq_len=model.seq_len,
                                  steps=args.width * args.batch_size,
                                  binary_threshold=args.binary_threshold)

    generate_func = generator.generate_inference(house=args.inference_house)

    # Forward
    inference_time = time.time()

    outputs = forward(model=model, generate_func=generate_func, cuda=cuda, has_target=False)
    outputs = np.concatenate([output[0] for output in outputs])
    if args.binary_threshold is not None:
        logging.info('----binary threshold is not none and binary metrics are returned----')
        targets = generator.get_target()
        logging.info('Inference time: {} s'.format(time.time() - inference_time))
        metric_dict = binary_metrics(outputs, targets)
        logging.info('Metrics: {}'.format(metric_dict))
    else:
        logging.info('----binary threshold is none and mae and sae metrics are returned----')
        outputs = generator.inverse_transform(outputs)

        logging.info('Inference time: {} s'.format(time.time() - inference_time))

        # Calculate metrics
        source = generator.get_source()
        targets = generator.get_target()

        valid_data = np.ones_like(source)
        for i in range(len(source)):
            if (source[i]==0) or (source[i] < targets[i]):
                valid_data[i] = 0

        mae = mean_absolute_error(outputs * valid_data, targets * valid_data)
        sae = signal_aggregate_error(outputs * valid_data, targets * valid_data)
        mae_allzero = mean_absolute_error(outputs*0, targets * valid_data)
        sae_allmean = signal_aggregate_error(outputs*0+generator.mean_y, targets * valid_data)

        # metric_dict = dict({'MAE': mae, 'MAE_zero': mae_allzero, 'SAE': sae, 'SAE_mean': sae_allmean}, **binary_metrics(((outputs - args.eval_binary_threshold) > 0).astype('float'), ((targets - args.eval_binary_threshold) > 0).astype('float')))
        # logging.info('Metrics: {}'.format(metric_dict))
        metric_dict = dict({'MAE': mae, 'MAE_zero': mae_allzero, 'SAE': sae, 'SAE_mean': sae_allmean})
        logging.info('Metrics: {}'.format(metric_dict))

    np.save(workspace+'/outputs/'+args.inference_model+'_'+args.inference_house+'_'+'prediction.npy', outputs)
    np.save(workspace+'/outputs/'+args.inference_model+'_'+args.inference_house+'_'+'groundtruth.npy', targets)
    np.save(workspace+'/outputs/'+args.inference_model+'_'+args.inference_house+'_'+'source.npy', source)


class DefaultNamespace(argparse.Namespace):
    """ When the requested attribute does not exists return None instead of throw AttributeError.
        Ex. Namespace().abc --> throw Attribute Error
            DefaultNamespace().abc --> return None
    """
    def __getattr__(self, name):
        return None


def consolidate_args(args):
    """ Merge different source of configuration. """
    # Loading config into args
    with open(args.config) as fin:
        config = json.load(fin)
        config.update({k: v for k, v in args.__dict__.items() if v is not None})
        args = DefaultNamespace(**config)
    # Loading commandline model parameters into model_params
    model_param_setting = {k: v for k, v in args.__dict__.items() if k.startswith('pm_')}
    if 'model_params' not in args.__dict__:
        args.__dict__['model_params'] = {}
    for k, v in model_param_setting.items():
        args.__dict__.pop(k)
        if v is not None:
            try:
                args.model_params[k[3:].replace('-', '_')] = int(v)
            except:
                args.model_params[k[3:]] = v
    if args.binary_threshold is not None:
        args.model_params['to_binary'] = True
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    model_params = set()
    for _, (_, mps) in MODELS.items():
        for p in mps:
            model_params.add(p)

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--config', type=str, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--width', type=int)
    parser_train.add_argument('--binary-threshold', type=float, default=None)
    parser_train.add_argument('--balance-threshold', type=float, default=None)
    parser_train.add_argument('--balance-positive', type=float, default=None)
    parser_train.add_argument('--train-model', type=str, default=None)
    for p in model_params:
        parser_train.add_argument('--pm-' + p.replace('_', '-'), type=str, metavar='<{}>'.format(p))

    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--workspace', type=str, required=True)
    parser_inference.add_argument('--config', type=str, required=True)
    parser_inference.add_argument('--inference-model', type=str)
    parser_inference.add_argument('--inference-house', type=str)
    parser_inference.add_argument('--binary-threshold', type=float, default=None)
    parser_inference.add_argument('--eval-binary-threshold', type=float, default=None)
    parser_inference.add_argument('--model-threshold', type=float, default=None)
    parser_inference.add_argument('--loss-weight-file', type=str, default=None) 
    parser_inference.add_argument('--data-file', type=str, default=None) 
    parser_inference.add_argument('--batch_size', type=int, default=1) 
    parser_inference.add_argument('--cuda', action='store_true', default=False) 
    for p in model_params:
        parser_inference.add_argument('--pm-' + p.replace('_', '-'), type=str, metavar='<{}>'.format(p))

    args = parser.parse_args()
    args = consolidate_args(args)

    # Write out log
    if args.mode == 'inference':
        logs_dir = os.path.join(args.workspace, 'logs', get_filename(__file__), 'inference_logs')
    else:
        logs_dir = os.path.join(args.workspace, 'logs', get_filename(__file__))
    logging = create_logging(logs_dir, filemode='w')
    logging.info(args)

    if args.mode == 'train':
        args.__dict__['basename'] = logging.getLoggerClass().root.handlers[0].baseFilename[:-4]
        config_to_save = args.basename + '.config.json'
        logging.info('Saving config to ' + config_to_save)
        ignores = set(['workspace', 'config', 'cuda', 'mode'])
        with open(config_to_save, 'w') as fout:
            json.dump({k: v for k, v in args.__dict__.items()
                       if k not in ignores}, fout)
        train(args)

    elif args.mode == 'inference':
        inference(args)

    else:
        raise Exception('Error!')

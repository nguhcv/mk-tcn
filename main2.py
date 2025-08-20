import argparse
import os

import numpy as np
import random
import torch
from Conferences.ICIT.utils.ds_reader import DatasetLoader
np.set_printoptions(threshold=np.inf)
from Conferences.MIWAI.core_functions import net_generator as ng
from Conferences.MIWAI.core_functions.model_testing import model_testing
from Conferences.MIWAI.core_functions.model_training import model_training

def main_app(opt, seed):

    set_seed(seed)

    #generate a network
    net = ng.generator(opt=opt)

    # set model_name
    # model_name= ['16-256-3-tcn-linear-gelu-0.2-[64, 64, 64, 64]-True-0.0-mse-NoneadamW']
    model_name = [str(opt.w)
                  +'-'+str(opt.batch_size)
                  +'-'+str(opt.encoder)
                  +'-'+str(opt.decoder)
                  +'-'+str(opt.activation)
                  +'-'+str(opt.dropout)
                  +'-'+str(opt.num_channels)
                  +'-'+str(opt.scheduling)
                  +'-'+str(opt.masked_value)
                  +'-'+str(opt.loss_type)
                  +'-'+str(opt.fuse_type)
                  +'-'+str(opt.lr)]

    #Dataset loader
    ds_loader = DatasetLoader(dataset=opt.dataset, data_path=opt.data_path, ts_num=opt.ts_num, window_size=opt.w)

    # visualize data
    if opt.visualize ==True:
        ds_loader.plot(data_type='train', dataset_name=opt.dataset)
        ds_loader.plot(data_type='test', dataset_name=opt.dataset)
        # breakpoint()

    #train or test
    if opt.mode =='test':
        model_testing(ds_loader= ds_loader, net=net, model_name=model_name, params=opt)

    elif opt.mode =='train':
        model_training (ds_loader = ds_loader, net=net,model_name = model_name, params=opt)
        model_testing(ds_loader=ds_loader, net=net, model_name=model_name, params=opt)



def set_seed(seed=50):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark=False


if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    # training or testing mode
    parser.add_argument('-mode', default='train', help='mode : train or test')
    parser.add_argument('-visualize', default=False, help='mode : train or test')

    # dataset setting
    parser.add_argument('--dimension', default=38,type= int, help= 'number of dimension of dataset')
    parser.add_argument('--dataset', default='smd')
    parser.add_argument('--ts_num', default=0)
    parser.add_argument('--data_pa'
                        'th', default='C:/Research/2025/TSAD/dataset/smd/npk/', help='data-path')
    parser.add_argument('--save_path', default='C:/Research/2025/TSAD/Conferences/MIWAI/saved_models/smd/machine1_1/',
                        help='save-data-path')

    # define net structure 
    parser.add_argument('-encoder', default='mktcn',type=str, help='model type: tcn, modified_tcn')
    parser.add_argument('-decoder', default='linear',type=str, help='model type: linear, nonlinear, tcn_linear, tcn_nonlinear, modified_tcn_linear,modified_tcn_nonlinear' )
    parser.add_argument('-activation', default='gelu',type=str, help='activation in TCN: relu, elu, gelu, leak_relu,swish')


    # training setting
    parser.add_argument('-batch_size', default=256, help='number of samples in each batch')  # 64 or 256
    parser.add_argument('-dropout', default=0.1, help='drop-out ratio')
    parser.add_argument('-w', default=16, type=int, help='window size of each sample')  # 16 or 32 or 64
    parser.add_argument('-n_epochs', default=300, type=int, help='number of training epochs')
    parser.add_argument('-lr', default=9e-5, type=float, help='learning rate ')
    parser.add_argument('-scheduling', default=False, type=bool, help='learning rate scheduling')
    parser.add_argument('-loss_type', default='mse', type=str, help='reconstruction loss')
    parser.add_argument('-masked_value', default='0.', type=float, help='masking value ')
    parser.add_argument('-predicted_length', default='1', type=int, help='prediction length')
    parser.add_argument('-num_channels', default=[32,32,32],  help='a list of number of channels in encoder')

    parser.add_argument('-regularization', default='None', help='regularization')
    parser.add_argument('-test_type', default='normal', type=str, help='test type: online, offline')
    parser.add_argument('-fuse_type', default=6, type=int, help='test type: 1:add, 2:avg, 3:mul, 4:concate, 5:attention, 6:bilinear')


    args = parser.parse_args()
    #print(args)

    main_app(args, seed=60)
    # run(args)

    pass

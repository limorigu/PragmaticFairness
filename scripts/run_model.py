from utils.experiments_utils import initialize_parameters, \
    generate_save_loc_from_args, generate_tensorboard_name
from torch.optim.lr_scheduler import StepLR
from utils import train, test, train_mu_dist, analytic_mu_model, \
    train_outcome_regression, \
        aug_lagrng_lambda_mu_update_step, \
            plot_learned_action_hist, \
            plt_joint_scatter_plots
from nets import MLP, decomposed_additive_MLP, policy_deterministic
from data_utils import get_nycschools_loaders_by_cov,\
                        get_IHDP_loaders_by_cov
from datetime import datetime
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import argparse
import time
import json
from utils import fill_config_args
import matplotlib.pyplot as plt
from pathlib import Path

def run_experiments():
    ### Wrapper function, loads args and 
    ### 1) generate slack plot or 2) run individual model 
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_config', type=str,
                    default='configs/training_configs.yml',
                    dest='training_config',
                    help='config file')
    # Utils
    parser.add_argument("--no-cuda", action="store_true", dest="no_cuda",
                    help="Cuda specification", default=False)

    args = parser.parse_args()
    args = fill_config_args(args)
    print("Using dataset", args.dataset_name)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.slack_vary_plot_generate:
        slack_plot_generate(args)
    else:
        main_routine(args)


def main_routine(args):
    start = time.time()

    # generate save_loc 
    args.save_loc = generate_save_loc_from_args(args)

    if args.const_choice == 'ModBrk':
        # for single model run (inc. hparam config)
        if args.arch == 'additive_outcome_and_determinstic_policy':
            _, Train_loss, Train_obj, Train_const, Train_group_diff, Train_loss_reg_out, *_ = run_model(args)
            if not args.stored_stage1_model:
                plot_single_loss_curve(Train_loss_reg_out, save_dir=args.save_loc)
        else:
            _, Train_loss, Train_obj, Train_const, Train_group_diff = run_model(args)
    
            print("Y_pi_group_diffs shape: ", len(Train_group_diff))
            plot_bbox_results(args, Train_loss, Train_obj, Train_const, Train_group_diff, save_dir=args.save_loc)
    
    if args.const_choice == 'EqB':
        if args.sensitive_binary:
            _, Train_loss, Train_obj, Train_const, Train_oracle_const, _, _ = run_model(args)
        else:
            _, Train_loss, Train_obj, Train_const, Train_oracle_const = run_model(args)
        plot_bbox_results(args, Train_loss, Train_obj, Train_const, Train_oracle_const, save_dir=args.save_loc)

    end = time.time()
    print("time spent on exp: ", end-start)


def plot_bbox_results(args, Train_loss, Train_obj, Train_const, Train_benchmark, save_dir='out/'):
    # plot black box training results
    figure, axis = plt.subplots(2, 2, figsize = (10,8))

    axis[0, 0].plot(list(range(len(Train_loss))), Train_loss)
    axis[0, 0].set_title("Loss Plot")

    axis[0, 1].plot(list(range(len(Train_obj))), Train_obj)
    axis[0, 1].set_title("Objective Plot")
    axis[0, 1].annotate('(%.3f)' %(Train_obj[-1]),
            xy=(len(Train_obj) - 1, Train_obj[-1]), textcoords='data')
    
    axis[1, 0].plot(list(range(len(Train_const))), Train_const)
    axis[1, 0].set_title("Constraints Plot")
    axis[1, 0].annotate('(%.3f)' %(Train_const[-1]),
            xy=(len(Train_const) - 1, Train_const[-1]), textcoords='data')
    
    if args.const_choice == 'ModBrk':
        axis[1, 1].plot(list(range(len(Train_benchmark))), Train_benchmark)
        axis[1, 1].set_title("|E[Y_pi|S=1] - E[Y_pi|S=0]| Plot")
    
    if args.const_choice == 'EqB':
        axis[1, 1].plot(list(range(len(Train_benchmark))), Train_benchmark)
        axis[1, 1].set_title("Oracle Constraints Plot")
        axis[1, 1].annotate('(%.6f)' %(Train_benchmark[-1]),
            xy=(len(Train_benchmark) - 1, Train_benchmark[-1]), textcoords='data')
        
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir + '/bbox_train_plot.png', bbox_inches='tight', dpi = (200))
    plt.clf()

def plot_single_loss_curve(Train_loss_reg_out, save_dir='out/'):
    plt.plot(list(range(len(Train_loss_reg_out))), Train_loss_reg_out)
    plt.title("Loss Plot train reg")

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir + '/reg_out_model_train_plot.png', bbox_inches='tight', dpi = (200))
    plt.clf()

def run_model(args):
    """
   Wrapper function for the training of a phis estimation model
   for as many epochs as user specified. Training per epoch done by
   train_phi_models() in train_test_utils file, but all relevant inputs to
   train_phi_models are initialized here.

     Input:
     - args (run config)
     - train (flag to specify whether to use train or test split of data)
     - nested_model (flag to specify if to train full phi~Z,W model
     or phi~Z model for cond. ind. test)

     Output: model (trained phis pred. model) """
     
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(
        "cuda" if use_cuda else "cpu")

    Train_loss = []
    Train_obj = []
    Train_const = []
    
    if args.sensitive_binary:
        Train_obj_group0 = []
        Train_obj_group1 = []
    if args.const_choice == "ModBrk":
        Train_group_diff = []
        if args.arch == 'additive_outcome_and_determinstic_policy':
            Train_loss_reg_out = []
            first_A_out = []
            last_A_out = []
    
    if args.const_choice == "EqB":
        Train_oracle_const = []
        Train_loss_mu_out = []
        Train_loss_policy_mu_out = []

    if args.dataset_name == 'NYCschools':
        loader_fn = get_nycschools_loaders_by_cov
    elif args.dataset_name == 'IHDP':
        loader_fn = get_IHDP_loaders_by_cov
    elif args.dataset_name == 'Warfarin':
        raise NotImplementedError
    else:
        raise NotImplementedError

    if args.split_dataset:
        train_loader, train_loader_policy = \
            loader_fn(args=args, split='2_splits')
    else:
        train_loader = loader_fn(args=args)

    model, optimizer, scheduler,\
           loss_criterion, epochs, device, lr = initialize_parameters(args)

    # define tensorboard logs saving path
    tensor_log_name = generate_tensorboard_name(args, len(train_loader))

    # initialize tesnorboard writer
    writer = SummaryWriter(
        args.tensorboard_dir + tensor_log_name)

    # Train stage1 for EqB const
    if args.const_choice == "EqB":
        if args.stored_mu_model:
            print("loading pretrained mu model")
            model[0].load_state_dict(torch.load(
                args.saved_model_dir+'mu_model_{}_epochs.pt'.format(args.stage1_epochs)))
            print(args.saved_model_dir+'mu_model_{}_epochs.pt'.format(args.stage1_epochs))
        else:
            print("training mu model")
            for epoch in range(1, args.stage1_epochs + 1):
                print("epoch pretrain mu: " + str(epoch) + ": " + str(datetime.now()))
                losses_mu, mu_model = train_mu_dist(args=args, 
                                                    mu_model=model[0], device=device, 
                                                    optimizer=optimizer[0], train_loader = train_loader, 
                                                    loss_criterion_mu_model=loss_criterion[0], scheduler = scheduler)
                
                writer.add_scalar('Train_loss_mu_out/',
                                            np.asarray(losses_mu).mean(), epoch)
                Train_loss_mu_out.append(np.asarray(losses_mu).mean())
            print("end pretrain mu")
            Path(args.saved_model_dir).mkdir(parents=True, exist_ok=True)
            torch.save(mu_model.state_dict(), 
            args.saved_model_dir+'mu_model_{}_epochs.pt'.format(args.stage1_epochs))
            model[0] = mu_model

        if args.stored_policy_mu_model:
            print("loading pretrained policy mu model")
            model[1].load_state_dict(torch.load(
                args.saved_model_dir+'policy_mu_model_{}_epochs.pt'.format(args.policy_mu_epochs)))
            print(args.saved_model_dir+'policy_mu_model_{}_epochs.pt'.format(args.policy_mu_epochs))
        else:
            print("training policy mu model")
            for epoch in range(1, args.policy_mu_epochs + 1):
                print("epoch pretrain policy mu: " + str(epoch) + ": " + str(datetime.now()))
                losses_mu, policy_mu_model = train_mu_dist(args=args, 
                                                    mu_model=model[1], device=device, 
                                                    optimizer=optimizer[1], train_loader = train_loader, 
                                                    loss_criterion_mu_model=loss_criterion[1], scheduler = scheduler, policymu = True)
                writer.add_scalar('Train_loss_policy_mu_out/',
                                            np.asarray(losses_mu).mean(), epoch)
            print("end pretrain mu")
            Path(args.saved_model_dir).mkdir(parents=True, exist_ok=True)
            torch.save(policy_mu_model.state_dict(), 
            args.saved_model_dir+'policy_mu_model_{}_epochs.pt'.format(args.policy_mu_epochs))
            model[1] = policy_mu_model

    ## if 2-stage process, complete stage 1
    if args.arch == 'additive_outcome_and_determinstic_policy':
        if args.stored_stage1_model:
            print("loading stage 1 model")
            model[0].load_state_dict(torch.load(
                args.saved_model_dir+'stage1_model_{}_epochs_{}_lr_{}_hd.pt'.
                format(args.stage1_epochs, args.stage1_lr, args.stage1_hidden_dim)))
        else:
            print("training stage 1 model")
            for epoch in range(1, args.stage1_epochs + 1):
                print("epoch stage 1: " + str(epoch) + ": " + str(datetime.now()))
                losses_out_reg, model_fixed_reg_outcome = train_outcome_regression(args=args, 
                                                            model_out_reg=model[0], device=device, 
                                                            optimizer=optimizer[0], train_loader=train_loader, 
                                                            loss_criterion_out_reg=loss_criterion[0])

                writer.add_scalar('Train_loss_reg_out/',
                                np.asarray(losses_out_reg).mean(), epoch)
                Train_loss_reg_out.append(np.asarray(losses_out_reg).mean())
            print("end stage 1")
            Path(args.saved_model_dir).mkdir(parents=True, exist_ok=True)
            torch.save(model[0].state_dict(), 
                args.saved_model_dir+'stage1_model_{}_epochs_{}_lr_{}_hd.pt'.
                format(args.stage1_epochs, args.stage1_lr, args.stage1_hidden_dim))
        for name, param in model[0].named_parameters():
            print("model param {} mean: {}".format(name, torch.mean(param)))
        # stage 1 model test performance
        print("************")
        print("model 0 eval")
        print("************")
        model[0].eval()
        if args.action_clip:
            test(model[0], device, train_loader_policy, loss_criterion[0], classification=False)
    
    for epoch in range(1, epochs + 1):
        print("epoch " + str(epoch) + ": " + str(datetime.now()))
        if args.augmented_lagrange:
            print("lambda' before update: ", args.last_lambda)
            args.last_lambda = args.lmbda
            print("lambda' after update: ", args.last_lambda)
        if args.const_choice == 'EqB':

            losses, objs, \
            constraints, oracle_consts, output_A = \
                train(args=args, model=model, device=device, 
                optimizer=optimizer[3], train_loader=train_loader,
                loss_criterion=loss_criterion[3], scheduler=scheduler)

        if args.const_choice == 'ModBrk':
            if args.arch == 'black_box':
                _, losses, objs, \
                constraints, Y_pi_group_diffs = \
                    train(args=args, model=model, device=device, 
                    optimizer=optimizer, train_loader=train_loader, 
                    loss_criterion=loss_criterion, scheduler=scheduler)
            elif args.arch == 'additive_outcome_and_determinstic_policy':
                ## stage 2 model training
                if args.split_dataset:
                    losses, objs, constraints, Y_pi_group_diffs, output_A = \
                        train(args=args, model=model, device=device, 
                        optimizer=optimizer[1], train_loader=train_loader_policy, 
                        loss_criterion=loss_criterion[1], scheduler=scheduler)
                else:
                    losses, objs, constraints, \
                    Y_pi_group_diffs, output_A = \
                            train(args=args, model=model, device=device, 
                            optimizer=optimizer[1], train_loader=train_loader, 
                            loss_criterion=loss_criterion[1], scheduler=scheduler)
                print("mean epoch constraint value: ", np.mean(constraints))
                print("mean epoch loss value: ", np.mean(losses))
        
        if args.sensitive_binary:
            objs_both = objs[0]
            objs_group0 = objs[1]
            objs_group1 = objs[2]
            
        if args.augmented_lagrange:
            args = aug_lagrng_lambda_mu_update_step(args, constraints)

        Train_loss.append(np.asarray(losses).mean())
        if args.sensitive_binary:
            Train_obj.append(np.asarray(objs_both).mean())
            Train_obj_group0.append(np.asarray(objs_group0).mean())
            Train_obj_group1.append(np.asarray(objs_group1).mean())
        else:
            Train_obj.append(np.asarray(objs).mean())
        Train_const.append(np.asarray(constraints).mean())
        if args.const_choice == 'ModBrk':
            Train_group_diff.append(np.asarray(Y_pi_group_diffs).mean())
        if args.const_choice == 'EqB':
            Train_oracle_const.append(np.asarray(oracle_consts).mean())

        # Store values for training monitoring and plotting
        Path(args.save_loc).mkdir(parents=True, exist_ok=True)
        columns = ['pred_A', 'sensitive'] + \
                    ['X'+str(i) for i in range(args.X_dim_total)] + \
                        ['Y_0', 'Y_pi']
        if epoch==1:
            output_A_df = pd.DataFrame(output_A, columns=columns)
            output_A_df.to_csv(args.save_loc+'/output_A_df_1st.csv')
        if epoch==(epochs/2):
            output_A_df = pd.DataFrame(output_A, columns=columns)
            output_A_df.to_csv(args.save_loc+'/output_A_df_half.csv')
        if epoch==epochs:
            output_A_df = pd.DataFrame(output_A, columns=columns)
            output_A_df.to_csv(args.save_loc+'/output_A_df_last.csv')

        writer.add_scalar('Train_loss_/',
                          np.asarray(losses).mean(), epoch)
        writer.add_scalar('Train_constraint/',
                          np.asarray(constraints).mean(), epoch)
        

    # plot action histogram
    plot_learned_action_hist(args.save_loc)
    # save args for reproducibility
    with open(args.save_loc+'args_state.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        
    writer.flush()
    writer.close()
    print("end of train/test", datetime.now())
    
    if args.const_choice == 'ModBrk':
        if args.arch == 'additive_outcome_and_determinstic_policy':
            if args.sensitive_binary:
                return model, Train_loss, Train_obj, Train_const, Train_group_diff, \
                    Train_loss_reg_out, Train_obj_group0, Train_obj_group1
            else:
                return model, Train_loss, Train_obj, Train_const, \
                                Train_group_diff, Train_loss_reg_out
        else:
            return model, Train_loss, Train_obj, Train_const, Train_group_diff
    if args.const_choice == 'EqB':
        if args.sensitive_binary:
            return model, Train_loss, Train_obj, Train_const, Train_oracle_const, \
                      Train_obj_group0, Train_obj_group1
        else:
            return model, Train_loss, Train_obj, Train_const, Train_oracle_const

def slack_plot_generate(args):
    start = time.time()
    slacks = np.linspace(args.min_slack, args.max_slack, args.num_slacks)
    ####
    #Start by generating unconstrained results
    ####
    args.augmented_lagrange = False
    if args.const_choice == 'EqB':
        Train_obj, Train_const, Train_oracle_const = \
                slack_plot_per_setting(args, plotting_stage1=True, plotting_stage2=True)
    else:
        Train_obj, Train_const = \
                slack_plot_per_setting(args, plotting_stage1=True, plotting_stage2=True)
    
    # can just copy the same results across slack values
    if args.sensitive_binary:
        curve_results_uncontrained = {'utility':[Train_obj[0][-1]]*len(slacks), 
                    'utility_group0': [Train_obj[1][-1]]*len(slacks),
                    'utility_group1': [Train_obj[2][-1]]*len(slacks),
                    'constraint': [Train_const[-1]]*len(slacks), 
                        'slack': list(slacks)}
    else:
        if args.const_choice == 'ModBrk':
            curve_results_uncontrained = {'utility':[Train_obj[-1]]*len(slacks), 
                                    'constraint': [Train_const[-1]]*len(slacks), 
                                        'slack': list(slacks)}
        if args.const_choice == 'EqB':
            curve_results_uncontrained = {'utility':[Train_obj[-1]]*len(slacks), 
                                    'constraint': [Train_const[-1]]*len(slacks), 
                                    'oracle constraint': [Train_oracle_const[-1]]*len(slacks),
                                        'slack': list(slacks)}

    #####
    # Next, Baseline 1: unconstrained results, dropping sensitive attribute from inputs
    #####
    # Since we used stage1 model in the above already, we can be certain we can load it
    args.stored_stage1_model = True
    args.drop_sensitive = True
    if args.const_choice == 'EqB':
        args.stored_mu_model = True
        args.stored_policy_mu_model = True

    if args.const_choice == 'EqB':
        Train_obj, Train_const, Train_oracle_const = \
                slack_plot_per_setting(args, plotting_stage1=True, plotting_stage2=True)
    else:
        Train_obj, Train_const = \
                slack_plot_per_setting(args, plotting_stage1=True, plotting_stage2=True)
    # can just copy the same results across slack values
    if args.sensitive_binary:
        curve_results_uncontrained_drop_sensitive = {'utility':[Train_obj[0][-1]]*len(slacks), 
                    'utility_group0': [Train_obj[1][-1]]*len(slacks),
                    'utility_group1': [Train_obj[2][-1]]*len(slacks),
                    'constraint': [Train_const[-1]]*len(slacks), 
                        'slack': list(slacks)}
    else:
        if args.const_choice == 'ModBrk':
            curve_results_uncontrained_drop_sensitive = {'utility':[Train_obj[-1]]*len(slacks), 
                                            'constraint': [Train_const[-1]]*len(slacks), 
                                                'slack': list(slacks)}
        if args.const_choice == 'EqB':
            curve_results_uncontrained_drop_sensitive = {'utility':[Train_obj[-1]]*len(slacks), 
                                    'constraint': [Train_const[-1]]*len(slacks), 
                                    'oracle constraint': [Train_oracle_const[-1]]*len(slacks),
                                        'slack': list(slacks)}
    
    # #####
    # # Continue to generating constrained results, varying slack values
    # #####
    if args.sensitive_binary:
        if args.const_choice == 'ModBrk':
            curve_results_contrained = {'utility':[], 'utility_group0': [], 
                                        'utility_group1': [], 'constraint': [], 'slack': list(slacks)}
        else:
            curve_results_contrained = {'utility':[], 'utility_group0': [], 
                            'utility_group1': [], 'constraint': [], 'oracle constraint': [], 'slack': list(slacks)}
    else:
        if args.const_choice == 'EqB':
            curve_results_contrained = {'utility':[], 'constraint': [], 'oracle constraint': [], 'slack': list(slacks)}
        else:
            curve_results_contrained = {'utility':[], 'constraint': [], 'slack': list(slacks)}
    
    args.augmented_lagrange = True
    args.drop_sensitive = False
    for i, slack in enumerate(slacks):
        args.constraint_thresh = slack
        args.lmbda = args.init_lmbda
        args.mu = args.init_mu
        if args.const_choice == 'EqB':
            Train_obj, Train_const, Train_oracle_const= \
                slack_plot_per_setting(args, plotting_stage1=False, plotting_stage2=True)
        else:
            Train_obj, Train_const = \
                slack_plot_per_setting(args, plotting_stage1=False, plotting_stage2=True)

        assert slack == curve_results_contrained['slack'][i]
        curve_results_contrained['constraint'].append(Train_const[-1])
        if args.const_choice == 'EqB':
            curve_results_contrained['oracle constraint'].append(Train_oracle_const[-1])
        if args.sensitive_binary:
            curve_results_contrained['utility'].append(Train_obj[0][-1])
            curve_results_contrained['utility_group0'].append(Train_obj[1][-1])
            curve_results_contrained['utility_group1'].append(Train_obj[2][-1])
        else:
            curve_results_contrained['utility'].append(Train_obj[-1])
    
    #####
    # Baseline 2: Constant Actions (no need for policy model training)
    #####
    args.constant_action = True
    args.augmented_lagrange = False
    args.lmbda = args.init_lmbda
    assert args.lmbda == 0
    args.epochs = 1
    curve_results_constant_actions = {}
    for action_value in args.constant_action_values:
        print("constant action:", action_value)
        args.constant_action_value = action_value
        if args.const_choice == 'EqB':
            Train_obj, Train_const, Train_oracle_const = \
                slack_plot_per_setting(args, plotting_stage1=False, plotting_stage2=False)
        if args.const_choice == 'ModBrk':
            Train_obj, Train_const = \
                slack_plot_per_setting(args, plotting_stage1=False, plotting_stage2=False)
        if args.sensitive_binary:
            if args.const_choice == 'ModBrk':
                curve_results_constant_actions[action_value] = \
                    {'utility':[Train_obj[0][-1]]*len(slacks), 
                        'utility_group0': [Train_obj[1][-1]]*len(slacks),
                        'utility_group1': [Train_obj[2][-1]]*len(slacks),
                        'constraint': [Train_const[-1]]*len(slacks), 
                            'slack': list(slacks)}
            else:
                curve_results_constant_actions[action_value] = \
                    {'utility':[Train_obj[0][-1]]*len(slacks), 
                        'utility_group0': [Train_obj[1][-1]]*len(slacks),
                        'utility_group1': [Train_obj[2][-1]]*len(slacks),
                        'constraint': [Train_const[-1]]*len(slacks), 
                        'oracle constraint': [Train_oracle_const[-1]]*len(slacks),
                            'slack': list(slacks)}
        else:
            if args.const_choice == 'ModBrk':
                curve_results_constant_actions[action_value] = \
                                    {'utility':[Train_obj[-1]]*len(slacks), 
                                        'constraint': [Train_const[-1]]*len(slacks), 
                                            'slack': list(slacks)}
            if args.const_choice == 'EqB':
                curve_results_constant_actions[action_value] = \
                        {'utility':[Train_obj[-1]]*len(slacks), 
                            'constraint': [Train_const[-1]]*len(slacks), 
                            'oracle constraint': [Train_oracle_const[-1]]*len(slacks),
                                'slack': list(slacks)}
    
    #####
    # Baseline 3: Original Actions (no need for policy model training)
    #####
    args.constant_action = False
    args.original_A = True
    assert args.lmbda == args.init_lmbda
    args.epochs = 1
    if args.const_choice == 'ModBrk':
        Train_obj, Train_const = \
                slack_plot_per_setting(args, plotting_stage1=False, plotting_stage2=False)
    if args.const_choice == 'EqB':
        Train_obj, Train_const, Train_oracle_const = \
                slack_plot_per_setting(args, plotting_stage1=False, plotting_stage2=False)
    # can just copy the same results across slack values
    if args.sensitive_binary:
        curve_results_original_actions = {'utility':[Train_obj[0][-1]]*len(slacks), 
                    'utility_group0': [Train_obj[1][-1]]*len(slacks),
                    'utility_group1': [Train_obj[2][-1]]*len(slacks),
                    'constraint': [Train_const[-1]]*len(slacks), 
                        'slack': list(slacks)}
    else:
        if args.const_choice == 'ModBrk':
            curve_results_original_actions = {'utility':[Train_obj[-1]]*len(slacks), 
                                        'constraint': [Train_const[-1]]*len(slacks), 
                                            'slack': list(slacks)}
        if args.const_choice == 'EqB':
            curve_results_original_actions = {'utility':[Train_obj[-1]]*len(slacks), 
                            'constraint': [Train_const[-1]]*len(slacks), 
                            'oracle constraint': [Train_oracle_const[-1]]*len(slacks),
                                'slack': list(slacks)}
        

    end = time.time()
    print("time spent on exp: ", end-start)

    plt_joint_scatter_plots(args, curve_results_uncontrained, 
                                curve_results_contrained, 
                                curve_results_uncontrained_drop_sensitive, 
                                curve_results_constant_actions, 
                                curve_results_original_actions, save_loc=args.save_loc)


def slack_plot_per_setting(args, plotting_stage1=False, plotting_stage2=False):
    args.save_loc = generate_save_loc_from_args(args)
    # for single model run (inc. hparam config)
    if args.const_choice == 'EqB':
        if args.sensitive_binary:
            _, Train_loss, Train_obj, Train_const, Train_oracle_const, \
                    Train_obj_group0, Train_obj_group1 = run_model(args)
            Train_obj_both = Train_obj
            Train_obj = [Train_obj_both, Train_obj_group0, Train_obj_group1]
        else:
            _, Train_loss, Train_obj, Train_const, Train_oracle_const = run_model(args)
        
        if plotting_stage2:
            if args.sensitive_binary:
                plot_bbox_results(args, Train_loss, Train_obj_both, Train_const, Train_oracle_const, save_dir=args.save_loc)
            else:
                plot_bbox_results(args, Train_loss, Train_obj, Train_const, Train_oracle_const, save_dir=args.save_loc)
            
        return Train_obj, Train_const, Train_oracle_const
    
    if args.const_choice == 'ModBrk':
        if args.arch == 'additive_outcome_and_determinstic_policy':
            if args.sensitive_binary:
                _, Train_loss, Train_obj, Train_const, Train_group_diff, Train_loss_reg_out, \
                        Train_obj_group0, Train_obj_group1 = run_model(args)
                Train_obj_both = Train_obj
                Train_obj = [Train_obj_both, Train_obj_group0, Train_obj_group1]
            else:
                _, Train_loss, Train_obj, Train_const, Train_group_diff, Train_loss_reg_out = run_model(args)
            if plotting_stage1:
                plot_single_loss_curve(Train_loss_reg_out, save_dir=args.save_loc)
        else:
            raise NotImplementedError
    
        if plotting_stage2:
            if args.sensitive_binary:
                plot_bbox_results(args, Train_loss, Train_obj_both, Train_const, Train_group_diff, save_dir=args.save_loc)
            else:
                plot_bbox_results(args, Train_loss, Train_obj, Train_const, Train_group_diff, save_dir=args.save_loc)
    
        return Train_obj, Train_const
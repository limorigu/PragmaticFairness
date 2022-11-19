import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from nets import MLP, decomposed_additive_MLP, policy_deterministic, mu_MLP, policy_probabilistic
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from torch.distributions.normal import Normal
import itertools
import pdb
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from pathlib import Path
import seaborn as sns
import pickle
import math
import random

def initialize_parameters(args):
    """
   Initialize model, loss criteria, optimizer, etc. needed for model training.

     Input:
     - args (run config)

     Output:
     - model (model and its parameters)
     - optimizer (optimizer object)
     - scheduler (scheduler for optimizer)
     - loss_criterion (which criterion to train with)
     - loss_criterion_test (which criterion to report test against)
     - epochs (number of epochs for training)
     - device (CPU/GPU)
     - lr (learning rate)
     - classification (flag for type of problem, T/F) """
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(
        "cuda" if use_cuda else "cpu")
        
    if (args.dataset_name == 'NYCschools'):
        classification = False
    elif (args.dataset_name == 'IHDP'):
        classification = False
    elif (args.dataset_name == 'Warfarin'):
        raise NotImplementedError
    else:
        raise NotImplementedError
    
    # initialize model
    if args.const_choice == 'EqB':
        
        # import oracle parameter for calculating oracle constraint
        file = open(args.save_oracle_model_loc + '.pkl', 'rb')
        mu_params = [item.astype(np.float32) for item in pickle.load(file)]
        file.close()

        # Initialize MLP_Y model using MLP
        mu_model = mu_MLP(args=args,
                            inputSize=args.X_dim_total+args.S_dim + args.A_dim,
                            outputSize=args.Y_dim,
                            hidden_dim=args.stage1_hidden_dim, policymu = False)
        
        # Initialize MLP^A_emptyset model using MLP
        policy_mu_model = mu_MLP(args=args,
                        inputSize=args.X_dim_total+args.S_dim,
                        outputSize=args.A_dim,
                        hidden_dim=args.stage1_hidden_dim, policymu = True)

        # Initialize MLP_Yvar model using MLP for future generality to heterogeneous case, now is not used
        var_model = mu_MLP(args=args,
                inputSize=args.X_dim_total+args.S_dim+ args.A_dim,
                outputSize=args.A_dim,
                hidden_dim=args.stage1_hidden_dim, policymu = False)
        
        # Initialize MLP^A_sigma_A model using MLP 
        if args.drop_sensitive:
            policy_model = policy_probabilistic(args=args, inputSize_A=args.X_dim_total, hidden_dim = args.hidden_dim,
                                outputSize_A=args.A_dim)
        else:
            policy_model = policy_probabilistic(args=args, inputSize_A= args.X_dim_total + args.S_dim, hidden_dim = args.hidden_dim,
                                outputSize_A=args.A_dim)
        
        # Initialize optimizers
        optimizer_mu_model = initialize_optimizer(args, mu_model, stage1 = True)
        optimizer_policy_mu_model = initialize_optimizer(args, policy_mu_model, policystage1 = True)
        optimizer_var_model = initialize_optimizer(args, var_model)
        optimizer_policy_model = initialize_optimizer(args, policy_model)
        
        optimizer = [optimizer_mu_model, optimizer_policy_mu_model,optimizer_var_model, optimizer_policy_model]
        model = [mu_model, policy_mu_model, var_model, mu_params, policy_model]
        
        mu_loss_criterion = F.mse_loss
        obj_loss_criterion = obj_IPW
        loss_criterion = [mu_loss_criterion, F.mse_loss, F.mse_loss, obj_loss_criterion]
            
    
    if args.const_choice == 'ModBrk':
        if args.arch == 'black_box':
            model = MLP(args=args,
                                inputSize=args.inputSize,
                                outputSize=args.outputSize,
                                hidden_dim=args.hidden_dim).to(device)
            print("model: ", model)
            optimizer = initialize_optimizer(args, model)
        elif args.arch == 'additive_outcome_and_determinstic_policy':
            regression_outcome_model = decomposed_additive_MLP(args=args,
                                inputSize_f=args.X_dim_total+args.S_dim, 
                                inputSize_g=args.X_dim_total+args.S_dim+args.A_dim, 
                                inputSize_h=args.X_dim_total+args.A_dim, 
                                hidden_dim=args.stage1_hidden_dim, 
                                outputSize_f=1, outputSize_g=1, outputSize_h=1).to(device)
            optimizer_outcome_model = initialize_optimizer(args, regression_outcome_model, stage1=True)
            if args.drop_sensitive:
                policy_model = policy_deterministic(args=args, inputSize_A=args.X_dim_total, 
                                                hidden_dim=args.hidden_dim, outputSize_A=args.A_dim).to(device)
            else:
                policy_model = policy_deterministic(args=args, inputSize_A=args.X_dim_total+args.S_dim, 
                                                hidden_dim=args.hidden_dim, outputSize_A=args.A_dim).to(device)
            optimizer_policy_model = initialize_optimizer(args, policy_model)
            model = [regression_outcome_model, policy_model]
            optimizer = [optimizer_outcome_model, optimizer_policy_model]
        
        # define loss criterion
        if args.loss_criterion == 'MSE':
            loss_criterion = F.mse_loss   
        elif args.loss_criterion == 'fix_outcome_regression':
            loss_criterion1 = F.mse_loss
            loss_criterion2 = fix_outcome_regression
            loss_criterion = [loss_criterion1, loss_criterion2]
        else:
            raise NotImplementedError

    epochs = args.epochs
    lr = args.lr

    # optional: activate optimizer lr scheduler
    if args.schd:
        if args.use_analytic_mu_model:
            scheduler = StepLR(optimizer, 
            step_size=args.schd_step_size, 
            gamma=args.gamma)
        else:
            scheduler = StepLR(optimizer[1], 
            step_size=args.schd_step_size, 
            gamma=args.gamma)
    else:
        scheduler = None
    return model, optimizer, scheduler,\
           loss_criterion, epochs, device, lr


def initialize_optimizer(args, model, stage1=False, policystage1 = False):
    """
   Helper function for the initialization of optimizer based on model parameters and args specification.

     Input:
     - args (run config)
     - model (contains parameters to initialize optimizer over)

     Output:
     - optimizer """
    # initialize optimizer
    if policystage1:
        lr = args.policy_lr
        print("set policystage1 lr: ", lr)
        weight_decay = args.stage1_weight_decay
    elif stage1:
        lr = args.stage1_lr
        print("set stage1 lr: ", lr)
        weight_decay = args.stage1_weight_decay
    else:
        lr = args.lr
        weight_decay = args.weight_decay
    if args.optimizer == 'sgd':
        optimizer_fn = optim.SGD
        optimizer = optimizer_fn(model.parameters(), lr=lr,
                                 weight_decay=weight_decay,
                                 momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer_fn = optim.Adam
        optimizer = optimizer_fn(model.parameters(), lr=lr,
                                 weight_decay=weight_decay,
                                 betas=(args.beta1, args.beta2))
    else:
        raise NotImplementedError

    return optimizer

def conditional_expectations(S, covs, exp):
    T = covs[:,0].unsqueeze(1)
    T_unique = T.unique()
    Cl = covs[:,1].unsqueeze(1)
    Cl_unique = Cl.unique()
    Cs = covs[:,2].unsqueeze(1)
    Cs_unique = Cs.unique()
    S_unique = np.unique(S.detach().cpu().numpy())
    return torch.stack([torch.mean(exp[(S==s)&(Cl==cl)&(Cs==cs)&(T==t)]) \
        for (s, cl, cs, t) in itertools.product(S_unique, Cl_unique, Cs_unique, T_unique) \
            if (len(exp[(S==s)&(Cl==cl)&(Cs==cs)&(T==t)]) > 0)])

def ecdf(y):
    """
    Plot Empirical CDF for array y

    Parameters
    ----------
    y : Array

    Returns
    -------
    x: An array of jumping points in empirical cdf
    cdf: An array of cumulative sums at point x

    """
    x, counts = np.unique(y, return_counts=True)
    cusum = np.cumsum(counts)
    return x, cusum / cusum[-1]

def evaluate_step(xs, ys, x):
    """
    Given an array of xs and its corresponding ys representing a step-wise function.
    Given a new point x, look for f(x) evaluated on stepwise function.
    Using Binary Search for time complexity here
    """
    # Binary Search
    L = 0 
    R = len(xs) - 1
    while R - L > 1:
        m = math.floor((L+R)/2)
        if xs[m] < x:
            L = m 
        elif xs[m] > x:
            R = m 
        else:
            return ys[m]
    if xs[L]<=x and xs[R]>x:
        return ys[L]
    if L == 0 and x < xs[L]:
        return 0
    if R == len(xs) - 1 and x > xs[R]:
        return ys[R]

def generate_g(beta, S, X, A, rescale=3):
    return torch.matmul(beta, torch.cat((S, X, S*X, A, X*A, S*A, S*A*X), axis=1).T)/torch.tensor(rescale)

def generate_h(gamma, A, X):
    return torch.matmul(gamma,torch.cat((X, A, A*X), axis=1).T)

def composed_Y_no_divide(y, g, h, noise, y0_scale=20):
    return (y0_scale*y + g + h + noise)

def calc_oracle_const(args, pi_baseline, sensitive, covs, mu_model, origy, pi_new = None, output_A_mu = None):
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(
        "cuda" if use_cuda else "cpu")
    res = []
    wsx_linear = torch.tensor(mu_model[0]).to(device)
    wx_linear = torch.tensor(mu_model[1]).to(device)
    noise_mean = torch.tensor(mu_model[2]).to(device)
    beta = torch.tensor(mu_model[3]).reshape(1, -1).to(device)
    gamma = torch.tensor(mu_model[4]).reshape(1, -1).to(device)

    for seed in range(1):
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        A_bsln_mean = (torch.matmul(wsx_linear, covs.T)*sensitive.reshape(-1))**2 + torch.max(torch.matmul(wx_linear, covs.T), torch.tensor(0.))
        noise_A = torch.normal(0.5, args.std, (A_bsln_mean.shape[0], )).to(device)
        A_bsln_w_noise = A_bsln_mean + noise_A
        A_bsln_bounded = A_bsln_w_noise - torch.min(A_bsln_w_noise)/(torch.max(A_bsln_w_noise) - torch.min(A_bsln_w_noise))
        
        if args.constant_action:
            A_new_bounded = output_A_mu 
        elif args.original_A:
            A_new_bounded = output_A_mu
        elif args.drop_sensitive:
            eval_policy_model = policy_probabilistic(args=args, inputSize_A=args.X_dim_total, hidden_dim=args.hidden_dim ,
                                outputSize_A=args.A_dim)
            eval_policy_model.eval()
            state_dict_new = pi_new
            eval_policy_model.load_state_dict(state_dict_new)
            A_new_mean = eval_policy_model(covs)
            A_new = A_new_mean + noise_A.reshape(-1, 1)
            A_new_bounded = A_new - torch.min(A_bsln_w_noise)/(torch.max(A_bsln_w_noise) - torch.min(A_bsln_w_noise))
        else:
            eval_policy_model = policy_probabilistic(args=args, inputSize_A=args.X_dim_total + args.S_dim, hidden_dim=args.hidden_dim ,
                                outputSize_A=args.A_dim).to(device)
            eval_policy_model.eval()
            state_dict_new = pi_new
            eval_policy_model.load_state_dict(state_dict_new)
            A_new_mean = eval_policy_model(covs, sensitive)
            A_new = A_new_mean + noise_A.reshape(-1, 1)
            A_new_bounded = A_new - torch.min(A_bsln_w_noise)/(torch.max(A_bsln_w_noise) - torch.min(A_bsln_w_noise))

        noise_composed_Y = torch.normal(1, args.std, (A_bsln_mean.size(0), 1)).to(device)
        g_bsln = generate_g(beta, sensitive, covs, A_bsln_bounded.reshape(-1, 1)).reshape(-1, 1)
        h_bsln = generate_h(gamma, A_bsln_bounded.reshape(-1, 1), covs).reshape(-1, 1)
        y_bsln = composed_Y_no_divide(origy, g_bsln, h_bsln, noise_composed_Y)
        g_new = generate_g(beta, sensitive, covs, A_new_bounded.reshape(-1, 1)).reshape(-1, 1)
        h_new = generate_h(gamma, A_new_bounded.reshape(-1, 1), covs).reshape(-1, 1)
        y_new = composed_Y_no_divide(origy, g_new, h_new, noise_composed_Y)


            
        y_ite = y_new - y_bsln
        y_ite_a = y_ite[sensitive.squeeze(1) == 1].detach().cpu().numpy()
        y_ite_na = y_ite[sensitive.squeeze(1) == 0].detach().cpu().numpy()
        # Build Empriical CDF
        zsa, cdfa = ecdf(y_ite_a)
        zsna, cdfna = ecdf(y_ite_na)
        empa = []
        empna = []
        for z in np.arange(args.search_range_start, args.search_range_end, 0.1):
            empa.append(evaluate_step(zsa, cdfa, z))
            empna.append(evaluate_step(zsna, cdfna, z))
        
        res.append(np.mean((np.array(empa) - np.array(empna))**2))
        
    return np.mean(res)


def obj_IPW(args, model, baseline_mean, baseline_std,
            obsrvd_action, sensitive, target, covs, pi_baseline, origy, output = None, pi_new = None):

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(
        "cuda" if use_cuda else "cpu")

    fitted_mu_model = model[0].to(device)
    fitted_policymu_model = model[1].to(device)
    mu_params = model[3]
    # Calculate oracle const
    if args.constant_action or args.original_A:
        oracle_const = calc_oracle_const(args, pi_baseline, sensitive, covs, mu_params, origy, output_A_mu = output)
    else:
        oracle_const = calc_oracle_const(args, pi_baseline, sensitive, covs, mu_params, origy, pi_new = pi_new)
        
    mu_bsln = fitted_policymu_model(covs, sensitive)
    # assume homoscedatic variance for A
    sigma_sqrt_bsln = torch.sqrt(torch.mean((obsrvd_action - mu_bsln)**2))
    sigma_sqrt = sigma_sqrt_bsln
    if args.constant_action:
        mu_adj = output 
        sigma_sqrt = 0.01
    elif args.original_A:
        mu_adj = mu_bsln
    else: 
        # adjust output to be a similar range as inputed A
        mu_adj = (output - torch.tensor(args.minA))/torch.tensor(args.rangeA)
    
    distribution = Normal(mu_adj, sigma_sqrt)
    bsln_distribution = Normal(mu_bsln, sigma_sqrt_bsln)
    std_dist = Normal(0, 1)

    IPW = torch.exp(distribution.log_prob(obsrvd_action) - \
        bsln_distribution.log_prob(obsrvd_action))
    IPW_before_mean = target*IPW 
    obj = torch.mean(IPW_before_mean)

    # Start calculating constraint
    search_range = torch.arange(args.search_range_start, args.search_range_end, 0.1).reshape(1, -1, 1).to(device)
    expected_Y_bsln = fitted_mu_model(covs, sensitive, mu_bsln)
    # Assume homoscedastic variance for Y
    std = torch.sqrt((target - expected_Y_bsln)**2).unsqueeze(2)
    expected_Y_pi = fitted_mu_model(covs, sensitive, mu_adj)
    expected_ITE = expected_Y_pi - expected_Y_bsln
    mudiff = expected_ITE.unsqueeze(2)
    numerator = search_range + mudiff 
    lb_inside = std_dist.cdf(numerator/(2.*std))
    lb = torch.where(numerator > 0, lb_inside, torch.zeros_like(numerator))
    ub = torch.where(numerator > 0, torch.ones_like(numerator), std_dist.cdf(numerator/(2.*std)))
    lb_a = torch.mean(lb[sensitive.squeeze(1) == 1], axis = 0)
    lb_na = torch.mean(lb[sensitive.squeeze(1)== 0], axis = 0)
    ub_a = torch.mean(ub[sensitive.squeeze(1)== 1], axis = 0)
    ub_na = torch.mean(ub[sensitive.squeeze(1)== 0], axis = 0)
    const = torch.sum((lb_a - lb_na)**2 + (ub_a - ub_na)**2)

    if args.augmented_lagrange:
        augmented_constraint = \
            aug_lagrng_theta_update_step(args, const)
        loss = -obj + augmented_constraint
    else:
        loss = -obj + args.lmbda * const
    
    if args.sensitive_binary:
        obj_s = torch.mean(IPW_before_mean[sensitive.squeeze(1) == 1])
        obj_ns = torch.mean(IPW_before_mean[sensitive.squeeze(1) == 0])
        objs = [obj, obj_ns, obj_s] 
        return loss, objs, const, oracle_const, expected_ITE
    else:
        return loss, obj, const, oracle_const, expected_ITE
    
def fix_outcome_regression(args, expected_Y_pi, g, sensitive):
    obj = torch.mean(expected_Y_pi)
    S_unique = np.unique(sensitive.detach().cpu().numpy())
    groups_diff = \
        torch.abs(torch.mean(expected_Y_pi[sensitive == 1]) - \
            torch.mean(expected_Y_pi[sensitive == 0]))
    
    if args.sensitive_binary:
        g_diffs_squared = \
            torch.stack([(torch.mean(g[sensitive==s])-
            torch.mean(g[sensitive==s_prime]))**2 
                for (s,s_prime) in itertools.combinations(S_unique, 2)])
        const = torch.sum(g_diffs_squared)
    else:
        const = torch.autograd.grad(g, sensitive, retain_graph=True)[0]

    if args.augmented_lagrange:
        augmented_constraint = \
            aug_lagrng_theta_update_step(args, const)
        loss = -obj + augmented_constraint
    else:
        loss = -obj + args.lmbda * const
    return loss, obj, const, groups_diff

#### Augmented Lagrangian update steps
def aug_lagrng_theta_update_step(args, constraint):
    # NOTE: eq. 9 in original writeup
    const_w_thresh = constraint-args.constraint_thresh
    if (const_w_thresh) <= (-args.last_lambda/args.mu):
        augmented_constraint = -(args.last_lambda**2)/(2*args.mu)
    else:
        augmented_constraint = (args.last_lambda*(const_w_thresh)) + \
                                ((args.mu*(const_w_thresh**2))/2)
    return augmented_constraint

def aug_lagrng_lambda_mu_update_step(args, constraints):
    mean_batches_const_w_thresh = \
            np.mean(np.array(constraints)-args.constraint_thresh)
    if mean_batches_const_w_thresh <= (-args.last_lambda/args.mu):
        args.lmbda = 0
    else:
        args.lmbda = args.last_lambda + \
            (args.mu*mean_batches_const_w_thresh)
    args.mu = args.mu + args.mu_update
    return args

#### location saving
def generate_save_loc_from_args(args):
    save_loc = args.save_loc_base
    if args.slack_vary_plot_generate:
        save_loc += 'slack_vary/'
    if args.drop_sensitive:
        save_loc += 'drop_sensitive/'
    if args.constant_action:
        save_loc += 'constant_action/value_{}_'.format(args.constant_action_values)
    if args.original_A:
        save_loc += 'original_A/'
    save_loc += 'arch_' + args.arch + '_data_split_' + str(args.split_dataset) + \
                     '_epochs_' + str(args.epochs) + '_lr_' + str(args.lr) + '_lmbda_' + str(args.lmbda)
    if args.augmented_lagrange:
            save_loc += '_aug_lagrange_' + str(args.augmented_lagrange) + \
                '_const_thresh_' + str(args.constraint_thresh) + '_mu_' + str(args.mu) + \
                    '_mu_update_' + str(args.mu_update) 
    if args.action_clip:
        save_loc += '_actions_clip_' + str(args.action_clip) + '_inter_criterion_' + str(args.interval_criterion)
        if args.adaptive_epsilon:
            save_loc += '_adaptive_epsilon_' + str(args.adaptive_epsilon) + \
                    '_perc_adapative_epsilon_' + str(args.perc_adapative_epsilon)
        else:
            save_loc += '_action_clip_epsilon_' + str(args.action_clip_epsilon)
    save_loc += '/'
    return save_loc
                      
def generate_tensorboard_name(args, train_loader_len):
    save_loc = '/' + args.dataset_name + \
                    '/arch_' + args.arch + '/data_split_' + str(args.split_dataset) + \
                    '/aug_lagrange_' + str(args.augmented_lagrange) + '/epochs_' + str(args.epochs) + \
                    '/lr_' + str(args.lr) + '/lmbda_' + str(args.lmbda) + \
                    '/loss_crit_' + args.loss_criterion + \
                    '/train_batch_size_' + str(args.train_batch_size) + \
                    '/num_batches_train_' + str(train_loader_len) + \
                    '/hidden_dim' + str(args.hidden_dim) + \
                    '/opt_' + args.optimizer + '/dropout_' + str(args.dropout)

    if args.augmented_lagrange:
            save_loc += '_aug_lagrange_' + str(args.augmented_lagrange) + \
                '_const_thresh_' + str(args.constraint_thresh) + '_mu_' + str(args.mu) + \
                    '_mu_update_' + str(args.mu_update) 
    if args.action_clip:
        save_loc += '_actions_clip_' + str(args.action_clip) + '_action_clip_epsilon_' + \
            str(args.action_clip_epsilon) + '_adaptive_epsilon_' + str(args.adaptive_epsilon) + \
                '_perc_adapative_epsilon_' + str(args.perc_adapative_epsilon) +\
                '_inter_criterion_' + str(args.interval_criterion)
    save_loc += '/'

    save_loc += datetime.now().strftime("%d-%m-%Y_%H-%M-%S")     
    return save_loc          

#### Plots
def plot_learned_action_hist(save_loc):
    output_A_first = pd.read_csv('{}/output_A_df_1st.csv'.format(save_loc))
    plt.hist(output_A_first[output_A_first['sensitive']==1]['pred_A'], label='s=1', alpha=0.3)
    plt.hist(output_A_first[output_A_first['sensitive']==0]['pred_A'], label='s=0', alpha=0.3)
    plt.legend()
    plt.savefig('{}/output_A_df_1st_by_s.png'.format(save_loc))
    plt.clf()

    plt.hist(output_A_first[output_A_first['sensitive']==1]['Y_pi'], label='s=1', alpha=0.3)
    plt.hist(output_A_first[output_A_first['sensitive']==0]['Y_pi'], label='s=0', alpha=0.3)
    plt.legend()
    plt.savefig('{}/Y_pi_df_1st_by_s.png'.format(save_loc))
    plt.clf()

    output_A_last = pd.read_csv('{}/output_A_df_last.csv'.format(save_loc))
    plt.hist(output_A_last[output_A_last['sensitive']==1]['pred_A'], label='s=1', alpha=0.3)
    plt.hist(output_A_last[output_A_last['sensitive']==0]['pred_A'], label='s=0', alpha=0.3)
    plt.legend()
    plt.savefig('{}/output_A_df_last_by_s.png'.format(save_loc))
    plt.clf()

    plt.hist(output_A_last[output_A_last['sensitive']==1]['Y_pi'], label='s=1', alpha=0.3)
    plt.hist(output_A_last[output_A_last['sensitive']==0]['Y_pi'], label='s=0', alpha=0.3)
    plt.legend()
    plt.savefig('{}/Y_pi_df_last_by_s.png'.format(save_loc))
    plt.clf()

def plt_joint_scatter_plots(args, curves_results_bsln, curves_results, 
                            curve_results_uncontrained_drop_sensitive, 
                            curve_results_constant_actions, 
                            curve_results_original_actions, 
                            save_loc):
    save_loc += "slack_vary/"
    Path(save_loc).mkdir(parents=True, exist_ok=True)
    if args.sensitive_binary:
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1,4)
    else:
        fig, (ax0, ax1) = plt.subplots(1,2)
    df_curves_bsln = pd.DataFrame(curves_results_bsln)
    df_curves = pd.DataFrame(curves_results)
    df_curves_bsln1_drop_sen = pd.DataFrame(curve_results_uncontrained_drop_sensitive)
    df_curves_bsln3_orig_A = pd.DataFrame(curve_results_original_actions)

    # plot constrained results
    sns.regplot(x='slack', y='utility', data=df_curves, ax=ax0, label='Const.', ci=None, scatter_kws={'alpha':0.5})
    sns.regplot(x='slack', y='constraint', data=df_curves, ax=ax1, label='Const.', ci=None, scatter_kws={'alpha':0.5})
    # plot unconstrained results
    sns.regplot(x='slack', y='utility', data=df_curves_bsln, ax=ax0, label='Uncost.', ci=None, scatter_kws={'alpha':0.5})
    sns.regplot(x='slack', y='constraint', data=df_curves_bsln, ax=ax1, label='Uncost.', ci=None, scatter_kws={'alpha':0.5})
    # plot baseline 1
    sns.regplot(x='slack', y='utility', data=df_curves_bsln1_drop_sen, ax=ax0, label='Drop S', ci=None, scatter_kws={'alpha':0.5})
    sns.regplot(x='slack', y='constraint', data=df_curves_bsln1_drop_sen, ax=ax1, label='Drop S', ci=None, scatter_kws={'alpha':0.5})
    # plot baseline 2 (iterate through constant_const_value)
    for action_value in args.constant_action_values:
        df_curves_bsln2_constant_values = pd.DataFrame(curve_results_constant_actions[action_value])
        df_curves_bsln2_constant_values.to_csv(save_loc+"df_curves_bsln2_constant_values_actval_{}_{}_epochs.csv".format(action_value, args.epochs))
        sns.regplot(x='slack', y='utility', data=df_curves_bsln2_constant_values, 
                                    ax=ax0, label='Const_A_{}'.format(action_value), ci=None, scatter_kws={'alpha':0.5})
        sns.regplot(x='slack', y='constraint', data=df_curves_bsln2_constant_values, 
                                    ax=ax1, label='Const_A_{}'.format(action_value), ci=None, scatter_kws={'alpha':0.5})
    # plot baseline 3 (original actions)
    sns.regplot(x='slack', y='utility', data=df_curves_bsln3_orig_A, ax=ax0, label='Original', ci=None, scatter_kws={'alpha':0.5})
    sns.regplot(x='slack', y='constraint', data=df_curves_bsln3_orig_A, ax=ax1, label='Original', ci=None, scatter_kws={'alpha':0.5})
    if args.sensitive_binary:
        # plot constrained results
        sns.regplot(x='slack', y='utility_group0', data=df_curves, ax=ax2, label='Const.', ci=None, scatter_kws={'alpha':0.5})
        sns.regplot(x='slack', y='utility_group1', data=df_curves, ax=ax3, label='Const.', ci=None, scatter_kws={'alpha':0.5})
        # plot unconstrained results
        sns.regplot(x='slack', y='utility_group0', data=df_curves_bsln, ax=ax2, label='Uncost.', ci=None, scatter_kws={'alpha':0.5})
        sns.regplot(x='slack', y='utility_group1', data=df_curves_bsln, ax=ax3, label='Uncost.', ci=None, scatter_kws={'alpha':0.5})

        # plot baseline 1
        sns.regplot(x='slack', y='utility_group0', data=df_curves_bsln1_drop_sen, ax=ax2, label='Drop S', ci=None, scatter_kws={'alpha':0.5})
        sns.regplot(x='slack', y='utility_group1', data=df_curves_bsln1_drop_sen, ax=ax3, label='Drop S', ci=None, scatter_kws={'alpha':0.5})
        # plot baseline 2 (iterate through constant_const_value)
        for action_value in args.constant_action_values:
            df_curves_bsln2_constant_values = pd.DataFrame(curve_results_constant_actions[action_value])
            sns.regplot(x='slack', y='utility_group0', data=df_curves_bsln2_constant_values, 
                                        ax=ax2, label='Const_A_{}'.format(action_value), ci=None, scatter_kws={'alpha':0.5})
            sns.regplot(x='slack', y='utility_group1', data=df_curves_bsln2_constant_values, 
                                        ax=ax3, label='Const_A_{}'.format(action_value), ci=None, scatter_kws={'alpha':0.5})
        # plot baseline 3 (original actions)
        sns.regplot(x='slack', y='utility_group0', data=df_curves_bsln3_orig_A, ax=ax2, label='Original', ci=None, scatter_kws={'alpha':0.5})
        sns.regplot(x='slack', y='utility_group1', data=df_curves_bsln3_orig_A, ax=ax3, label='Original', ci=None, scatter_kws={'alpha':0.5})
    
    plt.legend()
    plt.tight_layout()
    plt.legend(loc='center', bbox_to_anchor=(-3, -0.2),
        fancybox=True, shadow=True, ncol=4)
    plt.savefig(save_loc+"joint_scatter_plot_{}_epochs.pdf".format(args.epochs), 
                                                    format='pdf', bbox_inches="tight")

    df_curves_bsln.to_csv(save_loc+"df_curves_bsln_{}_epochs.csv".format(args.epochs))
    df_curves.to_csv(save_loc+"df_curves_{}_epochs.csv".format(args.epochs))
    df_curves_bsln1_drop_sen.to_csv(save_loc+"df_curves_uncontrained_drop_sensitive_{}_epochs.csv".format(args.epochs))
    df_curves_bsln3_orig_A.to_csv(save_loc+"df_curves_bsln3_orig_A_{}_epochs.csv".format(args.epochs))
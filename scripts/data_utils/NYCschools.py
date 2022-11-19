import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from .utils import get_action_min_max_per_discret_context

class NYCschools_data_by_cov(Dataset):
    """ NYCschools dataset class

     Input:
     - dataset_path (path to load data from)
     - T_dim (dimensions of covariates Z)
     - Cl_dim (dimensions of covariates W)
     - Cs_dim (dimensions of covariate X)
     - Y_dim (dimensions of covariate Y) """

    def __init__(self, dataset_path, const_choice, 
                 T_dim, Cl_dim, Cs_dim, S_dim, 
                 A_dim, A_mu_dim, Y_dim, behav_policy_dim,
                 action_clipping=False,
                 adaptive_clipping_epsilon=False,
                 perc_adapative_epsilon=0, 
                 loss_criterion='fix_outcome_regression',
                 interval_criterion='min_max',
                 non_negative_actions=False):
        
        read_in = np.load(dataset_path, allow_pickle=True)
        idx = read_in.files[0]
        self.nycschools = read_in[idx]
        print(self.nycschools.shape[1])
        self.action_clipping=action_clipping
        self.adaptive_clipping_epsilon = adaptive_clipping_epsilon
        self.perc_adapative_epsilon = perc_adapative_epsilon
        self.loss_criterion=loss_criterion
        self.interval_criterion = interval_criterion
        self.non_negative_actions = non_negative_actions
        self.const_choice = const_choice
        # Group all covariates together
        self.X = torch.tensor(self.nycschools[:, :T_dim + Cl_dim + Cs_dim]).float()
        self.S = torch.tensor(self.nycschools
                              [:, T_dim + Cl_dim + Cs_dim:
                                  T_dim + Cl_dim + Cs_dim + S_dim]).float()
        self.A = torch.tensor(self.nycschools
                              [:, T_dim + Cl_dim + Cs_dim + S_dim:
                                  T_dim + Cl_dim + Cs_dim + S_dim + A_dim]).float()
        self.A_mu = torch.tensor(self.nycschools
                                 [:, T_dim + Cl_dim + Cs_dim + S_dim + A_dim:
                                 T_dim + Cl_dim + Cs_dim + S_dim + A_dim + A_mu_dim]).float()
            
        if const_choice == 'EqB':
            
            self.Y = torch.tensor(self.nycschools
                         [:, T_dim + Cl_dim + Cs_dim + S_dim + A_dim + A_mu_dim: 
                          T_dim + Cl_dim + Cs_dim + S_dim + A_dim + A_mu_dim + Y_dim]).float()
            self.Orig_Y = torch.tensor(self.nycschools
                         [:, T_dim + Cl_dim + Cs_dim + S_dim + A_dim + A_mu_dim + Y_dim: 
                          T_dim + Cl_dim + Cs_dim + S_dim + A_dim + A_mu_dim + Y_dim + 1]).float()
            
            if behav_policy_dim != 0:
                self.behav_policy = torch.tensor(self.nycschools
                                                 [:, T_dim + Cl_dim + Cs_dim + S_dim + A_dim + A_mu_dim + Y_dim + 1:]).float()

            assert ((self.nycschools.shape[1]-
            (T_dim + Cl_dim + Cs_dim + S_dim + 
            A_dim + A_mu_dim + behav_policy_dim + 1))==Y_dim)

        
        if const_choice == 'ModBrk':
            if self.action_clipping:
                if self.adaptive_clipping_epsilon:
                    self.mins, self.maxs, self.epsilons = \
                        get_action_min_max_per_discret_context(self.X, self.S, self.A, 
                                                                XS_dim=T_dim + Cl_dim + Cs_dim + S_dim,
                                                                adaptive_epsilon=True, 
                                                                perc_adapative_epsilon=self.perc_adapative_epsilon,
                                                                interval_criterion=self.interval_criterion)
                else:
                    self.mins, self.maxs = \
                        get_action_min_max_per_discret_context(self.X, self.S, self.A, 
                                                                XS_dim=T_dim + Cl_dim + Cs_dim + S_dim,
                                                                interval_criterion=self.interval_criterion)
            self.Y = torch.tensor(self.nycschools
                                     [:, T_dim + Cl_dim + Cs_dim + S_dim + A_dim + A_mu_dim:]).float()
    
            assert ((self.nycschools.shape[1]-
            (T_dim + Cl_dim + Cs_dim + S_dim + 
            A_dim + A_mu_dim))==Y_dim)
    
        print("self.nycschools.shape: ", self.nycschools.shape)
        print("self.X.shape: ", self.X.shape)
        print("self.S.shape: ", self.S.shape)
        print("self.A.shape: ", self.A.shape)
        print("self.A_mu.shape: ", self.A_mu.shape)
        print("self.Y.shape: ", self.Y.shape)
        print("Original mean of target Y: %.4f" %(torch.mean(self.Y)))
        

    def __getitem__(self, index):
        if not isinstance(index, int):
            index = index.cpu().numpy()
        X = self.X[index]
        S = self.S[index]
        A = self.A[index]
        A_mu = self.A_mu[index]
        Y = self.Y[index]

        if self.const_choice == 'EqB':
            Pi_emptyset = self.behav_policy[index]
            orig_Y = self.Orig_Y[index]
            return X, S, A, A_mu, Pi_emptyset, Y, orig_Y, index
        
        if self.const_choice == 'ModBrk':
            if self.action_clipping:
                mins = self.mins[index]
                maxs = self.maxs[index]
                if self.adaptive_clipping_epsilon:
                    epsilons = self.epsilons[index]
                    return X, S, A, mins, maxs, epsilons, Y, index    
                return X, S, A, mins, maxs, Y, index 
            return X, S, A, Y, index

    def __len__(self):
        return len(self.nycschools)

def get_nycschools_loaders_by_cov(args, split='all', 
                                    split_size = 0.5, **kwargs):
    """ helper function to load dataset
    for NYCschools phis model training.
    If used for training, return only train and dev loader.
    Else, return test loader.

     Input:
     - args (run configs from user)
     - split (flag to specify split of interest)
    Output:
    - loader (loader for relevant splot by batch) """
    dataclass = NYCschools_data_by_cov(args.data_save_dir + "_all.npz", args.const_choice, 
                                                        args.T_dim, args.Cl_dim,
                                                        args.Cs_dim, args.S_dim,
                                                        args.A_dim, args.A_mu_dim, 
                                                        args.Y_dim, args.behav_policy_dim,
                                                        action_clipping=args.action_clip,
                                                        adaptive_clipping_epsilon=args.adaptive_epsilon,
                                                        perc_adapative_epsilon=args.perc_adapative_epsilon,
                                                        loss_criterion=args.loss_criterion,
                                                        interval_criterion=args.interval_criterion,
                                                        non_negative_actions=args.non_negative_actions)

    dataset_len = len(dataclass)
    all_loader = DataLoader(dataclass,
                            batch_size=args.train_batch_size, 
                            **kwargs)
    
    if split == 'all':
        return all_loader
    elif split == '2_splits':
        print("all_loader len: ", dataset_len)
        split_size = int(np.floor(split_size * dataset_len))
        print("split_size len: ", split_size)
        remaining_size = dataset_len - split_size
        split1_ds, split2_ds = \
            torch.utils.data.random_split(all_loader.dataset, 
                                    (split_size, remaining_size))

        trainloader_1 = torch.utils.data.DataLoader(split1_ds, 
                                            batch_size=args.train_batch_size,
                                            shuffle=True, **kwargs)
        trainloader_2 = torch.utils.data.DataLoader(split2_ds, 
                                            args.train_batch_size,
                                            shuffle=True, **kwargs)
        return trainloader_1, trainloader_2
    else:
        raise NotImplementedError


def get_full_vector_nycschools(args, split='all'):
    """ helper function to load values for target of interest
    (i.e. covariate of interest) for entire split of interest.

     Input:
     - args (run configs from user)
     - split (split of interest to obtain values from)
    Output:
    - Z, W, X, Y, phis (covariates from dataset) """

    if split == 'all':
        dataset_path = args.data_save_dir + "_all.npz"
    else:
        raise NotImplementedError

    read_in = np.load(dataset_path, allow_pickle=True)
    idx = read_in.files[0]
    nycschools_df = read_in[idx]
    X = torch.tensor(nycschools_df[:, :args.T_dim + args.Cl_dim + args.Cs_dim]).float()
    S = torch.tensor(nycschools_df
                        [:, args.T_dim + args.Cl_dim + args.Cs_dim:
                            args.T_dim + args.Cl_dim + args.Cs_dim + args.S_dim]).float()
    A = torch.tensor(nycschools_df
                        [:, args.T_dim + args.Cl_dim + args.Cs_dim + args.S_dim:
                            args.T_dim + args.Cl_dim + args.Cs_dim + args.S_dim + args.A_dim]).float()
    A_mu = torch.tensor(nycschools_df
                            [:, args.T_dim + args.Cl_dim + args.Cs_dim + \
                                args.S_dim + args.A_dim:
                            args.T_dim + args.Cl_dim + args.Cs_dim + \
                                args.S_dim + args.A_dim + args.A_mu_dim]).float()
    if args.action_clip:
        if args.adaptive_epsilon:
            mins, maxs, epsilons = \
                get_action_min_max_per_discret_context(X, S, A, 
                                                        XS_dim=args.Con_cov_dim + args.Cat_cov_dim + args.S_dim,
                                                        adaptive_epsilon=args.adaptive_epsilon, 
                                                        perc_adapative_epsilon=args.perc_adapative_epsilon,
                                                        interval_criterion=args.interval_criterion,
                                                        non_negative_actions=args.non_negative_actions)
        else:
            mins, maxs = \
                get_action_min_max_per_discret_context(X, S, A, 
                XS_dim=args.T_dim + args.Cl_dim + args.Cs_dim + args.S_dim,
                interval_criterion=args.interval_criterion,
                non_negative_actions=args.non_negative_actions)
    Y = torch.tensor(nycschools_df
                                [:, args.T_dim + args.Cl_dim + args.Cs_dim + \
                                    args.S_dim + args.A_dim + args.A_mu_dim:]).float()

    assert ((nycschools_df.shape[1]-
        (args.T_dim + args.Cl_dim + args.Cs_dim + args.S_dim + 
        args.A_dim + args.A_mu_dim))==args.Y_dim)

    if args.action_clip:
        if args.adaptive_epsilon:
            return X, S, A, mins, maxs, epsilons, Y    
        return X, S, A, mins, maxs, Y   
    return X, S, A, Y

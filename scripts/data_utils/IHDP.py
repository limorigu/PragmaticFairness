import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from .utils import \
    get_action_min_max_per_discret_context,\
        discretize_cont_covs


class IHDP_data_by_cov(Dataset):
    """ IHDP dataset class

     Input:
     - dataset_path (path to load data from)
     - T_dim (dimensions of covariates Z)
     - Cl_dim (dimensions of covariates W)
     - Cs_dim (dimensions of covariate X)
     - Y_dim (dimensions of covariate Y) """

    def __init__(self, dataset_path,
                 Con_cov_dim, Cat_cov_dim, 
                 S_dim, A_dim, Y_dim, 
                 action_clipping=False, 
                 adaptive_clipping_epsilon=False,
                 perc_adapative_epsilon=0,
                 interval_criterion='min_max',
                 non_negative_actions=False):
        read_in = np.load(dataset_path, allow_pickle=True)
        idx = read_in.files[0]
        self.action_clipping = action_clipping
        self.adaptive_clipping_epsilon = adaptive_clipping_epsilon
        self.perc_adapative_epsilon = perc_adapative_epsilon
        self.interval_criterion = interval_criterion
        self.non_negative_actions = non_negative_actions
        self.IHDP = read_in[idx]
        # Group all covariates together
        self.X = torch.tensor(self.IHDP[:, :Con_cov_dim + Cat_cov_dim]).float()
        self.S = torch.tensor(self.IHDP
                              [:, Con_cov_dim + Cat_cov_dim:
                                  Con_cov_dim + Cat_cov_dim + S_dim]).float()
        self.A = torch.tensor(self.IHDP
                              [:, Con_cov_dim + Cat_cov_dim + S_dim:
                                  Con_cov_dim + Cat_cov_dim + S_dim + A_dim]).float()

        if self.action_clipping:
            self.X_discretized = discretize_cont_covs(self.X, Con_cov_dim)

            if self.adaptive_clipping_epsilon:
                self.mins, self.maxs, self.epsilons = \
                    get_action_min_max_per_discret_context(self.X_discretized, self.S, self.A, 
                                                            XS_dim=Con_cov_dim + Cat_cov_dim + S_dim,
                                                            adaptive_epsilon=True, 
                                                            perc_adapative_epsilon=self.perc_adapative_epsilon,
                                                            interval_criterion=self.interval_criterion)
            else:
                self.mins, self.maxs = \
                    get_action_min_max_per_discret_context(self.X_discretized, self.S, self.A, 
                                                            XS_dim=Con_cov_dim + Cat_cov_dim + S_dim,
                                                            interval_criterion=self.interval_criterion)
        self.Y = torch.tensor(self.IHDP
                                 [:, Con_cov_dim + Cat_cov_dim + S_dim + A_dim:]).float()

        assert ((self.IHDP.shape[1]-
        (Con_cov_dim + Cat_cov_dim + S_dim + 
        A_dim))==Y_dim)

        print("self.IHDP.shape: ", self.IHDP.shape)
        print("self.X.shape: ", self.X.shape)
        print("self.S.shape: ", self.S.shape)
        print("self.A.shape: ", self.A.shape)
        print("self.Y.shape: ", self.Y.shape)
        

    def __getitem__(self, index):
        if not isinstance(index, int):
            index = index.cpu().numpy()
        X = self.X[index]
        S = self.S[index]
        A = self.A[index]
        Y = self.Y[index]
        if self.action_clipping:
            mins = self.mins[index]
            maxs = self.maxs[index]
            if self.adaptive_clipping_epsilon:
                epsilons = self.epsilons[index]
                return X, S, A, mins, maxs, epsilons, Y, index    
            return X, S, A, mins, maxs, Y, index    
        return X, S, A, Y, index

    def __len__(self):
        return len(self.IHDP)


def get_IHDP_loaders_by_cov(args, split='all', 
                            split_size = 0.5, **kwargs):
    """ helper function to load dataset
    for IHDP phis model training.
    If used for training, return only train and dev loader.
    Else, return test loader.

     Input:
     - args (run configs from user)
     - split (flag to specify split of interest)
    Output:
    - loader (loader for relevant splot by batch) """
    dataclass = IHDP_data_by_cov(args.data_save_dir + "_all.npz",
                                args.Con_cov_dim, args.Cat_cov_dim, 
                                args.S_dim, args.A_dim, args.Y_dim, 
                                action_clipping=args.action_clip,
                                adaptive_clipping_epsilon=args.adaptive_epsilon,
                                perc_adapative_epsilon=args.perc_adapative_epsilon,
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


def get_full_vector_IHDP(args, split='all'):
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
    IHDP_df = read_in[idx]

    X = torch.tensor(IHDP_df[:, :args.Con_cov_dim + args.Cat_cov_dim]).float()
    S = torch.tensor(IHDP_df
                        [:, args.Con_cov_dim + args.Cat_cov_dim:
                            args.Con_cov_dim + args.Cat_cov_dim + args.S_dim]).float()
    A = torch.tensor(IHDP_df
                        [:, args.Con_cov_dim + args.Cat_cov_dim + args.S_dim:
                            args.Con_cov_dim + args.Cat_cov_dim + \
                                args.S_dim + args.A_dim]).float()
    if args.action_clip:
        X_discretized = discretize_cont_covs(X, args.Con_cov_dim)
        if args.adaptive_epsilon:
            mins, maxs, epsilons = \
                get_action_min_max_per_discret_context(X, S, A, 
                                                        XS_dim=args.Con_cov_dim + args.Cat_cov_dim + args.S_dim,
                                                        adaptive_epsilon=args.adaptive_epsilon, 
                                                        perc_adapative_epsilon=args.perc_adapative_epsilon,
                                                        interval_criterion=args.interval_criterion)
        else:
            mins, maxs = \
                get_action_min_max_per_discret_context(X, S, A, 
                XS_dim=args.Con_cov_dim + args.Cat_cov_dim + args.S_dim,
                interval_criterion=args.interval_criterion)
    Y = torch.tensor(IHDP_df
                            [:, args.Con_cov_dim + args.Cat_cov_dim + \
                                args.S_dim + args.A_dim:]).float()

    assert ((IHDP_df.shape[1]-
        (args.Con_cov_dim + args.Cat_cov_dim + 
        args.S_dim + args.A_dim))==args.Y_dim)
    
    if args.action_clip:
        if args.adaptive_epsilon:
            return X, S, A, mins, maxs, epsilons, Y    
        return X, S, A, mins, maxs, Y
    return X, S, A, Y

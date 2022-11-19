from numpy import not_equal
import numpy as np
import torch
import torch.nn.functional as F

class MLP(torch.nn.Module):
    """ Generic regression MLP model class.

     Input:
     - args (arguments from user)
     - inputSize (input dim)
     - hidden_dim (hidden dim)
     - outputSize (output dim)"""

    def __init__(self, args, inputSize,
                 hidden_dim, outputSize):
        super().__init__()

        self.hidden = torch.nn.Linear(inputSize, hidden_dim)
        self.relu = F.relu
        self.drop = torch.nn.Dropout(p=args.dropout)
        self.hidden2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = F.relu
        self.drop2 = torch.nn.Dropout(p=args.dropout)
        self.output = torch.nn.Linear(hidden_dim, outputSize)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.hidden2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        out = self.output(x)
        return out

class mu_MLP(torch.nn.Module):
    """ regression MLP model to pretrain E[Y|x, s, a] and E[A|s, x]

     Input:
     - args (arguments from user)
     - inputSize (input dim)
     - hidden_dim (hidden dim)
     - outputSize (output dim)"""
     
    def __init__(self, args, inputSize,
             hidden_dim, outputSize, policymu):
        super().__init__()
    
        self.hidden = torch.nn.Linear(inputSize, hidden_dim)
        self.relu = F.relu
        self.drop = torch.nn.Dropout(p=args.dropout)
        self.hidden2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = F.relu
        self.hidden3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = F.relu
        self.drop2 = torch.nn.Dropout(p=args.dropout)
        self.output = torch.nn.Linear(hidden_dim, outputSize)
        self.policymu = policymu
        
    def forward(self, x, s, a=None):
        if self.policymu:
            xsa = torch.cat([x, s], dim=1)
        else:
            xsa = torch.cat([x,s,a], dim = 1)
        xsa = self.hidden(xsa)
        xsa = self.relu(xsa)
        out = self.output(xsa)
        return out
    
# Outcome MLPs
class decomposed_additive_MLP(torch.nn.Module):
    """ decomposed MLP model class, to fit 
    E[Y_pi | x, s] via a shared weights network
    with mid-level representations of f,g,h.

     Input:
     - args (arguments from user)
     - inputSize (input dim)
     - hidden_dim (hidden dim)
     - outputSize (output dim)"""

    def __init__(self, args, inputSize_f, inputSize_g, inputSize_h, 
                hidden_dim, outputSize_f, outputSize_g, outputSize_h):
        super().__init__()
        self.args = args
        # decomposed E[Y_xpi|x,s] into f, g, h
        self.f1 = torch.nn.Linear(inputSize_f, hidden_dim)
        self.f_relu = F.relu
        if args.stage1_extra_layer:
            self.f2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.f_relu2 = F.relu
        self.f3 = torch.nn.Linear(hidden_dim, outputSize_f)

        self.g1 = torch.nn.Linear(inputSize_g, hidden_dim)
        self.g_relu = F.relu
        if args.stage1_extra_layer:
            self.g2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.g_relu2 = F.relu
        self.g3 = torch.nn.Linear(hidden_dim, outputSize_g)

        self.h1 = torch.nn.Linear(inputSize_h, hidden_dim)
        self.h_relu = F.relu
        if args.stage1_extra_layer:
            self.h2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.h_relu2 = F.relu
        self.h3 = torch.nn.Linear(hidden_dim, outputSize_h)


    def forward(self, x, s, a):
        xs = torch.cat([x, s], dim=1)
        xsa = torch.cat([x, s, a], dim=1)
        xa = torch.cat([x, a], dim=1)
        
        f = self.f1(xs)
        f = self.f_relu(f)
        if self.args.stage1_extra_layer:
            f = self.f2(f)
            f = self.f_relu2(f)
        f = self.f3(f)

        g = self.g1(xsa)
        g = self.g_relu(g)
        if self.args.stage1_extra_layer:
            g = self.g2(g)
            g = self.g_relu2(g)
        g = self.g3(g)

        h = self.h1(xa)
        h = self.h_relu(h)
        if self.args.stage1_extra_layer:
            h = self.h2(h)
            h = self.h_relu2(h)
        h = self.h3(h)

        expected_Y = f+g+h
        
        return f, g, h, expected_Y


# NonLinear Policy probabilistic
class policy_probabilistic(torch.nn.Module):
    """ General policy model which returns mean action

    Input:
    - args (arguments from user)
    - inputSize (input dim)
    - outputSize (output dim)
    """
    def __init__(self, args, inputSize_A, hidden_dim, outputSize_A):
        super().__init__()
        self.args = args
        self.A_hidden = torch.nn.Linear(inputSize_A, hidden_dim)
        self.A_relu = F.relu
        self.batch_norm = torch.nn.BatchNorm1d(hidden_dim)

        self.A_hidden2 = torch.nn.Linear(hidden_dim, hidden_dim)
 
        self.A_relu2 = F.relu
        self.A_hidden3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.A_relu3 = F.relu

        self.A_output = torch.nn.Linear(hidden_dim, outputSize_A)
    # A model
    def forward(self, x, s = None):
        if self.args.drop_sensitive:
            xs = x
        else:
            xs = torch.cat([x, s], dim=1)
        x = self.A_hidden(xs)
        x = self.A_relu(x)

        A_out = self.A_output(x)

        return A_out

# Policy MLP
class policy_deterministic(torch.nn.Module):
    """ decomposed MLP model class, to fit 
    E[Y_pi | x, s] via a shared weights network
    with mid-level representations of f,g,h.

     Input:
     - args (arguments from user)
     - inputSize (input dim)
     - hidden_dim (hidden dim)
     - outputSize (output dim)"""

    def __init__(self, args, inputSize_A, 
                hidden_dim, outputSize_A):
        super().__init__()
        self.args = args
        # A network 
        self.A_hidden = torch.nn.Linear(inputSize_A, hidden_dim)
        self.batch_norm = torch.nn.BatchNorm1d(hidden_dim)
        self.A_relu = F.relu
        self.A_drop = torch.nn.Dropout(p=args.dropout)
        self.A_hidden2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_dim)
        self.A_relu2 = F.relu
        self.A_drop2 = torch.nn.Dropout(p=args.dropout)
        self.A_output = torch.nn.Linear(hidden_dim, outputSize_A)

    # A model
    def forward(self, x, s=None, mins=None, maxs=None, epsilons=None):
        if self.args.drop_sensitive:
            x = self.A_hidden(x)
        else:
            xs = torch.cat([x, s], dim=1)
            x = self.A_hidden(xs)

        x = self.batch_norm(x)
        x = self.A_relu(x)
        x = self.A_drop(x)
        x = self.A_hidden2(x)
        x = self.A_relu2(x)
        A_out = self.A_output(x)
        if self.args.discrete_action:
            A_out = self.softmax(A_out)
            if self.args.action_clip:
                raise NotImplementedError
        else:
            print("min A out before sigmoid shift: ", torch.min(A_out))
            print("max A out before sigmoid shift: ", torch.max(A_out))
            if self.args.action_clip:
                if self.args.adaptive_epsilon:
                    A_out = \
                        shift_scale_sigmoid(self.args, mins,
                                                maxs, A_out, epsilons, 
                                                non_negative_actions=self.args.non_negative_actions)
                    # The additions of the small epsilon is to avoid float-point impercisions comparison issues
                    assert torch.all((A_out >= (mins-epsilons-1e-3)) & \
                            (A_out <= (maxs+epsilons+1e-3))), \
                            "violations: min {}, max {}, min_interval: {}, max_interval: {}".format(A_out[(A_out<mins-epsilons).nonzero()],
                                                                                            A_out[(A_out>maxs+epsilons).nonzero()],
                                                                                            (mins-epsilons)[(A_out<mins-epsilons).nonzero()],
                                                                                            (maxs+epsilons)[(A_out>maxs+epsilons).nonzero()])
                else:
                    A_out = \
                        shift_scale_sigmoid(self.args, mins,
                                                maxs, A_out,
                                                non_negative_actions=self.args.non_negative_actions)
                    print("interval for sigmoid: ", mins-self.args.action_clip_epsilon, maxs+self.args.action_clip_epsilon)
                    assert torch.all((A_out >= (mins-epsilons-1e-3)) & \
                            (A_out <= (maxs+epsilons+1e-3))), \
                            "violations: min {}, max {}, min_interval: {}, max_interval: {}".format(A_out[(A_out<mins-self.args.action_clip_epsilon).nonzero()],
                                                                    A_out[(A_out>maxs+self.args.action_clip_epsilon).nonzero()],
                                                                    (mins-self.args.action_clip_epsilon)[(A_out<mins-self.args.action_clip_epsilon).nonzero()],
                                                                    (maxs+self.args.action_clip_epsilon)[(A_out>maxs+self.args.action_clip_epsilon).nonzero()])

        return A_out

# utils
def shift_scale_sigmoid(args, min_value, max_value, 
                        x, epsilons=None, 
                        non_negative_actions=False):
    if args.adaptive_epsilon:
        min_value = min_value - epsilons
        max_value = max_value + epsilons 
    else:
        min_value = min_value - args.action_clip_epsilon
        max_value = max_value + args.action_clip_epsilon  
    if non_negative_actions:
        min_value = min_value.double()
        min_value = torch.where(min_value<0., 0., min_value).float()
        print("min_value type: ", min_value.type()) 
    return min_value + ((max_value-min_value)/(1 + torch.exp(-x)))

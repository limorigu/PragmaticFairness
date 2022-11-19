from pathlib import Path
import matplotlib.pyplot as plt
import torch
import logging
import pdb
import numpy as np
from sklearn.metrics import explained_variance_score, r2_score
    
def analytic_mu_model(args, mu_params, covs, sensitive, A_mean):

    bias = torch.ones(size = (A_mean.size(0), 1))
    X_lin = torch.cat([bias, covs, A_mean, sensitive], axis = 1)
    itrx_term = (args.intrx_coef*A_mean*sensitive).reshape(-1, 1)
    y_mean = torch.matmul(X_lin, mu_params).reshape(-1, 1) + itrx_term
    
    return y_mean


def train_mu_dist(args, mu_model, device, optimizer, 
                  train_loader, loss_criterion_mu_model, fitted_Ymean = None, 
                  scheduler = None, policymu = False, var = False):
    
    losses = []
    
    for batch_idx, (X, S, A, A_mu, Pi, Y, Orig_Y, idx) in enumerate(train_loader):
        if var:
            Ymu_model, _,  Yvar_model, _, _ = mu_model
            fitted_Ymean = Ymu_model(X, S, A)
            losses, Yvar_model = \
                batch_train_mu_model(X, S, A, Y, 
                                     Yvar_model, device, 
                                     optimizer, loss_criterion_mu_model, 
                                     losses, fitted_Ymean = fitted_Ymean, 
                                     scheduler = scheduler, policymu = policymu, var = var)

            return losses, Yvar_model
        else:
            losses, mu_model = \
                batch_train_mu_model(X, S, A, Y, 
                                     mu_model, device, 
                                     optimizer, loss_criterion_mu_model, 
                                     losses, fitted_Ymean = fitted_Ymean, 
                                     scheduler = scheduler, policymu = policymu, var = var)
            return losses, mu_model


def train_outcome_regression(args, model_out_reg, device, optimizer,
                            train_loader, loss_criterion_out_reg, scheduler=None):

    losses = []
    if args.action_clip:
        if args.adaptive_epsilon:
            for batch_idx, (X, S, A, _, _, _, Y, idx) in enumerate(train_loader):
                losses, model_out_reg = \
                                batch_train_outcome_regression(X, S, A, Y, 
                                                                model_out_reg, device,
                                                                optimizer, loss_criterion_out_reg, 
                                                                args, losses, scheduler=scheduler)
        else:    
            for batch_idx, (X, S, A, _, _, Y, idx) in enumerate(train_loader):
                losses, model_out_reg = \
                                batch_train_outcome_regression(X, S, A, Y, 
                                                                model_out_reg, device,
                                                                optimizer, loss_criterion_out_reg, 
                                                                args, losses, scheduler=scheduler)
            
    else:
        for batch_idx, (X, S, A, Y, idx) in enumerate(train_loader):
            losses, model_out_reg = \
                            batch_train_outcome_regression(X, S, A, Y,
                                                            model_out_reg, device,
                                                            optimizer, loss_criterion_out_reg, 
                                                            args, losses, scheduler=scheduler)
    return losses, model_out_reg

def batch_train_outcome_regression(X, S, A, Y, model_out_reg, device, optimizer,
                                    loss_criterion_out_reg, args, losses, scheduler=None):
    X, S, A, Y = X.to(device), S.to(device), \
            A.to(device), Y.to(device)

    optimizer.zero_grad()
    _, _, _, output = model_out_reg(X, S, A)
    loss = loss_criterion_out_reg(output, Y)

    losses.append(loss.detach().cpu().numpy())
    loss.backward(retain_graph=True)
    optimizer.step()
    if scheduler:
        scheduler.step()

    return losses, model_out_reg

def batch_train_mu_model(X, S, A, Y, mu_model, device, optimizer, 
                         loss_criterion_mu_model, losses, fitted_Ymean = None, 
                         scheduler = None, policymu = False, var = False):
    
    X, S, A, Y = X.to(device), S.to(device), A.to(device), Y.to(device)
    mu_model = mu_model.to(device)
    optimizer.zero_grad()
    if policymu:
        output = mu_model(X, S)
        loss = loss_criterion_mu_model(output, A)
    elif var:
        fitted_Ymean = fitted_Ymean.to(device)
        target = (Y - fitted_Ymean)**2
        output = mu_model(X, S, A)
        loss = loss_criterion_mu_model(output, target)
    else:
        output = mu_model(X, S, A)
        loss = loss_criterion_mu_model(output, Y)
    losses.append(loss.detach().cpu().numpy())
    loss.backward()
    optimizer.step()
    if scheduler:
        scheduler.step()
    
    return losses, mu_model


def train(args, model, device, optimizer,
          train_loader, loss_criterion,
          scheduler=None):
    """ Train given model in batches.

     Input:
     - model (model to train),
     - device (device to be used),
     - optimizer (optimizer initialized with model's parameters),
     - train_loader (loader to iterate over batches from),
     - loss_criterion (form of objective function)
     - scheduler (optional, if passed, scheduler object)
     - option: toggle lr scheduler as needed """

    losses = []
    objs = []
    consts = []
    if args.const_choice == 'EqB':
        oracle_consts = []
        policy_model = model[4]
    else:
        Y_pi_group_diffs = []
        
    if args.sensitive_binary:
        objs_group0 = []
        objs_group1 = []
        objs_both = []
        objs = [objs_both, objs_group0, objs_group1]
    
    if args.const_choice == 'EqB':
        for batch_idx, (X, S, A, A_mu, Pi , Y, Orig_Y, idx) in enumerate(train_loader):
            X, S, A, A_mu, Pi, Y, Orig_Y = X.to(device), S.to(device), \
                A.to(device), A_mu.to(device), Pi.to(device), Y.to(device), Orig_Y.to(device)
            policy_model = policy_model.to(device)
            S.requires_grad = True
            if args.constant_action:
                output_A_mu = torch.full(A.shape, args.constant_action_value, 
                                                device=device, dtype=torch.float)
                loss, obj, constraint, oracle_const, expected_ITE = loss_criterion(
                    args, model, A_mu, args.std, A, S, Y, X, Pi, Orig_Y, output = output_A_mu)
            elif args.original_A: 
                output_A_mu = A
                loss, obj, constraint, oracle_const, expected_ITE = loss_criterion(
                    args, model, A_mu, args.std, A, S, Y, X, Pi, Orig_Y, output = output_A_mu)
            else:
                optimizer.zero_grad()
                if args.drop_sensitive:
                    output_A_mu = policy_model(X)
                else:
                    output_A_mu = policy_model(X, S) # output =[new_A_mu, new_A]
               
                pi_new = policy_model.state_dict()
                loss, obj, constraint, oracle_const, expected_ITE = loss_criterion(
                    args, model, A_mu, args.std, A, S, Y, X, Pi, Orig_Y, output = output_A_mu, pi_new = pi_new)
           
            if not args.constant_action:
                output_A_mu = (output_A_mu - torch.tensor(args.minA))/torch.tensor(args.rangeA)
            losses, objs, consts, oracle_consts = \
                batch_wise_train_optimizer_and_record(args, losses, objs, 
                consts, oracle_consts, loss, optimizer, obj, constraint, 
                oracle_const, scheduler = scheduler)

        return losses, objs, consts, oracle_consts , \
                np.column_stack((output_A_mu.detach().cpu().numpy(), 
                S.detach().cpu().numpy(), X.detach().cpu().numpy(), 
                Y.detach().cpu().numpy(), expected_ITE.detach().cpu().numpy()))
                
    if args.const_choice == 'ModBrk': 
        if args.action_clip:
            if args.adaptive_epsilon:
                for batch_idx, (X, S, A, mins, maxs, epsilons, Y, idx) in enumerate(train_loader):
                    loss, obj, constraint, group_diff, output_A, expected_Y_pi = \
                        batch_wise_train_policy(args, S, X, A, Y,
                                                        model, device, 
                                                        optimizer, loss_criterion,
                                                        mins=mins, maxs=maxs, epsilons=epsilons)
                    losses, objs, consts, Y_pi_group_diffs = \
                        batch_wise_train_optimizer_and_record(args, losses, objs, 
                                                        consts, Y_pi_group_diffs, 
                                                        loss, optimizer, obj, 
                                                        constraint, group_diff, scheduler=scheduler)
            else:
                for batch_idx, (X, S, A, mins, maxs, Y, idx) in enumerate(train_loader):
                    loss, obj, constraint, group_diff, output_A, expected_Y_pi = \
                        batch_wise_train_policy(args, S, X, A, Y,
                                                        model, device, 
                                                        optimizer, loss_criterion,
                                                        mins=mins, maxs=maxs)
                    losses, objs, consts, Y_pi_group_diffs = \
                        batch_wise_train_optimizer_and_record(args, losses, objs, 
                                                        consts, Y_pi_group_diffs, 
                                                        loss, optimizer, obj, 
                                                        constraint, group_diff, scheduler=scheduler)
        else:
            for batch_idx, (X, S, A, Y, idx) in enumerate(train_loader):      
                loss, obj, constraint, group_diff, output_A, expected_Y_pi = \
                    batch_wise_train_policy(args, S, X, A, Y,
                                                    model, device, 
                                                    optimizer, loss_criterion)

                losses, objs, consts, Y_pi_group_diffs = \
                    batch_wise_train_optimizer_and_record(args, losses, objs, 
                                                        consts, Y_pi_group_diffs, 
                                                        loss, optimizer, obj, 
                                                        constraint, group_diff, scheduler=scheduler)

    if args.loss_criterion == 'fix_outcome_regression':
        return losses, objs, consts, Y_pi_group_diffs, \
            np.column_stack((output_A.detach().cpu().numpy(), 
            S.detach().cpu().numpy(), X.detach().cpu().numpy(), 
            Y.detach().cpu().numpy(), expected_Y_pi.detach().cpu().numpy()))
    else:
        return losses, objs, consts, Y_pi_group_diffs


def batch_wise_train_optimizer_and_record(args, losses, objs, 
                                            consts, Y_pi_group_diffs, 
                                            loss, optimizer, obj, 
                                            constraint, group_diff, scheduler=None):
    # Note: this is a wrapper function that just means to clean up the 
    # saving process of all batch-speicifc stats, and take the optimizer step. 
    # It is meant to save the need to write these lines for each loop, targeted by differe
    # flags for modes of running the code
    losses.append(loss.detach().cpu().numpy())
    if args.sensitive_binary:
        objs[0].append(obj[0].detach().cpu().numpy())
        objs[1].append(obj[1].detach().cpu().numpy())
        objs[2].append(obj[2].detach().cpu().numpy())
    else:
        objs.append(obj.detach().cpu().numpy())
    consts.append(constraint.detach().cpu().numpy())
    if args.const_choice == 'EqB':
        Y_pi_group_diffs.append(group_diff)
    if args.const_choice == 'ModBrk':
        Y_pi_group_diffs.append(group_diff.detach().cpu().numpy())

    if not args.constant_action:
        loss.backward(retain_graph=True)
        optimizer.step()
        if scheduler:
            scheduler.step()

    return losses, objs, consts, Y_pi_group_diffs

def batch_wise_train_policy(args, S, X, A, Y, model, device, optimizer,
                            loss_criterion, mins=None, maxs=None, epsilons=None):
    outcome_regression_model, policy_model = model[0], model[1]
    print("S type: ", S.type())
    print("X type: ", X.type())
    print("A type: ", A.type())
    print("Y type: ", Y.type())
    S.requires_grad = True
    X, S, A, Y = X.to(device), S.to(device), \
                    A.to(device), Y.to(device)
    if type(mins) != type(None):
        mins = mins.to(device)
    if type(maxs) != type(None):
        maxs = maxs.to(device)
    if type(epsilons) != type(None):
        epsilons = epsilons.to(device)
    print("device: ", device)
    if args.constant_action:
        output_A = torch.full(A.shape, args.constant_action_value, 
                                        device=device, dtype=torch.float)
        print("output_A shape: ", output_A.shape)
        print("output_A dtype: ", output_A.dtype)
    elif args.original_A: 
        output_A = A
    else:
        optimizer.zero_grad()
        if args.action_clip:
            if args.adaptive_epsilon:
                output_A = policy_model(X, S, 
                                        mins=mins, maxs=maxs, 
                                        epsilons=epsilons)
            else:
                output_A = policy_model(X, S, 
                                        mins=mins, maxs=maxs)
        else:
        ### Note for args.drop_sensitive: this is applied within policy_deterministic() in NN.py
        ### This was a choice model to avoid explosion 
        ### of even more cases that includes both action clipping and sensitive_drop. 
        ### Will consider making more interpretable in the future
            output_A = policy_model(X,S)
    print("Output_A: ", output_A)
    f, g, h, expected_Y_pi = outcome_regression_model(X, S, output_A)
    print("expected_Y_pi: ", expected_Y_pi)
    if args.augmented_lagrange:
        loss, obj, constraint, group_diff = \
            loss_criterion(args, expected_Y_pi, g, S)
    else:
        loss, obj, constraint, group_diff = \
            loss_criterion(args, expected_Y_pi, g, S)

        print("loss: ", loss)
        print("obj: ", obj)
    if args.sensitive_binary:
        obj0 = torch.mean(expected_Y_pi[S==0])
        obj1 = torch.mean(expected_Y_pi[S==1])
        obj_both = obj
        obj = [obj_both, obj0, obj1]
    return loss, obj, constraint, group_diff, output_A, expected_Y_pi

def test(model, device, test_loader,
         loss_criterion, classification):
    """ Test given model in batches.

     Input:
     - model (model to test),
     - device (device to be used),
     - test_loader (loader to iterate over batches from),
     - loss_criterion (form of objective function)
     - classification (flag to indicate whether the model is class. or reg.
     this will matter for reporting accuracy/variance exp.)
     Output: test_loss (avg. loss across batches, optional) """
    test_loss = 0
    k = 0
    if classification:
        correct = 0
    else:
        outputs = []
        targets = []
    with torch.no_grad():
        for batch_idx, (X, S, A, mins, maxs, epsilons, Y, idx) in enumerate(test_loader):
            X, S, A, Y = X.to(device), S.to(device), \
                A.to(device), Y.to(device)
            _, _, _, output = model(X, S, A)
            print("Y.shape: ", Y.shape)
            print("output.shape: ", output.shape)
            # sum up batch loss
            test_loss += loss_criterion(output, Y).item()
            print("test loss: ", test_loss)
            # classification perf.: count correct pred.
            if classification:
                pred = output > 0.5
                if batch_idx == 0:
                    k = Y.shape[1]
                correct += pred.eq(Y.view_as(pred)).sum().item()
            # regression perf.: accum. outputs for var. exp.
            else:
                if output.shape[1] > 1:
                    if type(outputs) == list:
                        outputs = output.cpu().numpy()
                        targets = Y.cpu().numpy()
                    else:
                        outputs = np.concatenate((outputs,
                                                  output.cpu().numpy()))
                        targets = np.concatenate((targets,
                                                  Y.cpu().numpy()))
                else:
                    outputs.append(output.cpu().numpy())
                    targets.append(Y.cpu().numpy())
    # avg. batch args, losses
    test_loss /= len(test_loader.dataset)

    print('Test results ...')

    if classification:
        print('\nTest on 20% train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, k * len(test_loader.dataset),
                                100. * (correct / (k * len(test_loader.dataset)))))

    else:
        targets = np.concatenate(targets).ravel()
        outputs = np.concatenate(outputs).ravel()
        print('\nTest set: Average loss: {:.4f}, explained var.: {:.4f}, r2 score: {:.4f}\n'.format(
            test_loss, explained_variance_score(targets, outputs), r2_score(targets, outputs)))
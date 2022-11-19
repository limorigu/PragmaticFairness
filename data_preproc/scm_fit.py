from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import statsmodels.formula.api as sm
import statsmodels.formula.api as smf
import numpy as np
from sklearn.utils import resample

class SCM:
    def __init__(self, X, S, A, Y, \
                X_cat_columns, X_cont_columns, X_count_columns, \
                S_type, A_type, Y_type, rmse_noise_add_to_reg=True, \
                    duplicate_exogenous=False, bootstrap_exogenous=False):
        self.X_cat_columns = X_cat_columns
        self.X_cont_columns = X_cont_columns
        self.X_count_columns = X_count_columns
        self.A_type = A_type
        self.S_type = S_type
        self.Y_type = Y_type
        self.rmse_noise_add_to_reg = rmse_noise_add_to_reg
        self.duplicate_exogenous = duplicate_exogenous
        self.bootstrap_exogenous = bootstrap_exogenous
        self.X = X
        self.S = S
        self.A = A
        self.Y = Y

        self.fitted_SCM = {}
        self.fit_SCM()

    def fit_target(self, data, target, var_type, var_name):
        # wrapper function, fit model based on target variable type
        if var_type == 'cat':
            fit_fn = self.fit_catgoritcal_target(data, target, var_name)
        elif var_type == 'cont':
            fit_fn = self.fit_continuos_target(data, target, var_name)
        elif var_type == 'count':
            fit_fn = self.fit_count_target(data, target, var_name)
        return fit_fn

    def fit_catgoritcal_target(self, data, target, var_name):
        # generic helper function, fit categorical target with a Decision Tree Classifier
        # input: data and target to fit
        # output: fitted model
        fit_fn = DecisionTreeClassifier(random_state=42, max_depth=12)
        fit_fn.fit(data, target)
        self.fitted_SCM[var_name+'_fn'] = fit_fn

    def fit_continuos_target(self, data, target, var_name):
        # generic helper function, fit continuous target with a Decision Tree Regressor
        # input: data and target to fit
        # output: fitted model
        fit_fn = DecisionTreeRegressor(random_state=42, max_depth=12)
        fit_fn.fit(data, target)
        predict = fit_fn.predict(data)
        resid = target-predict
        rmse = np.sqrt(np.mean(resid ** 2))
        r2 = r2_score(target, predict)
        print("r2 score {}: {}".format(var_name, r2))
        if r2 == 1:
            print("rmse: ", rmse)
        self.fitted_SCM[var_name+'_fn'] = fit_fn
        self.fitted_SCM[var_name+'_rmse'] = rmse

    def fit_count_target(self, data, target, var_name):
        # generic helper function, fit continuous target with a Support Vector Regression
        # input: data and target to fit
        # output: fitted model
        fit_fn = HistGradientBoostingRegressor(loss="poisson", random_state=42, max_leaf_nodes=128)
        fit_fn.fit(data, target)
        predict = fit_fn.predict(data)
        resid = target-predict
        rmse = np.sqrt(np.mean(resid ** 2))
        r2 = r2_score(target, predict)
        print("r2 score {}: {}".format(var_name, r2))
        self.fitted_SCM[var_name+'_fn'] = fit_fn
        self.fitted_SCM[var_name+'_rmse'] = rmse
        self.fitted_SCM[var_name+'_mean'] = np.mean(target)

    def sample_categorical_target(self, data, variable_name):
        predict_proba_variable = self.fitted_SCM[variable_name+'_fn'].predict_proba(data)
        variable_SCM = np.vstack([np.random.choice(predict_proba_variable.shape[1], 
                            size=(1, 1), p=predict_proba_variable[row, :]) 
                            for row in range(predict_proba_variable.shape[0])])

        return variable_SCM
    
    def sample_continuous_target(self, data, variable_name):
        print("variable_name from cont sample: ", variable_name)
        prediction = self.fitted_SCM[variable_name+'_fn'].predict(data)
        if self.rmse_noise_add_to_reg:
            if 'count' in variable_name:
                data_adaptive_noise = \
                    np.random.poisson(lam=self.fitted_SCM[variable_name+'_rmse'], 
                                                        size=prediction.shape)
            else:
                data_adaptive_noise = \
                    np.random.normal(scale=self.fitted_SCM[variable_name+'_rmse'], 
                                                    size=prediction.shape)
            variable_SCM = (prediction + data_adaptive_noise).reshape(-1, 1)
        else:
            variable_SCM = prediction.reshape(-1, 1)
        return variable_SCM

    def sample_target(self, data, var_type, variable_name):
        if (var_type == 'cat'):
            sampled_target = self.sample_categorical_target(data, variable_name)
        elif (var_type == 'cont') or (var_type == 'count'):
            sampled_target = self.sample_continuous_target(data, variable_name)
        else:
            raise NotImplementedError
        return sampled_target

    def fit_SCM(self):
        np.random.seed(42)
        S_reshaped = np.array(self.S).reshape(-1, 1)
        A_reshaped = np.array(self.A).reshape(-1, 1)
        assert np.all(np.vstack((self.S))==S_reshaped)
        if (not self.duplicate_exogenous) and (not self.bootstrap_exogenous):
            # fit X_categorical (col by col)
            [self.fit_target(S_reshaped, 
                self.X[self.X_cat_columns].iloc[:, col], 'cat', 'X_cat_'+str(col)) 
                for col in range(len(self.X_cat_columns))]
            
            # fit X_continuous (col by col)
            [self.fit_target(S_reshaped, 
                self.X[self.X_cont_columns].iloc[:, col], 'cont', 'X_cont_'+str(col)) 
                for col in range(len(self.X_cont_columns))]

            # fit X_count (col by col)
            [self.fit_target(S_reshaped, 
                self.X[self.X_count_columns].iloc[:, col], 'count', 'X_count_'+str(col)) 
                for col in range(len(self.X_count_columns))]
        
        # fit A
        self.fit_target(np.hstack((S_reshaped, self.X)), self.A, self.A_type, 'A')
        # fit Y
        self.fit_target(np.hstack((S_reshaped, self.X, A_reshaped)), self.Y, self.Y_type, 'Y')

    def exogenous_generate(self, n):
        if self.duplicate_exogenous:
            times_to_dup = int(n/len(self.S))
            dup_S = np.array(self.S)
            dup_S = np.tile(dup_S, times_to_dup).reshape(-1, 1)

            dup_X = np.array(self.X)
            dup_X = np.tile(dup_X, (times_to_dup, 1))

            return dup_S, dup_X
        elif self.bootstrap_exogenous:
            np_S = np.array(self.S)
            boot_S = resample(np_S, replace=True, n_samples=n, random_state=42).reshape(-1, 1)

            np_X = np.array(self.X)
            boot_X = resample(np_X, replace=True, n_samples=n, random_state=42)

            return boot_S, boot_X
        else:
            if self.S_type == 'cat':
                # check if binary sensitive attribute
                if np.all(np.unique(np.array(self.S)) == [0,1]):
                    num_1s = np.sum(np.array(self.S))
                    prob_1 = num_1s/len(self.S)
                    sampled_S = np.random.binomial(1, prob_1, size=n)
            else:
                raise NotImplementedError

            return sampled_S

    def sample_SCM(self, n):
        if (self.duplicate_exogenous) or (self.bootstrap_exogenous):
            sampled_S, sampled_X = self.exogenous_generate(n)
        else:
            # sample S
            sampled_S = self.exogenous_generate(n)
            sampled_S = sampled_S.reshape(-1, 1)
            # sample X cat
            sampled_X_cat = \
                np.hstack([self.sample_target(sampled_S, 'cat', 'X_cat_'+str(col)) 
                                            for col in range(len(self.X_cat_columns))])
            assert sampled_X_cat.shape==(n, len(self.X_cat_columns))
            # sample X cont
            samples_cont = [self.sample_target(sampled_S, 'cont', 'X_cont_'+str(col)) 
                                            for col in range(len(self.X_cont_columns))]
            sampled_X_cont = \
                np.hstack(samples_cont)
            assert sampled_X_cont.shape==(n, len(self.X_cont_columns))
            # sample X count
            samples_count = [self.sample_target(sampled_S, 'count', 'X_count_'+str(col)) 
                                            for col in range(len(self.X_count_columns))]
            sampled_X_count = \
                np.hstack(samples_count)
            assert sampled_X_count.shape==(n, len(self.X_count_columns))

            sampled_X = np.hstack((sampled_X_cat, sampled_X_cont, sampled_X_count))
        
        # sample A
        sampled_A = \
            self.sample_target(
                np.hstack((sampled_S, sampled_X)), self.A_type, 'A')
        # find A_col size
        if len(self.A.shape)==1:
            A_size = 1
        else:
            A_size = self.A.shape[1]
        assert sampled_A.shape==(n, A_size)
        # sample Y
        sampled_Y = \
            self.sample_target(
                np.hstack((sampled_S, sampled_X, sampled_A)), self.Y_type, 'Y')
        # find Y_col size
        if len(self.Y.shape)==1:
            Y_size = 1
        else:
            Y_size = self.Y.shape[1]
        assert sampled_Y.shape==(n, Y_size)

        if (self.duplicate_exogenous) or (self.bootstrap_exogenous):
            return sampled_S, sampled_X, sampled_A, sampled_Y
        else:
            return sampled_S, sampled_X_cat, sampled_X_cont, sampled_X_count, sampled_A, sampled_Y

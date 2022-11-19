# Pragmatic Fairness: Developing Policies with Outcome Disparity Control

## Steps to reproduce results from the submission.

0. Set up Environment (optional)
   - `conda env create -f EqB_environment.yml` (or ModBrk_envrionment.yml) 
   - `conda activate EqB_environment` (or ModBrk_envrionment)
1. Paste experiment specific configs into training_configs.yml
   - NYCSchools_EqB_configs.yml (for EqB const)
   - NYCSchools_ModBrk_configs.yml (for ModBrk const)
2. Run `python main.py`

We will see results saved in out/ folder. We also include the results we obtained by running the code, and how we plot them in manuscript, in the plot/ folder.

## Code to reproduce results from the submission.

### data_preproc
  - NYCSchools_dataset_EqB_clean.ipynb - contains data preprocessing needed for EqB constraint
  - NYCSchools_dataset_ModBrk_clean.ipynb - contains data preprocessing needed for ModBrk constraint, NYC dataset
  - IHDP_dataset_ModBrk_clean.ipynb - contains data preprocessing needed for EqB constraint, IHDP dataset
  
### scripts
  - main.py - includes main function, where the entire pipeline is run from. 
  - run_model.py - contains the main routine for the training of the policy models as well as record results
  - configs
    - training_config.yml - should paste from experiment specific configs and run with python main.py
    - NYCSchools_EqB_configs.yml
    - NYCSchools_ModBrk_configs.yml
  - data - contains processed data from according to data_preproc/ notebooks
  - data_utils
    - NYCschools.py or IHDP.py - load preprocessed data to torch-ready formats
  - utils
    - experiments_utils.py - consists most helper functions for model training
    - train_test_utils.py - contains the core train and test functions for a single epoch
  - nets
    - NNs.py - contains class of NN that makes up our pretrained MLPs and policy models


############################################################################
##                 boilerplate.py                                        ###
##   All the functions that are used daily, written here for ease of use ###
############################################################################
import pandas as pd
import numpy as np


def get_data(dataset_name):
    """
    Input 'quac' or 'squad' to get either of the data 
    Returns the datset with keys 'train' and 'validation'
    """
    # Colab setup
    import os, sys, subprocess
    # if "google.colab" in sys.modules:
    cmd = "pip install datasets -q"
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    from datasets import load_dataset
    dataset = load_dataset(dataset_name)
    return dataset

def data_to_dataframe(dataset):
    """
    Converts the data to pandas dataframe
    Returns one dataframe which has all columns and another which has context
    and questions only
    """
    data_dict = dataset.shape
    if (data_dict["train"] == (11567,11)):
    	data_name = "quac"
    else:
    	data_name = "squad"
    train = dataset["train"]
    valid = dataset["validation"]
    df_train = pd.DataFrame(train)
    df_valid = pd.DataFrame(valid)
    ques = "questions" if data_name=="quac" else "question"
    simple_traindf = pd.DataFrame(columns = ["source", "target"])
    simple_traindf["source"] = df_train["context"]
    simple_traindf["target"] = df_train[ques]
    simple_validdf = pd.DataFrame(columns = ["source", "target"])
    simple_validdf["source"] = df_valid["context"]
    simple_validdf["target"] = df_valid[ques]
    return df_train, df_valid, simple_traindf, simple_validdf

def preprocess_sentence(sentence):
    pass

def create_dataset(training_df):
    pass

def wandb_tracker(wandb_tracker_variable,
				  project_name="Automated-Question-Generation",
				  entity="saty"):

	import wandb
	wandb.init(project=project_name, entity=entity)
	wandb_tracker_variable = True
	return wandb_tracker_variable

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

hydrothermal_deposits_df = pd.read_csv("Hypodermal_deposits.csv")
porphyry_deposits_df = pd.read_csv("Phorphyry_deposits.csv")
sedex_deposits_df = pd.read_csv("Sedex_Deposits.csv")
vms_deposits_df = pd.read_csv("VMS_deposits.csv")
epithermal_deposits_df = pd.read_csv("epithermal_Deposits.csv")

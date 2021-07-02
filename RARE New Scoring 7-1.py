#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing packages
import os
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import random
from random import choice
from random import seed
from random import randrange
import collections
import statistics
from sklearn.pipeline import make_pipeline
from skrebate import MultiSURF
import math
import numpy as numpy
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
import itertools
from itertools import chain


# In[2]:


#Step 1: Initialize Population of Candidate Bins

#Random initialization of candidate bins, which are groupings of multiple features
#The value of each bin/feature is the sum of values for each amino acid in the bin (0 for match, 1 for different)
#Adding a function that can be an option to automatically separate rare and common variables


# In[3]:


#Defining a function to delete variables with MAF = 0
def Remove_Empty_Variables (original_feature_matrix, label_name):
    #Removing the label column to create a list of features
    feature_df = original_feature_matrix.drop(columns = [label_name])
    
    #Creating a list of features 
    feature_list = []
    for column in feature_df:
        feature_list.append(str(column))

    feature_matrix_no_empty_variables = pd.DataFrame()
    
    #Creating a list of features with MAF = 0 to delete
    MAF_0_features = []
    
    for i in range (0, len(feature_list)):
        #If the MAF of the feature is less than the cutoff, it will be removed
        if feature_df[feature_list[i]].sum()/(2*len(feature_df.index)) == 0:
            MAF_0_features.append(feature_list[i])
    
    #Removing the features
    for j in range (0, len(MAF_0_features)):
        feature_list.remove(MAF_0_features[j])
    
    #Updating the feature matrix accordingly
    for k in range (0, len(feature_list)):
        feature_matrix_no_empty_variables[feature_list[k]] = feature_df[feature_list[k]]
    
    #Adding the class label to the feature matrix
    feature_matrix_no_empty_variables['Class'] = original_feature_matrix[label_name]
    
    #Saving the feature list of nonempty features
    nonempty_feature_list = feature_list
    
    return feature_matrix_no_empty_variables, MAF_0_features, nonempty_feature_list


# In[4]:


#Defining a function to group features randomly, each feature can be in a number of groups up to a set max

def Random_Feature_Grouping(feature_matrix, label_name, number_of_groups, min_features_per_group, 
                            max_number_of_groups_with_feature):
    
    #Removing the label column to create a list of features
    feature_df = feature_matrix.drop(columns = [label_name])
    
    #Creating a list of features 
    feature_list = []
    for column in feature_df:
        feature_list.append(str(column))
    
    #Adding a random number of repeats of the features so that features can be in more than one group
    for w in range (0, len(feature_list)):
        repeats = randrange(max_number_of_groups_with_feature)
        for i in range (0, repeats):
            feature_list.append(feature_list[w])
    
    #Shuffling the feature list to enable random groups
    random.shuffle(feature_list)
    
    #Creating a dictionary of the groups
    feature_groups = {}
    
    #Assigns the minimum number of features to all the groups
    for x in range (0, min_features_per_group*number_of_groups, min_features_per_group):
        feature_groups[x/min_features_per_group] = feature_list[x:x+min_features_per_group]
    
    #Randomly distributes the remaining features across the set number of groups
    for y in range (min_features_per_group*number_of_groups, len(feature_list)):
        feature_groups[random.choice(list(feature_groups.keys()))].append(feature_list[y])
    
    #Removing duplicates of features in the same bin
    for z in range (0, len(feature_groups)):
        unique = []
        for a in range (0, len(feature_groups[z])):
            if feature_groups[z][a] not in unique:
                unique.append(feature_groups[z][a])
        feature_groups[z] = unique
    
    #Creating a dictionary with bin labels
    binned_feature_groups = {}
    for index in range (0, len(feature_groups)):
        binned_feature_groups["Bin " + str(index + 1)] = feature_groups[index]
    
    return feature_list, binned_feature_groups


# In[5]:


#Defining a function to create a feature matrix where each feature is a bin of features from the original feature matrix

def Grouped_Feature_Matrix(feature_matrix, label_name, binned_feature_groups):
    
    #Creating an empty data frame for the feature matrix with bins
    bins_df = pd.DataFrame()
    
    #Creating a list of 0s, where the number of 0s is the number of instances in the original feature matrix
    zero_list = []
    for a in range (0, len(feature_matrix.index)):
        zero_list.append(0)
        
    #Creating a dummy data frame
    dummy_df = pd.DataFrame()
    dummy_df['Zeros'] = zero_list
    #The list and dummy data frame will be used for adding later
    
   #For each feature group/bin, the values of the amino acid in the bin will be summed to create a value for the bin 
    count = 0
    for key in binned_feature_groups:
        sum_column = dummy_df['Zeros']
        for j in range (0, len(binned_feature_groups[key])):
            sum_column = sum_column + feature_matrix[binned_feature_groups[key][j]]
        count = count + 1
        bins_df[key] = sum_column
    
    #Adding the class label to the data frame
    bins_df['Class'] = feature_matrix[label_name]
    return bins_df


# In[6]:


#Defining a function to separate rare and common variables based on a rare variant minor allele frequency (MAF) cutoff
def Rare_and_Common_Variable_Separation (original_feature_matrix, label_name, rare_variant_MAF_cutoff):
    
    #Removing the label column to create a list of features
    feature_df = original_feature_matrix.drop(columns = [label_name])
    
    #Creating a list of features 
    feature_list = []
    for column in feature_df:
        feature_list.append(str(column))
    
    #Creating lists of rare and common features
    rare_feature_list = []
    common_feature_list = []
    MAF_0_features = []
    
    #Creating dictionaries of rare and common features, as the MAF of the features will be useful later
    rare_feature_MAF_dict = {}
    common_feature_MAF_dict = {}
    
    #Creating an empty data frames for feature matrices of rare features and common features
    rare_feature_df = pd.DataFrame()
    common_feature_df = pd.DataFrame()
    
    for i in range (0, len(feature_list)):
        #If the MAF of the feature is less than the cutoff, it will be designated as a rare variant
        if feature_df[feature_list[i]].sum()/(2*len(feature_df.index)) < rare_variant_MAF_cutoff and feature_df[feature_list[i]].sum()/(2*len(feature_df.index)) > 0:
            rare_feature_list.append(feature_list[i])
            rare_feature_MAF_dict[feature_list[i]] = feature_df[feature_list[i]].sum()/(2*len(feature_df.index))
            rare_feature_df[feature_list[i]] = feature_df[feature_list[i]]
        
        elif feature_df[feature_list[i]].sum()/(2*len(feature_df.index)) == 0:
            MAF_0_features.append(feature_list[i])

        
        #Otherwise, it will be considered as a common feature
        elif feature_df[feature_list[i]].sum()/(2*len(feature_df.index)) > rare_variant_MAF_cutoff:
            common_feature_list.append(feature_list[i])
            common_feature_MAF_dict[feature_list[i]] = feature_df[feature_list[i]].sum()/(2*len(feature_df.index))
            common_feature_df[feature_list[i]] = feature_df[feature_list[i]]
        
        #In case the MAF is exactly the cutoff 
        elif feature_df[feature_list[i]].sum()/(2*len(feature_df.index)) == rare_variant_MAF_cutoff:
            common_feature_list.append(feature_list[i])
            common_feature_MAF_dict[feature_list[i]] = feature_df[feature_list[i]].sum()/(2*len(feature_df.index))
            common_feature_df[feature_list[i]] = feature_df[feature_list[i]]
    
    #Adding the class label to each data frame
    rare_feature_df['Class'] = original_feature_matrix[label_name]
    common_feature_df['Class'] = original_feature_matrix[label_name]
    return rare_feature_list, rare_feature_MAF_dict, rare_feature_df, common_feature_list, common_feature_MAF_dict, common_feature_df, MAF_0_features


# In[7]:


#Step 2: Genetic Algorithm with Relief-based Feature Scoring (repeated for a given number of iterations) 


# In[8]:


#Step 2a: Relief-based Feature Importance Scoring and Bin Deletion

#Use MultiSURF to calculate the feature importance of each candidate bin 
#If the population size > the set max population size then bins will be probabilistically deleted based on fitness


# In[9]:


#Defining a function to calculate the feature importance of each bin using MultiSURF
def MultiSURF_Feature_Importance(bin_feature_matrix, label_name):
    
    #Converting to float to prevent any errors with the MultiSURF algorithm
    float_feature_matrix = bin_feature_matrix.astype(float)
    
    #Using MultiSURF and storing the feature importances in a dictionary
    features, labels = float_feature_matrix.drop(label_name, axis=1).values, float_feature_matrix[label_name].values
    fs = MultiSURF()
    fs.fit(features, labels)
    feature_scores = {}
    for feature_name, feature_score in zip(float_feature_matrix.drop(label_name, axis=1).columns,
                                       fs.feature_importances_):
        feature_scores[feature_name] = feature_score
        
    return feature_scores


# In[10]:


#Defining a function to calculate MultiSURF feature importance based on a sample of instances
def MultiSURF_Feature_Importance_Instance_Sample(bin_feature_matrix, label_name, sample_size):
    
    #Taking a random sample of the instances based on the sample size paramter to calculate MultiSURF 
    bin_feature_matrix_sample = bin_feature_matrix.sample(sample_size)
    
    #Converting to float to prevent any errors with the MultiSURF algorithm
    float_feature_matrix = bin_feature_matrix_sample.astype(float)
    
    #Using MultiSURF and storing the feature importances in a dictionary
    features, labels = float_feature_matrix.drop(label_name, axis=1).values, float_feature_matrix[label_name].values
    fs = MultiSURF()
    fs.fit(features, labels)
    feature_scores = {}
    for feature_name, feature_score in zip(float_feature_matrix.drop(label_name, axis=1).columns,
                                       fs.feature_importances_):
        feature_scores[feature_name] = feature_score
        
    return feature_scores


# In[11]:


#Defining a function to calculate the feature importance of each bin using MultiSURF for rare variants in context with common variables
def MultiSURF_Feature_Importance_Rare_Variants(bin_feature_matrix, common_feature_list, common_feature_matrix, label_name):
    
    #Creating a feature matrix with both binned rare variants and common features when calculating feature importance
    common_features_and_bins_matrix = bin_feature_matrix.copy()
    
    for i in range (0, len(common_feature_list)):
        common_features_and_bins_matrix[common_feature_list[i]] = common_feature_matrix[common_feature_list[i]]

    #Converting to float to prevent any errors with the MultiSURF algorithm
    float_feature_matrix = common_features_and_bins_matrix.astype(float)
    
    #Using MultiSURF and storing the feature importances in a dictionary
    features, labels = float_feature_matrix.drop(label_name, axis=1).values, float_feature_matrix[label_name].values
    fs = MultiSURF()
    fs.fit(features, labels)
    feature_scores = {}
    for feature_name, feature_score in zip(float_feature_matrix.drop(label_name, axis=1).columns,
                                       fs.feature_importances_):
        feature_scores[feature_name] = feature_score
        
    for i in range (0, len(common_feature_list)):
        del feature_scores[common_feature_list[i]]
        
    return feature_scores


# In[12]:


#Defining a function to calculate MultiSURF feature importance for rare variants in context with common variables based on a sample of instances
def MultiSURF_Feature_Importance_Rare_Variants_Instance_Sample(bin_feature_matrix, common_feature_list, common_feature_matrix,
                                                               label_name, sample_size):
    
    #Creating a feature matrix with both binned rare variants and common features when calculating feature importance
    common_features_and_bins_matrix = bin_feature_matrix.copy()
    
    for i in range (0, len(common_feature_list)):
        common_features_and_bins_matrix[common_feature_list[i]] = common_feature_matrix[common_feature_list[i]]
    
    #Taking a random sample of the instances based on the sample size paramter to calculate MultiSURF 
    common_features_and_bins_matrix_sample = common_features_and_bins_matrix.sample(sample_size)
    
    #Converting to float to prevent any errors with the MultiSURF algorithm
    float_feature_matrix = common_features_and_bins_matrix_sample.astype(float)
    
    #Using MultiSURF and storing the feature importances in a dictionary
    features, labels = float_feature_matrix.drop(label_name, axis=1).values, float_feature_matrix[label_name].values
    fs = MultiSURF()
    fs.fit(features, labels)
    feature_scores = {}
    for feature_name, feature_score in zip(float_feature_matrix.drop(label_name, axis=1).columns,
                                       fs.feature_importances_):
        feature_scores[feature_name] = feature_score
    
    for i in range (0, len(common_feature_list)):
        del feature_scores[common_feature_list[i]]
        
    return feature_scores


# In[13]:


#Defining a function to calculate MultiSURF feature importance only considering the bin and common feature(s)
def MultiSURF_Feature_Importance_Bin_and_Common_Features(bin_feature_matrix, amino_acid_bins, common_feature_list, common_feature_matrix, label_name):
    #Creating a feature matrix with both binned rare variants and common features when calculating feature importance
    common_features_and_bins_matrix = bin_feature_matrix.copy()
    
    for i in range (0, len(common_feature_list)):
        common_features_and_bins_matrix[common_feature_list[i]] = common_feature_matrix[common_feature_list[i]]
    
    bin_scores = {}
    for i in amino_acid_bins.keys():
        #Only taking the bin and the common features for the feature importance calculation
        bin_and_common_features = []
        bin_and_common_features.append(i)
        bin_and_common_features.extend(common_feature_list)
        bin_and_common_features.append(label_name)
        bin_and_cf_df = common_features_and_bins_matrix[bin_and_common_features]
        float_feature_matrix = bin_and_cf_df.astype(float)
        features, labels = float_feature_matrix.drop(label_name, axis=1).values, float_feature_matrix[label_name].values
        fs = MultiSURF()
        fs.fit(features, labels)
        feature_scores = {}
        for feature_name, feature_score in zip(float_feature_matrix.drop(label_name, axis=1).columns,
                                       fs.feature_importances_):
            feature_scores[feature_name] = feature_score
        bin_scores[i] = feature_scores[i]
    
    return bin_scores


# In[14]:


#Defining a function to calculate MultiSURF feature importance only considering the bin and common feature(s)
def MultiSURF_Feature_Importance_Bin_and_Common_Features_Instance_Sample(bin_feature_matrix, amino_acid_bins, common_feature_list, common_feature_matrix, label_name, sample_size):
    #Creating a feature matrix with both binned rare variants and common features when calculating feature importance
    common_features_and_bins_matrix = bin_feature_matrix.copy()
    
    for i in range (0, len(common_feature_list)):
        common_features_and_bins_matrix[common_feature_list[i]] = common_feature_matrix[common_feature_list[i]]
    
    bin_scores = {}
    for i in amino_acid_bins.keys():
        #Only taking the bin and the common features for the feature importance calculation
        bin_and_common_features = []
        bin_and_common_features.append(i)
        bin_and_common_features.extend(common_feature_list)
        bin_and_common_features.append(label_name)
        bin_and_cf_df = common_features_and_bins_matrix[bin_and_common_features]
        
        #Taking a sample to run MultiSURF on
        bin_and_cf_df_sample = bin_and_cf_df.sample(sample_size)
        float_feature_matrix = bin_and_cf_df_sample.astype(float)
        features, labels = float_feature_matrix.drop(label_name, axis=1).values, float_feature_matrix[label_name].values
        fs = MultiSURF()
        fs.fit(features, labels)
        feature_scores = {}
        for feature_name, feature_score in zip(float_feature_matrix.drop(label_name, axis=1).columns,
                                       fs.feature_importances_):
            feature_scores[feature_name] = feature_score
        bin_scores[i] = feature_scores[i]
    
    return bin_scores


# In[15]:


#Defining a function to score bins based on chi squared value
def Chi_Square_Feature_Importance(bin_feature_matrix, label_name, amino_acid_bins):
    
   #Calculating the chisquare and p values of each of the bin features in the bin feature matrix
    feature_matrix = bin_feature_matrix.copy()
    X = bin_feature_matrix.drop(label_name,axis=1)
    y = bin_feature_matrix[label_name]
    chi_scores, p_values = chi2(X,y)

    #Creating a dictionary with each bin and the chi-square value and p-value
    bin_scores = {}
    bin_names_list = list(amino_acid_bins.keys())
    for i in range (0, len(bin_names_list)):
        bin_scores[bin_names_list[i]] = chi_scores[i]
        
    for i in bin_scores.keys():
        if np.isnan(bin_scores[i]) == True:
            bin_scores[i] = 0
    
    return bin_scores


# In[16]:


#Defining a function to score bins based on chi squared value
def Mutual_Information_Feature_Importance(bin_feature_matrix, label_name, amino_acid_bins):
    
   #Calculating the mutual information for each of the bin features in the bin feature matrix
    feature_matrix = bin_feature_matrix.copy()
    X = bin_feature_matrix.drop(label_name,axis=1)
    y = bin_feature_matrix[label_name]
    mi = mutual_info_classif(X, y, True)

    #Creating a dictionary with each bin and the chi-square value and p-value
    bin_scores = {}
    bin_names_list = list(amino_acid_bins.keys())
    for i in range (0, len(bin_names_list)):
        bin_scores[bin_names_list[i]] = mi[i]
        
    for i in bin_scores.keys():
        if np.isnan(bin_scores[i]) == True:
            bin_scores[i] = 0
    
    return bin_scores


# In[17]:


def GlassRBC(X,y):
    #First step is to convert all continuous or interval X values into ranks (which may include ties)

    #Creating a list Xranks that will store the ranks of each element in X
    Xranks = X.copy()

    #Creating a list called remaining that will be used to generate the ranks of each element in X
    remaining = X.copy()
    
    while len(remaining) > 0:

        #Starting with the lowest value (would be ranked 1 if there's no ties)
        Xvalue_toberanked = min(remaining)

        #Allows for ties
        number_toberanked = remaining.count(Xvalue_toberanked)
        starting_rank = len(X) - len(remaining) + 1
        rank = (sum(list(range(starting_rank,starting_rank + number_toberanked))))/number_toberanked
        for i in range (0, len(X)):
            if X[i] == Xvalue_toberanked:
                #Adding the rank to the Xranks list with all the ranks of elements in X
                Xranks[i] = rank
            
        #Removing the value from the remaining list so it can move on to ranking the next lowest element of X
        remaining = [value for value in remaining if value != Xvalue_toberanked]
    
    
    #Finding the mean rank of X values in class 0 and class 1
    y0_Xranks = []
    y1_Xranks = []
    for i in range (0, len(Xranks)):
        if y[i] == 0:
            y0_Xranks.append(Xranks[i])
        elif y[i] == 1:
            y1_Xranks.append(Xranks[i])
    
    M1 = sum(y1_Xranks)/len(y1_Xranks)
    M0 = sum(y0_Xranks)/len(y0_Xranks)
    
    rank_biserial_correlation = 2*(M1-M0)/len(y)
    
    return rank_biserial_correlation


# In[18]:


def Rank_Biserial_Correlation_Feature_Importance(bin_feature_matrix, label_name, amino_acid_bins):
    #Using rank-biserial correlation to score bins
    feature_matrix = bin_feature_matrix.copy()
    bin_names_list = list(amino_acid_bins.keys())
    y = bin_feature_matrix[label_name].to_list()
    bin_scores = {}
    
    for i in range (0, len(bin_names_list)):
        X = bin_feature_matrix[bin_names_list[i]].to_list()
        bin_scores[bin_names_list[i]] = GlassRBC(X,y)
    
    for i in bin_scores.keys():
        if np.isnan(bin_scores[i]) == True:
            bin_scores[i] = 0
    
    return bin_scores


# In[19]:


def Logistic_Regression_Odds_Ratio_Feature_Importance(bin_feature_matrix, label_name, amino_acid_bins):
    
    #Calculating the odds ratio after a logistic regression with each bin feature on the target class
    #Accessing the target variable
    feature_matrix = bin_feature_matrix.copy()
    bin_names_list = list(amino_acid_bins.keys())
    y = bin_feature_matrix[label_name]
   
    #Creating a dictionary to score bin scores
    bin_scores = {}
    for i in range (0, len(bin_names_list)):
        Xlist = bin_feature_matrix[bin_names_list[i]]
        X = np.array(Xlist)
        clf = LogisticRegression().fit(X[:,None],y)
        odds_ratio_array = np.exp(clf.coef_)
        bin_scores[bin_names_list[i]] = odds_ratio_array[0,0]
    
    for i in bin_scores.keys():
        if np.isnan(bin_scores[i]) == True:
            bin_scores[i] = 0
    
    return bin_scores
        


# In[20]:


def Logistic_Regression_Score_Feature_Importance(bin_feature_matrix, label_name, amino_acid_bins):
    
    #Calculating the odds ratio after a logistic regression with each bin feature on the target class
    #Accessing the target variable
    feature_matrix = bin_feature_matrix.copy()
    bin_names_list = list(amino_acid_bins.keys())
    y = bin_feature_matrix[label_name]
   
    #Creating a dictionary to score bin scores
    bin_scores = {}
    for i in range (0, len(bin_names_list)):
        Xlist = bin_feature_matrix[bin_names_list[i]]
        X = np.array(Xlist)
        clf = LogisticRegression().fit(X[:,None],y)
        y_pred=clf.predict(X[:,None])
        bin_scores[bin_names_list[i]] = balanced_accuracy_score(y,y_pred)

    
    for i in bin_scores.keys():
        if np.isnan(bin_scores[i]) == True:
            bin_scores[i] = 0
    
    return bin_scores
        


# In[21]:


#Step 2b: Genetic Algorithm 

#Parent bins are probabilistically selected based on fitness (score calculated in Step 2a)
#New offspring bins will be created through cross over and mutation and are added to the next generation's population
#Based on the value of the elitism parameter, a number of high scoring parent bins will be preserved for the next gen


# In[22]:


#Defining a function to probabilitically select 2 parent bins based on their feature importance rank
#Tournament Selection works in this case by choosing a random sample of the bins and choosing the best two scores
def Tournament_Selection_Parent_Bins(bin_scores):
    
    #Choosing a random sample of 5% of the bin population or if that would be too small, choosing a sample of 50%
    if round(0.05*len(bin_scores)) < 2:
        samplekeys = random.sample(bin_scores.keys(), round(0.5*len(bin_scores)))
    else: 
        samplekeys = random.sample(bin_scores.keys(), round(0.05*len(bin_scores)))
    
    sample = {}
    for key in samplekeys:
        sample[key] = bin_scores[key]
    
    #Sorting the bins from best score to worst score
    sorted_bin_scores = dict(sorted(sample.items(), key=lambda item: item[1], reverse=True))
    sorted_bin_list = list(sorted_bin_scores.keys())
    
    #Choosing the parent bins and adding them to a list of parent bins
    parent_bins = [sorted_bin_list[0], sorted_bin_list[1]]
    
    return parent_bins


# In[23]:


#Defining a function for crossover and mutation that creates n offspring based on crossover of selected parents
#n is the max number of bins (but not all the offspring will carry on, as the worst will be deleted in Step 2a next time)
def Crossover_and_Mutation(max_population_of_bins, elitism_parameter, feature_list, binned_feature_groups, bin_scores, 
                           crossover_probability, mutation_probability):

    #Creating a list for offspring
    offspring_list = []
    
    #Creating a number of offspring equal to the the number needed to replace the nonelites
    #Each pair of parents will produce two offspring
    for i in range (0, int((max_population_of_bins - (elitism_parameter*max_population_of_bins))/2)):
        #Choosing the two parents and gettings the list of features in each parent bin
        parent_bins = Tournament_Selection_Parent_Bins(bin_scores)
        parent1_features = binned_feature_groups[parent_bins[0]].copy()
        parent2_features = binned_feature_groups[parent_bins[1]].copy()

        #Creating two lists for the offspring bins
        offspring1 = []
        offspring2 = []
        
        #CROSSOVER
        #Each feature in the parent bin will crossover based on the given probability (uniform crossover)
        for j in range (0, len(parent1_features)):
            if crossover_probability > random.random():
                offspring2.append(parent1_features[j])
            else:
                offspring1.append(parent1_features[j])
        
        for k in range (0, len(parent2_features)):
            if crossover_probability > random.random():
                offspring1.append(parent2_features[k])
            else:
                offspring2.append(parent2_features[k])
        
        #Ensuring that each of the offspring is no more than twice the size of the other offspring
        while len(offspring1) > len(offspring2):
            switch = random.choice(offspring1)
            offspring1.remove(switch)
            offspring2.append(switch)
            
        while len(offspring2) > len(offspring1):
            switch = random.choice(offspring2)
            offspring2.remove(switch)
            offspring1.append(switch)
        
        
        #MUTATION
        #Mutation only occurs with a certain probability on each feature in the original feature space
        
        #Applying the mutation operation to the first offspring
        #Creating a probability for adding a feature that accounts for the ratio between the feature list and the size of the bin
        if len(offspring1) > 0 and len(offspring1) != len(feature_list):            
            mutation_addition_prob = (mutation_probability)*(len(offspring1))/((len(feature_list)-len(offspring1)))
        elif len(offspring1) == 0 and len(offspring1) != len(feature_list):
            mutation_addition_prob = mutation_probability
        elif len(offspring1) == len(feature_list):
            mutation_addition_prob = 0
        
        deleted_list = []
        #Deletion form of mutation
        for l in range (0, len(offspring1)):
            #Mutation (deletion) occurs on this feature with probability equal to the mutation parameter
            if mutation_probability > random.random():
                deleted_list.append(offspring1[l])
                
        
        for l in range (0, len(deleted_list)):
            offspring1.remove(deleted_list[l])
            
        #Creating a list of features outside the offspring
        features_not_in_offspring = [item for item in feature_list if item not in offspring1]
        
        #Addition form of mutation
        for l in range (0, len(features_not_in_offspring)):
            #Mutation (addiiton) occurs on this feature with probability proportional to the mutation parameter
            #The probability accounts for the ratio between the feature list and the size of the bin
            if mutation_addition_prob > random.random():
                    offspring1.append(features_not_in_offspring[l])
        
        #Applying the mutation operation to the second offspring
        #Creating a probability for adding a feature that accounts for the ratio between the feature list and the size of the bin
        if len(offspring2) > 0 and len(offspring2) != len(feature_list):            
            mutation_addition_prob = (mutation_probability)*(len(offspring2))/((len(feature_list)-len(offspring2)))
        elif len(offspring2) == 0 and len(offspring2) != len(feature_list):
            mutation_addition_prob = mutation_probability
        elif len(offspring2) == len(feature_list):
            mutation_addition_prob = 0
        
        deleted_list = []
        #Deletion form of mutation
        for l in range (0, len(offspring2)):
            #Mutation (deletion) occurs on this feature with probability equal to the mutation parameter
            if mutation_probability > random.random():
                deleted_list.append(offspring2[l])
        
        for l in range (0, len(deleted_list)):
            offspring2.remove(deleted_list[l])
        
        #Creating a list of features outside the offspring
        features_not_in_offspring = [item for item in feature_list if item not in offspring2]
        
        #Addition form of mutation
        for l in range (0, len(features_not_in_offspring)):
            #Mutation (addiiton) occurs on this feature with probability proportional to the mutation parameter
            #The probability accounts for the ratio between the feature list and the size of the bin
            if mutation_addition_prob > random.random():
                    offspring2.append(features_not_in_offspring[l])
            
        #CLEANUP
        #Deleting any repeats of an amino acid in a bin
        #Removing duplicates of features in the same bin that may arise due to crossover
        unique = []
        for a in range (0, len(offspring1)):
            if offspring1[a] not in unique:
                unique.append(offspring1[a])
        
        #Adding random features from outside the bin to replace the deleted features in the bin
        replace_number = len(offspring1) - len(unique)
        features_not_in_offspring = []
        features_not_in_offspring = [item for item in feature_list if item not in offspring1]
        offspring1 = unique.copy()
        if len(features_not_in_offspring) > replace_number:
            replacements = random.sample(features_not_in_offspring, replace_number)
        else:
            replacements = features_not_in_offspring.copy()
        offspring1.extend(replacements)
        
        unique = []
        for a in range (0, len(offspring2)):
            if offspring2[a] not in unique:
                unique.append(offspring2[a])
        
        #Adding random features from outside the bin to replace the deleted features in the bin
        replace_number = len(offspring2) - len(unique)
        features_not_in_offspring = []
        features_not_in_offspring = [item for item in feature_list if item not in offspring2]
        offspring2 = unique.copy()
        if len(features_not_in_offspring) > replace_number:
            replacements = random.sample(features_not_in_offspring, replace_number)
        else:
            replacements = features_not_in_offspring.copy()
        offspring2.extend(replacements)
        
        #Adding the new offspring to the list of feature bins
        offspring_list.append(offspring1)
        offspring_list.append(offspring2)
        
   
    return offspring_list


# In[24]:


def Create_Next_Generation(binned_feature_groups, bin_scores, max_population_of_bins, elitism_parameter, offspring_list):
    
    #Sorting the bins from best score to worst score
    sorted_bin_scores = dict(sorted(bin_scores.items(), key=lambda item: item[1], reverse=True))
    sorted_bin_list = list(sorted_bin_scores.keys())
        
    #Determining the number of elite bins
    number_of_elite_bins = round(max_population_of_bins*elitism_parameter)
    elites = []
    #Adding the elites to a list of elite feature bins
    for a in range (0, number_of_elite_bins):
        elites.append(binned_feature_groups[sorted_bin_list[a]])
    
    #Creating a list of feature bins (without labels because those will be changed as things get deleted and added)
    feature_bin_list = elites.copy()
    
    #Adding the offspring to the feature bin list
    feature_bin_list.extend(offspring_list)
    return feature_bin_list


# In[25]:


#Defining a function to recreate the feature matrix (add up values of amino a cids from original dataset)
def Regroup_Feature_Matrix(feature_list, feature_matrix, label_name, feature_bin_list):
    
    
    #First deleting any bins that are empty
    #Creating a list of bins to delete
    bins_to_delete = []
    for i in feature_bin_list:
        if not i:
            bins_to_delete.append(i)
    for i in bins_to_delete:
        feature_bin_list.remove(i)
    
    #The length of the bin will be equal to the average length of nonempty bins in the population
    bin_lengths = []
    for i in feature_bin_list:
        if len(i) > 0:
            bin_lengths.append(len(i))      
    replacement_length = round(statistics.mean(bin_lengths))
    
    #Replacing each deleted bin with a bin with random features
    for i in range (0, len(bins_to_delete)):
        replacement = random.sample(feature_list, replacement_length)
        feature_bin_list.append(replacement)

    #Checking each pair of bins, if the bins are duplicates then one of the copies will be deleted
    seen = set()
    unique = []
    for x in feature_bin_list:
        srtd = tuple(sorted(x))
        if srtd not in seen:
            unique.append(x)
            seen.add(srtd)
    
    #Replacing each deleted bin with a bin with random features
    replacement_number = len(feature_bin_list) - len(unique)
    feature_bin_list = unique.copy()
    
    for i in feature_bin_list:
        if len(i) > 0:
            bin_lengths.append(len(i))      
    replacement_length = round(statistics.mean(bin_lengths))
    
    for i in range(0, replacement_number):
        replacement = random.sample(feature_list, replacement_length)
        feature_bin_list.append(replacement)
    
    
    #Deleting duplicate features in the same bin and replacing them with random features
    for Bin in range (0, len(feature_bin_list)):
        unique = []
        for a in range (0, len(feature_bin_list[Bin])):
            if feature_bin_list[Bin][a] not in unique:
                unique.append(feature_bin_list[Bin][a])
    
        replace_number = len(feature_bin_list[Bin]) - len(unique)
        
        features_not_in_offspring = []
        features_not_in_offspring = [item for item in feature_list if item not in feature_bin_list[Bin]]
        
        bin_replacement = unique.copy()
        if len(features_not_in_offspring) > replace_number:
            replacements = random.sample(features_not_in_offspring, replace_number)
        else:
            replacements = features_not_in_offspring.copy()
        bin_replacement.extend(replacements)
        
        feature_bin_list[Bin] = bin_replacement.copy()

    
    #Creating an empty data frame for the feature matrix with bins
    bins_df = pd.DataFrame()
    
    #Creating a list of 0s, where the number of 0s is the number of instances in the original feature matrix
    zero_list = []
    for a in range (0, len(feature_matrix.index)):
        zero_list.append(0)
        
    #Creating a dummy data frame
    dummy_df = pd.DataFrame()
    dummy_df['Zeros'] = zero_list
    #The list and dummy data frame will be used for adding later
    
   #For each feature group/bin, the values of the amino acid in the bin will be summed to create a value for the bin
   #This will be used to create a a feature matrix for the bins and a dictionary of binned feature groups

    count = 0
    binned_feature_groups = {}
    
    for i in range (0, len(feature_bin_list)):
        sum_column = dummy_df['Zeros']
        for j in range (0, len(feature_bin_list[i])):
            sum_column = sum_column + feature_matrix[feature_bin_list[i][j]]
        count = count + 1
        bins_df["Bin " + str(count)] = sum_column
        binned_feature_groups["Bin " + str(count)] = feature_bin_list[i]
    
    #Adding the class label to the data frame
    bins_df['Class'] = feature_matrix[label_name]
    return bins_df, binned_feature_groups


# In[26]:


#Defining a function for the RAFE algorithm (Relief-based Association Feature-bin Evolver)
#Same as RARE but it bins all features not just rare features
def RAFE (given_starting_point, amino_acid_start_point, amino_acid_bins_start_point, iterations, original_feature_matrix, 
          label_name, set_number_of_bins, min_features_per_group, max_number_of_groups_with_feature, 
          scoring_method, score_based_on_sample, instance_sample_size, 
          crossover_probability, mutation_probability, elitism_parameter):

    #Step 0: Deleting Empty Features (MAF = 0)
    feature_matrix_no_empty_variables, MAF_0_features, nonempty_feature_list = Remove_Empty_Variables(original_feature_matrix, 
                                                                                                      'Class')

    #Step 1: Initialize Population of Candidate Bins 
    #Initialize Feature Groups
    
    #If there is a starting point, use that for the amino acid list and the amino acid bins list
    if given_starting_point == True:
        amino_acid_bins = amino_acid_bins_start_point.copy()
        amino_acids = amino_acid_start_point.copy()
        
        features_to_remove = [item for item in amino_acids if item not in nonempty_feature_list]
        for feature in features_to_remove:
            amino_acids.remove(feature)
                
        bin_names = amino_acid_bins.keys()
        for bin_name in bin_names:
            for feature in features_to_remove:
                if feature in amino_acid_bins[bin_name]:
                    amino_acid_bins[bin_name].remove(feature)
                    
    #Otherwise randomly initialize the bins
    elif given_starting_point == False:
        amino_acids, amino_acid_bins = Random_Feature_Grouping(feature_matrix_no_empty_variables, label_name, 
                                                               set_number_of_bins, min_features_per_group, 
                                                               max_number_of_groups_with_feature)
    
    #Create Initial Binned Feature Matrix
    bin_feature_matrix = Grouped_Feature_Matrix(feature_matrix_no_empty_variables, label_name, amino_acid_bins)
    
    #Step 2: Genetic Algorithm with Feature Scoring (repeated for a given number of iterations)
    for i in range (0, iterations):
        
        #Step 2a: Feature Importance Scoring and Bin Deletion
        
        #Feature scoring can be done with Relief or a univariate chi squared test
        if scoring_method == 'Relief':
            #Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
            if score_based_on_sample == False:
                amino_acid_bin_scores = MultiSURF_Feature_Importance(bin_feature_matrix, label_name)
            elif score_based_on_sample == True:
                amino_acid_bin_scores = MultiSURF_Feature_Importance_Instance_Sample(bin_feature_matrix, label_name,
                                                                                instance_sample_size)
        elif scoring_method == 'Chi_Square':
            amino_acid_bin_scores = Chi_Square_Feature_Importance(bin_feature_matrix, label_name, amino_acid_bins)
        
        elif scoring_method == 'Mutual_Information':
            amino_acid_bin_scores = Mutual_Information_Feature_Importance(bin_feature_matrix, label_name, amino_acid_bins)
        
        elif scoring_method == 'Rank-Biserial_Correlation':
            amino_acid_bin_scores = Rank_Biserial_Correlation_Feature_Importance(bin_feature_matrix, label_name, amino_acid_bins)
        
        elif scoring_method == 'Logistic_Regression_Odds_Ratio':    
            amino_acid_bin_scores = Logistic_Regression_Odds_Ratio_Feature_Importance(bin_feature_matrix, label_name, amino_acid_bins)
        
           
        elif scoring_method == 'Logistic_Regression_Score':
            amino_acid_bin_scores = Logistic_Regression_Score_Feature_Importance(bin_feature_matrix, label_name, amino_acid_bins)
    
        #Step 2b: Genetic Algorithm 
        #Creating the offspring bins through crossover and mutation
        offspring_bins = Crossover_and_Mutation(set_number_of_bins, elitism_parameter, amino_acids, amino_acid_bins, amino_acid_bin_scores,
                                                crossover_probability, mutation_probability)
        
        #Creating the new generation by preserving some elites and adding the offspring
        feature_bin_list = Create_Next_Generation(amino_acid_bins, amino_acid_bin_scores, set_number_of_bins, 
                                                  elitism_parameter, offspring_bins)
        
        bin_feature_matrix, amino_acid_bins = Regroup_Feature_Matrix(amino_acids, original_feature_matrix, label_name, feature_bin_list)
    
    #Creating the final amino acid bin scores
    #Feature scoring can be done with Relief or a univariate chi squared test
    if scoring_method == 'Relief':
        #Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
        if score_based_on_sample == False:
            amino_acid_bin_scores = MultiSURF_Feature_Importance(bin_feature_matrix, label_name)
        elif score_based_on_sample == True:
            amino_acid_bin_scores = MultiSURF_Feature_Importance_Instance_Sample(bin_feature_matrix, label_name,
                                                                                instance_sample_size)
    elif scoring_method == 'Chi_Square':
        amino_acid_bin_scores = Chi_Square_Feature_Importance(bin_feature_matrix, label_name, amino_acid_bins)
    
    elif scoring_method == 'Mutual_Information':
        amino_acid_bin_scores = Mutual_Information_Feature_Importance(bin_feature_matrix, label_name, amino_acid_bins)
    
    elif scoring_method == 'Rank-Biserial_Correlation':
        amino_acid_bin_scores = Rank_Biserial_Correlation_Feature_Importance(bin_feature_matrix, label_name, amino_acid_bins)
   
        
    elif scoring_method == 'Logistic_Regression_Odds_Ratio':    
        amino_acid_bin_scores = Logistic_Regression_Odds_Ratio_Feature_Importance(bin_feature_matrix, label_name, amino_acid_bins)
    
       
    elif scoring_method == 'Logistic_Regression_Score':
        amino_acid_bin_scores = Logistic_Regression_Score_Feature_Importance(bin_feature_matrix, label_name, amino_acid_bins)
    
    
    return bin_feature_matrix, amino_acid_bins, amino_acid_bin_scores, MAF_0_features


# In[27]:


#Defining a function to present the top bins 
def Top_Bins_Summary(original_feature_matrix, label_name, bin_feature_matrix, bins, bin_scores, number_of_top_bins):
    
    #Ordering the bin scores from best to worst
    sorted_bin_scores = dict(sorted(bin_scores.items(), key=lambda item: item[1], reverse=True))
    sorted_bin_list = list(sorted_bin_scores.keys())
    sorted_bin_feature_importance_values = list(sorted_bin_scores.values())

    #Calculating the chi square and p values of each of the features in the original feature matrix
    df = original_feature_matrix
    X = df.drop('Class',axis=1)
    y = df['Class']
    chi_scores, p_values = chi2(X,y)
    
    #Removing the label column to create a list of features
    feature_df = original_feature_matrix.drop(columns = [label_name])
    
    #Creating a list of features 
    feature_list = []
    for column in feature_df:
        feature_list.append(str(column))
        
    #Creating a dictionary with each feature and the chi-square value and p-value
    Univariate_Feature_Stats = {}
    for i in range (0, len(feature_list)):
        list_of_stats = []
        list_of_stats.append(chi_scores[i])
        list_of_stats.append(p_values[i])
        Univariate_Feature_Stats[feature_list[i]] = list_of_stats
        #There will be features with nan for their chi-square value and p-value because the whole column is zeroes
    
    #Calculating the chisquare and p values of each of the features in the bin feature matrix
    X = bin_feature_matrix.drop('Class',axis=1)
    y = bin_feature_matrix['Class']
    chi_scores, p_values = chi2(X,y)

    #Creating a dictionary with each bin and the chi-square value and p-value
    Bin_Stats = {}
    bin_names_list = list(amino_acid_bins.keys())
    for i in range (0, len(bin_names_list)):
        list_of_stats = []
        list_of_stats.append(chi_scores[i])
        list_of_stats.append(p_values[i])
        Bin_Stats[bin_names_list[i]] = list_of_stats

    for i in range (0, number_of_top_bins):
        #Printing the bin Name
        print ("Bin Rank " + str(i+1) + ": " + sorted_bin_list[i])
        #Printing the bin's MultiSURF/Univariate score, chi-square value, and p-value
        print ("Score: " + str(sorted_bin_feature_importance_values[i]) + "; chi-square value: " + str(Bin_Stats[sorted_bin_list[i]][0]) + "; p-value: " + str(Bin_Stats[sorted_bin_list[i]][1]))
        #Printing each of the features in the bin and also printing the univariate stats of that feature
        for j in range (0, len(bins[sorted_bin_list[i]])):
            print ("Feature Name: " + bins[sorted_bin_list[i]][j] + "; chi-square value: " + str(Univariate_Feature_Stats[bins[sorted_bin_list[i]][j]][0]) + "; p-value: " + str(Univariate_Feature_Stats[bins[sorted_bin_list[i]][j]][1]))
        print ('---------------------------')                                              
                        


# In[28]:


#Defining a function for the RARE algorithm (Relief-based Association Rare-variant-bin Evolver)
def RARE (given_starting_point, amino_acid_start_point, amino_acid_bins_start_point, iterations, original_feature_matrix, 
          label_name, rare_variant_MAF_cutoff, set_number_of_bins, 
          min_features_per_group, max_number_of_groups_with_feature, 
          scoring_method, score_based_on_sample, score_with_common_variables, 
          instance_sample_size, crossover_probability, mutation_probability, elitism_parameter):
    
    #Step 0: Separate Rare Variants and Common Features
    rare_feature_list, rare_feature_MAF_dict, rare_feature_df, common_feature_list, common_feature_MAF_dict, common_feature_df, MAF_0_features = Rare_and_Common_Variable_Separation (original_feature_matrix, label_name, rare_variant_MAF_cutoff)

    #Step 1: Initialize Population of Candidate Bins 
    #Initialize Feature Groups
    if given_starting_point == True:
        amino_acid_bins = amino_acid_bins_start_point.copy()
        amino_acids = amino_acid_start_point.copy()
        
        features_to_remove = [item for item in amino_acids if item not in rare_feature_list]
        for feature in features_to_remove:
            amino_acids.remove(feature)
                
        bin_names = amino_acid_bins.keys()
        for bin_name in bin_names:
            for feature in features_to_remove:
                if feature in amino_acid_bins[bin_name]:
                    amino_acid_bins[bin_name].remove(feature)
    
    #Otherwise randomly initialize the bins
    elif given_starting_point == False:
        amino_acids, amino_acid_bins = Random_Feature_Grouping(rare_feature_df, label_name, 
                                                               set_number_of_bins, min_features_per_group, 
                                                               max_number_of_groups_with_feature)
    #Create Initial Binned Feature Matrix
    bin_feature_matrix = Grouped_Feature_Matrix(rare_feature_df, label_name, amino_acid_bins)
    
    #Step 2: Genetic Algorithm with Feature Scoring (repeated for a given number of iterations)
    for i in range (0, iterations):
        
        #Step 2a: Feature Importance Scoring and Bin Deletion
        
        #Feature importance can be scored with Relief or a univariate metric (chi squared value)
        if scoring_method == 'Relief':
            #Feature importance is calculating either with common variables
            if score_with_common_variables == True:
                #Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
                if score_based_on_sample == False:
                    amino_acid_bin_scores = MultiSURF_Feature_Importance_Rare_Variants(bin_feature_matrix, common_feature_list,
                                                                                      common_feature_df, label_name)
                elif score_based_on_sample == True:
                    amino_acid_bin_scores = MultiSURF_Feature_Importance_Rare_Variants_Instance_Sample(bin_feature_matrix, 
                                                                                                  common_feature_list,
                                                                                                  common_feature_df,
                                                                                                  label_name, 
                                                                                                  instance_sample_size)
        
            #Or feauture importance is calculated only based on rare variant bins
            elif score_with_common_variables == False:
                #Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
                if score_based_on_sample == False:
                    amino_acid_bin_scores = MultiSURF_Feature_Importance(bin_feature_matrix, label_name)
                elif score_based_on_sample == True:
                    amino_acid_bin_scores = MultiSURF_Feature_Importance_Instance_Sample(bin_feature_matrix, label_name, 
                                                                                         instance_sample_size)
        elif scoring_method == 'Relief only on bin and common features':
            #Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
            if score_based_on_sample == True:
                amino_acid_bin_scores = MultiSURF_Feature_Importance_Bin_and_Common_Features_Instance_Sample(bin_feature_matrix, amino_acid_bins, common_feature_list, common_feature_df, label_name, instance_sample_size)
            elif score_based_on_sample == False:
                amino_acid_bin_scores = MultiSURF_Feature_Importance_Bin_and_Common_Features(bin_feature_matrix, amino_acid_bins, common_feature_list, common_feature_df, label_name)
        
        elif scoring_method == 'Chi_Square':
            amino_acid_bin_scores = Chi_Square_Feature_Importance(bin_feature_matrix, label_name, amino_acid_bins)
        
        elif scoring_method == 'Mutual_Information':
            amino_acid_bin_scores = Mutual_Information_Feature_Importance(bin_feature_matrix, label_name, amino_acid_bins)
                
        elif scoring_method == 'Logistic_Regression_Odds_Ratio':    
            amino_acid_bin_scores = Logistic_Regression_Odds_Ratio_Feature_Importance(bin_feature_matrix, label_name, amino_acid_bins)
         
           
        elif scoring_method == 'Logistic_Regression_Score':
            amino_acid_bin_scores = Logistic_Regression_Score_Feature_Importance(bin_feature_matrix, label_name, amino_acid_bins)
        
        elif scoring_method == 'Rank-Biserial_Correlation':
            amino_acid_bin_scores = Rank_Biserial_Correlation_Feature_Importance(bin_feature_matrix, label_name, amino_acid_bins)
   
    
        #Step 2b: Genetic Algorithm 
        #Creating the offspring bins through crossover and mutation
        offspring_bins = Crossover_and_Mutation(set_number_of_bins, elitism_parameter, amino_acids, amino_acid_bins, amino_acid_bin_scores,
                                                crossover_probability, mutation_probability)
        
        #Creating the new generation by preserving some elites and adding the offspring
        feature_bin_list = Create_Next_Generation(amino_acid_bins, amino_acid_bin_scores, set_number_of_bins, 
                                                  elitism_parameter, offspring_bins)
        
        #Updating the binned feature matrix
        bin_feature_matrix, amino_acid_bins = Regroup_Feature_Matrix(amino_acids, rare_feature_df, label_name, feature_bin_list)
    
    
    #Creating the final amino acid bin scores
    if scoring_method == 'Relief':
            #Feature importance is calculating either with common variables
            if score_with_common_variables == True:
                #Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
                if score_based_on_sample == False:
                    amino_acid_bin_scores = MultiSURF_Feature_Importance_Rare_Variants(bin_feature_matrix, common_feature_list,
                                                                                      common_feature_df, label_name)
                elif score_based_on_sample == True:
                    amino_acid_bin_scores = MultiSURF_Feature_Importance_Rare_Variants_Instance_Sample(bin_feature_matrix, 
                                                                                                  common_feature_list,
                                                                                                  common_feature_df,
                                                                                                  label_name, 
                                                                                                  instance_sample_size)
        
            #Or feauture importance is calculated only based on rare variant bins
            elif score_with_common_variables == False:
                #Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
                if score_based_on_sample == False:
                    amino_acid_bin_scores = MultiSURF_Feature_Importance(bin_feature_matrix, label_name)
                elif score_based_on_sample == True:
                    amino_acid_bin_scores = MultiSURF_Feature_Importance_Instance_Sample(bin_feature_matrix, label_name, 
                                                                                         instance_sample_size)
    elif scoring_method == 'Chi_Square':
        amino_acid_bin_scores = Chi_Square_Feature_Importance(bin_feature_matrix, label_name, amino_acid_bins)
    
    elif scoring_method == 'Mutual_Information':
        amino_acid_bin_scores = Mutual_Information_Feature_Importance(bin_feature_matrix, label_name, amino_acid_bins)
        
    elif scoring_method == 'Rank-Biserial_Correlation':
        amino_acid_bin_scores = Rank_Biserial_Correlation_Feature_Importance(bin_feature_matrix, label_name, amino_acid_bins)
   
    
    elif scoring_method == 'Logistic_Regression_Odds_Ratio':    
        amino_acid_bin_scores = Logistic_Regression_Odds_Ratio_Feature_Importance(bin_feature_matrix, label_name, amino_acid_bins)
   
    elif scoring_method == 'Logistic_Regression_Score':
        amino_acid_bin_scores = Logistic_Regression_Score_Feature_Importance(bin_feature_matrix, label_name, amino_acid_bins)
    
    elif scoring_method == 'Relief only on bin and common features':
        #Calculating feature importance with MultiSURF on whole dataset or MultiSURF on a sample
        if score_based_on_sample == True:
            amino_acid_bin_scores = MultiSURF_Feature_Importance_Bin_and_Common_Features_Instance_Sample(bin_feature_matrix, amino_acid_bins, common_feature_list, common_feature_df, label_name, instance_sample_size)
        elif score_based_on_sample == False:
            amino_acid_bin_scores = MultiSURF_Feature_Importance_Bin_and_Common_Features(bin_feature_matrix, amino_acid_bins, common_feature_list, common_feature_df, label_name)
        
    #Creating a final feature matrix with both rare variant bins and common features
    common_features_and_bins_matrix = bin_feature_matrix.copy()    
    for i in range (0, len(common_feature_list)):
        common_features_and_bins_matrix[common_feature_list[i]] = common_feature_df[common_feature_list[i]]
    
    bin_feature_matrix['Class'] = original_feature_matrix[label_name]
    common_features_and_bins_matrix['Class'] = original_feature_matrix[label_name]

    return bin_feature_matrix, common_features_and_bins_matrix, amino_acid_bins, amino_acid_bin_scores, rare_feature_MAF_dict, common_feature_MAF_dict, rare_feature_df, common_feature_df, MAF_0_features


# In[29]:


#Loading in the SRTR data
srtr_data = pd.read_csv('C:\\Users\\satvi\\Downloads\\SRTR_data.csv')

#A list of the amino acid positions
SFVT_aa = ['MM_A_1', 'MM_A_2', 'MM_A_3', 'MM_A_4', 'MM_A_5', 'MM_A_6', 'MM_A_7', 'MM_A_8', 'MM_A_9', 'MM_A_10', 'MM_A_11', 'MM_A_12', 'MM_A_13', 'MM_A_14', 'MM_A_15', 'MM_A_16', 'MM_A_17', 'MM_A_18', 'MM_A_19', 'MM_A_20', 'MM_A_21', 'MM_A_22', 'MM_A_23', 'MM_A_24', 'MM_A_25', 'MM_A_26', 'MM_A_27', 'MM_A_28', 'MM_A_29', 'MM_A_30', 'MM_A_31', 'MM_A_32', 'MM_A_33', 'MM_A_34', 'MM_A_35', 'MM_A_36', 'MM_A_37', 'MM_A_38', 'MM_A_39', 'MM_A_40', 'MM_A_41', 'MM_A_42', 'MM_A_43', 'MM_A_44', 'MM_A_45', 'MM_A_46', 'MM_A_47', 'MM_A_48', 'MM_A_49', 'MM_A_50', 'MM_A_51', 'MM_A_52', 'MM_A_53', 'MM_A_54', 'MM_A_55', 'MM_A_56', 'MM_A_57', 'MM_A_58', 'MM_A_59', 'MM_A_60', 'MM_A_61', 'MM_A_62', 'MM_A_63', 'MM_A_64', 'MM_A_65', 'MM_A_66', 'MM_A_67', 'MM_A_68', 'MM_A_69', 'MM_A_70', 'MM_A_71', 'MM_A_72', 'MM_A_73', 'MM_A_74', 'MM_A_75', 'MM_A_76', 'MM_A_77', 'MM_A_78', 'MM_A_79', 'MM_A_80', 'MM_A_81', 'MM_A_82', 'MM_A_83', 'MM_A_84', 'MM_A_95', 'MM_A_96', 'MM_A_97', 'MM_A_99', 'MM_A_113', 'MM_A_114', 'MM_A_116', 'MM_A_123', 'MM_A_124', 'MM_A_133', 'MM_A_142', 'MM_A_143', 'MM_A_146', 'MM_A_147', 'MM_A_150', 'MM_A_152', 'MM_A_155', 'MM_A_156', 'MM_A_159', 'MM_A_160', 'MM_A_163', 'MM_A_167', 'MM_A_170', 'MM_A_171', 'MM_A_145', 'MM_A_149', 'MM_A_151', 'MM_A_154', 'MM_A_158', 'MM_A_162', 'MM_A_166', 'MM_A_169', 'MM_A_85', 'MM_A_86', 'MM_A_87', 'MM_A_88', 'MM_A_89', 'MM_A_90', 'MM_A_91', 'MM_A_92', 'MM_A_93', 'MM_A_94', 'MM_A_98', 'MM_A_100', 'MM_A_101', 'MM_A_102', 'MM_A_103', 'MM_A_104', 'MM_A_105', 'MM_A_106', 'MM_A_107', 'MM_A_108', 'MM_A_109', 'MM_A_110', 'MM_A_111', 'MM_A_112', 'MM_A_115', 'MM_A_117', 'MM_A_118', 'MM_A_119', 'MM_A_120', 'MM_A_121', 'MM_A_122', 'MM_A_125', 'MM_A_126', 'MM_A_127', 'MM_A_128', 'MM_A_129', 'MM_A_130', 'MM_A_131', 'MM_A_132', 'MM_A_134', 'MM_A_135', 'MM_A_136', 'MM_A_137', 'MM_A_138', 'MM_A_139', 'MM_A_140', 'MM_A_141', 'MM_A_144', 'MM_A_148', 'MM_A_153', 'MM_A_157', 'MM_A_161', 'MM_A_164', 'MM_A_165', 'MM_A_168', 'MM_A_172', 'MM_A_173', 'MM_A_174', 'MM_A_176', 'MM_A_177', 'MM_A_178', 'MM_A_179', 'MM_A_175', 'MM_A_180', 'MM_A_181', 'MM_A_182', 'MM_B_1', 'MM_B_2', 'MM_B_3', 'MM_B_4', 'MM_B_5', 'MM_B_6', 'MM_B_7', 'MM_B_8', 'MM_B_9', 'MM_B_10', 'MM_B_11', 'MM_B_12', 'MM_B_13', 'MM_B_14', 'MM_B_15', 'MM_B_16', 'MM_B_17', 'MM_B_18', 'MM_B_19', 'MM_B_20', 'MM_B_21', 'MM_B_22', 'MM_B_23', 'MM_B_24', 'MM_B_25', 'MM_B_26', 'MM_B_27', 'MM_B_28', 'MM_B_29', 'MM_B_30', 'MM_B_31', 'MM_B_32', 'MM_B_33', 'MM_B_34', 'MM_B_35', 'MM_B_36', 'MM_B_37', 'MM_B_38', 'MM_B_39', 'MM_B_40', 'MM_B_41', 'MM_B_42', 'MM_B_43', 'MM_B_44', 'MM_B_45', 'MM_B_46', 'MM_B_47', 'MM_B_48', 'MM_B_49', 'MM_B_50', 'MM_B_51', 'MM_B_52', 'MM_B_53', 'MM_B_54', 'MM_B_55', 'MM_B_56', 'MM_B_57', 'MM_B_58', 'MM_B_59', 'MM_B_60', 'MM_B_61', 'MM_B_62', 'MM_B_63', 'MM_B_64', 'MM_B_65', 'MM_B_66', 'MM_B_67', 'MM_B_68', 'MM_B_69', 'MM_B_70', 'MM_B_71', 'MM_B_72', 'MM_B_73', 'MM_B_74', 'MM_B_75', 'MM_B_76', 'MM_B_77', 'MM_B_78', 'MM_B_79', 'MM_B_80', 'MM_B_81', 'MM_B_82', 'MM_B_83', 'MM_B_84', 'MM_B_94', 'MM_B_95', 'MM_B_97', 'MM_B_99', 'MM_B_114', 'MM_B_116', 'MM_B_123', 'MM_B_143', 'MM_B_146', 'MM_B_147', 'MM_B_152', 'MM_B_155', 'MM_B_156', 'MM_B_159', 'MM_B_163', 'MM_B_167', 'MM_B_171', 'MM_B_85', 'MM_B_86', 'MM_B_87', 'MM_B_88', 'MM_B_89', 'MM_B_90', 'MM_B_91', 'MM_B_92', 'MM_B_93', 'MM_B_96', 'MM_B_98', 'MM_B_100', 'MM_B_101', 'MM_B_102', 'MM_B_103', 'MM_B_104', 'MM_B_105', 'MM_B_106', 'MM_B_107', 'MM_B_108', 'MM_B_109', 'MM_B_110', 'MM_B_111', 'MM_B_112', 'MM_B_113', 'MM_B_115', 'MM_B_117', 'MM_B_118', 'MM_B_121', 'MM_B_122', 'MM_B_119', 'MM_B_120', 'MM_B_124', 'MM_B_125', 'MM_B_126', 'MM_B_127', 'MM_B_128', 'MM_B_129', 'MM_B_130', 'MM_B_131', 'MM_B_132', 'MM_B_133', 'MM_B_134', 'MM_B_135', 'MM_B_136', 'MM_B_137', 'MM_B_138', 'MM_B_139', 'MM_B_140', 'MM_B_141', 'MM_B_142', 'MM_B_144', 'MM_B_145', 'MM_B_148', 'MM_B_149', 'MM_B_150', 'MM_B_151', 'MM_B_153', 'MM_B_154', 'MM_B_157', 'MM_B_158', 'MM_B_160', 'MM_B_161', 'MM_B_164', 'MM_B_165', 'MM_B_166', 'MM_B_168', 'MM_B_169', 'MM_B_170', 'MM_B_172', 'MM_B_173', 'MM_B_174', 'MM_B_176', 'MM_B_177', 'MM_B_178', 'MM_B_179', 'MM_B_162', 'MM_B_175', 'MM_B_180', 'MM_B_181', 'MM_B_182', 'MM_C_1', 'MM_C_2', 'MM_C_3', 'MM_C_4', 'MM_C_5', 'MM_C_6', 'MM_C_7', 'MM_C_8', 'MM_C_9', 'MM_C_10', 'MM_C_11', 'MM_C_12', 'MM_C_13', 'MM_C_14', 'MM_C_15', 'MM_C_16', 'MM_C_17', 'MM_C_18', 'MM_C_19', 'MM_C_20', 'MM_C_21', 'MM_C_22', 'MM_C_23', 'MM_C_24', 'MM_C_25', 'MM_C_26', 'MM_C_27', 'MM_C_28', 'MM_C_29', 'MM_C_30', 'MM_C_31', 'MM_C_32', 'MM_C_33', 'MM_C_34', 'MM_C_35', 'MM_C_36', 'MM_C_37', 'MM_C_38', 'MM_C_39', 'MM_C_40', 'MM_C_41', 'MM_C_42', 'MM_C_43', 'MM_C_44', 'MM_C_45', 'MM_C_46', 'MM_C_47', 'MM_C_59', 'MM_C_62', 'MM_C_63', 'MM_C_66', 'MM_C_67', 'MM_C_69', 'MM_C_70', 'MM_C_73', 'MM_C_74', 'MM_C_77', 'MM_C_80', 'MM_C_81', 'MM_C_84', 'MM_C_95', 'MM_C_97', 'MM_C_99', 'MM_C_116', 'MM_C_123', 'MM_C_124', 'MM_C_143', 'MM_C_146', 'MM_C_147', 'MM_C_150', 'MM_C_152', 'MM_C_155', 'MM_C_156', 'MM_C_159', 'MM_C_163', 'MM_C_167', 'MM_C_171', 'MM_C_72', 'MM_C_75', 'MM_C_76', 'MM_C_79', 'MM_C_142', 'MM_C_145', 'MM_C_148', 'MM_C_149', 'MM_C_151', 'MM_C_48', 'MM_C_49', 'MM_C_50', 'MM_C_51', 'MM_C_52', 'MM_C_53', 'MM_C_54', 'MM_C_55', 'MM_C_56', 'MM_C_57', 'MM_C_58', 'MM_C_60', 'MM_C_61', 'MM_C_64', 'MM_C_65', 'MM_C_68', 'MM_C_71', 'MM_C_78', 'MM_C_82', 'MM_C_83', 'MM_C_94', 'MM_C_96', 'MM_C_98', 'MM_C_85', 'MM_C_86', 'MM_C_87', 'MM_C_88', 'MM_C_89', 'MM_C_90', 'MM_C_91', 'MM_C_92', 'MM_C_93', 'MM_C_115', 'MM_C_117', 'MM_C_121', 'MM_C_122', 'MM_C_100', 'MM_C_101', 'MM_C_102', 'MM_C_103', 'MM_C_119', 'MM_C_120', 'MM_C_104', 'MM_C_105', 'MM_C_106', 'MM_C_107', 'MM_C_108', 'MM_C_109', 'MM_C_110', 'MM_C_111', 'MM_C_112', 'MM_C_113', 'MM_C_114', 'MM_C_118', 'MM_C_125', 'MM_C_126', 'MM_C_127', 'MM_C_128', 'MM_C_129', 'MM_C_130', 'MM_C_131', 'MM_C_132', 'MM_C_133', 'MM_C_134', 'MM_C_135', 'MM_C_136', 'MM_C_137', 'MM_C_138', 'MM_C_139', 'MM_C_140', 'MM_C_141', 'MM_C_144', 'MM_C_153', 'MM_C_154', 'MM_C_157', 'MM_C_158', 'MM_C_160', 'MM_C_161', 'MM_C_164', 'MM_C_165', 'MM_C_166', 'MM_C_168', 'MM_C_169', 'MM_C_170', 'MM_C_172', 'MM_C_173', 'MM_C_174', 'MM_C_176', 'MM_C_177', 'MM_C_178', 'MM_C_179', 'MM_C_162', 'MM_C_175', 'MM_C_180', 'MM_C_181', 'MM_C_182', 'MM_DQB1_9', 'MM_DQB1_11', 'MM_DQB1_12', 'MM_DQB1_13', 'MM_DQB1_14', 'MM_DQB1_15', 'MM_DQB1_26', 'MM_DQB1_27', 'MM_DQB1_28', 'MM_DQB1_30', 'MM_DQB1_37', 'MM_DQB1_47', 'MM_DQB1_56', 'MM_DQB1_57', 'MM_DQB1_60', 'MM_DQB1_61', 'MM_DQB1_67', 'MM_DQB1_71', 'MM_DQB1_74', 'MM_DQB1_77', 'MM_DQB1_78', 'MM_DQB1_79', 'MM_DQB1_81', 'MM_DQB1_82', 'MM_DQB1_85', 'MM_DQB1_86', 'MM_DQB1_88', 'MM_DQB1_70', 'MM_DQB1_8', 'MM_DQB1_10', 'MM_DQB1_16', 'MM_DQB1_17', 'MM_DQB1_18', 'MM_DQB1_19', 'MM_DQB1_20', 'MM_DQB1_21', 'MM_DQB1_22', 'MM_DQB1_23', 'MM_DQB1_24', 'MM_DQB1_25', 'MM_DQB1_29', 'MM_DQB1_31', 'MM_DQB1_32', 'MM_DQB1_35', 'MM_DQB1_36', 'MM_DQB1_33', 'MM_DQB1_34', 'MM_DQB1_83', 'MM_DQB1_87', 'MM_DQB1_90', 'MM_DQB1_91', 'MM_DQB1_51', 'MM_DQB1_53', 'MM_DQB1_54', 'MM_DQB1_38', 'MM_DQB1_39', 'MM_DQB1_40', 'MM_DQB1_41', 'MM_DQB1_42', 'MM_DQB1_43', 'MM_DQB1_44', 'MM_DQB1_45', 'MM_DQB1_46', 'MM_DQB1_48', 'MM_DQB1_49', 'MM_DQB1_50', 'MM_DQB1_52', 'MM_DQB1_55', 'MM_DQB1_58', 'MM_DQB1_59', 'MM_DQB1_62', 'MM_DQB1_63', 'MM_DQB1_64', 'MM_DQB1_65', 'MM_DQB1_66', 'MM_DQB1_68', 'MM_DQB1_69', 'MM_DQB1_72', 'MM_DQB1_73', 'MM_DQB1_75', 'MM_DQB1_76', 'MM_DQB1_80', 'MM_DQB1_84', 'MM_DQB1_89', 'MM_DQB1_92', 'MM_DRB1_8', 'MM_DRB1_9', 'MM_DRB1_10', 'MM_DRB1_11', 'MM_DRB1_12', 'MM_DRB1_13', 'MM_DRB1_14', 'MM_DRB1_15', 'MM_DRB1_16', 'MM_DRB1_17', 'MM_DRB1_18', 'MM_DRB1_19', 'MM_DRB1_20', 'MM_DRB1_21', 'MM_DRB1_22', 'MM_DRB1_26', 'MM_DRB1_28', 'MM_DRB1_30', 'MM_DRB1_37', 'MM_DRB1_47', 'MM_DRB1_56', 'MM_DRB1_57', 'MM_DRB1_60', 'MM_DRB1_61', 'MM_DRB1_67', 'MM_DRB1_70', 'MM_DRB1_71', 'MM_DRB1_74', 'MM_DRB1_77', 'MM_DRB1_78', 'MM_DRB1_81', 'MM_DRB1_82', 'MM_DRB1_85', 'MM_DRB1_86', 'MM_DRB1_89', 'MM_DRB1_90', 'MM_DRB1_64', 'MM_DRB1_65', 'MM_DRB1_66', 'MM_DRB1_69', 'MM_DRB1_73', 'MM_DRB1_76', 'MM_DRB1_80', 'MM_DRB1_84', 'MM_DRB1_23', 'MM_DRB1_24', 'MM_DRB1_25', 'MM_DRB1_27', 'MM_DRB1_29', 'MM_DRB1_31', 'MM_DRB1_32', 'MM_DRB1_33', 'MM_DRB1_34', 'MM_DRB1_88', 'MM_DRB1_91', 'MM_DRB1_93', 'MM_DRB1_35', 'MM_DRB1_36', 'MM_DRB1_38', 'MM_DRB1_39', 'MM_DRB1_40', 'MM_DRB1_41', 'MM_DRB1_52', 'MM_DRB1_53', 'MM_DRB1_54', 'MM_DRB1_55', 'MM_DRB1_42', 'MM_DRB1_43', 'MM_DRB1_44', 'MM_DRB1_45', 'MM_DRB1_46', 'MM_DRB1_48', 'MM_DRB1_49', 'MM_DRB1_83', 'MM_DRB1_50', 'MM_DRB1_51', 'MM_DRB1_58', 'MM_DRB1_59', 'MM_DRB1_62', 'MM_DRB1_63', 'MM_DRB1_68', 'MM_DRB1_72', 'MM_DRB1_75', 'MM_DRB1_79']

#A list of the SFVT expert knowledge bins
RARESFVT = {'Bin 1': ['MM_DRB1_12', 'MM_DQB1_74', 'MM_DRB1_32', 'MM_DRB1_11', 'MM_DRB1_37', 'MM_DRB1_13', 'MM_DRB1_74', 'MM_DRB1_47', 'MM_DRB1_10', 'MM_DQB1_71', 'MM_DQB1_55', 'MM_DRB1_28', 'MM_DRB1_70', 'MM_DRB1_67', 'MM_DRB1_30'], 'Bin 2': ['MM_DQB1_74', 'MM_DRB1_30', 'MM_DRB1_10', 'MM_DRB1_70', 'MM_DRB1_32', 'MM_DRB1_13', 'MM_DRB1_74', 'MM_DRB1_47', 'MM_DRB1_11', 'MM_DRB1_28', 'MM_DRB1_26', 'MM_DRB1_12', 'MM_DQB1_55', 'MM_DRB1_67', 'MM_DRB1_37'], 'Bin 3': ['MM_DQB1_74', 'MM_DQB1_55', 'MM_DRB1_11', 'MM_DRB1_28', 'MM_DRB1_37', 'MM_DRB1_47', 'MM_DRB1_70', 'MM_DQB1_75', 'MM_DRB1_32', 'MM_DRB1_10', 'MM_DRB1_67', 'MM_DRB1_30', 'MM_DRB1_74', 'MM_DRB1_13', 'MM_DRB1_12'], 'Bin 4': ['MM_DQB1_74', 'MM_DRB1_13', 'MM_DRB1_70', 'MM_DRB1_67', 'MM_DRB1_47', 'MM_DRB1_11', 'MM_DRB1_32', 'MM_B_32', 'MM_DRB1_28', 'MM_DRB1_12', 'MM_DRB1_30', 'MM_DRB1_10', 'MM_DQB1_55', 'MM_DRB1_74', 'MM_DRB1_37'], 'Bin 5': ['MM_DRB1_30', 'MM_DRB1_26', 'MM_DQB1_71', 'MM_DQB1_55', 'MM_DRB1_70', 'MM_DRB1_13', 'MM_DRB1_32', 'MM_DRB1_10', 'MM_DRB1_74', 'MM_DRB1_11', 'MM_DRB1_47', 'MM_DRB1_12', 'MM_DRB1_37', 'MM_DRB1_28', 'MM_DRB1_67'], 'Bin 6': ['MM_DQB1_74', 'MM_DRB1_37', 'MM_DRB1_74', 'MM_DRB1_10', 'MM_DRB1_12', 'MM_DRB1_67', 'MM_DQB1_55', 'MM_DRB1_32', 'MM_DRB1_30', 'MM_DQB1_77', 'MM_DRB1_13', 'MM_DRB1_28', 'MM_DRB1_70', 'MM_DRB1_47', 'MM_DRB1_11'], 'Bin 7': ['MM_DRB1_11', 'MM_DQB1_75', 'MM_DRB1_28', 'MM_DRB1_30', 'MM_DRB1_10', 'MM_DRB1_74', 'MM_DQB1_71', 'MM_DRB1_13', 'MM_DQB1_55', 'MM_DRB1_67', 'MM_DRB1_12', 'MM_DRB1_37', 'MM_DRB1_47', 'MM_DRB1_32', 'MM_DRB1_70'], 'Bin 8': ['MM_DRB1_67', 'MM_DRB1_37', 'MM_DRB1_26', 'MM_DQB1_55', 'MM_DRB1_74', 'MM_DRB1_32', 'MM_DRB1_13', 'MM_DRB1_30', 'MM_DQB1_71', 'MM_DRB1_70', 'MM_DRB1_10', 'MM_DRB1_47', 'MM_DRB1_12', 'MM_DQB1_74', 'MM_DRB1_28'], 'Bin 9': ['MM_DQB1_74', 'MM_DQB1_75', 'MM_DRB1_28', 'MM_DRB1_30', 'MM_DRB1_10', 'MM_DRB1_74', 'MM_DQB1_71', 'MM_DRB1_13', 'MM_DQB1_55', 'MM_DRB1_67', 'MM_DRB1_12', 'MM_DRB1_37', 'MM_DRB1_47', 'MM_DRB1_32', 'MM_DRB1_70'], 'Bin 10': ['MM_DRB1_26', 'MM_DRB1_11', 'MM_DQB1_75', 'MM_DRB1_28', 'MM_DRB1_30', 'MM_DRB1_10', 'MM_DRB1_74', 'MM_DRB1_13', 'MM_DQB1_55', 'MM_DRB1_67', 'MM_DRB1_12', 'MM_DRB1_37', 'MM_DRB1_47', 'MM_DRB1_32', 'MM_DRB1_70'], 'Bin 11': ['MM_DRB1_26', 'MM_DRB1_30', 'MM_DRB1_28', 'MM_DRB1_47', 'MM_DRB1_13', 'MM_DQB1_74', 'MM_DRB1_32', 'MM_DRB1_12', 'MM_DRB1_37', 'MM_DRB1_70', 'MM_DRB1_10', 'MM_DQB1_71', 'MM_DQB1_55', 'MM_DRB1_67', 'MM_DRB1_11'], 'Bin 12': ['MM_DRB1_70', 'MM_DQB1_74', 'MM_DQB1_71', 'MM_DRB1_10', 'MM_DQB1_55', 'MM_DRB1_67', 'MM_DRB1_12', 'MM_DRB1_11', 'MM_DRB1_47', 'MM_DRB1_13', 'MM_DRB1_26', 'MM_DRB1_30', 'MM_DRB1_32', 'MM_DRB1_37', 'MM_DRB1_74'], 'Bin 13': ['MM_DRB1_11', 'MM_B_32', 'MM_DRB1_30', 'MM_DRB1_28', 'MM_DRB1_67', 'MM_DRB1_37', 'MM_DRB1_12', 'MM_DRB1_13', 'MM_DRB1_10', 'MM_DQB1_55', 'MM_DQB1_71', 'MM_DRB1_74', 'MM_DRB1_70', 'MM_DRB1_32', 'MM_DRB1_47'], 'Bin 14': ['MM_DRB1_12', 'MM_DQB1_74', 'MM_DRB1_13', 'MM_DRB1_28', 'MM_DRB1_67', 'MM_DRB1_32', 'MM_DRB1_74', 'MM_DRB1_11', 'MM_DRB1_47', 'MM_DQB1_55', 'MM_DQB1_71', 'MM_DRB1_10', 'MM_DRB1_30', 'MM_DRB1_37', 'MM_DRB1_26'], 'Bin 15': ['MM_DRB1_26', 'MM_DRB1_28', 'MM_DQB1_75', 'MM_DRB1_67', 'MM_DRB1_10', 'MM_DRB1_74', 'MM_DRB1_47', 'MM_DRB1_37', 'MM_DQB1_74', 'MM_DQB1_55', 'MM_DRB1_13', 'MM_DRB1_70', 'MM_DRB1_30', 'MM_DRB1_12', 'MM_DRB1_32'], 'Bin 16': ['MM_DRB1_28', 'MM_DQB1_55', 'MM_DQB1_74', 'MM_DRB1_47', 'MM_DRB1_10', 'MM_DRB1_70', 'MM_DRB1_32', 'MM_DRB1_13', 'MM_DRB1_30', 'MM_DQB1_75', 'MM_DRB1_12', 'MM_DRB1_11', 'MM_DRB1_67', 'MM_DQB1_71', 'MM_DRB1_37'], 'Bin 17': ['MM_DQB1_75', 'MM_DQB1_71', 'MM_DRB1_67', 'MM_DRB1_10', 'MM_DRB1_74', 'MM_DRB1_11', 'MM_DRB1_47', 'MM_DRB1_37', 'MM_DQB1_74', 'MM_DQB1_55', 'MM_DRB1_13', 'MM_DRB1_70', 'MM_DRB1_30', 'MM_DRB1_12', 'MM_DRB1_32'], 'Bin 18': ['MM_DRB1_37', 'MM_DQB1_74', 'MM_DQB1_75', 'MM_DQB1_71', 'MM_DQB1_55', 'MM_DRB1_67', 'MM_DRB1_13', 'MM_DRB1_12', 'MM_DRB1_10', 'MM_DRB1_30', 'MM_DRB1_28', 'MM_DRB1_47', 'MM_DRB1_32', 'MM_DRB1_11', 'MM_DRB1_74'], 'Bin 19': ['MM_DQB1_71', 'MM_DRB1_28', 'MM_DQB1_77', 'MM_DRB1_70', 'MM_DRB1_13', 'MM_DRB1_67', 'MM_DRB1_47', 'MM_DRB1_11', 'MM_DRB1_32', 'MM_DRB1_12', 'MM_DRB1_30', 'MM_DRB1_10', 'MM_DQB1_55', 'MM_DRB1_74', 'MM_DRB1_37'], 'Bin 20': ['MM_DRB1_37', 'MM_DQB1_74', 'MM_DRB1_67', 'MM_DRB1_28', 'MM_B_32', 'MM_DQB1_71', 'MM_DRB1_74', 'MM_DQB1_55', 'MM_DRB1_13', 'MM_DRB1_10', 'MM_DRB1_70', 'MM_DRB1_30', 'MM_DRB1_12', 'MM_DRB1_32', 'MM_DRB1_47'], 'Bin 21': ['MM_DRB1_28', 'MM_DQB1_74', 'MM_DRB1_37', 'MM_DRB1_13', 'MM_DRB1_12', 'MM_DQB1_55', 'MM_DRB1_11', 'MM_DQB1_56', 'MM_DRB1_67', 'MM_DRB1_32', 'MM_DRB1_74', 'MM_DRB1_30', 'MM_DRB1_26', 'MM_DRB1_10', 'MM_A_152'], 'Bin 22': ['MM_DRB1_47', 'MM_DRB1_70', 'MM_DQB1_75', 'MM_DQB1_71', 'MM_DRB1_10', 'MM_DRB1_13', 'MM_DQB1_74', 'MM_DRB1_11', 'MM_DRB1_30', 'MM_DRB1_67', 'MM_DRB1_12', 'MM_DRB1_32', 'MM_C_73', 'MM_C_99', 'MM_C_174'], 'Bin 23': ['MM_DRB1_13', 'MM_DRB1_47', 'MM_DRB1_11', 'MM_DRB1_74', 'MM_DQB1_71', 'MM_DRB1_12', 'MM_DRB1_10', 'MM_DRB1_32', 'MM_DQB1_67', 'MM_DRB1_28', 'MM_C_96', 'MM_DQB1_52', 'MM_C_93', 'MM_DRB1_58', 'MM_A_67'], 'Bin 24': ['MM_DQB1_75', 'MM_DRB1_30', 'MM_DQB1_55', 'MM_DRB1_70', 'MM_DRB1_10', 'MM_DRB1_28', 'MM_DRB1_13', 'MM_DRB1_12', 'MM_DRB1_37', 'MM_DRB1_74', 'MM_C_25', 'MM_A_77', 'MM_DRB1_11', 'MM_B_114', 'MM_B_151'], 'Bin 25': ['MM_DRB1_26', 'MM_DQB1_75', 'MM_DRB1_13', 'MM_DQB1_55', 'MM_DRB1_37', 'MM_DRB1_11', 'MM_DRB1_67', 'MM_DRB1_47', 'MM_DRB1_32', 'MM_DQB1_71', 'MM_DRB1_30', 'MM_A_81', 'MM_B_32', 'MM_C_73', 'MM_A_79'], 'Bin 26': ['MM_DRB1_28', 'MM_DRB1_74', 'MM_DRB1_12', 'MM_DRB1_70', 'MM_DQB1_74', 'MM_DRB1_67', 'MM_DRB1_32', 'MM_DRB1_11', 'MM_DQB1_55', 'MM_DRB1_10', 'MM_DRB1_30', 'MM_C_144', 'MM_C_175', 'MM_C_14', 'MM_A_144'], 'Bin 27': ['MM_DRB1_12', 'MM_DQB1_55', 'MM_DRB1_30', 'MM_DQB1_75', 'MM_DRB1_28', 'MM_DRB1_67', 'MM_DRB1_37', 'MM_DRB1_32', 'MM_DRB1_10', 'MM_DRB1_11', 'MM_DRB1_26', 'MM_C_157', 'MM_DRB1_47', 'MM_B_82', 'MM_B_75'], 'Bin 28': ['MM_DQB1_74', 'MM_DRB1_74', 'MM_DQB1_77', 'MM_DRB1_70', 'MM_DRB1_47', 'MM_DRB1_11', 'MM_DQB1_71', 'MM_DRB1_37', 'MM_DRB1_67', 'MM_DRB1_32', 'MM_DRB1_13', 'MM_DRB1_30', 'MM_DRB1_28', 'MM_DRB1_10', 'MM_B_177'], 'Bin 29': ['MM_B_32', 'MM_DRB1_28', 'MM_DRB1_47', 'MM_DRB1_74', 'MM_DQB1_71', 'MM_DQB1_74', 'MM_DRB1_67', 'MM_DRB1_13', 'MM_DRB1_37', 'MM_A_81', 'MM_DRB1_32', 'MM_A_125', 'MM_C_120', 'MM_C_102', 'MM_C_91'], 'Bin 30': ['MM_DRB1_11', 'MM_DRB1_30', 'MM_DQB1_55', 'MM_DRB1_70', 'MM_DRB1_74', 'MM_DRB1_47', 'MM_DRB1_37', 'MM_DRB1_12', 'MM_DRB1_32', 'MM_DQB1_30', 'MM_DQB1_71', 'MM_DRB1_13', 'MM_A_97', 'MM_A_92', 'MM_C_77'], 'Bin 31': ['MM_DQB1_74', 'MM_DRB1_67', 'MM_DRB1_12', 'MM_DRB1_10', 'MM_DRB1_47', 'MM_DRB1_32', 'MM_DRB1_11', 'MM_DRB1_13', 'MM_DRB1_37', 'MM_DRB1_70', 'MM_DRB1_74', 'MM_DQB1_55', 'MM_C_137', 'MM_C_131', 'MM_B_163'], 'Bin 32': ['MM_DQB1_75', 'MM_DQB1_71', 'MM_DRB1_28', 'MM_DRB1_37', 'MM_DRB1_74', 'MM_DQB1_55', 'MM_DRB1_67', 'MM_DRB1_12', 'MM_DRB1_47', 'MM_DRB1_32', 'MM_C_136', 'MM_DRB1_66', 'MM_DRB1_76', 'MM_DRB1_11', 'MM_DRB1_34'], 'Bin 33': ['MM_DRB1_47', 'MM_DRB1_37', 'MM_DRB1_30', 'MM_DRB1_13', 'MM_DRB1_11', 'MM_DQB1_55', 'MM_DQB1_75', 'MM_DRB1_28', 'MM_DRB1_67', 'MM_DRB1_12', 'MM_DRB1_32', 'MM_DRB1_10', 'MM_DRB1_74', 'MM_B_156', 'MM_A_107'], 'Bin 34': ['MM_DQB1_74', 'MM_DRB1_70', 'MM_DRB1_26', 'MM_DRB1_11', 'MM_DRB1_28', 'MM_DRB1_30', 'MM_DRB1_74', 'MM_DQB1_55', 'MM_DRB1_67', 'MM_DRB1_12', 'MM_DRB1_37', 'MM_DRB1_32', 'MM_C_119', 'MM_DQB1_23', 'MM_C_97'], 'Bin 35': ['MM_DRB1_37', 'MM_DRB1_30', 'MM_DRB1_67', 'MM_DRB1_10', 'MM_B_32', 'MM_A_107', 'MM_DRB1_74', 'MM_DRB1_47', 'MM_DQB1_90', 'MM_DRB1_13', 'MM_DRB1_12', 'MM_DRB1_70', 'MM_B_94', 'MM_C_134', 'MM_A_145'], 'Bin 36': ['MM_DRB1_32', 'MM_DRB1_28', 'MM_DRB1_11', 'MM_DRB1_70', 'MM_DRB1_74', 'MM_DQB1_74', 'MM_DRB1_12', 'MM_DRB1_10', 'MM_DQB1_55', 'MM_DRB1_13', 'MM_B_80', 'MM_DRB1_78', 'MM_B_69', 'MM_B_66', 'MM_C_106'], 'Bin 37': ['MM_DQB1_74', 'MM_DQB1_55', 'MM_DRB1_47', 'MM_DRB1_10', 'MM_DRB1_70', 'MM_DRB1_28', 'MM_DRB1_74', 'MM_C_106', 'MM_DRB1_37', 'MM_DQB1_71', 'MM_DRB1_26', 'MM_DRB1_16', 'MM_A_107', 'MM_C_136', 'MM_A_9'], 'Bin 38': ['MM_B_32', 'MM_DRB1_13', 'MM_DRB1_30', 'MM_DRB1_12', 'MM_DRB1_32', 'MM_DRB1_28', 'MM_DQB1_71', 'MM_DRB1_70', 'MM_DRB1_11', 'MM_DRB1_10', 'MM_DRB1_74', 'MM_DRB1_37', 'MM_DRB1_67', 'MM_C_25', 'MM_C_126'], 'Bin 39': ['MM_DQB1_74', 'MM_DRB1_37', 'MM_DRB1_30', 'MM_DRB1_28', 'MM_DQB1_55', 'MM_DRB1_70', 'MM_DRB1_13', 'MM_C_154', 'MM_DRB1_67', 'MM_DRB1_10', 'MM_DRB1_47', 'MM_C_123', 'MM_DQB1_90', 'MM_C_25', 'MM_C_134'], 'Bin 40': ['MM_DRB1_26', 'MM_DRB1_32', 'MM_DRB1_74', 'MM_DQB1_85', 'MM_A_152', 'MM_DQB1_77', 'MM_DRB1_47', 'MM_DRB1_30', 'MM_DQB1_55', 'MM_DRB1_13', 'MM_DRB1_12', 'MM_DRB1_28', 'MM_A_127', 'MM_DQB1_14', 'MM_B_178'], 'Bin 41': ['MM_DRB1_67', 'MM_DRB1_30', 'MM_DRB1_32', 'MM_DQB1_74', 'MM_DQB1_75', 'MM_DRB1_37', 'MM_C_164', 'MM_B_162', 'MM_A_17', 'MM_DRB1_70', 'MM_DQB1_55', 'MM_DRB1_12', 'MM_DRB1_74', 'MM_DRB1_11', 'MM_B_30'], 'Bin 42': ['MM_DRB1_28', 'MM_DRB1_13', 'MM_DRB1_47', 'MM_DRB1_10', 'MM_C_118', 'MM_DQB1_77', 'MM_C_165', 'MM_A_107', 'MM_DRB1_70', 'MM_DQB1_55', 'MM_DRB1_67', 'MM_DRB1_12', 'MM_DRB1_11', 'MM_DRB1_37', 'MM_DQB1_67'], 'Bin 43': ['MM_DRB1_47', 'MM_DRB1_28', 'MM_A_105', 'MM_DRB1_13', 'MM_A_102', 'MM_DRB1_70', 'MM_DQB1_74', 'MM_DQB1_55', 'MM_DRB1_12', 'MM_DQB1_85', 'MM_DRB1_30', 'MM_C_145', 'MM_DRB1_37', 'MM_B_66', 'MM_C_121'], 'Bin 44': ['MM_DRB1_74', 'MM_DQB1_77', 'MM_DRB1_10', 'MM_C_123', 'MM_A_127', 'MM_C_154', 'MM_DRB1_13', 'MM_DRB1_30', 'MM_DRB1_37', 'MM_DRB1_12', 'MM_DRB1_70', 'MM_DQB1_55', 'MM_DRB1_76', 'MM_B_127', 'MM_A_70'], 'Bin 45': ['MM_DRB1_70', 'MM_DRB1_28', 'MM_DRB1_67', 'MM_DQB1_71', 'MM_DRB1_30', 'MM_DQB1_74', 'MM_DQB1_55', 'MM_DQB1_75', 'MM_A_107', 'MM_DRB1_47', 'MM_DRB1_37', 'MM_C_124', 'MM_DRB1_32', 'MM_C_80', 'MM_DRB1_35'], 'Bin 46': ['MM_DRB1_10', 'MM_DRB1_74', 'MM_DRB1_13', 'MM_DQB1_74', 'MM_DRB1_32', 'MM_DQB1_75', 'MM_DRB1_11', 'MM_DRB1_67', 'MM_DQB1_71', 'MM_DRB1_30', 'MM_DRB1_37', 'MM_DRB1_12', 'MM_DQB1_55', 'MM_DRB1_47', 'MM_B_83'], 'Bin 47': ['MM_DRB1_70', 'MM_DRB1_12', 'MM_C_91', 'MM_B_58', 'MM_DRB1_37', 'MM_DRB1_30', 'MM_DRB1_11', 'MM_DRB1_13', 'MM_DQB1_55', 'MM_DRB1_10', 'MM_DRB1_67', 'MM_DQB1_75', 'MM_B_7', 'MM_DQB1_77', 'MM_A_114'], 'Bin 48': ['MM_DRB1_74', 'MM_DRB1_47', 'MM_DRB1_32', 'MM_A_74', 'MM_C_73', 'MM_DRB1_11', 'MM_DRB1_10', 'MM_DRB1_13', 'MM_DRB1_37', 'MM_DQB1_74', 'MM_DRB1_67', 'MM_C_49', 'MM_C_99', 'MM_C_119', 'MM_B_75'], 'Bin 49': ['MM_DRB1_10', 'MM_DRB1_74', 'MM_DQB1_71', 'MM_DRB1_13', 'MM_DRB1_67', 'MM_B_32', 'MM_DRB1_32', 'MM_DQB1_75', 'MM_A_144', 'MM_A_81', 'MM_B_138', 'MM_B_147', 'MM_C_21', 'MM_C_124', 'MM_C_80'], 'Bin 50': ['MM_DRB1_11', 'MM_DRB1_28', 'MM_DRB1_30', 'MM_DRB1_12', 'MM_DRB1_37', 'MM_DRB1_47', 'MM_DQB1_75', 'MM_DRB1_26', 'MM_DRB1_13', 'MM_DRB1_70', 'MM_DQB1_55', 'MM_DRB1_73', 'MM_A_149', 'MM_B_136', 'MM_A_17'], 'Bin 51': ['MM_DQB1_71', 'MM_DRB1_11', 'MM_DRB1_26', 'MM_DRB1_67', 'MM_DQB1_47', 'MM_DRB1_12', 'MM_A_144', 'MM_DRB1_70', 'MM_A_125', 'MM_DQB1_55', 'MM_A_156', 'MM_C_122', 'MM_A_150', 'MM_C_103', 'MM_C_180'], 'Bin 52': ['MM_DQB1_74', 'MM_DRB1_13', 'MM_DRB1_74', 'MM_A_167', 'MM_C_149', 'MM_DQB1_38', 'MM_DRB1_70', 'MM_DRB1_32', 'MM_DRB1_47', 'MM_DRB1_28', 'MM_DQB1_55', 'MM_DRB1_12', 'MM_C_136', 'MM_C_66', 'MM_B_67'], 'Bin 53': ['MM_DRB1_30', 'MM_DQB1_55', 'MM_DQB1_71', 'MM_DRB1_11', 'MM_DQB1_74', 'MM_DRB1_28', 'MM_DQB1_75', 'MM_DRB1_12', 'MM_C_124', 'MM_DRB1_37', 'MM_A_107', 'MM_DRB1_32', 'MM_DRB1_70', 'MM_DRB1_47', 'MM_C_154'], 'Bin 54': ['MM_DRB1_13', 'MM_DRB1_67', 'MM_B_32', 'MM_DRB1_10', 'MM_DRB1_74', 'MM_DQB1_74', 'MM_DRB1_32', 'MM_DRB1_11', 'MM_DRB1_28', 'MM_DRB1_37', 'MM_DRB1_12', 'MM_DRB1_70', 'MM_DRB1_6', 'MM_C_104', 'MM_A_31'], 'Bin 55': ['MM_DRB1_47', 'MM_B_32', 'MM_DRB1_70', 'MM_DRB1_67', 'MM_DQB1_90', 'MM_DRB1_37', 'MM_DRB1_12', 'MM_DQB1_74', 'MM_DRB1_32', 'MM_DRB1_74', 'MM_DRB1_13', 'MM_DRB1_28', 'MM_C_6', 'MM_C_161', 'MM_DRB1_52'], 'Bin 56': ['MM_DRB1_30', 'MM_DRB1_26', 'MM_DQB1_71', 'MM_DRB1_10', 'MM_DRB1_70', 'MM_DRB1_74', 'MM_DRB1_32', 'MM_DRB1_28', 'MM_DQB1_55', 'MM_DRB1_37', 'MM_DRB1_13', 'MM_A_107', 'MM_C_80', 'MM_C_155', 'MM_A_79'], 'Bin 57': ['MM_DRB1_74', 'MM_DRB1_37', 'MM_DRB1_28', 'MM_DQB1_71', 'MM_DQB1_77', 'MM_DRB1_30', 'MM_DRB1_67', 'MM_DRB1_13', 'MM_DRB1_70', 'MM_DRB1_47', 'MM_DQB1_55', 'MM_C_118', 'MM_DRB1_10', 'MM_DQB1_52', 'MM_B_151'], 'Bin 58': ['MM_DRB1_26', 'MM_DRB1_12', 'MM_A_35', 'MM_B_59', 'MM_DRB1_13', 'MM_DRB1_47', 'MM_DRB1_70', 'MM_DQB1_55', 'MM_DRB1_67', 'MM_DRB1_10', 'MM_DRB1_30', 'MM_DRB1_11', 'MM_C_165', 'MM_A_107', 'MM_DRB1_85'], 'Bin 59': ['MM_DRB1_70', 'MM_DRB1_13', 'MM_DRB1_10', 'MM_C_24', 'MM_DRB1_26', 'MM_DQB1_74', 'MM_C_132', 'MM_C_114', 'MM_DRB1_47', 'MM_DRB1_28', 'MM_DRB1_37', 'MM_DRB1_30', 'MM_B_95', 'MM_DRB1_67', 'MM_DQB1_57'], 'Bin 60': ['MM_DRB1_32', 'MM_DRB1_11', 'MM_DRB1_74', 'MM_C_123', 'MM_DRB1_12', 'MM_DQB1_71', 'MM_DQB1_55', 'MM_DRB1_47', 'MM_DRB1_37', 'MM_DRB1_10', 'MM_DRB1_67', 'MM_DQB1_74', 'MM_DRB1_33', 'MM_C_113', 'MM_A_62'], 'Bin 61': ['MM_DRB1_28', 'MM_DRB1_32', 'MM_DQB1_55', 'MM_DRB1_11', 'MM_DRB1_12', 'MM_DQB1_74', 'MM_DRB1_30', 'MM_DQB1_71', 'MM_DRB1_47', 'MM_DQB1_84', 'MM_B_74', 'MM_B_113', 'MM_B_65', 'MM_B_116', 'MM_C_115'], 'Bin 62': ['MM_DRB1_26', 'MM_DRB1_70', 'MM_DRB1_10', 'MM_DQB1_75', 'MM_DQB1_71', 'MM_DRB1_67', 'MM_DRB1_74', 'MM_DRB1_47', 'MM_DRB1_37', 'MM_DQB1_74', 'MM_DRB1_13', 'MM_DRB1_30', 'MM_DRB1_12', 'MM_B_76', 'MM_C_11'], 'Bin 63': ['MM_DRB1_70', 'MM_DRB1_13', 'MM_DRB1_47', 'MM_DRB1_11', 'MM_DQB1_55', 'MM_DRB1_28', 'MM_DQB1_71', 'MM_DQB1_75', 'MM_A_107', 'MM_DRB1_37', 'MM_DRB1_32', 'MM_DRB1_12', 'MM_DQB1_90', 'MM_A_163', 'MM_DQB1_13'], 'Bin 64': ['MM_DRB1_30', 'MM_DRB1_26', 'MM_DRB1_10', 'MM_DRB1_74', 'MM_DRB1_67', 'MM_DQB1_74', 'MM_DRB1_32', 'MM_DRB1_11', 'MM_DRB1_37', 'MM_DRB1_12', 'MM_DQB1_55', 'MM_DRB1_47', 'MM_C_6', 'MM_A_73', 'MM_C_122'], 'Bin 65': ['MM_DRB1_11', 'MM_DRB1_30', 'MM_DQB1_55', 'MM_DRB1_12', 'MM_DRB1_74', 'MM_DRB1_70', 'MM_DQB1_75', 'MM_DRB1_37', 'MM_DRB1_13', 'MM_DRB1_10', 'MM_DRB1_32', 'MM_DRB1_67', 'MM_A_145', 'MM_B_70', 'MM_C_137'], 'Bin 66': ['MM_DRB1_28', 'MM_DQB1_71', 'MM_DRB1_47', 'MM_DRB1_26', 'MM_DQB1_75', 'MM_DRB1_30', 'MM_DRB1_10', 'MM_DRB1_74', 'MM_DRB1_67', 'MM_DRB1_37', 'MM_DRB1_32', 'MM_DRB1_70', 'MM_C_144', 'MM_DQB1_11', 'MM_C_91'], 'Bin 67': ['MM_DRB1_11', 'MM_C_157', 'MM_DRB1_13', 'MM_DQB1_75', 'MM_DRB1_37', 'MM_C_136', 'MM_DRB1_70', 'MM_DRB1_67', 'MM_DRB1_28', 'MM_DRB1_74', 'MM_C_137', 'MM_C_169', 'MM_DRB1_50', 'MM_C_149', 'MM_B_136'], 'Bin 68': ['MM_DRB1_47', 'MM_DQB1_71', 'MM_DRB1_32', 'MM_DRB1_26', 'MM_DRB1_30', 'MM_DRB1_10', 'MM_DQB1_75', 'MM_DRB1_28', 'MM_DQB1_55', 'MM_DRB1_67', 'MM_DRB1_12', 'MM_DRB1_70', 'MM_B_178', 'MM_C_143', 'MM_DRB1_14'], 'Bin 69': ['MM_DRB1_74', 'MM_DRB1_47', 'MM_DRB1_28', 'MM_C_99', 'MM_DRB1_30', 'MM_DQB1_46', 'MM_DQB1_55', 'MM_DQB1_74', 'MM_DQB1_75', 'MM_DRB1_12', 'MM_DRB1_70', 'MM_DRB1_32', 'MM_A_90', 'MM_B_162', 'MM_C_114'], 'Bin 70': ['MM_DQB1_71', 'MM_DRB1_67', 'MM_DRB1_11', 'MM_DRB1_37', 'MM_DRB1_10', 'MM_DQB1_74', 'MM_DQB1_55', 'MM_DQB1_75', 'MM_DRB1_30', 'MM_DRB1_13', 'MM_DRB1_70', 'MM_DRB1_32', 'MM_DRB1_9', 'MM_C_101', 'MM_C_112'], 'Bin 71': ['MM_DRB1_30', 'MM_DRB1_32', 'MM_DRB1_70', 'MM_DQB1_71', 'MM_DQB1_74', 'MM_DRB1_67', 'MM_DRB1_47', 'MM_DRB1_10', 'MM_DQB1_55', 'MM_DRB1_28', 'MM_DRB1_13', 'MM_DQB1_75', 'MM_C_176', 'MM_DRB1_31', 'MM_DRB1_25'], 'Bin 72': ['MM_DRB1_12', 'MM_DRB1_74', 'MM_DRB1_26', 'MM_DRB1_28', 'MM_DQB1_55', 'MM_DQB1_74', 'MM_DRB1_47', 'MM_DRB1_32', 'MM_DRB1_13', 'MM_DRB1_11', 'MM_DRB1_67', 'MM_DQB1_71', 'MM_DRB1_37', 'MM_C_173', 'MM_DQB1_26'], 'Bin 73': ['MM_DRB1_47', 'MM_DQB1_55', 'MM_DRB1_74', 'MM_DRB1_14', 'MM_C_116', 'MM_DRB1_67', 'MM_DQB1_74', 'MM_DRB1_70', 'MM_DRB1_26', 'MM_DRB1_30', 'MM_DRB1_13', 'MM_DRB1_28', 'MM_DRB1_10', 'MM_C_159', 'MM_C_177'], 'Bin 74': ['MM_DRB1_12', 'MM_DQB1_71', 'MM_DRB1_32', 'MM_C_15', 'MM_B_151', 'MM_DRB1_30', 'MM_DRB1_47', 'MM_DRB1_10', 'MM_DRB1_26', 'MM_DRB1_11', 'MM_DRB1_70', 'MM_DRB1_74', 'MM_A_114', 'MM_B_156', 'MM_A_43'], 'Bin 75': ['MM_DRB1_74', 'MM_DQB1_71', 'MM_DRB1_70', 'MM_DRB1_12', 'MM_DRB1_13', 'MM_DRB1_32', 'MM_DQB1_30', 'MM_DRB1_67', 'MM_DQB1_55', 'MM_B_97', 'MM_C_143', 'MM_C_175', 'MM_A_74', 'MM_C_99', 'MM_A_70'], 'Bin 76': ['MM_DRB1_37', 'MM_DRB1_26', 'MM_DRB1_30', 'MM_DRB1_10', 'MM_DQB1_74', 'MM_DRB1_28', 'MM_B_32', 'MM_A_81', 'MM_DRB1_67', 'MM_DRB1_32', 'MM_DRB1_13', 'MM_DQB1_55', 'MM_B_71', 'MM_B_65', 'MM_C_136'], 'Bin 77': ['MM_DRB1_37', 'MM_DRB1_11', 'MM_DRB1_74', 'MM_DRB1_13', 'MM_DRB1_32', 'MM_DRB1_12', 'MM_DRB1_70', 'MM_DRB1_67', 'MM_DRB1_28', 'MM_B_156', 'MM_DRB1_30', 'MM_B_32', 'MM_DRB1_47', 'MM_A_9', 'MM_DQB1_86'], 'Bin 78': ['MM_DRB1_10', 'MM_B_7', 'MM_DRB1_26', 'MM_DQB1_55', 'MM_DRB1_28', 'MM_DRB1_67', 'MM_DRB1_74', 'MM_DRB1_47', 'MM_DQB1_74', 'MM_DRB1_30', 'MM_DRB1_12', 'MM_DRB1_32', 'MM_A_166', 'MM_B_4', 'MM_B_70'], 'Bin 79': ['MM_DRB1_30', 'MM_DQB1_55', 'MM_DQB1_75', 'MM_B_90', 'MM_DQB1_74', 'MM_DRB1_12', 'MM_DRB1_32', 'MM_DRB1_13', 'MM_C_148', 'MM_DRB1_74', 'MM_DRB1_47', 'MM_DRB1_26', 'MM_DRB1_67', 'MM_C_112', 'MM_C_173'], 'Bin 80': ['MM_DRB1_37', 'MM_DRB1_11', 'MM_DRB1_70', 'MM_DQB1_71', 'MM_B_167', 'MM_DQB1_74', 'MM_DRB1_13', 'MM_DRB1_10', 'MM_DRB1_32', 'MM_DRB1_67', 'MM_DRB1_12', 'MM_DQB1_75', 'MM_DQB1_55', 'MM_C_111', 'MM_DRB1_92'], 'Bin 81': ['MM_B_7', 'MM_DQB1_53', 'MM_DQB1_52', 'MM_C_123', 'MM_DRB1_74', 'MM_DRB1_13', 'MM_B_59', 'MM_DRB1_37', 'MM_A_99', 'MM_DRB1_10', 'MM_C_138', 'MM_DRB1_28', 'MM_DRB1_47', 'MM_DRB1_67', 'MM_DQB1_71'], 'Bin 82': ['MM_DRB1_11', 'MM_DQB1_75', 'MM_DRB1_30', 'MM_DQB1_55', 'MM_DQB1_74', 'MM_DRB1_12', 'MM_B_58', 'MM_A_114', 'MM_DRB1_37', 'MM_DRB1_67', 'MM_DRB1_13', 'MM_DRB1_10', 'MM_DRB1_70', 'MM_A_90', 'MM_C_177'], 'Bin 83': ['MM_DRB1_70', 'MM_DRB1_47', 'MM_DRB1_28', 'MM_DRB1_30', 'MM_DRB1_37', 'MM_DRB1_74', 'MM_DRB1_10', 'MM_DRB1_11', 'MM_DRB1_67', 'MM_DQB1_55', 'MM_DRB1_26', 'MM_DRB1_12', 'MM_DQB1_71', 'MM_A_74', 'MM_B_11'], 'Bin 84': ['MM_DQB1_74', 'MM_DRB1_32', 'MM_DRB1_30', 'MM_DRB1_26', 'MM_DQB1_55', 'MM_DRB1_13', 'MM_DRB1_10', 'MM_DRB1_11', 'MM_DRB1_12', 'MM_DRB1_37', 'MM_DRB1_28', 'MM_DRB1_67', 'MM_C_157', 'MM_C_35', 'MM_A_12'], 'Bin 85': ['MM_DRB1_28', 'MM_DRB1_74', 'MM_DRB1_70', 'MM_DQB1_46', 'MM_DRB1_9', 'MM_DQB1_71', 'MM_DQB1_55', 'MM_DRB1_13', 'MM_DRB1_30', 'MM_DRB1_10', 'MM_C_99', 'MM_DRB1_12', 'MM_C_103', 'MM_DQB1_26', 'MM_B_138'], 'Bin 86': ['MM_DRB1_11', 'MM_DRB1_67', 'MM_DRB1_37', 'MM_DRB1_47', 'MM_DRB1_32', 'MM_DRB1_10', 'MM_DQB1_74', 'MM_DQB1_75', 'MM_DRB1_30', 'MM_DRB1_13', 'MM_DRB1_70', 'MM_DRB1_12', 'MM_C_45', 'MM_A_114', 'MM_A_167'], 'Bin 87': ['MM_DQB1_74', 'MM_DRB1_13', 'MM_DRB1_74', 'MM_DRB1_11', 'MM_DQB1_55', 'MM_DRB1_32', 'MM_DRB1_28', 'MM_DRB1_67', 'MM_DRB1_37', 'MM_DQB1_75', 'MM_DRB1_47', 'MM_DRB1_70', 'MM_DRB1_30', 'MM_DQB1_52', 'MM_A_105'], 'Bin 88': ['MM_DRB1_10', 'MM_DRB1_26', 'MM_DRB1_12', 'MM_DRB1_28', 'MM_DQB1_55', 'MM_DRB1_47', 'MM_DRB1_70', 'MM_DRB1_32', 'MM_DRB1_13', 'MM_DRB1_67', 'MM_DQB1_71', 'MM_DRB1_37', 'MM_C_145', 'MM_DQB1_53', 'MM_B_52'], 'Bin 89': ['MM_DQB1_74', 'MM_DRB1_32', 'MM_DRB1_13', 'MM_DRB1_28', 'MM_DQB1_55', 'MM_DRB1_30', 'MM_DRB1_67', 'MM_DRB1_10', 'MM_DRB1_11', 'MM_DRB1_26', 'MM_DRB1_47', 'MM_C_149', 'MM_C_168', 'MM_A_67', 'MM_A_63'], 'Bin 90': ['MM_DRB1_12', 'MM_DRB1_37', 'MM_DRB1_74', 'MM_DRB1_70', 'MM_DRB1_47', 'MM_DRB1_11', 'MM_DRB1_67', 'MM_DRB1_13', 'MM_DRB1_30', 'MM_DRB1_10', 'MM_A_144', 'MM_C_73', 'MM_C_178', 'MM_C_6', 'MM_C_165'], 'Bin 91': ['MM_DRB1_67', 'MM_DRB1_12', 'MM_DRB1_74', 'MM_DQB1_77', 'MM_DRB1_32', 'MM_DQB1_71', 'MM_C_150', 'MM_DQB1_55', 'MM_DRB1_28', 'MM_B_171', 'MM_DRB1_47', 'MM_A_99', 'MM_B_77', 'MM_C_146', 'MM_A_31'], 'Bin 92': ['MM_DQB1_75', 'MM_DRB1_30', 'MM_DRB1_70', 'MM_DQB1_74', 'MM_DRB1_11', 'MM_DRB1_13', 'MM_C_164', 'MM_A_17', 'MM_B_162', 'MM_DRB1_74', 'MM_DQB1_55', 'MM_DRB1_37', 'MM_DQB1_26', 'MM_DQB1_14', 'MM_DRB1_9'], 'Bin 93': ['MM_DRB1_32', 'MM_DRB1_10', 'MM_DRB1_47', 'MM_DRB1_12', 'MM_C_24', 'MM_DQB1_55', 'MM_DQB1_74', 'MM_C_123', 'MM_DQB1_71', 'MM_DRB1_67', 'MM_DRB1_11', 'MM_DRB1_70', 'MM_DQB1_75', 'MM_A_3', 'MM_B_98'], 'Bin 94': ['MM_DRB1_28', 'MM_DRB1_37', 'MM_DQB1_75', 'MM_DRB1_13', 'MM_DRB1_70', 'MM_DRB1_67', 'MM_DRB1_11', 'MM_DRB1_74', 'MM_DRB1_12', 'MM_DQB1_55', 'MM_DQB1_74', 'MM_C_99', 'MM_B_83', 'MM_C_77', 'MM_A_81'], 'Bin 95': ['MM_DQB1_55', 'MM_DRB1_47', 'MM_DRB1_70', 'MM_DRB1_10', 'MM_DRB1_30', 'MM_DQB1_71', 'MM_C_138', 'MM_DRB1_13', 'MM_DRB1_28', 'MM_C_123', 'MM_B_59', 'MM_DRB1_74', 'MM_A_166', 'MM_B_127', 'MM_C_147'], 'Bin 96': ['MM_DQB1_74', 'MM_DRB1_11', 'MM_DQB1_75', 'MM_DRB1_32', 'MM_DRB1_37', 'MM_DRB1_67', 'MM_DRB1_13', 'MM_DRB1_12', 'MM_DRB1_28', 'MM_DRB1_10', 'MM_DRB1_74', 'MM_DRB1_70', 'MM_DQB1_52', 'MM_A_99', 'MM_DQB1_85'], 'Bin 97': ['MM_DRB1_28', 'MM_DRB1_13', 'MM_DRB1_37', 'MM_DRB1_47', 'MM_DRB1_30', 'MM_DRB1_12', 'MM_DQB1_55', 'MM_DRB1_32', 'MM_DQB1_71', 'MM_DRB1_10', 'MM_DRB1_74', 'MM_DRB1_11', 'MM_A_116', 'MM_C_109', 'MM_A_152'], 'Bin 98': ['MM_DQB1_74', 'MM_DRB1_70', 'MM_DRB1_67', 'MM_DRB1_26', 'MM_DRB1_10', 'MM_DRB1_11', 'MM_DQB1_71', 'MM_DQB1_55', 'MM_DRB1_32', 'MM_DRB1_47', 'MM_DRB1_30', 'MM_C_25', 'MM_DQB1_67', 'MM_A_109', 'MM_C_94'], 'Bin 99': ['MM_DRB1_28', 'MM_DRB1_70', 'MM_DRB1_10', 'MM_DRB1_12', 'MM_DQB1_74', 'MM_DRB1_30', 'MM_DQB1_55', 'MM_DRB1_37', 'MM_DRB1_32', 'MM_DRB1_67', 'MM_DRB1_13', 'MM_C_94', 'MM_B_167', 'MM_B_66', 'MM_C_99'], 'Bin 100': ['MM_DRB1_11', 'MM_DQB1_75', 'MM_DRB1_74', 'MM_DRB1_37', 'MM_DRB1_26', 'MM_DQB1_55', 'MM_DRB1_32', 'MM_DRB1_13', 'MM_DRB1_30', 'MM_DRB1_70', 'MM_DRB1_47', 'MM_DQB1_74', 'MM_B_30', 'MM_C_168', 'MM_DRB1_91']}

bin_feature_matrix, amino_acid_bins, amino_acid_bin_scores, MAF_0_feautures = RAFE(True, SFVT_aa, RARESFVT, 0, srtr_data, 
                                                                                   'Class', 100, 
                                                                                   None, None, 'Rank-Biserial_Correlation', True, 
                                                                                   500, 0.8, 0.1, 0.2)

Top_Bins_Summary(srtr_data, 'Class', bin_feature_matrix, amino_acid_bins, amino_acid_bin_scores, 10)


# In[30]:


#Defining a function to present the top bins 
def Top_Rare_Variant_Bins_Summary(rare_feature_matrix, label_name, bins, bin_scores, 
                                  rare_feature_MAF_dict, number_of_top_bins):
    
    #Ordering the bin scores from best to worst
    sorted_bin_scores = dict(sorted(bin_scores.items(), key=lambda item: item[1], reverse=True))
    sorted_bin_list = list(sorted_bin_scores.keys())
    sorted_bin_feature_importance_values = list(sorted_bin_scores.values())

    #Calculating the chi square and p values of each of the features in the rare feature matrix
    df = rare_feature_matrix
    X = df.drop('Class',axis=1)
    y = df['Class']
    chi_scores, p_values = chi2(X,y)
    
    #Removing the label column to create a list of features
    feature_df = rare_feature_df.drop(columns = [label_name])
    
    #Creating a list of features
    feature_list = []
    for column in feature_df:
        feature_list.append(str(column))
        
    #Creating a dictionary with each feature and the chi-square value and p-value
    Univariate_Feature_Stats = {}
    for i in range (0, len(feature_list)):
        list_of_stats = []
        list_of_stats.append(chi_scores[i])
        list_of_stats.append(p_values[i])
        Univariate_Feature_Stats[feature_list[i]] = list_of_stats
        #There will be features with nan for their chi-square value and p-value because the whole column is zeroes
    
    #Calculating the chisquare and p values of each of the features in the bin feature matrix
    X = bin_feature_matrix.drop('Class',axis=1)
    y = bin_feature_matrix['Class']
    chi_scores, p_values = chi2(X,y)

    #Creating a dictionary with each bin and the chi-square value and p-value
    Bin_Stats = {}
    bin_names_list = list(amino_acid_bins.keys())
    for i in range (0, len(bin_names_list)):
        list_of_stats = []
        list_of_stats.append(chi_scores[i])
        list_of_stats.append(p_values[i])
        Bin_Stats[bin_names_list[i]] = list_of_stats
    
    for i in range (0, number_of_top_bins):
        #Printing the bin Name
        print ("Bin Rank " + str(i+1) + ": " + sorted_bin_list[i])
        #Printing the bin's MultiSURF/Univariate score, chi-square value, and p-value
        print ("Score: " + str(sorted_bin_feature_importance_values[i]) + "; chi-square value: " + str(Bin_Stats[sorted_bin_list[i]][0]) + "; p-value: " + str(Bin_Stats[sorted_bin_list[i]][1]))
        #Printing each of the features in the bin and also printing the univariate stats of that feature
        for j in range (0, len(bins[sorted_bin_list[i]])):
            print ("Feature Name: " + str(bins[sorted_bin_list[i]][j]) + "; minor allele frequency: " + str(rare_feature_MAF_dict[bins[sorted_bin_list[i]][j]]) + "; chi-square value: " + str(Univariate_Feature_Stats[bins[sorted_bin_list[i]][j]][0]) + "; p-value: " + str(Univariate_Feature_Stats[bins[sorted_bin_list[i]][j]][1]))
        print ('---------------------------')                                              


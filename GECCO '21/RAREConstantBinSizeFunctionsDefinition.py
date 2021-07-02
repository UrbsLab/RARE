#This file contains all the code defining the version of RARE with Constant Bin Size
#No function is ever called in this file, please see the "RAREExperiment6" file for an example of how to run RARE with Constant Bin Size on the Rare Variant Data Simulators
#Readers please note: amino acid refers to a rare variant feature, while amino acid bin is the equivalent of a bin of rare variant features



#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing packages
import os
import numpy as np
import matplotlib.pyplot as plt
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

def Random_Feature_Grouping_Constant_Bin_Size(feature_matrix, label_name, number_of_groups, number_of_features_in_group):
    
    #Removing the label column to create a list of features
    feature_df = feature_matrix.drop(columns = [label_name])
    
    #Creating a list of features 
    feature_list = []
    for column in feature_df:
        feature_list.append(str(column))
    
    #Creating the bins/groups
    feature_groups = []
    
    #Finding the number of feature copies needed to meet the user specified parameters
    feature_copies = (number_of_groups * number_of_features_in_group) - len(feature_list)
    
    #Creating the copies of features
    #This is so that bins can be created with every feature appearing at least once and the user parameters are met
    copies_added = 0
    copies = []
    
    #Shuffling the feature list so that the features with copies in the bin population are random
    random.shuffle(feature_list)
    while copies_added < feature_copies:
        #using modular arithmetic to ensure that once all the features are copied, it will start from the beginning again
        list_index = int(math.fmod(copies_added,len(feature_list)))
        copies.append(feature_list[list_index])
        copies_added = copies_added +1
        
    #Creating the pool of features from which bins can be constructed from
    feature_pool = feature_list.copy()
    feature_pool.extend(copies)
    
    #Shuffling the entire feature pool to ensure complete randomness in the bins
    random.shuffle(feature_pool)
    
    #Creating the feature groups (ie bins)
    n = number_of_features_in_group
    feature_groups = [feature_pool[i:i + n] for i in range(0, len(feature_pool), n)] 

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


#Step 2b: Genetic Algorithm 

#Parent bins are probabilistically selected based on fitness (score calculated in Step 2a)
#New offspring bins will be created through cross over and mutation and are added to the next generation's population
#Based on the value of the elitism parameter, a number of high scoring parent bins will be preserved for the next gen


# In[17]:


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


# In[18]:


#Defining a function for crossover and mutation that creates n offspring based on crossover of selected parents
#n is the max number of bins (but not all the offspring will carry on, as the worst will be deleted in Step 2a next time)
def Crossover_and_Mutation_Constant_Bin_Size(max_population_of_bins, elitism_parameter, feature_list, binned_feature_groups, bin_scores, 
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
        #Making features crossover from offspring bin 1 to offspring bin 2
        for j in range (0, len(parent1_features)):
            if crossover_probability > random.random():
                offspring2.append(parent1_features[j])
            else:
                offspring1.append(parent1_features[j])
        
        #An equal number of features will crossover from offspring bin 2 to offspring bin 1
        crossover_amount = len(offspring2)
        offspring2.extend(parent2_features)
        crossover_from2to1 = random.sample(parent2_features, crossover_amount)
        for feature in crossover_from2to1:
            offspring1.append(feature)
            offspring2.remove(feature)
        
        #MUTATION
        #Mutation only occurs with a certain probability on each feature in the original feature space
        #In this form of RARE (constant bin size), mutation can only be a swap and occurs based on a probability

        #Applying the mutation operation to the first offspring
        #Creating a list of features outside the offspring
        features_not_in_offspring = [item for item in feature_list if item not in offspring1]
        for l in range (0, len(offspring1)):
            delete_list = []
            add_list = []
            if mutation_probability > random.random():
                delete_list.append(offspring1[l])
                replacement = random.choice(features_not_in_offspring)
                add_list.append(replacement)
            
            for feature in delete_list:
                offspring1.remove(feature)
                
            for feature in add_list:
                offspring1.append(feature)
            
        #Applying the mutation operation to the second offspring
        features_not_in_offspring = [item for item in feature_list if item not in offspring2]
        for l in range (0, len(offspring2)):
            delete_list = []
            add_list = []
            if mutation_probability > random.random():
                delete_list.append(offspring2[l])
                replacement = random.choice(features_not_in_offspring)
                add_list.append(replacement)
            
            for feature in delete_list:
                offspring2.remove(feature)
                
            for feature in add_list:
                offspring2.append(feature)
            
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


# In[19]:


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


# In[20]:


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


# In[21]:


#Defining a function for the RAFE algorithm (Relief-based Association Feature-bin Evolver)
#Same as RARE but it bins all features not just rare features
def RAFE_Constant_Bin_Size (given_starting_point, amino_acid_start_point, amino_acid_bins_start_point, iterations, original_feature_matrix, 
          label_name, set_number_of_bins, number_of_features_in_group, 
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
        amino_acids, amino_acid_bins = Random_Feature_Grouping_Constant_Bin_Size(feature_matrix_no_empty_variables, label_name, 
                                                               set_number_of_bins, number_of_features_in_group)
    
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
        elif scoring_method == 'Univariate':
            amino_acid_bin_scores = Chi_Square_Feature_Importance(bin_feature_matrix, 'Class', amino_acid_bins)
            
        #Step 2b: Genetic Algorithm 
        #Creating the offspring bins through crossover and mutation
        offspring_bins = Crossover_and_Mutation_Constant_Bin_Size(set_number_of_bins, elitism_parameter, amino_acids, amino_acid_bins, amino_acid_bin_scores,
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
    elif scoring_method == 'Univariate':
        amino_acid_bin_scores = Chi_Square_Feature_Importance(bin_feature_matrix, 'Class', amino_acid_bins)
    
    return bin_feature_matrix, amino_acid_bins, amino_acid_bin_scores, MAF_0_features


# In[22]:


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
        print ("MultiSURF or Univariate Score: " + str(sorted_bin_feature_importance_values[i]) + "; chi-square value: " + str(Bin_Stats[sorted_bin_list[i]][0]) + "; p-value: " + str(Bin_Stats[sorted_bin_list[i]][1]))
        #Printing each of the features in the bin and also printing the univariate stats of that feature
        for j in range (0, len(bins[sorted_bin_list[i]])):
            print ("Feature Name: " + bins[sorted_bin_list[i]][j] + "; chi-square value: " + str(Univariate_Feature_Stats[bins[sorted_bin_list[i]][j]][0]) + "; p-value: " + str(Univariate_Feature_Stats[bins[sorted_bin_list[i]][j]][1]))
        print ('---------------------------')                                              
                        


# In[23]:


#Defining a function for the RARE algorithm (Relief-based Association Rare-variant-bin Evolver)
def RARE_Constant_Bin_Size (given_starting_point, amino_acid_start_point, amino_acid_bins_start_point, iterations, original_feature_matrix, 
          label_name, rare_variant_MAF_cutoff, set_number_of_bins, 
          number_of_features_in_group, 
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
        amino_acids, amino_acid_bins = Random_Feature_Grouping_Constant_Bin_Size(rare_feature_df, label_name, 
                                                               set_number_of_bins, number_of_features_in_group)
        
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
        
        elif scoring_method == 'Univariate':
            amino_acid_bin_scores = Chi_Square_Feature_Importance(bin_feature_matrix, 'Class', amino_acid_bins)
        
        #Step 2b: Genetic Algorithm 
        #Creating the offspring bins through crossover and mutation
        offspring_bins = Crossover_and_Mutation_Constant_Bin_Size(set_number_of_bins, elitism_parameter, amino_acids, amino_acid_bins, amino_acid_bin_scores,
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
    elif scoring_method == 'Univariate':
            amino_acid_bin_scores = Chi_Square_Feature_Importance(bin_feature_matrix, 'Class', amino_acid_bins)
    
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


# In[24]:


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
        print ("MultiSURF or Univariate Score: " + str(sorted_bin_feature_importance_values[i]) + "; chi-square value: " + str(Bin_Stats[sorted_bin_list[i]][0]) + "; p-value: " + str(Bin_Stats[sorted_bin_list[i]][1]))
        #Printing each of the features in the bin and also printing the univariate stats of that feature
        for j in range (0, len(bins[sorted_bin_list[i]])):
            print ("Feature Name: " + str(bins[sorted_bin_list[i]][j]) + "; minor allele frequency: " + str(rare_feature_MAF_dict[bins[sorted_bin_list[i]][j]]) + "; chi-square value: " + str(Univariate_Feature_Stats[bins[sorted_bin_list[i]][j]][0]) + "; p-value: " + str(Univariate_Feature_Stats[bins[sorted_bin_list[i]][j]][1]))
        print ('---------------------------')



# In[ ]:


#ADDING FUNCTIONS TO TEST RARE 


# In[ ]:


#Defining a function to create an artificial dataset with parameters, there will be one ideal/strong bin
#Note: MAF (minor allele frequency) cutoff refers to the threshold separating rare variant features from common features
def RVDS_One_Bin(number_of_instances, number_of_features, number_of_features_in_bin, 
                 rare_variant_MAF_cutoff, endpoint_cutoff_parameter, endpoint_variation_probability):
    
    #Creating an empty dataframe to use as a starting point for the eventual feature matrix
    #Adding one to number of features to give space for the class column
    df = pd.DataFrame(np.zeros((number_of_instances, number_of_features+1)))
    
    #Creating a list of features
    feature_list = []
    
    #Creating a list of predictive features in the strong bin
    predictive_features = []
    for a in range (0, number_of_features_in_bin):
        predictive_features.append("P_" + str(a+1))
    
    for b in range (0, len(predictive_features)):
        feature_list.append(predictive_features[b])
            
    #Creating a list of randomly created features
    random_features = []
    for c in range (0, number_of_features - number_of_features_in_bin):
        random_features.append("R_" + str(c+1))
    
    for d in range (0, len(random_features)):
        feature_list.append(random_features[d])
    
    #Adding the features and the class/endpoint 
    features_and_class = feature_list.copy()
    features_and_class.append('Class')
    df.columns = features_and_class
    
    #Creating a list of numbers with the amount of numbers equal to the number of instances
    #This will be used when assigning values to the values of features that are in the bin
    instance_list = []
    for number in range (0, number_of_instances):
        instance_list.append(number)
    
    #ASSIGNING VALUES TO PREDICTIVE FEATURES
    #Randomly assigning instances in each of the predictive features the value of 1 or 2
    #Ensuring that the MAF (minor allele frequency) of each feature is a random value between 0 and the cutoff
    #Multiplying by 2 because there are two alleles for each instance
    for e in range (0, number_of_features_in_bin):
        #Calculating the sum of instances with minor alleles
        MA_sum = round((random.uniform(0, 2*rare_variant_MAF_cutoff))*number_of_instances)
        #Between 0 and 50% of the minor allele sum will be from instances with value 2
        number_of_MA2_instances = round(0.5*(random.uniform(0, MA_sum*0.5)))
        #The remaining MA instances will have a value of 1
        number_of_MA1_instances = MA_sum - 2*number_of_MA2_instances
        MA1_instances = random.sample(instance_list, number_of_MA1_instances)
        for f in MA1_instances:
            df.at[f, predictive_features[e]] = 1
        instances_wo_MA1 = list(set(instance_list) - set(MA1_instances))
        MA2_instances = random.sample(instances_wo_MA1, number_of_MA2_instances)
        for f in MA2_instances:
            df.at[f, predictive_features[e]] = 2
    
    #ASSIGNING ENDPOINT (CLASS) VALUES 
    #Creating a list of bin values for the sum of values across predictive features in the bin
    bin_values = []
    for g in range (0, number_of_instances):
        sum_of_values = 0
        for h in predictive_features:
            sum_of_values += df.iloc[g][h]
        bin_values.append(sum_of_values)
    
    #User input for the cutoff between 0s and 1s is either mean or median. 
    if endpoint_cutoff_parameter == 'mean':
        #Finding the mean to make a cutoff for 0s and 1s in the class colum
        endpoint_cutoff = statistics.mean(bin_values)
    
        #If the sum of feature values in an instance is greater than or equal to the mean, then the endpoint will be 1
        for i in range (0, number_of_instances):
            if bin_values[i] > endpoint_cutoff:
                df.at[i, 'Class'] = 1
        
            elif bin_values[i] == endpoint_cutoff:
                df.at[i, 'Class'] = 1
            
            elif bin_values[1] < endpoint_cutoff:
                df.at[i, 'Class'] = 0
    
    elif endpoint_cutoff_parameter == 'median':
        #Finding the median to make a cutoff for 0s and 1s in the class colum
        endpoint_cutoff = statistics.median(bin_values)
    
        #If the sum of feature values in an instance is greater than or equal to the mean, then the endpoint will be 1
        for i in range (0, number_of_instances):
            if bin_values[i] > endpoint_cutoff:
                df.at[i, 'Class'] = 1
        
            elif bin_values[i] == endpoint_cutoff:
                df.at[i, 'Class'] = 1
            
            elif bin_values[1] < endpoint_cutoff:
                df.at[i, 'Class'] = 0
        
    #Applying the "noise" parameter to introduce endpoint variability
    for instance in range (0, number_of_instances):
        if endpoint_variation_probability > random.uniform(0,1):
            if df.loc[instance]['Class'] == 0:
                df.at[instance, 'Class'] = 1
            
            elif df.loc[instance]['Class'] == 1:
                df.at[instance,'Class'] = 0
    
    #ASSIGNING VALUES TO RANDOM FEATURES
    #Randomly assigning instances in each of the random features the value of 1
    #Ensuring that the MAF of each feature is a random value between 0 and the cutoff (probably 0.05)
    #Multiplying by 2 because there are two alleles for each instance
    for e in range (0, len(random_features)):
        #Calculating the sum of instances with minor alleles
        MA_sum = round((random.uniform(0, 2*rare_variant_MAF_cutoff))*number_of_instances)
        #Between 0 and 50% of the minor allele sum will be from instances with value 2
        number_of_MA2_instances = round(0.5*(random.uniform(0, MA_sum*0.5)))
        #The remaining MA instances will have a value of 1
        number_of_MA1_instances = MA_sum - 2*number_of_MA2_instances
        MA1_instances = random.sample(instance_list, number_of_MA1_instances)
        for f in MA1_instances:
            df.at[f, random_features[e]] = 1
        instances_wo_MA1 = list(set(instance_list) - set(MA1_instances))
        MA2_instances = random.sample(instances_wo_MA1, number_of_MA2_instances)
        for f in MA2_instances:
            df.at[f, random_features[e]] = 2
    
    return df, endpoint_cutoff


# In[ ]:


#Defining a function to create an artificial dataset with parameters
#There will be an epistatic relationship between a bin and a common feature
#Note: MAF (minor allele frequency) cutoff refers to the threshold separating rare variant features from common features
#Common feature genotype frequencies list should be of the form [0.25, 0.5, 0.25] (numbers should add up to 1)
#List of predictive MLGs (multi-locus genotypes) should contain any of the 9 possibilites and be of the form below:
#[AABB, AABb, AAbb, AaBB, AaBb, Aabb, aaBB, aaBb, aabb]
#Genotype Cutoff Metric can be mean or median
def RVDS_Bin_Epistatic_Interaction_with_Common_Feature(number_of_instances, number_of_rare_features, 
                                                       number_of_features_in_bin, 
                                                       rare_variant_MAF_cutoff, common_feature_genotype_frequencies_list, 
                                                       genotype_cutoff_metric, 
                                                       endpoint_variation_probability,
                                                       list_of_MLGs_predicting_disease, print_summary):
    
    #Creating an empty dataframe to use as a starting point for the eventual feature matrix
    #Adding two to number of rare features to give space for the class column and give space for the common feature
    df = pd.DataFrame(np.zeros((number_of_instances, number_of_rare_features+2)))
    
    #Creating a list of features
    feature_list = []
    
    #Creating a list of features in the bin that interacts epistatically with the common feature
    predictive_features = []
    for a in range (0, number_of_features_in_bin):
        predictive_features.append("P_" + str(a+1))
    
    for b in range (0, len(predictive_features)):
        feature_list.append(predictive_features[b])
            
    #Creating a list of randomly created features
    random_features = []
    for c in range (0, number_of_rare_features - number_of_features_in_bin):
        random_features.append("R_" + str(c+1))
    
    for d in range (0, len(random_features)):
        feature_list.append(random_features[d])
    
    #Adding the common feature to the feature list
    feature_list.append('Common_Feature')
    
    #Adding the features and the class/endpoint 
    features_and_class = feature_list
    features_and_class.append('Class')
    df.columns = features_and_class
    
    #Creating a list of numbers with the amount of numbers equal to the number of instances
    #This will be used when assigning values to the values of features that are in the bin
    instance_list = []
    for number in range (0, number_of_instances):
        instance_list.append(number)
    
    #RANDOMLY ASSIGNING VALUES TO THE FEATURES IN THE BIN
    #Randomly assigning instances in each of the predictive features the value of 1 or 2
    #Ensuring that the MAF (minor allele frequency) of each feature is a random value between 0 and the cutoff
    for e in range (0, number_of_features_in_bin):
        #Calculating the sum of instances with minor alleles
        MA_sum = round((random.uniform(0, 2*rare_variant_MAF_cutoff))*number_of_instances)
        #Between 0 and 50% of the minor allele sum will be from instances with value 2
        number_of_MA2_instances = round(0.5*(random.uniform(0, MA_sum*0.5)))
        #The remaining MA instances will have a value of 1
        number_of_MA1_instances = MA_sum - 2*number_of_MA2_instances
        MA1_instances = random.sample(instance_list, number_of_MA1_instances)
        for f in MA1_instances:
            df.at[f, predictive_features[e]] = 1
        instances_wo_MA1 = list(set(instance_list) - set(MA1_instances))
        MA2_instances = random.sample(instances_wo_MA1, number_of_MA2_instances)
        for f in MA2_instances:
            df.at[f, predictive_features[e]] = 2
    
    #DETERMING GENOTYPE CUTOFF FOR THE BIN (AA vs. Aa vs. aa) 
    #Creating a list of bin values for the sum of values across predictive features in the bin
    bin_values = []
    for g in range (0, number_of_instances):
        sum_of_values = 0
        for h in predictive_features:
            sum_of_values += df.iloc[g][h]
        bin_values.append(sum_of_values)
    
    #Bin values that are 0 will be AA
    
    nonzero_bin_values = []
    #Creating a list of nonzero bin values, the mean of median of this will determine the cutoff between Aa and aa
    for i in range (0, len(bin_values)):
        if bin_values[i] != 0:
            nonzero_bin_values.append(bin_values[i])
    
    if genotype_cutoff_metric == 'mean':
        Aa_aa_cutoff = statistics.mean(nonzero_bin_values)
        
    elif genotype_cutoff_metric == 'median':
        Aa_aa_cutoff = statistics.median(nonzero_bin_values)
    
    #Creating a list for each of the bin genotypes
    bin_AA_genotype_list = []
    bin_Aa_genotype_list = []
    bin_aa_genotype_list = []
    
    for g in range (0, number_of_instances):
        if bin_values[g] == 0:
            bin_AA_genotype_list.append(g)
        
        elif bin_values[g] > 0  and bin_values[g] < Aa_aa_cutoff:
            bin_Aa_genotype_list.append(g)
            
        elif bin_values[g] == Aa_aa_cutoff or bin_values[g] > Aa_aa_cutoff:
            bin_aa_genotype_list.append(g)
    
    #ASSIGNING VALUES FOR THE COMMON FEATURE 
    #Based on the given allele frequencies the user inputs, randomly choosing instances for each genotype (BB, Bb, bb)
    #of the common feature
    
    instances_left = instance_list.copy()
    common_feature_BB_genotype_list = random.sample(instance_list, (round(number_of_instances * float(common_feature_genotype_frequencies_list[0]))))
    for instance in common_feature_BB_genotype_list:
        df.at[instance, 'Common_Feature'] = 0
        instances_left.remove(instance)
    
    common_feature_Bb_genotype_list = random.sample(instances_left, (round(number_of_instances * float(common_feature_genotype_frequencies_list[1]))))
    for instance in common_feature_Bb_genotype_list:
        df.at[instance, 'Common_Feature'] = 1
        instances_left.remove(instance)
    
    common_feature_bb_genotype_list = random.sample(instances_left, (round(number_of_instances * float(common_feature_genotype_frequencies_list[2]))))
    for instance in common_feature_bb_genotype_list:
        df.at[instance, 'Common_Feature'] = 2
    
    #ASSIGNING CLASS/ENDPOINT VALUES 
    
    #First assigning a value of 0 for all the classes (this will be overwrote later)
    for instance in instance_list:
        df.at[instance, 'Class'] = 0
        
    #Assigning a value of 1 for each instance that matches the MLG (multi-locus genotype) that the user specifies should
    #result in a diseased state
    disease_instances = []
    for instance in instance_list:
        if 'AABB' in list_of_MLGs_predicting_disease:
            if instance in bin_AA_genotype_list and instance in common_feature_BB_genotype_list:
                df.at[instance, 'Class'] = 1
                disease_instances.append(instance)
        
        if 'AABb' in list_of_MLGs_predicting_disease:
            if instance in bin_AA_genotype_list and instance in common_feature_Bb_genotype_list:
                df.at[instance, 'Class'] = 1
                disease_instances.append(instance)
                
        if 'AAbb' in list_of_MLGs_predicting_disease:
            if instance in bin_AA_genotype_list and instance in common_feature_bb_genotype_list:
                df.at[instance, 'Class'] = 1
                disease_instances.append(instance)
        
        if 'AaBB' in list_of_MLGs_predicting_disease:
            if instance in bin_Aa_genotype_list and instance in common_feature_BB_genotype_list:
                df.at[instance, 'Class'] = 1
                disease_instances.append(instance)
                
        if 'AaBb' in list_of_MLGs_predicting_disease:
            if instance in bin_Aa_genotype_list and instance in common_feature_Bb_genotype_list:
                df.at[instance, 'Class'] = 1
                disease_instances.append(instance)
        
        if 'Aabb' in list_of_MLGs_predicting_disease:
            if instance in bin_Aa_genotype_list and instance in common_feature_bb_genotype_list:
                df.at[instance, 'Class'] = 1
                disease_instances.append(instance)

        if 'aaBB' in list_of_MLGs_predicting_disease:
            if instance in bin_aa_genotype_list and instance in common_feature_BB_genotype_list:
                df.at[instance, 'Class'] = 1
                disease_instances.append(instance)
        
        if 'aaBb' in list_of_MLGs_predicting_disease:
            if instance in bin_aa_genotype_list and instance in common_feature_Bb_genotype_list:
                df.at[instance, 'Class'] = 1
                disease_instances.append(instance)
        
        if 'aabb' in list_of_MLGs_predicting_disease:
            if instance in bin_aa_genotype_list and instance in common_feature_bb_genotype_list:
                df.at[instance, 'Class'] = 1
                disease_instances.append(instance)
        
    #Applying the "noise" parameter to introduce endpoint variability
    for instance in range (0, number_of_instances):
        if endpoint_variation_probability > random.uniform(0,1):
            if df.loc[instance]['Class'] == 0:
                df.at[instance, 'Class'] = 1
            
            elif df.loc[instance]['Class'] == 1:
                df.at[instance,'Class'] = 0
    
    #ASSIGNING VALUES TO RANDOM FEATURES
    #Randomly assigning instances in each of the random features the value of 1
    #Ensuring that the MAF of each feature is a random value between 0 and the cutoff (probably 0.05)
    for e in range (0, len(random_features)):
        #Calculating the sum of instances with minor alleles
        MA_sum = round((random.uniform(0, 2*rare_variant_MAF_cutoff))*number_of_instances)
        #Between 0 and 50% of the minor allele sum will be from instances with value 2
        number_of_MA2_instances = round(0.5*(random.uniform(0, MA_sum*0.5)))
        #The remaining MA instances will have a value of 1
        number_of_MA1_instances = MA_sum - 2*number_of_MA2_instances
        MA1_instances = random.sample(instance_list, number_of_MA1_instances)
        for f in MA1_instances:
            df.at[f, random_features[e]] = 1
        instances_wo_MA1 = list(set(instance_list) - set(MA1_instances))
        MA2_instances = random.sample(instances_wo_MA1, number_of_MA2_instances)
        for f in MA2_instances:
            df.at[f, random_features[e]] = 2
    
    if print_summary == True:
        #PRINTING INFORMATION ABOUT THE DATASET THAT CAN BE INPUTTED IN A PENETRANCE TABLE
        print("Probability of Disease (Class = 1) for Multi-Locus Genotypes:")
        MLG_list = ['AABB', 'AABb', 'AAbb', 'AaBB', 'AaBb', 'Aabb', 'aaBB', 'aaBb', 'aabb']
        MLG_penetrance_dict = {}
        for MLG in MLG_list:
            if MLG in list_of_MLGs_predicting_disease:
                print(MLG + ': 1')
                MLG_penetrance_dict[MLG] = 1
            elif MLG not in list_of_MLGs_predicting_disease:
                print(MLG + ': 0')
                MLG_penetrance_dict[MLG] = 0
        print("---")
        print('Marginal Penetrance for Genotypes:')
        AA_penetrance = ((len(common_feature_BB_genotype_list)*MLG_penetrance_dict['AABB']/number_of_instances) + 
                          (len(common_feature_Bb_genotype_list)*MLG_penetrance_dict['AABb']/number_of_instances) + 
                          (len(common_feature_bb_genotype_list)*MLG_penetrance_dict['AAbb']/number_of_instances))
        print("AA: " + str(AA_penetrance))
    
        Aa_penetrance = ((len(common_feature_BB_genotype_list)*MLG_penetrance_dict['AaBB']/number_of_instances) + 
                          (len(common_feature_Bb_genotype_list)*MLG_penetrance_dict['AaBb']/number_of_instances) + 
                          (len(common_feature_bb_genotype_list)*MLG_penetrance_dict['Aabb']/number_of_instances))
        print("Aa: " + str(Aa_penetrance))
          
        aa_penetrance = ((len(common_feature_BB_genotype_list)*MLG_penetrance_dict['aaBB']/number_of_instances) + 
                          (len(common_feature_Bb_genotype_list)*MLG_penetrance_dict['aaBb']/number_of_instances) + 
                          (len(common_feature_bb_genotype_list)*MLG_penetrance_dict['aabb']/number_of_instances))
        print("aa: " + str(aa_penetrance))
    
        BB_penetrance = ((len(bin_AA_genotype_list)*MLG_penetrance_dict['AABB']/number_of_instances) + 
                          (len(bin_Aa_genotype_list)*MLG_penetrance_dict['AaBB']/number_of_instances) + 
                          (len(bin_aa_genotype_list)*MLG_penetrance_dict['aaBB']/number_of_instances))
        print("BB: " + str(BB_penetrance))
    
        Bb_penetrance = ((len(bin_AA_genotype_list)*MLG_penetrance_dict['AABb']/number_of_instances) + 
                          (len(bin_Aa_genotype_list)*MLG_penetrance_dict['AaBb']/number_of_instances) + 
                          (len(bin_aa_genotype_list)*MLG_penetrance_dict['aaBb']/number_of_instances))
        print("Bb: " + str(Bb_penetrance))
    
        bb_penetrance = ((len(bin_AA_genotype_list)*MLG_penetrance_dict['AAbb']/number_of_instances) + 
                          (len(bin_Aa_genotype_list)*MLG_penetrance_dict['Aabb']/number_of_instances) + 
                          (len(bin_aa_genotype_list)*MLG_penetrance_dict['aabb']/number_of_instances))
        print("bb: " + str(bb_penetrance))
        print('---')
        print("Population Prevalence of Disease (K): " + str(len(disease_instances)/number_of_instances))
    
    return df, Aa_aa_cutoff





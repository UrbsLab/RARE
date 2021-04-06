#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


#Step 1: Initialize Population of Candidate Bins

#Random initialization of candidate bins, which are groupings of multiple features
#The value of each bin/feature is the sum of values for each amino acid in the bin (0 for match, 1 for different)
#Adding a function that can be an option to automatically separate rare and common variables


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


#Step 2: Genetic Algorithm with Relief-based Feature Scoring (repeated for a given number of iterations) 


# In[ ]:


#Step 2a: Relief-based Feature Importance Scoring and Bin Deletion

#Use MultiSURF to calculate the feature importance of each candidate bin 
#If the population size > the set max population size then bins will be probabilistically deleted based on fitness


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


#Step 2b: Genetic Algorithm 

#Parent bins are probabilistically selected based on fitness (score calculated in Step 2a)
#New offspring bins will be created through cross over and mutation and are added to the next generation's population
#Based on the value of the elitism parameter, a number of high scoring parent bins will be preserved for the next gen


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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
        elif scoring_method == 'Univariate':
            amino_acid_bin_scores = Chi_Square_Feature_Importance(bin_feature_matrix, 'Class', amino_acid_bins)
            
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
    elif scoring_method == 'Univariate':
        amino_acid_bin_scores = Chi_Square_Feature_Importance(bin_feature_matrix, 'Class', amino_acid_bins)
    
    return bin_feature_matrix, amino_acid_bins, amino_acid_bin_scores, MAF_0_features


# In[ ]:


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
                        


# In[ ]:


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
        
        elif scoring_method == 'Univariate':
            amino_acid_bin_scores = Chi_Square_Feature_Importance(bin_feature_matrix, 'Class', amino_acid_bins)
        
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


# In[ ]:


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


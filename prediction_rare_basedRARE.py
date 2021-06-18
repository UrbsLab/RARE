#!/usr/bin/env python
# coding: utf-8

# In[1]:


#This file contains all the code defining the functions for RARE and its Rare Variant Data Simulators
#No function is ever called in this file, please see the "RARE Experiment 1" file for an example of how to run RARE on the Rare Variant Data Simulators
#Readers please note: amino acid refers to a rare variant feature, while amino acid bin is the equivalent of a bin of rare variant features

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


#Step 1: Initialize Population of Candidate Bins

#Random initialization of candidate bins, which are groupings of multiple features
#The value of each bin/feature is the sum of values for each feature in the bin 
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





# In[ ]:




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
    
   #For each feature group/bin, the values of the features in the bin will be summed to create a value for the bin
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





#Defining a function to score bins based on predictive metrics (accuracy, precision, recall/sensitivity, f1, or specificity)
def PredictionBased_Bin_Scoring(bin_feature_matrix, label_name, amino_acid_bins, prediction_metric, adjust_for_proportion, proportion_cutoff):
    
    #Creating a list of 0s, where the number of 0s is the number of instances in the original feature matrix
    zero_list = []
    for a in range (0, len(bin_feature_matrix.index)):
        zero_list.append(0)
        
    #Creating a dict to store bin name and the optimal score for that bin
    bin_scores = {}
    
    bin_name_list = amino_acid_bins.keys()
    
    #Running a loop for every bin to find the score for that bin
    for bin_name in bin_name_list:
        bin_value_list = bin_feature_matrix[bin_name].tolist()
        class_value_list = bin_feature_matrix[label_name].tolist()
        
        max_bin_value = max(bin_value_list)
        min_bin_value = min(bin_value_list)
        
        #Creating a dictionary to store the cutoff and the corresponding bin value for all possible cutoffs
        cutoff_score_dict = {}
        
        #Running a loop for each possible bin value cutoff
        for i in range(0, int(max_bin_value)+1):
            cutoff = i
            above_cutoff_instances = []
            at_under_cutoff_instances = []
                
            #Putting each instance to the category it belongs in (above cutoff or below cutoff)
            for k in range (0, len(zero_list)):
                if bin_value_list[k] > cutoff:
                    above_cutoff_instances.append(k)
                elif bin_value_list[k] == cutoff:
                    at_under_cutoff_instances.append(k)
                elif bin_value_list[k] < cutoff:
                    at_under_cutoff_instances.append(k)
                        

            predicted_positives = above_cutoff_instances.copy()
            proportion = len(predicted_positives)/len(zero_list)
            predicted_negatives = at_under_cutoff_instances.copy()

                
            actual_positives = []
            actual_negatives = []
            for l in range (0, len(zero_list)):
                if class_value_list[l] == 0:
                    actual_negatives.append(l)
                elif class_value_list[l] == 1:
                    actual_positives.append(l)
                    
            
            
                
            #Finding the number of true positives, true negatives, false positives, and false negatives
            true_positives = list(set(actual_positives) & set(predicted_positives))
            true_negatives = list(set(actual_negatives) & set(predicted_negatives))
            false_positives = list(set(actual_negatives) & set(predicted_positives))
            false_negatives = list(set(actual_positives) & set(predicted_negatives))
                
            #Calculating accuracy, precision, recall, specificity, and f1 based on the tp, tn, fp, and fn
            accuracy = (len(true_positives) + len(true_negatives))/((len(true_positives) + len(true_negatives) + len(false_positives) + len(false_negatives)))
            if (len(true_positives)+len(false_positives)) > 0:
                precision = (len(true_positives))/(len(true_positives) + len(false_positives))
            else:
                precision = 0
            if (len(true_positives) + len(false_negatives)) > 0:
                recall = (len(true_positives))/(len(true_positives) + len(false_negatives))
            else:
                recall = 0
            if (len(true_negatives) + len(false_positives)) > 0:
                specificity = (len(true_negatives))/(len(true_negatives) + len(false_positives))
            else:
                specificity = 0
            if (precision + recall) > 0:
                f1 = (2*precision*recall)/(precision + recall)
            else: 
                f1 = 0
            if recall > 0 and specificity > 0:
                balanced_accuracy = (recall + specificity)/2
            else:
                balanced_accuracy = 0
                    
                
            #Choosing the score based on the user specified metric
            if prediction_metric == 'accuracy':
                score = accuracy
            elif prediction_metric == 'precision':
                score = precision
            elif prediction_metric == 'recall':
                score = recall
            elif prediction_metric == 'specificity':
                score = specificity
            elif prediction_metric == 'f1':
                score = f1
            elif prediction_metric == 'balanced_accuracy':
                score = balanced_accuracy

            #If the the user wants to apply a cutoff for the proportion of patients that must be above cutoff and that is not the case, then score is changed to 0
            if adjust_for_proportion == True:
                if proportion < proportion_cutoff:
                    score = 0

            #Adding the score to the dictionary
            cutoff_score_dict[cutoff] = score
            
        #Finding the best score and the corresponding cutoff
        v=list(cutoff_score_dict.values())
        k=list(cutoff_score_dict.keys())
        bestkey = k[v.index(max(v))]
            
        #Adding that to the overall dictionary with the best score for each bin
        bin_scores[bin_name] = cutoff_score_dict[bestkey]
        
    return bin_scores
            
        
        

          


# In[3]:


def PredictionBased_Bin_Scoring_Output(bin_feature_matrix, label_name, amino_acid_bins, prediction_metric, adjust_for_proportion, proportion_cutoff):
    
    zero_list = []
    for a in range (0, len(bin_feature_matrix.index)):
        zero_list.append(0)
    
    #Creating dicts to store bin name; optimal cutoff, corresponding score, and proportion for that bin
    bin_scores = {}
    bin_optimal_cutoffs = {}
    bin_proportions = {}
    
    bin_name_list = amino_acid_bins.keys()
    
    #Running a loop for every bin to find the score for that bin
    for bin_name in bin_name_list:
        bin_value_list = bin_feature_matrix[bin_name].tolist()
        class_value_list = bin_feature_matrix[label_name].tolist()
        
        max_bin_value = max(bin_value_list)
        min_bin_value = min(bin_value_list)
        
        #Creating a dictionary to store the cutoff and the corresponding bin score
        cutoff_score_dict = {}

        #Creating a dictionary to store the proportion and the corresponding bin score
        proportion_score_dict = {}

        
        #Running a loop for each possible bin value cutoff
        for i in range(0, int(max_bin_value)+1):
            cutoff = i
            above_cutoff_instances = []
            at_under_cutoff_instances = []
                
            #Putting each instance to the category it belongs in (above cutoff or below cutoff)
            for k in range (0, len(zero_list)):
                if bin_value_list[k] > cutoff:
                    above_cutoff_instances.append(k)
                elif bin_value_list[k] == cutoff:
                    at_under_cutoff_instances.append(k)
                elif bin_value_list[k] < cutoff:
                    at_under_cutoff_instances.append(k)
                        
            predicted_positives = above_cutoff_instances.copy()
            proportion = len(predicted_positives)/len(zero_list)
            predicted_negatives = at_under_cutoff_instances.copy()
                
            actual_positives = []
            actual_negatives = []
            for l in range (0, len(zero_list)):
                if class_value_list[l] == 0:
                    actual_negatives.append(l)
                elif class_value_list[l] == 1:
                    actual_positives.append(l)
                
            #Finding the number of true positives, true negatives, false positives, and false negatives
            true_positives = list(set(actual_positives) & set(predicted_positives))
            true_negatives = list(set(actual_negatives) & set(predicted_negatives))
            false_positives = list(set(actual_negatives) & set(predicted_positives))
            false_negatives = list(set(actual_positives) & set(predicted_negatives))
                
            #Calculating accuracy, precision, recall, specificity, and f1 based on the tp, tn, fp, and fn
            accuracy = (len(true_positives) + len(true_negatives))/((len(true_positives) + len(true_negatives) + len(false_positives) + len(false_negatives)))
            if (len(true_positives)+len(false_positives)) > 0:
                precision = (len(true_positives))/(len(true_positives) + len(false_positives))
            else:
                precision = 0
            if (len(true_positives) + len(false_negatives)) > 0:
                recall = (len(true_positives))/(len(true_positives) + len(false_negatives))
            else:
                recall = 0
            if (len(true_negatives) + len(false_positives)) > 0:
                specificity = (len(true_negatives))/(len(true_negatives) + len(false_positives))
            else:
                specificity = 0
            if (precision + recall) > 0:
                f1 = (2*precision*recall)/(precision + recall)
            else: 
                f1 = 0
            if recall > 0 and specificity > 0:
                balanced_accuracy = (recall + specificity)/2
            else:
                balanced_accuracy = 0            
                    
                
            #Choosing the score based on the user specified metric
            if prediction_metric == 'accuracy':
                score = accuracy
            elif prediction_metric == 'precision':
                score = precision
            elif prediction_metric == 'recall':
                score = recall
            elif prediction_metric == 'specificity':
                score = specificity
            elif prediction_metric == 'f1':
                score = f1
            elif prediction_metric == 'balanced_accuracy':
                score = balanced_accuracy

            #If the the user wants to apply a cutoff for the proportion of patients that must be above cutoff and that is not the case, then score is changed to 0
            if adjust_for_proportion == True:
                if proportion < proportion_cutoff:
                    score = -1
                    
            #Adding the score to the dictionary
            cutoff_score_dict[cutoff] = score
            proportion_score_dict[proportion] = score

            

            
        #Finding the best score and the corresponding cutoff
        v=list(cutoff_score_dict.values())
        k=list(cutoff_score_dict.keys())
        bestkey = k[v.index(max(v))]

        v=list(proportion_score_dict.values())
        k=list(proportion_score_dict.keys())
        bestprop = k[v.index(max(v))]
            
        #Adding that to the overall dictionary with the best score and cutoff for each bin
        bin_scores[bin_name] = [cutoff_score_dict[bestkey]]
        bin_optimal_cutoffs[bin_name] = bestkey
        bin_proportions[bin_name] = bestprop
        
    return bin_scores, bin_optimal_cutoffs, bin_proportions







def RiskBased_Bin_Scoring_Output(bin_feature_matrix, label_name, amino_acid_bins, risk_metric, adjust_for_proportion, proportion_cutoff):
    
    zero_list = []
    for a in range (0, len(bin_feature_matrix.index)):
        zero_list.append(0)
    
    #Creating dicts to store bin name; optimal cutoff, corresponding score, and proportion for that bin
    bin_scores = {}
    bin_optimal_cutoffs = {}
    bin_proportions = {}
    
    bin_name_list = amino_acid_bins.keys()
    
    #Running a loop for every bin to find the score for that bin
    for bin_name in bin_name_list:
        bin_value_list = bin_feature_matrix[bin_name].tolist()
        class_value_list = bin_feature_matrix[label_name].tolist()
        
        max_bin_value = max(bin_value_list)
        min_bin_value = min(bin_value_list)
        
        #Creating a dictionary to store the cutoff and the corresponding bin score
        cutoff_score_dict = {}

        #Creating a dictionary to store the proportion and the corresponding bin score
        proportion_score_dict = {}

        
        #Running a loop for each possible bin value cutoff
        for i in range(0, int(max_bin_value)+1):
            cutoff = i
            above_cutoff_instances = []
            at_under_cutoff_instances = []
                
            #Putting each instance to the category it belongs in (above cutoff or below cutoff)
            for k in range (0, len(zero_list)):
                if bin_value_list[k] > cutoff:
                    above_cutoff_instances.append(k)
                elif bin_value_list[k] == cutoff:
                    at_under_cutoff_instances.append(k)
                elif bin_value_list[k] < cutoff:
                    at_under_cutoff_instances.append(k)
                        
            predicted_positives = above_cutoff_instances.copy()
            proportion = len(predicted_positives)/len(zero_list)
            predicted_negatives = at_under_cutoff_instances.copy()
                
            actual_positives = []
            actual_negatives = []
            for l in range (0, len(zero_list)):
                if class_value_list[l] == 0:
                    actual_negatives.append(l)
                elif class_value_list[l] == 1:
                    actual_positives.append(l)
                
                
            #Calculating risk, relative risk, and odds ratio based on the tp, tn, fp, and fn
            if (len(true_positives)+len(false_positives)) > 0:
                risk = (len(true_positives))/(len(true_positives) + len(false_positives))
            else:
                risk = 0
            
            
            if (len(true_positives) + len(false_positives)) > 0 and (len(false_negatives) + len(true_negatives)) > 0 and (len(false_negatives)/(len(false_negatives) + len(true_negatives))) > 0:
                relative_risk = ((len(true_positives)/(len(true_positives) + len(false_positives))))/(len(false_negatives)/(len(false_negatives) + len(true_negatives)))
            else:
                relative_risk = 1
            
            if len(false_positives) > 0 and len(true_negatives) > 0 and (len(false_negatives)/len(true_negatives)) > 0:
                odds_ratio = (len(true_positives)/len(false_positives))/(len(false_negatives)/len(true_negatives))
            else:
                odds_ratio = 1
           
                
            #Choosing the score based on the user specified metric
            if risk_metric == 'risk':
                score = risk
            elif risk_metric == 'relative_risk':
                score = relative_risk
            elif risk_metric == 'odds_ratio':
                score = odds_ratio

            #If the the user wants to apply a cutoff for the proportion of patients that must be above cutoff and that is not the case, then score is changed to 0
            if adjust_for_proportion == True:
                if proportion < proportion_cutoff:
                    score = -1
                    
            #Adding the score to the dictionary
            cutoff_score_dict[cutoff] = score
            proportion_score_dict[proportion] = score

            

            
        #Finding the best score and the corresponding cutoff
        v=list(cutoff_score_dict.values())
        k=list(cutoff_score_dict.keys())
        bestkey = k[v.index(max(v))]

        v=list(proportion_score_dict.values())
        k=list(proportion_score_dict.keys())
        bestprop = k[v.index(max(v))]
            
        #Adding that to the overall dictionary with the best score and cutoff for each bin
        bin_scores[bin_name] = [cutoff_score_dict[bestkey]]
        bin_optimal_cutoffs[bin_name] = bestkey
        bin_proportions[bin_name] = bestprop
        
    return bin_scores, bin_optimal_cutoffs, bin_proportions

            
        
        


# In[5]:


def RiskBased_Bin_Scoring(bin_feature_matrix, label_name, amino_acid_bins, risk_metric, adjust_for_proportion, proportion_cutoff):
    
    zero_list = []
    for a in range (0, len(bin_feature_matrix.index)):
        zero_list.append(0)
    
    #Creating a dict to store bin name and the optimal score for that bin
    bin_scores = {}
    
    bin_name_list = amino_acid_bins.keys()
    
    #Running a loop for every bin to find the score for that bin
    for bin_name in bin_name_list:
        bin_value_list = bin_feature_matrix[bin_name].tolist()
        class_value_list = bin_feature_matrix[label_name].tolist()
        
        max_bin_value = max(bin_value_list)
        min_bin_value = min(bin_value_list)
        
        #Creating a dictionary to store the cutoff and the corresponding bin value for all possible cutoffs
        cutoff_score_dict = {}
        
        #Running a loop for each possible bin value cutoff
        for i in range(0, int(max_bin_value)+1):
            cutoff = i
            above_cutoff_instances = []
            at_under_cutoff_instances = []
                
            #Putting each instance to the category it belongs in (above cutoff or below cutoff)
            for k in range (0, len(zero_list)):
                if bin_value_list[k] > cutoff:
                    above_cutoff_instances.append(k)
                elif bin_value_list[k] == cutoff:
                    at_under_cutoff_instances.append(k)
                elif bin_value_list[k] < cutoff:
                    at_under_cutoff_instances.append(k)
                        
            predicted_positives = above_cutoff_instances.copy()
            proportion = len(predicted_positives)/len(zero_list)
            predicted_negatives = at_under_cutoff_instances.copy()
        
                
            actual_positives = []
            actual_negatives = []
            for l in range (0, len(zero_list)):
                if class_value_list[l] == 0:
                    actual_negatives.append(l)
                elif class_value_list[l] == 1:
                    actual_positives.append(l)
                
            #Finding the number of true positives, true negatives, false positives, and false negatives
            true_positives = list(set(actual_positives) & set(predicted_positives))
            true_negatives = list(set(actual_negatives) & set(predicted_negatives))
            false_positives = list(set(actual_negatives) & set(predicted_positives))
            false_negatives = list(set(actual_positives) & set(predicted_negatives))
                
            #Calculating risk, relative risk, and odds ratio based on the tp, tn, fp, and fn
            if (len(true_positives)+len(false_positives)) > 0:
                risk = (len(true_positives))/(len(true_positives) + len(false_positives))
            else:
                risk = 0
            
            
            if (len(true_positives) + len(false_positives)) > 0 and (len(false_negatives) + len(true_negatives)) > 0 and (len(false_negatives)/(len(false_negatives) + len(true_negatives))) > 0:
                relative_risk = ((len(true_positives)/(len(true_positives) + len(false_positives))))/(len(false_negatives)/(len(false_negatives) + len(true_negatives)))
            else:
                relative_risk = 1
            
            if len(false_positives) > 0 and len(true_negatives) > 0 and (len(false_negatives)/len(true_negatives)) > 0:
                odds_ratio = (len(true_positives)/len(false_positives))/(len(false_negatives)/len(true_negatives))
            else:
                odds_ratio = 1
           
                
            #Choosing the score based on the user specified metric
            if risk_metric == 'risk':
                score = risk
            elif risk_metric == 'relative_risk':
                score = relative_risk
            elif risk_metric == 'odds_ratio':
                score = odds_ratio

            #If the the user wants to apply a cutoff for the proportion of patients that must be above cutoff and that is not the case, then score is changed to 0
            if adjust_for_proportion == True:
                if proportion < proportion_cutoff:
                    score = -1
                    
            #Adding the score to the dictionary
            cutoff_score_dict[cutoff] = score
            
        #Finding the best score and the corresponding cutoff
        v=list(cutoff_score_dict.values())
        k=list(cutoff_score_dict.keys())
        bestkey = k[v.index(max(v))]
            
        #Adding that to the overall dictionary with the best score for each bin
        bin_scores[bin_name] = cutoff_score_dict[bestkey]
        
    return bin_scores
            


# In[6]:


def predictRAFE (given_starting_point, amino_acid_start_point, amino_acid_bins_start_point, iterations, original_feature_matrix, 
          label_name, set_number_of_bins, min_features_per_group, max_number_of_groups_with_feature, prediction_metric, adjust_for_proportion, proportion_cutoff,
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
        amino_acid_bin_scores = PredictionBased_Bin_Scoring(bin_feature_matrix, label_name, amino_acid_bins, prediction_metric, adjust_for_proportion, proportion_cutoff)
        
            
        #Step 2b: Genetic Algorithm 
        #Creating the offspring bins through crossover and mutation
        offspring_bins = Crossover_and_Mutation(set_number_of_bins, elitism_parameter, amino_acids, amino_acid_bins, amino_acid_bin_scores,
                                                crossover_probability, mutation_probability)
        
        #Creating the new generation by preserving some elites and adding the offspring
        feature_bin_list = Create_Next_Generation(amino_acid_bins, amino_acid_bin_scores, set_number_of_bins, 
                                                  elitism_parameter, offspring_bins)
        
        bin_feature_matrix, amino_acid_bins = Regroup_Feature_Matrix(amino_acids, original_feature_matrix, label_name, feature_bin_list)
    
    #Creating the final bin scores
    amino_acid_bin_scores, bin_optimal_cutoffs, bin_proportions = PredictionBased_Bin_Scoring_Output(bin_feature_matrix, label_name, amino_acid_bins, prediction_metric, adjust_for_proportion, proportion_cutoff)

    
    return bin_feature_matrix, amino_acid_bins, amino_acid_bin_scores, bin_optimal_cutoffs, bin_proportions, MAF_0_features



# In[ ]:


#Defining a function to present the top bins 
def prTop_Bins_Summary(original_feature_matrix, label_name, bin_feature_matrix, bins, bin_scores, bin_optimal_cutoffs, bin_proportions, number_of_top_bins, scoring_method):
    
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

    #Printing the scoring method
    print(scoring_method)
    
    for i in range (0, number_of_top_bins):
        #Printing the bin Name
        print ("Bin Rank " + str(i+1) + ": " + sorted_bin_list[i])
        #Printing the bin's score, chi-square value, and p-value
        print ("Score: " + str(sorted_bin_feature_importance_values[i]) + "; chi-square value: " + str(Bin_Stats[sorted_bin_list[i]][0]) + "; p-value: " + str(Bin_Stats[sorted_bin_list[i]][1]))
        #Printing the bin's optimal cutoff and proportion of instances with bin value above the cutoff
        print ("Optimal Bin Cutoff: " + str(bin_optimal_cutoffs[sorted_bin_list[i]]) + "; Proportion of Instances Above Cutoff: " + str(bin_proportions[sorted_bin_list[i]]))
        #Printing each of the features in the bin and also printing the univariate stats of that feature
        for j in range (0, len(bins[sorted_bin_list[i]])):
            print ("Feature Name: " + bins[sorted_bin_list[i]][j] + "; chi-square value: " + str(Univariate_Feature_Stats[bins[sorted_bin_list[i]][j]][0]) + "; p-value: " + str(Univariate_Feature_Stats[bins[sorted_bin_list[i]][j]][1]))
        print ('---------------------------')                                              
                        


# In[ ]:


# In[7]:


def riskRAFE (given_starting_point, amino_acid_start_point, amino_acid_bins_start_point, iterations, original_feature_matrix, 
          label_name, set_number_of_bins, min_features_per_group, max_number_of_groups_with_feature, risk_metric, adjust_for_proportion, proportion_cutoff,
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
        amino_acid_bin_scores = RiskBased_Bin_Scoring(bin_feature_matrix, label_name, amino_acid_bins, risk_metric, adjust_for_proportion, proportion_cutoff)
        
            
        #Step 2b: Genetic Algorithm 
        #Creating the offspring bins through crossover and mutation
        offspring_bins = Crossover_and_Mutation(set_number_of_bins, elitism_parameter, amino_acids, amino_acid_bins, amino_acid_bin_scores,
                                                crossover_probability, mutation_probability)
        
        #Creating the new generation by preserving some elites and adding the offspring
        feature_bin_list = Create_Next_Generation(amino_acid_bins, amino_acid_bin_scores, set_number_of_bins, 
                                                  elitism_parameter, offspring_bins)
        
        bin_feature_matrix, amino_acid_bins = Regroup_Feature_Matrix(amino_acids, original_feature_matrix, label_name, feature_bin_list)
    
    #Creating the final bin scores
    amino_acid_bin_scores, bin_optimal_cutoffs, bin_proportions = RiskBased_Bin_Scoring_Output(bin_feature_matrix, label_name, amino_acid_bins, risk_metric, adjust_for_proportion, proportion_cutoff)

    
    return bin_feature_matrix, amino_acid_bins, amino_acid_bin_scores, bin_optimal_cutoffs, bin_proportions, MAF_0_features


# In[13]:

#Code running example
#df = pd.read_csv('/home/satvikd/Datasets/srtrdata.csv')
#bin_feature_matrix, amino_acid_bins, amino_acid_bin_scores, bin_optimal_cutoffs, bin_proportions, MAF_0_features = predictRAFE(False, None, None, 1000, df, 'Class', 100, 5, 10, 'accuracy', True, 0.1, 0.8, 0.1, 0.3)
#prTop_Bins_Summary(df, 'Class', bin_feature_matrix, amino_acid_bins, amino_acid_bin_scores, bin_optimal_cutoffs, bin_proportions, 10, 'accuracy')


# In[14]:



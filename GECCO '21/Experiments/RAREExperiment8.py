#Code for running Experiment 8
#Must be run after running the code in "RARE_Methods"

# In[ ]:


#Experiment 8: 1 Bin Interacting Epistatically with a Common Feature, 0 Noise, Univariate Scoring, No Expert Knowledge
for replicate in range (0, 1):
    print('Experiment 8')

    #Creating the simulated dataset
    sim_epi_data, cutoff = RVDS_Bin_Epistatic_Interaction_with_Common_Feature(1000, 15, 5, 0.05, [0.25, 0.5, 0.25], 'mean', 0, ['AAbb', 'AABB', 'AaBb', 'aaBB', 'aabb'], True)
        
    #Running RARE (univariate scoring) on the dataset
    bin_feature_matrix, common_features_and_bins_matrix, amino_acid_bins, amino_acid_bin_scores, rare_feature_MAF_dict, common_feature_MAF_dict, rare_feature_df, common_feature_df, MAF_0_features = RARE(False, None, None, 3000, sim_epi_data, 'Class', 0.05, 50, 3, 35, 'Univariate', True, True, 500, 0.8, 0.1, 0.4)    
    
    #Summary of the best bins
    Top_Rare_Variant_Bins_Summary(rare_feature_df, 'Class', amino_acid_bins, amino_acid_bin_scores, rare_feature_MAF_dict, 10) 
      
    #Inputting the correct answer to compare it to RARE's result
    partial_answers = {}
    correct_answer = ["P_1", "P_2", "P_3", "P_4", "P_5"]
    partial_answers["Bin 1"] = correct_answer
    all_features = ["P_1", "P_2", "P_3", "P_4", "P_5", 
                    "R_1", "R_2", "R_3", "R_4", "R_5", "R_6", "R_7", "R_8", "R_9", "R_10"]
    for i in range (2, 51):
        feature_selection = []
        label = "Bin " + str(i)
        feature_selection = random.sample(all_features, 5)
        partial_answers[label] = feature_selection
    
    #Running RARE just to compare the optimal bin (0 iterations)
    bin_feature_matrix, common_features_and_bins_matrix, amino_acid_bins, amino_acid_bin_scores, rare_feature_MAF_dict, common_feature_MAF_dict, rare_feature_df, common_feature_df, MAF_0_features = RARE(False, None, None, 0, sim_epi_data, 'Class', 0.05, 50, 3, 35, 'Univariate', True, True, 500, 0.8, 0.1, 0.4)    
    
    #Summary of the best bins to compare the correct answer with RARE's result
    Top_Rare_Variant_Bins_Summary(rare_feature_df, 'Class', amino_acid_bins, amino_acid_bin_scores, rare_feature_MAF_dict, 10)
    
    print('-----------------------------------------------------------------')    


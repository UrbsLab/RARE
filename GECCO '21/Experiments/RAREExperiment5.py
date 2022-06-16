#Code for running Experiment 5
#Must be run after running the code in "RARE_Methods"


# In[ ]:


#Experiment 5: 1 Bin, 0 Noise, No Expert Knowlege, Univariate Scoring
for replicate in range (0,1):
    print('Experiment 5')
        
    #Creating the simulated dataset
    sim_data, cutoff = RVDS_One_Bin(1000, 50, 10, 0.05, 'mean', 0)
    
    #Running RARE and checking if 80% of predictive features are reached at each iteration
    #This will be used for Trial 2 to see how fast RARE bins when given/not given partial expert knowledge
    bin_feature_matrix, common_features_and_bins_matrix, amino_acid_bins, amino_acid_bin_scores, rare_feature_MAF_dict, common_feature_MAF_dict, rare_feature_df, common_feature_df, MAF_0_features = RARE_check_for_80_pct(False, None, None, 3000, sim_data, 'Class', 0.05, 50, 5, 25, 'Univariate', True, False, 500, 0.8, 0.1, 0.4)
    
    #Summary of the best bins
    Top_Rare_Variant_Bins_Summary(rare_feature_df, 'Class', amino_acid_bins, amino_acid_bin_scores, rare_feature_MAF_dict, 5) 
    
    #Inputting the correct answer to compare it to RARE's result
    partial_answers = {}
    correct_answer = ["P_1", "P_2", "P_3", "P_4", "P_5", "P_6", "P_7", "P_8", "P_9", "P_10"]
    partial_answers["Bin 1"] = correct_answer
    all_features = ["P_1", "P_2", "P_3", "P_4", "P_5", "P_6", "P_7", "P_8", "P_9", "P_10", 
                    "R_1", "R_2", "R_3", "R_4", "R_5", "R_6", "R_7", "R_8", "R_9", "R_10",
                    "R_11", "R_12", "R_13", "R_14", "R_15", "R_16", "R_17", "R_18", "R_19", "R_20",
                    "R_21", "R_22", "R_23", "R_24", "R_25", "R_26", "R_27", "R_28", "R_29", "R_30",
                    "R_31", "R_32", "R_33", "R_34", "R_35", "R_36", "R_37", "R_38", "R_39", "R_40"]
    for i in range (2, 51):
        feature_selection = []
        label = "Bin " + str(i)
        feature_selection = random.sample(all_features, 10)
        partial_answers[label] = feature_selection
    
    #Running RARE just to compare the optimal bin (0 iterations)
    bin_feature_matrix, common_features_and_bins_matrix, amino_acid_bins, amino_acid_bin_scores, rare_feature_MAF_dict, common_feature_MAF_dict, rare_feature_df, common_feature_df, MAF_0_features = RARE(True, all_features, partial_answers, 0, sim_data, 'Class', 0.05, 50, None, None, 'Univariate', True, False, 500, 0.8, 0.1, 0.4)    
    
    #Summary of the best bins to compare the correct answer with RARE's result
    Top_Rare_Variant_Bins_Summary(rare_feature_df, 'Class', amino_acid_bins, amino_acid_bin_scores, rare_feature_MAF_dict, 5, bin_feature_matrix) 
    
    print('-----------------------------------------------------------------') 


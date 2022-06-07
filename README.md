# RARE
Citation: Dasariraju, S., & Urbanowicz, R. J. (2021). RARE: Evolutionary feature engineering for rare-variant bin discovery. Proceedings of the Genetic and Evolutionary Computation Conference Companion, 1335–1343. https://doi.org/10.1145/3449726.3463174

**RARE: Relevant Association Rare-variant-bin Evolver** (under development) is an evolutionary algorithm approach to binning rare variants as a rare variant association analysis tool. Please email sdasariraju23@lawrenceville.org and Ryan.Urbanowicz@cshs.org for any inquiries related to RARE. 

RARE is an evolutionary algorithm that constructs bins of rare variant features with relevant association to class (univariate and/or multivariate interactions)
through the following steps:

1) Random bin initializaiton or expert knowledge input
2) Repeated evolutionary cycles consisting of:
  a) Candidate bin evaluation with univariate scoring (chi-square test) or Relief-based scoring (MultiSURF algorithm); Note: new scoring options currently under testing
  b) Genetic operations (parent selection, crossover, and mutation) to generate the next generation of candidate bins
3) Final bin evaluation and summary of top bins

In the GECCO '21 folder, please see the RARE_Methods.py file for code definition the RARE function and its subfunctions. The RAREConstantBinSizeFunctionsDefinition.py file contains code for a modified version of RARE that preserves a constant bin size through initilaization and evolutionary cycles (these files also contain code defining the RVDS functions for the data simulators used to test RARE)

<ins>Parameters for RARE:</ins>
1) given_starting_point: whether or not expert knowledge is being inputted (True or False)
2) amino_acid_start_point: if RARE is starting with expert knowledge, input the list of features here; otherwise None
3) amino_acid_bins_start_point: if RARE is starting with expert knowledge, input the list of bins of features here; otherwise None
4) iterations: the number of evolutionary cycles RARE will run
5) original_feature_matrix: the dataset 
6) label_name: label for the class/endpoint column in the dataset (e.g., 'Class')
7) rare_variant_MAF_cutoff: the minor allele frequency cutoff separating common features from rare variant features
8) set_number_of_bins: the population size of candidate bins
9) min_features_per_group: the minimum number of features in a bin
10) max_number_of_groups_with_feature: the maximum number of bins containing a feature
11) scoring_method: 'Univariate', 'Relief', or 'Relief only on bin and common features'
12) score_based_on_sample: if Relief scoring is used, whether or not bin evaluation is done based on a sample of instances rather than the whole dataset
13) score_with_common_variables: if Relief scoring is used, whether or not common features should be used as context for evaluating rare variant bins
14) instance_sample_size: if bin evaluation is done based on a sample of instances, input the sample size here
15) crossover_probability: the probability of each feature in an offspring bin to crossover to the paired offspring bin (recommendation: 0.5 to 0.8)
16) mutation_probability: the probability of each feature in a bin to be deleted (a proportionate probability is automatically applied on each feature outside the bin to be added (recommendation: 0.03 to 0.1)
17) elitism_parameter: the proportion of elite bins in the current generation to be preserved for the next evolutionary cycle (recommendation: 0.2 to 0.4)

**RARE Variant Data Simulators (RVDSs)** are functions that create simulated data for testing/evaluating RARE.
1) The RVDS for Univariate Association Bin (called RVDS_One_Bin) creates a dataset such that no rare variant feature is 100% predictive of class, but an additive bin of features is fully penetrant to class.
2) The RVDS for Epistatic Interaction Bin creates a dataset such that no rare variant feature or bin of rare variant features is predictive of class, but an epistatic interaction between a common feature and an additive bin of rare variant features is 100% predictive of class.

In the GECCO '21 folder, please see the RARE_Variant_Data_Simulator_Methods.py file for the code of the two RVDSs.

<ins>Parameters for RVDS for Univariate Association Bin:
1) number_of_instances: number of instances (i.e., rows) desired in the simulated dataset
2) number_of_features: the total number of rare variant features that should be in the simulated dataset
3) number_of_features_in_bin: of the number_of_features, how many rare variant features should be binned additively for univariate association to class
4) rare_variant_MAF_cutoff: the minor allele frequency that all rare variant features but be below
5) endpoint_cutoff_parameter: "mean" or "median" (recommended "mean")
6) endpoint_variation_probability: how much noise is desired in the dataset (0 produces a bin with a 100% clear signal, 0.5 can be used as a negative control where class value is randomly assigned)

<ins>Parameters for RVDS for Epistatic Interaction Bin:
1) number_of_instances: number of instances (i.e., rows) desired in the simulated dataset
2) number_of_rare_features: the total number of rare variant features that should be in the simulated dataset
3) number_of_features_in_bin: of the number_of_features, how many rare variant features should be binned additively for univariate association to class
4) rare_variant_MAF_cutoff: the minor allele frequency that all rare variant features but be below
5) common_feature_genotype_frequencies_list: a list with the genotype frequencies of each of the common feature genotypes (BB, Bb, bb). Should be of the form [0.25, 0.5, 0.25], where 0.25 is the frequency of the BB genotype, 0.5 is the frequency of the Bb genotype, and 0.25 is the frequency of the bb genotype. [0.25, 0.5, 0.25] is recommended
6) genotype_cutoff_metric: "mean" or "median" (recommended "mean")
7) endpoint_variation_probability: how much noise is desired in the dataset (0 produces a bin that interacts with a common feature to be fully penetrant, 0.5 can be used as a negative control where class value is randomly assigned)
8) list of MLGs_predicting_disease: which of the nine MLGs (AABB, AaBB, aaBB, AABb, AaBb, aaBb, AAbb, Aabb, aabb) correspond to a value of 1 in the class column. [AABB, aaBB, AaBb, AAbb, aabb] should be paired with [0.25, 0.5, 0.25] for the common feature genotype frequencies list to create a dataset with pure, strict epistasis
9) print_summary: whether or not a summary of the simulated datasets with penetrance and frequency values for each of the bin genotypes, common feature genotypes, and MLGs should be printed (True or False)
                                                                                               
                                
We evaluate RARE with 9 Experiments contained in the RAREExperiments file, located in the GECCO '21 Folder. Each file contains an example of using an RVDS to create a simulated dataset and also shows how to apply the RARE algorithm on a dataset. 

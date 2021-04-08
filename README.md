# RARE
RARE: Relevant Association Rare-variant-bin Evolver (under development); an evolutionary algorithm approach to binning rare variants as a rare variant association analysis tool.

RARE is an evolutionary algorithm that constructs bins of rare variant features with relevant association to class (univariate and/or multivariate interactions)
through the following steps:

(1) Random bin initializaiton or expert knowledge input
(2) Repeated evolutionary cycles consisting of:
  (a) Candidate bin evaluation with univariate scoring (chi-square test) or Relief-based scoring (MultiSURF algorithm)
  (b) Genetic operations (parent selection, crossover, and mutation) to generate the next generation of candidate bins
(3) Final bin evaluation and summary of top bins

Please see the RARE_Methods.py file for code definition the RARE function and its subfunctions.
The RAREConstantBinSizeFunctionsDefinition.py file contains code for a modified version of RARE that preserves a constant bin size through initilaization and evolutionary cycles.
(these files also contain code defining the RVDS functions for the data simulators used to test RARE)

The RARE Variant Data Simulators (RVDSs) are functions that create simulated data for testing/evaluating RARE.
1) The RVDS for Univariate Association Bin (called RVDS_One_Bin) creates a dataset such that no rare variant feature is 100% predictive of class, but an additive bin of features is
  fully penetrant to class.
2) The RVDS for Epistatic Interaction Bin creates a dataset such that no rare variant feature or bin of rare variant features is predictive of class, but an epistatic interaction
  between a common feature and an additive bin of rare variant features is 100% predictive of class.

Please see the RARE_Variant_Data_Simulator_Methods.py file for the code of the two RVDSs.

We evaluate RARE with 9 Experiments contained in the RAREExperiments file. Each file contains an example of using an RVDS to create a simulated dataset and also shows 
how to apply the RARE algorithm on a dataset. 

Please email satvik.dasariraju@pennmedicine.upenn.edu and ryanurb@pennmedicine.upenn.edu for any inquiries related to RARE. 

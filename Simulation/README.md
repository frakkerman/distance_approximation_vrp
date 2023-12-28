# Simulation

This folder contains two seperate simulation models, one for the Stylized case study and one for the Amsterdam waste collection case study. The Amsterdam waste collection code is written in Java, for more details we refer to the respective folder.

The stylized case runs using a single file. We used our own VRP solver to obtain vehicle routes and obtain data, for code simplicity we left out this solver, but you can easily add your own, e.g. HGS-CVRP (Vidal et al. 2012). 

The code implements a loop of continuously updating a ML-model during data collection. The ML-model is used to make distance approximation and subsequent customer selection decisions. For code clarity, we only implemented a linear regression model and limited the number of calculated features, but you can easily add these elements from the other projects in this codebase.

Reference to HGS-CVRP: Vidal, T., Crainic, T. G., Gendreau, M., Lahrichi, N., Rei, W. (2012). A hybrid genetic algorithm for multidepot and periodic vehicle routing problems. Operations Research, 60(3), 611-624. https://doi.org/10.1287/opre.1120.1048.
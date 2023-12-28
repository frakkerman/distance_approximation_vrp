# Distance Approximation for Vehicle Routing Problems

This project contains code for the paper titled "Distance approximation to support customer selection in vehicle routing problems" by Fabian Akkerman and Martijn Mes, see: https://doi.org/10.1007/s10479-022-04674-8.

Other related works:

- "Distance Approximation for Dynamic Waste Collection Planning" by Fabian Akkerman, Martijn Mes, and Wouter Heijnen, see: https://doi.org/10.1007/978-3-030-59747-4_23
- "Dynamic Time Slot Pricing Using Delivery Costs Approximations" by Fabian Akkerman, Martijn Mes, and Eduardo Lalla-Ruiz, see: https://doi.org/10.1007/978-3-031-16579-5_15

## Citation

When using the code or data in this repo, please cite the following work:

```
@Article{Akkerman2022,
	author={Akkerman, Fabian
	and Mes, Martijn},
	title={Distance approximation to support customer selection in vehicle routing problems},
	journal={Annals of Operations Research},
	year={2022}
}
```

## Environment

The code is written in Python 3.10. A requirements.txt details further requirements of our project. We tested our project on a Windows 11 environment.


## Folder Structure
The repository contains the following folders:

- **`data/`**: Contains the data used by FeatureEngineering and outputted by FetaureEngineering.
- **`FeatureEngineering/`**: Contains the main code for engineering features based on simulation data.
- **`MachineLearning/`**: Contains the main code for training various distance approximation machine learning models on the data.
  - **`Models/`**: Contains the machine learning models we used (linear regression, random forests, XGBoost, lightGBM, and neural networks), here we also save the models.
  - **`Preprocessing/`**: Contains various preprocessing functions, e.g., loading data, normalization, and feature selection.
  - **`Utils/`**: Contains utility functions, e.g., data loading and hyperparameter tuning using grid search and Bayesian optimization.
- **`Simulation/`**: Contains the main code for simulation: both for the stylized experiments and the Amsterdam waste collection case study (see paper).

Each folder in the root is a seperate project with a seperate `main.py`, such that users can use only the functionality that they want, e.g., only use the ML-models. If you want to test the complete pipeline, you will first need to obtain data using the simulation model, next engineer features, and finally train a ML-model. Next, you can use this ML model again in the simulation. Note that we already provide a dataset for the Amstedam case in the `data` folder.

We provide more details and usage instructions for each project in seperate README's in the folders in this project.


## To the make the code work

 * Create a local python environment by subsequently executing the following commands in the root folder
	* `python3 -m venv venv`
	* `source venv/bin/activate`
	* `python -m pip install -r requirements.txt`
	* `deactivate`
 
## Contributing

If you have proposed extensions to this codebase, feel free to do a pull request! If you experience issues, feel free to send us an email.

## License
* [MIT license](https://opensource.org/license/mit/)
* Copyright 2023 © [Fabian Akkerman](https://people.utwente.nl/f.r.akkerman), [Martijn Mes](https://www.utwente.nl/en/bms/iebis/staff/mes/)
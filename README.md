# Distance Approximation

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

- **`Environments/`**: Contains the problem and all related data.
- **`Src/`**: Contains the main code for all algorithms.
  - **`Algorithms/`**: Contains all algorithmic implementations.
  - **`Utils/`**: Contains utility functions, e.g., data loading, actor and critic class structure, prediction models.


On the first level you can see run.py which implements the overall policy training and evaluation loop. For running PPO we use a seperate file called run_ppo.py.

### Src 

On the first level you can see a parser.py, wherein we set hyperparameters and environment variables, and config.py, which preprocesses inputs.


`Algorithms`: 
* Agent.py: Groups several high-level agent functionalities
* Baseline.py: Contains the baseline, StaticPricing.
* DSPO.py: Contains our proposed contribution, Dynamic Selection and Pricing of OOH (DSPO)
* Heuristic.py: Conntains the benchmark heuristics by Yang et al. (2016).
* PPO.py: Contains the Gaussian PPO policy, as proposed in Schulman et al. (2017)

`Utils`: 
* Actor.py and Critic.py: Contain the neural network architectures for actor and critic respectively.
* Basis.py: Contains the state representation module.
* Predictors.py: Contains the prediction models used for DSPO and the linear benchmark.
* Utils.py: Contains several helper functions such as plotting.

### Environments
`OOH` Contains the implementation of the OOH environment and the used data (Amazon_data and HombergerGehring_data).
* containers.py: container @dataclasses for storing during simulation.
* customerchoice.py: the MNL choice model.
* env_utils.py: some utility functions related to the environment.
* Parcelpoint_py.py:the main problem implementation, following Gymnasium implementation structure mainly.


## To the make the code work

 * Create a local python environment by subsequently executing the following commands in the root folder
	* `python3 -m venv venv`
	* `source venv/bin/activate`
	* `python -m pip install -r requirements.txt`
	* `deactivate`

 * `Src/parser.py` Set your study's hyperparameters in this file.
 
 * `run.py` Execute this file using the command line `python3 run.py`. Run the PPO algorithm with `python3 run_ppo.py`
 
 * Note that you might have to adapt your root folder's name to `ooh_code`
 
 * Note that `hygese.py` requires a slight change to one source file when running with `--load_data=True`, this change is indicated when running the code
 
## Contributing

If you have proposed extensions to this codebase, feel free to do a pull request! If you experience issues, feel free to send us an email.

## License
* [MIT license](https://opensource.org/license/mit/)
* Copyright 2023 © [Fabian Akkerman](https://people.utwente.nl/f.r.akkerman), [Martijn Mes](https://www.utwente.nl/en/bms/iebis/staff/mes/)
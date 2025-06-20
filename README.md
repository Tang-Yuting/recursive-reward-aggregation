# recursive-reward-aggregation
Code of Recursive Reward Aggregate.
Built upon Stable-Baselines3.


## Running with Docker
You can build and run the Docker container:
```sh
cd docker
docker build -t rra_image . -f Dockerfile 
```
### **Tips**
If you encounter issues with **Cython**, try the following:
```sh
pip uninstall Cython
pip install Cython==3.0.0a10
```
This can resolve version conflicts or compatibility issues with certain dependencies.


## Running Experiments

### 1. Gird-world environment
The `grid-world` environment can be executed in two ways:

#### Using Jupyter Notebook

The `gird-world` environment can be executed directly using the Jupyter Notebook **`grid.ipynb`**.
```sh
cd grid_world
jupyter notebook grid.ipynb
```

#### Using Shell Script
You can also run the environment from the command line using the provided shell script.

```sh
cd grid_world
bash run_grid.sh [aggregation]
```

Replace `[aggregation]` with your desired aggregation method, such as: ``dsum`` (default), ``max``, ``mean``, ``dmax``.

### 2. Wind environment

### 3. Continuous control experiment

### 4. Portfolio experiment

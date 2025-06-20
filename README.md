# recursive-reward-aggregation
Code of [Recursive Reward Aggregation](https://openreview.net/forum?id=13lUcKpWy8#discussion).
Built upon Stable-Baselines3.


## Running with Docker
You can build and run the Docker container as follows.

#### Step 1: Build the Docker image
```sh
cd docker
docker build -t rra_image . -f Dockerfile 
```

#### Step 2: Run the Docker container
```sh
docker run -dit -p 8888:22 --mount type=bind,source=/path/to/your/RRA,destination=/workspace/RRA --name rra_container -m 16g --gpus all rra_image /bin/bash
```
Replace `/path/to/your/RRA` with the absolute path to your local RRA directory.

#### Step 3: Enter the running container
Once the container is up and running, you can access its shell with:
```sh
docker exec -it rra_container bash
```


### Additional Setup for Experiments
To run Continuous Control and Portfolio experiments inside the Docker container, you will also need to manually install:
```sh
pip install torch stable-baselines3
```
Make sure to run this inside the container after building it.

### Tips
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
bash run_grid.sh [RECURSIVE_TYPE]
```
Replace `[RECURSIVE_TYPE]` with your desired aggregation method. Default: `dsum`. Available options: `dsum`, `dmax`, `min`, `mean`.

### 2. Wind environment
The `wind` environment can be executed directly using the Jupyter Notebook **`wind.ipynb`**.
```sh
cd wind
jupyter notebook wind.ipynb
```

### 3. Continuous control experiment
You can run the continuous control experiments using the provided shell script.
```sh
cd continuous_control
bash run_td3.sh [SEED] [ENV] [RECURSIVE_TYPE]
```
- `SEED` (optional): Random seed for training. Default: `42`.
- `ENV` (optional): OpenAI Gym environment name. Default: `Ant-v3`. Available options: `Ant-v3`, `Walker2d-v3`, `Hopper-v3`.
- `RECURSIVE_TYPE` (optional): Type of recursive aggregation to use in training. Default: `dsum`. Available options: `dsum`, `dmax`, `min`, `dsum_dmax`, `dsum_variance`.

If no arguments are provided, the script will use the default values.

### 4. Portfolio experiment
```sh
cd portfolio
./run_portfolio.sh
```


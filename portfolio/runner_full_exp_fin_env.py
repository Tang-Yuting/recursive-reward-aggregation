import argparse
import random
import sys

import os
import copy
import pickle
import torch as th
import numpy as np
import empyrical as ep

# sys.path.insert(0, "/workspace/stable-baselines3")
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

from recursive_stable_baselines3 import Recursive_PPO_multi_output

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from own_eval_callback import OwnEvalCallback, own_eval_policy

from fin_env import FinEnv, FinEnv_resursive
from fin_utils import find_closest_date_before, find_closest_date_after, str_to_bool



def check_and_make_directories(directories: list[str]):
    for directory in directories:
        if not os.path.exists("./" + directory):
            os.makedirs("./" + directory)


def main():
    parser = argparse.ArgumentParser(description="Runner Parser",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed_start', type=int, default=5, help='Random seed')
    parser.add_argument('--adapt_state', type=str, default="False", help='adapt state')
    parser.add_argument('--adapt_reward', type=str, default="False", help='adapt reward')
    parser.add_argument('--result_dir', type=str, default="./full_exp", help='output directory')

    args = vars(parser.parse_args())
    # We used seed_start [0, 5, 10, 15, 20] for the experiments
    seed_start=args["seed_start"]
    n_seeds = 1
    seeds = np.arange(seed_start, seed_start+n_seeds, 1, dtype=int)
    adapt_state=str_to_bool(args["adapt_state"])
    adapt_reward=str_to_bool(args["adapt_reward"])

    print(f"{seed_start=}, {adapt_state=}, {adapt_reward=}", flush=True)


    start_date_list_train = ["2006-01-01", "2007-01-01", "2008-01-01", "2009-01-01", "2010-01-01", 
                    "2011-01-01", "2012-01-01", "2013-01-01", "2014-01-01", "2015-01-01",]
    end_date_list_train = ["2010-12-31", "2011-12-31", "2012-12-31", "2013-12-31", "2014-12-31",
                    "2015-12-31", "2016-12-31", "2017-12-31", "2018-12-31", "2019-12-31",]

    start_date_list_eval = ["2011-01-01", "2012-01-01", "2013-01-01", "2014-01-01", "2015-01-01",
                            "2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01",]
    end_date_list_eval = ["2011-12-31", "2012-12-31", "2013-12-31", "2014-12-31", "2015-12-31",
                            "2016-12-31", "2017-12-31", "2018-12-31", "2019-12-31", "2020-12-31"]

    start_date_list_test = ["2012-01-01", "2013-01-01", "2014-01-01", "2015-01-01", "2016-01-01",
                            "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01",]
    end_date_list_test = ["2012-12-31", "2013-12-31", "2014-12-31", "2015-12-31", "2016-12-31",
                        "2017-12-31", "2018-12-31", "2019-12-31", "2020-12-31", "2021-12-31"]


    exp_dir = os.path.abspath(args["result_dir"])
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    log_dir=os.path.join(exp_dir, f'log_{adapt_reward=}_{adapt_state=}')
    tensorboard_dir =os.path.join(exp_dir, f"tensorboard_{adapt_reward=}_{adapt_state=}")

    check_and_make_directories([log_dir, tensorboard_dir])


    verbose = 1

    n_evals = 200

    # PPO paras
    training_steps = 7.5e6
    n_envs = 10
    n_steps =  252*3
    batch_size = 252*5
    n_epochs = 16
    gamma=0.9
    gae_lambda=0.9
    clip_range=0.25

    lr_start = 3e-4
    lr_end = 1e-5

    eval_freq = int(training_steps / n_evals / n_envs)

    deterministic_eval = True
    if deterministic_eval:
        n_traj_eval = 1
    else:
        n_traj_eval = n_envs


    # shape (n_steps)
    best_model_seed = []

    # shapes (n_steps, n_seeds)
    sharpe_list_train = []
    sharpe_list_eval = []
    sharpe_list_test = []

    # shapes (n_steps, n_seeds, n_steps)
    all_R_train = []
    all_R_eval = []
    all_R_test = []


    start_step = 0
    last_best_model_path = None

    data_folder = "./preproc_data/"
    with open(data_folder + "date_list.txt", "r") as f:
        date_list = f.read().splitlines()

    for step in range(start_step, len(start_date_list_train)):
        best_model_save_path = os.path.join(log_dir, f"best_model_{step}")
        tensorboard_dir_log = os.path.join(tensorboard_dir, f"tensorboard_{step}")
        check_and_make_directories([best_model_save_path, tensorboard_dir_log])

        sharpe_list_train_step = []
        sharpe_list_eval_step = []
        sharpe_list_test_step = []

        all_R_train_step = []
        all_R_eval_step = []
        all_R_test_step = []

        for seed in seeds:
            seed = int(seed)
            th.manual_seed(seed)
            th.cuda.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

            def lr_schedule(progress):
                return lr_end + (lr_start - lr_end) * progress

            policy_kwargs = dict(
                        net_arch=dict(pi=[64, 64], vf=[64, 64]),
                        activation_fn=th.nn.Tanh,
                        log_std_init=-1.,
            )

            start_date = find_closest_date_after(start_date_list_train[step], date_list)
            end_date = find_closest_date_before(end_date_list_train[step], date_list)

            env_kwargs = {  "start_date": start_date, 
                            "end_date": end_date, 
                            "data_folder": data_folder, 
                            "adapt_state": adapt_state, 
                            "adapt_reward": adapt_reward,
                            "lookback_window":59,
                            "eta": 1/252,
                            "init_P": 1e5,
                            "init_past": False,
                        }
            
            to_add=f"_{seed=}, {start_date=}, {end_date=}, {step=}"
            
            

            eval_env_kwargs_0 = copy.deepcopy(env_kwargs)
            eval_env_kwargs_0["start_date"] = find_closest_date_after(start_date_list_train[step], date_list) 
            eval_env_kwargs_0["end_date"] = find_closest_date_before(end_date_list_train[step], date_list)
            eval_env_kwargs_0["eval"] = True
            eval_env_0 = make_vec_env(FinEnv_resursive, n_envs=n_envs, monitor_dir=log_dir, vec_env_cls=SubprocVecEnv,
                                    env_kwargs=eval_env_kwargs_0)
            # eval_env_0 = FinEnv_resursive(**eval_env_kwargs_0)
            eval_callback_0 = OwnEvalCallback(eval_env_0, best_model_save_path=best_model_save_path,
                                        log_path=best_model_save_path, eval_freq=eval_freq, n_eval_episodes=n_traj_eval,
                                        deterministic=deterministic_eval, render=False, to_add= "_same" + to_add, verbose=verbose,
                                        tens_name=f"same_{step}",)

            eval_env_kwargs_1 = copy.deepcopy(env_kwargs)
            eval_env_kwargs_1["start_date"] = find_closest_date_after(start_date_list_eval[step], date_list) 
            eval_env_kwargs_1["end_date"] = find_closest_date_before(end_date_list_eval[step], date_list)
            eval_env_kwargs_1["eval"] = True
            eval_env_1 = make_vec_env(FinEnv_resursive, n_envs=n_envs, monitor_dir=log_dir, vec_env_cls=SubprocVecEnv,
                                    env_kwargs=eval_env_kwargs_1)
            # eval_env_1 = FinEnv_resursive(**eval_env_kwargs_1)

            eval_callback_1 = OwnEvalCallback(eval_env_1, best_model_save_path=best_model_save_path,
                                        log_path=best_model_save_path, eval_freq=eval_freq, n_eval_episodes=n_traj_eval,
                                        deterministic=deterministic_eval, render=False, to_add="_eval" + to_add, verbose=verbose,
                                        tens_name=f"eval_{step}",)

            eval_env_kwargs_2 = copy.deepcopy(env_kwargs)
            eval_env_kwargs_2["start_date"] = find_closest_date_after(start_date_list_test[step], date_list) 
            eval_env_kwargs_2["end_date"] = find_closest_date_before(end_date_list_test[step], date_list)
            eval_env_kwargs_2["eval"] = True
            eval_env_2 = make_vec_env(FinEnv_resursive, n_envs=n_envs, monitor_dir=log_dir, vec_env_cls=SubprocVecEnv,
                                    env_kwargs=eval_env_kwargs_2)
            # eval_env_2 = FinEnv_resursive(**eval_env_kwargs_2)
            eval_callback_2 = OwnEvalCallback(eval_env_2, best_model_save_path=best_model_save_path,
                                        log_path=best_model_save_path, eval_freq=eval_freq, n_eval_episodes=n_traj_eval,
                                        deterministic=deterministic_eval, render=False, to_add= "_test" + to_add, verbose=verbose,
                                        tens_name=f"test_{step}",)

            vec_env = make_vec_env(FinEnv_resursive, n_envs=n_envs, monitor_dir=log_dir, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
            # vec_env = FinEnv_resursive(**env_kwargs)

            model = Recursive_PPO_multi_output("MlpPolicy_multi_output", vec_env, tensorboard_log=tensorboard_dir_log,
                        learning_rate = copy.deepcopy(lr_schedule),
                        n_steps = n_steps,
                        batch_size = batch_size,
                        n_epochs = n_epochs,
                        gamma = gamma,
                        gae_lambda = gae_lambda,
                        clip_range = clip_range,
                        policy_kwargs = policy_kwargs,
                        seed = seed,
                        recursive_type="sharpe_3_multi_env",
                        output_feature_num=3,
                                               )
            
            if last_best_model_path:
                model.set_parameters(last_best_model_path)
            model.set_random_seed(seed)

            model.learn(total_timesteps=training_steps,
                        callback=[eval_callback_0, eval_callback_1, eval_callback_2,])

            eval_env_0.close()
            eval_env_1.close()
            eval_env_2.close()
            vec_env.close()

            # Eval results -------------------------------------------------------------------------------------------------------
            model.set_parameters(os.path.join(best_model_save_path, "best_model" + "_same" + to_add))
            eval_env_0 = FinEnv_resursive(**eval_env_kwargs_0)
            _, _, R_same = own_eval_policy(
                model,
                eval_env_0,
                deterministic=deterministic_eval,
            )
            sharpe_same = ep.sharpe_ratio(R_same)
            sharpe_same = np.nan_to_num(sharpe_same, nan=0.0)

            # K = np.sqrt(len(all_R_same) / (len(all_R_same) - 1))
            # if not np.isclose(sharpe_same, np.mean(all_R_same) / (np.std(all_R_same) * K) * np.sqrt(252)):
            #     print(f"ERROR, {sharpe_same=}, {np.mean(all_R_same) / (np.std(all_R_same) * K) * np.sqrt(252)=}", flush=True)

            model.set_parameters(os.path.join(best_model_save_path, "best_model" + "_eval" + to_add))
            eval_env_1 = FinEnv_resursive(**eval_env_kwargs_1)
            _, _, R_eval = own_eval_policy(
                model,
                eval_env_1,
                deterministic=deterministic_eval,
            )
            sharpe_eval = ep.sharpe_ratio(R_eval)
            sharpe_eval = np.nan_to_num(sharpe_eval, nan=0.0)

            model.set_parameters(os.path.join(best_model_save_path, "best_model" + "_eval" + to_add))
            eval_env_2 = FinEnv_resursive(**eval_env_kwargs_2)
            _, _, R_test = own_eval_policy(
                model,
                eval_env_2,
                deterministic=deterministic_eval,
            )
            sharpe_test = ep.sharpe_ratio(R_test)
            sharpe_test = np.nan_to_num(sharpe_test, nan=0.0)


            sharpe_list_train_step.append(sharpe_same)
            sharpe_list_eval_step.append(sharpe_eval)
            sharpe_list_test_step.append(sharpe_test)

            all_R_train_step.append(R_same)
            all_R_eval_step.append(R_eval)
            all_R_test_step.append(R_test)

            
        sharpe_list_train.append(sharpe_list_train_step)
        sharpe_list_test.append(sharpe_list_test_step)
        sharpe_list_eval.append(sharpe_list_eval_step)

        all_R_train.append(all_R_train_step)
        all_R_eval.append(all_R_eval_step)
        all_R_test.append(all_R_test_step)

        # find best eval model
        best_seed = seeds[np.argmax(sharpe_list_eval_step)]
        best_model_seed.append(best_seed)
        to_add_best=f"_seed={best_seed}, {start_date=}, {end_date=}, {step=}"
        last_best_model_path = os.path.join(best_model_save_path, "best_model" + "_eval" + to_add_best)  

        # Save sharpe ratios
        sharpe_train_np = np.array(sharpe_list_train)
        sharpe_test_np = np.array(sharpe_list_test)
        sharpe_eval_np = np.array(sharpe_list_eval)

        best_model_seed_np = np.array(best_model_seed)

        with open(os.path.join(log_dir, "sharpe_train.npy"), "wb") as f:
            np.save(f, sharpe_train_np)
        with open(os.path.join(log_dir, "sharpe_test.npy"), "wb") as f:
            np.save(f, sharpe_test_np)
        with open(os.path.join(log_dir, "sharpe_eval.npy"), "wb") as f:
            np.save(f, sharpe_eval_np)

        with open(os.path.join(log_dir, "all_R_train.pkl"), "wb") as f:
            pickle.dump(all_R_train, f)
        with open(os.path.join(log_dir, "all_R_test.pkl"), "wb") as f:
            pickle.dump(all_R_test, f)
        with open(os.path.join(log_dir, "all_R_eval.pkl"), "wb") as f:
            pickle.dump(all_R_eval, f)

        with open(os.path.join(log_dir, "best_model_seed.npy"), "wb") as f:
            np.save(f, best_model_seed_np)

if __name__ == '__main__':
    main()


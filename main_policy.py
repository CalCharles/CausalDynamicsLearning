import os
import numpy as np
import torch
import sys
import time
from torch.utils.tensorboard import SummaryWriter
np.set_printoptions(precision=3, suppress=True, threshold=3000)
torch.set_printoptions(precision=3, sci_mode=False, threshold=3000)

from Baselines.CDL.model.inference_mlp import InferenceMLP
from Baselines.CDL.model.inference_gnn import InferenceGNN
from Baselines.CDL.model.inference_reg import InferenceReg
from Baselines.CDL.model.inference_nps import InferenceNPS
from Baselines.CDL.model.inference_cmi import InferenceCMI

from Baselines.CDL.model.random_policy import RandomPolicy
from Baselines.CDL.model.hippo import HiPPO
from Baselines.CDL.model.model_based import ModelBased

from Baselines.CDL.model.encoder import make_encoder

from Baselines.CDL.utils.utils import TrainingParams, update_obs_act_spec, set_seed_everywhere, get_env, get_start_step_from_model_loading
from Baselines.CDL.utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from Baselines.CDL.utils.plot import plot_adjacency_intervention_mask
from Baselines.CDL.utils.scripted_policy import get_scripted_policy, get_is_demo

from Baselines.CDL.env.chemical_env import Chemical

from Environment.Environments.initialize_environment import initialize_environment

def process_breakout(obs):
    obs = obs['factored_state']
    for n in obs.keys():
        if n.find("Block") != -1:
            obs[n] = np.array([obs[n][-1]])
        else:
            obs[n][:2] = np.array(obs[n])[:2] / 84 - 0.5 if type(obs[n]) == list else obs[n][:2] / 84.0 - 0.5
            obs[n] = obs[n][:4]
    return obs


def train(params, args):
    device = torch.device("cuda:{}".format(params.cuda_id) if torch.cuda.is_available() else "cpu")
    set_seed_everywhere(params.seed)

    print(device)

    params.device = device
    training_params = params.training_params
    inference_params = params.inference_params
    policy_params = params.policy_params
    cmi_params = inference_params.cmi_params

    # init environment
    render = False
    num_env = params.env_params.num_env
    is_vecenv = num_env > 1
    
    if args.environment.env in ["Breakout", "RoboPushing"]:
        env, rec = initialize_environment(args.environment, args.record)
        if args.environment.env == "Breakout": 
            env.get_state_cdl = lambda: process_breakout(env.get_state())
            env.object_sizes["Block"] = 1
            env.object_sizes["Ball"] = 4
            env.object_sizes["Paddle"] = 4
        else:
            env.get_state_cdl = lambda: env.get_state()['factored_state']

        params.obs_keys = env.all_names
        params.obs_keys.remove("Action")
        params.obs_keys.remove("Done")
        params.obs_keys.remove("Reward")
    else:
        env = get_env(params, render)
    if isinstance(env, Chemical):
        torch.save(env.get_save_information(), os.path.join(params.rslts_dir, "chemical_env_params"))

    # init model
    update_obs_act_spec(env, params)
    encoder = make_encoder(params)
    decoder = None

    inference_algo = params.training_params.inference_algo
    use_cmi = inference_algo == "cmi"
    if inference_algo == "mlp":
        Inference = InferenceMLP
    elif inference_algo == "gnn":
        Inference = InferenceGNN
    elif inference_algo == "reg":
        Inference = InferenceReg
    elif inference_algo == "nps":
        Inference = InferenceNPS
    elif inference_algo == "cmi":
        Inference = InferenceCMI
    else:
        raise NotImplementedError
    inference = Inference(encoder, params)

    scripted_policy = get_scripted_policy(env, params)
    rl_algo = params.training_params.rl_algo
    is_task_learning = rl_algo == "model_based"
    if rl_algo == "random":
        policy = RandomPolicy(params)
    elif rl_algo == "hippo":
        policy = HiPPO(encoder, inference, params)
    elif rl_algo == "model_based":
        policy = ModelBased(encoder, inference, params)
    else:
        raise NotImplementedError

    # init replay buffer
    use_prioritized_buffer = getattr(training_params.replay_buffer_params, "prioritized_buffer", False)
    if use_prioritized_buffer:
        assert is_task_learning
        replay_buffer = PrioritizedReplayBuffer(params)
    else:
        replay_buffer = ReplayBuffer(params)

    # init saving
    writer = SummaryWriter(os.path.join(params.rslts_dir, "tensorboard"))
    model_dir = os.path.join(params.rslts_dir, "trained_models")
    os.makedirs(model_dir, exist_ok=True)

    start_step = get_start_step_from_model_loading(params)
    total_steps = training_params.total_steps
    collect_env_step = training_params.collect_env_step
    inference_gradient_steps = training_params.inference_gradient_steps
    policy_gradient_steps = training_params.policy_gradient_steps
    train_prop = inference_params.train_prop

    # init episode variables
    episode_num = 0
    obs = env.reset()
    n_steps_per_model_train = 1000 # training_params.n_steps_per_model_train
    if args.environment.env in ["Breakout"]:
        obs = process_breakout(obs)
    if args.environment.env in ["RoboPushing"]:
        obs = obs["factored_state"]
    scripted_policy.reset(obs)

    done = np.zeros(num_env, dtype=bool) if is_vecenv else False
    success = False
    episode_reward = np.zeros(num_env) if is_vecenv else 0
    episode_step = np.zeros(num_env) if is_vecenv else 0
    is_train = (np.random.rand(num_env) if is_vecenv else np.random.rand()) < train_prop
    is_demo = np.array([get_is_demo(0, params) for _ in range(num_env)]) if is_vecenv else get_is_demo(0, params)

    for step in range(start_step, total_steps):
        is_init_stage = step < training_params.init_steps
        print("{}/{}, init_stage: {}".format(step + 1, total_steps, is_init_stage))
        loss_details = {"inference": [],
                        "inference_eval": [],
                        "policy": []}

        # env interaction and transition saving
        if collect_env_step:
            # reset in the beginning of an episode
            if is_vecenv and done.any():
                for i, done_ in enumerate(done):
                    if not done_:
                        continue
                    is_train[i] = np.random.rand() < train_prop
                    is_demo[i] = get_is_demo(step, params)
                    if rl_algo == "hippo":
                        policy.reset(i)
                    scripted_policy.reset(obs, i)

                    if writer is not None:
                        writer.add_scalar("policy_stat/episode_reward", episode_reward[i], episode_num)
                    episode_reward[i] = 0
                    episode_step[i] = 0
                    episode_num += 1
            elif not is_vecenv and done:
                obs = env.reset()
                if args.environment.env in ["Breakout"]:
                    obs = process_breakout(obs)
                if args.environment.env in ["RoboPushing"]:
                    obs = obs["factored_state"]

                if rl_algo == "hippo":
                    policy.reset()
                scripted_policy.reset(obs)

                if writer is not None:
                    if is_task_learning:
                        if not is_demo:
                            writer.add_scalar("policy_stat/episode_reward", episode_reward, episode_num)
                            writer.add_scalar("policy_stat/success", float(success), episode_num)
                    else:
                        writer.add_scalar("policy_stat/episode_reward", episode_reward, episode_num)
                is_train = np.random.rand() < train_prop
                is_demo = get_is_demo(step, params)
                episode_reward = 0
                episode_step = 0
                success = False
                episode_num += 1

            # get action
            inference.eval()
            policy.eval()
            if is_init_stage:
                if is_vecenv:
                    action = np.array([policy.act_randomly() for _ in range(num_env)])
                else:
                    action = policy.act_randomly()
            else:
                if is_vecenv:
                    action = policy.act(obs)
                    if is_demo.any():
                        demo_action = scripted_policy.act(obs)
                        action[is_demo] = demo_action[is_demo]
                else:
                    action_policy = scripted_policy if is_demo else policy
                    action = action_policy.act(obs)

            next_obs, env_reward, done, info = env.step(action)
            if args.environment.env in ["Breakout"]:
                next_obs = process_breakout(next_obs)
            if args.environment.env in ["RoboPushing"]:
                next_obs = next_obs["factored_state"]
            if is_task_learning and not is_vecenv:
                success = success or info["success"]

            inference_reward = np.zeros(num_env) if is_vecenv else 0
            episode_reward += env_reward if is_task_learning else inference_reward
            episode_step += 1

            # is_train: if the transition is training data or evaluation data for inference_cmi
            print(env_reward)
            replay_buffer.add(obs, action, env_reward, next_obs, done, is_train, info)

            # ppo uses its own buffer
            if rl_algo == "hippo" and not is_init_stage:
                policy.update_trajectory_list(obs, action, done, next_obs, info)

            obs = next_obs

        # training and logging
        if is_init_stage:
            continue
        if step % n_steps_per_model_train != 0:
            continue

        if inference_gradient_steps > 0:
            inf_time = time.time()
            inference.train()
            inference.setup_annealing(step)
            upd = time.time()
            for i_grad_step in range(inference_gradient_steps):
                obs_batch, actions_batch, next_obses_batch = \
                    replay_buffer.sample_inference(inference_params.batch_size, "train")
                loss_detail = inference.update(obs_batch, actions_batch, next_obses_batch)
                loss_details["inference"].append(loss_detail)
            print("upd", time.time() - upd)

            inference.eval()
            if (step + 1) % cmi_params.eval_freq == 0:
                if use_cmi:
                    # if do not update inference, there is no need to update inference eval mask
                    inference.reset_causal_graph_eval()
                    for _ in range(cmi_params.eval_steps):
                        obs_batch, actions_batch, next_obses_batch = \
                            replay_buffer.sample_inference(cmi_params.eval_batch_size, use_part="eval")
                        eval_pred_loss = inference.update_mask(obs_batch, actions_batch, next_obses_batch)
                        loss_details["inference_eval"].append(eval_pred_loss)
                else:
                    obs_batch, actions_batch, next_obses_batch = \
                        replay_buffer.sample_inference(cmi_params.eval_batch_size, use_part="eval")
                    loss_detail = inference.update(obs_batch, actions_batch, next_obses_batch, eval=True)
                    loss_details["inference_eval"].append(loss_detail)
            print("inference_training", time.time() - inf_time)

        if policy_gradient_steps > 0 and rl_algo != "random":
            pol_time = time.time()
            policy.train()
            if rl_algo in ["ppo", "hippo"]:
                loss_detail = policy.update()
                loss_details["policy"].append(loss_detail)
            else:
                policy.setup_annealing(step)
                for i_grad_step in range(policy_gradient_steps):
                    obs_batch, actions_batch, rewards_batch, idxes_batch = \
                        replay_buffer.sample_model_based(policy_params.batch_size)

                    loss_detail = policy.update(obs_batch, actions_batch, rewards_batch)
                    if use_prioritized_buffer:
                        replay_buffer.update_priorties(idxes_batch, loss_detail["priority"])
                    loss_details["policy"].append(loss_detail)
            print("policy_training", time.time() - inf_time)
            # print("loss details", loss_details, policy)
            policy.eval()

        if writer is not None:
            for module_name, module_loss_detail in loss_details.items():
                if not module_loss_detail:
                    continue
                # list of dict to dict of list
                if isinstance(module_loss_detail, list):
                    keys = set().union(*[dic.keys() for dic in module_loss_detail])
                    module_loss_detail = {k: [dic[k].item() for dic in module_loss_detail if k in dic]
                                          for k in keys if k not in ["priority"]}
                for loss_name, loss_values in module_loss_detail.items():
                    writer.add_scalar("{}/{}".format(module_name, loss_name), np.mean(loss_values), step)

            if (step + 1) % training_params.plot_freq == 0 and inference_gradient_steps > 0:
                plot_adjacency_intervention_mask(params, inference, writer, step)

        if (step + 1) % training_params.saving_freq == 0:
            if inference_gradient_steps > 0:
                inference.save(os.path.join(model_dir, "inference_{}".format(step + 1)))
            if policy_gradient_steps > 0:
                policy.save(os.path.join(model_dir, "policy_{}".format(step + 1)))


if __name__ == "__main__":
    params = TrainingParams(training_params_fname="policy_params.json", train=True)
    train(params)

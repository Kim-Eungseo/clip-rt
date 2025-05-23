"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
import time

sys.path.append("./openvla-clip-rt")
sys.path.append("./openvla-clip-rt/LIBERO")

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

# import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)

from clip_rt_utils import get_clip_rt, get_tokenizer
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
    get_clip_rt_action,
)


@dataclass
class GenerateConfig:
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "clip_rt"  # Model family
    pretrained_checkpoint: Union[str, Path] = ""  # Pretrained checkpoint path
    load_in_8bit: bool = False  # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False  # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True  # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # CLIP-RT environment-specific parameters
    #################################################################################################################
    data_portion: int = 10
    save_video: str = "y"
    model_ckpt: str = "1"
    chunk_cut: int = 8
    zero_action_exception: bool = True
    # model_path: str = "/data/jhkim/cliprt/ckpt/{}/epoch_{}.pt".format(
    #     task_suite_name.split("_")[-1], model_ckpt
    # )
    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None  # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"  # Local directory for eval logs

    use_wandb: bool = False  # Whether to also log results in Weights & Biases
    wandb_project: str = ""  # Name of W&B project to log to (use default!)
    wandb_entity: str = ""  # Name of entity to log under

    seed: int = 7  # Random Seed (for reproducibility)


@draccus.wrap()

def eval_libero(cfg: GenerateConfig) -> None:

    model_path = "/data/jhkim/cliprt/cliprt_libero_{}_reg_epoch_{}.pt".format(
        cfg.task_suite_name.split("_")[-1], cfg.model_ckpt
    )

    lines = []
    lines.append(str(cfg.model_family))
    lines.append(f"model epoch: {cfg.model_ckpt}")
    lines.append(f"Model path: {model_path}")
    lines.append(f"Data portion: {100 / cfg.data_portion}%")
    lines.append(f"Zero action exception: {cfg.zero_action_exception}")

    print("\n".join(lines))
    
    assert (
        cfg.pretrained_checkpoint is not None
    ), "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert (
            cfg.center_crop
        ), "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (
        cfg.load_in_8bit and cfg.load_in_4bit
    ), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # [OpenVLA] Get Hugging Face processor
    processor = None

    if cfg.model_family == "clip_rt":
        model, preprocess, action_classes, lookup_table = get_clip_rt(
            model_path=model_path,
            task_split=cfg.task_suite_name,
        )
        tokenizer = get_tokenizer()

    # Initialize local logging
    mm = model_path.split("epoch_")[-1]
    run_id = f"EVAL-{cfg.model_family}-{mm}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    log_dir = os.path.join(cfg.local_log_dir,"ac",cfg.task_suite_name)
    os.makedirs(log_dir, exist_ok=True)
    local_log_filepath = os.path.join(log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    if cfg.model_family == "openvla":
        resize_size = get_image_resize_size(cfg)
    elif cfg.model_family == "clip_rt":
        resize_size = 224

    # Start evaluation
    ssssss = 0
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # ssssss += 1
        # if ssssss not in [3, 4]:
        #     continue

        print(f"Task {task_id} of {num_tasks_in_suite}")
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        tot_inf_time = 0 
        tot_steps = 0

        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            if episode_idx % cfg.data_portion != 0:
                continue
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")

            actions = []

            prev_action = None
            repeat_count = 0

            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(
                            get_libero_dummy_action(cfg.model_family)
                        )
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    # replay_images.append(img)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                    }
                    
                    stime = time.time()

                    action_chunks = get_clip_rt_action(
                        model,
                        preprocess,
                        tokenizer,
                        action_classes,
                        lookup_table,
                        observation,
                        task_description,
                        zero_action_exception = cfg.zero_action_exception
                    )
                    etime = time.time()
                    runtime = etime-stime
                    tot_inf_time += runtime
                    tot_steps += 1

                    # print(f"Action_chunks: {action_chunks}")
                    
                    log_file.write(f"Action_chunks: {action_chunks}\n")
                    actions.extend(action_chunks)

                    done_flag = False

                    # Execute action in environment
                    for cut, action_chunk in enumerate(action_chunks):
                        # if cut > cfg.chunk_cut-1:
                        #     break
                        # action_chunk = normalize_gripper_action(action_chunk, binarize=True)
                        # action_chunk = invert_gripper_action(action_chunk)
                        
                        # Get preprocessed image
                        img = get_libero_image(obs, resize_size)

                        # Save preprocessed image for replay video
                        replay_images.append(img)
                        obs, reward, done, info = env.step(action_chunk)
                        if done:
                            done_flag = True
                            break
                        t += 1
                        
                    if done_flag:
                        task_successes += 1
                        total_successes += 1
                        break
                    

                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    # raise e
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            if cfg.save_video == "y":
                save_rollout_video(
                    cfg.task_suite_name,
                    cfg.model_ckpt,
                    replay_images,
                    total_episodes,
                    success=done,
                    task_description=task_description,
                    log_file=log_file,
                )
            if cfg.model_family == "clip_rt":
                import json

                os.makedirs(
                    f"./actions/{cfg.task_suite_name}/epoch_{cfg.model_ckpt}/",
                    exist_ok=True,
                )

                with open(
                    f"./actions/{cfg.task_suite_name}/epoch_{cfg.model_ckpt}/actions_{task_description}_{episode_idx}.json",
                    "w",
                ) as f:
                    json.dump(actions, f, indent=4)

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
            )
            print("model epoch_{}".format(cfg.model_ckpt))
            print("chunk cut: {}".format(cfg.chunk_cut))

            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n"
            )
            log_file.write("model epoch_{}\n".format(cfg.model_ckpt))
            log_file.write("chunk cut: {}".format(cfg.chunk_cut))
            log_file.flush()

        # Log final results
        print("latency (sec): {}".format(float(tot_inf_time)/int(tot_steps)))

        print(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}"
        )
        print(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}"
        )
        log_file.write("latency (sec): {}\n".format(float(tot_inf_time)/int(tot_steps)))
        log_file.write(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}\n"
        )
        log_file.write(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}\n"
        )
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes)
                    / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )
    print("\n".join(lines))
    log_file.write("\n".join(lines))
    log_file.write(action_classes)
    log_file.flush()
    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()

# %%
import sys
from huggingface_hub import snapshot_download

# 프로젝트 경로 추가
sys.path.append("/home/ngseo/clip-rt/openvla")

from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset, EpisodicRLDSDataset

vla_download_path = snapshot_download(repo_id="openvla/openvla-7b")

from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
)

import torch

processor = AutoProcessor.from_pretrained(vla_download_path, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    vla_download_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to("cuda:0")

from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.datasets import RLDSBatchTransform

action_tokenizer = ActionTokenizer(processor.tokenizer)
use_wrist_image = False

batch_transform = RLDSBatchTransform(
    action_tokenizer=action_tokenizer,
    base_tokenizer=processor.tokenizer,
    image_transform=processor.image_processor.apply_transform,
    prompt_builder_fn=PurePromptBuilder,
)

dataset_name = "libero_goal"
# RLDSDataset 인스턴스 생성 (데이터셋 로딩)
train_dataset = EpisodicRLDSDataset(
    "/home/ngseo/clip-rt/openvla-dataset/modified_libero_rlds/",
    f"{dataset_name}_no_noops",
    batch_transform,
    resize_resolution=(224, 224),
    shuffle_buffer_size=1000,
    image_aug=False,
)

# %%
import os
import re
import json
from tqdm import tqdm

for i, sample in enumerate(tqdm(train_dataset)):
    language_instruction = sample[0]["language_instruction"]

    # language_instruction 폴더 경로
    language_dir = f"/home/ngseo/clip-rt/data/{dataset_name}/{language_instruction}"

    # 폴더가 없으면 생성
    if not os.path.exists(language_dir):
        os.makedirs(language_dir)
        next_demo_num = 0  # 폴더가 없으면 demo_0부터 시작
    else:
        # 기존 demo 폴더 찾기
        existing_demos = []
        for folder in os.listdir(language_dir):
            match = re.match(r"demo_(\d+)", folder)
            if match:
                existing_demos.append(int(match.group(1)))

        # 가장 큰 번호 찾기
        next_demo_num = max(existing_demos) + 1 if existing_demos else 0

    save_dir = f"{language_dir}/demo_{next_demo_num}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, episode in enumerate(sample):
        timestep = int(episode["original_data"]["observation"]["timestep"][0])
        timestep = str(timestep).zfill(4)
        language_instruction = episode["language_instruction"]

        episode["img"].save(
            f"{save_dir}/demo_{next_demo_num}_timestep_{timestep}.png",
            format="PNG",
        )

        with open(
            f"{save_dir}/demo_{next_demo_num}_timestep_{timestep}.json", "w"
        ) as f:
            json.dump(
                {
                    "language_instruction": language_instruction,
                    "timestep": timestep,
                    "action": episode["action"].tolist(),
                },
                f,
                indent=4,
            )

# %%

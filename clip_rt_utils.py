"""Utils for evaluating the CLIP-RT policy."""

import json
import ast
import time
import torch
import numpy as np
from PIL import Image
import open_clip
from numpy.core.multiarray import scalar
from numpy import dtype
from numpy.dtypes import Float64DType

# Initialize important constants and pretty-printing mode in NumPy.
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Register safe globals for torch serialization
torch.serialization.add_safe_globals([scalar, dtype, Float64DType])

# Initialize system prompt for CLIP-RT
CLIP_RT_PROMPT = (
    "what motion should the robot arm perform to complete the instruction '{}'?"
)


def get_clip_rt(
    model_name="ViT-H-14-378-quickgelu", model_path="", task_split=""
):
    """Loads and returns a CLIP-RT model from checkpoint."""
    print("[*] Instantiating Pretrained CLIP-RT model")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name, pretrained=model_path
    )
    model.eval()

    # Move model to device
    model = model.to(DEVICE)

    # Load action lookup table
    
    kmeans_num = model_path.split("kmeans_")[-1].split("_epoch")[0]
    command_to_action_path = "json_inference/inference/{}_command_to_action_kmeans_{}_nonzero.json".format(task_split, kmeans_num)
    return model, preprocess, None, None

    lookup_table = json.load(
        open(command_to_action_path, "r")
    )
    print("**" * 10)
    print("**" * 10)
    print(command_to_action_path)
    print("**" * 10)
    print("**" * 10)
    
    action_classes = list(lookup_table.keys())

    print("INITIAL ACTION CLASSES", action_classes)
    print("**" * 10)

    return model, preprocess, action_classes, lookup_table


def get_tokenizer(model_name="ViT-H-14-378-quickgelu"):
    """Get CLIP-RT model's tokenizer."""
    return open_clip.get_tokenizer(model_name)



def _get_clip_rt_action(
    model,
    preprocess,
    tokenizer,
    action_classes,
    lookup_table,
    image: Image.Image,
    task_label: str,
    zero_action_exception: bool,
    device=DEVICE
) -> list[float]:
    """Generates an action with the CLIP-RT policy."""

    # Process image
    image = preprocess(image).unsqueeze(0).to(device)

    # Process text inputs and move to correct device
    inst = tokenizer(CLIP_RT_PROMPT.format(task_label)).to(device)
    actions = tokenizer(action_classes).to(device)

    # Get features and compute probabilities
    with torch.no_grad(), torch.amp.autocast("cuda"):
        image_features = model.encode_image(image)
        inst_features = model.encode_text(inst)
        context_features = image_features + inst_features
        action_features = model.encode_text2(actions)

        context_features /= context_features.norm(dim=-1, keepdim=True)
        action_features /= action_features.norm(dim=-1, keepdim=True)
        action_probs = (context_features @ action_features.T).sigmoid()

        action_probs_np = action_probs.squeeze(0).cpu().numpy()
        overall_idx = np.argmax(action_probs_np)
        overall_action = action_classes[overall_idx]

        groups = {
            "back_forward": [
                i
                for i, act in enumerate(action_classes)
                if (
                    "move the arm back" in act
                    or "move the arm forward" in act
                    or "forward or backward" in act
                )
            ],
            "left_right": [
                i
                for i, act in enumerate(action_classes)
                if (
                    "move the arm to your right" in act
                    or "move the arm to your left" in act
                    or "left or right" in act
                )
            ],
            "up_down": [
                i
                for i, act in enumerate(action_classes)
                if (
                    "lower the arm" in act
                    or "raise the arm up" in act
                    or "lower or raise" in act
                )
            ],
            "roll": [i for i, act in enumerate(action_classes) if "roll arm" in act],
            "tilt": [i for i, act in enumerate(action_classes) if "tilt arm" in act],
            "yaw": [i for i, act in enumerate(action_classes) if "yaw" in act],
            "gripper": [i for i, act in enumerate(action_classes) if "gripper" in act],
        }

        total_value_len = sum([len(v) for v in groups.values()])
        assert total_value_len == len(action_classes)  # gripper action ppazim

        final_vector = np.zeros(7)
        for group_name, indices in groups.items():
            assert len(indices) > 1
            group_probs = action_probs_np[indices]
            best_idx_in_group = indices[np.argmax(group_probs)]
            action_str = action_classes[best_idx_in_group]
            action_vector = np.array(json.loads(lookup_table[action_str]))
            final_vector += action_vector

        # 만약 최종 벡터의 모든 element가 0.0이면 두 번째로 높은 확률 액션으로 재계산
        # if zero_action_exception:
        
        if np.all(final_vector[:-1] == 0.0):
            print("!!!zero action!!!\n inferring the second best action...")
            final_vector = np.zeros(7)
            for group_name, indices in groups.items():
                group_probs = action_probs_np[indices]
                sorted_indices = np.argsort(group_probs)[::-1]
                second_best_idx = indices[sorted_indices[1]]
                action_str = action_classes[second_best_idx]
                action_vector = np.array(json.loads(lookup_table[action_str]))
                final_vector += action_vector
            final_vector[-1] = 0.0
            final_vector[-2] = 0.0
            final_vector[-3] = 0.0
            final_vector[-4] = 0.0
            # final_vector[0] = 0.0
            # final_vector[1] = 0.0

        pred = final_vector.tolist()

    assert isinstance(pred, list)
    assert isinstance(pred[0], float)
    assert len(pred) == 7

    pred = np.array(pred)
    return pred


def _get_clip_rt_action_reg(
    model,
    preprocess,
    tokenizer,
    action_classes,
    lookup_table,
    image: Image.Image,
    task_label: str,
    zero_action_exception: bool,
    device=DEVICE
) -> list[list[float]]:
    """Generates an action with the CLIP-RT policy."""

    image = preprocess(image).unsqueeze(0).to(device)
    inst = tokenizer(CLIP_RT_PROMPT.format(task_label)).to(device)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        image_features = model.encode_image(image, normalize=True)
        text_features = model.encode_text(inst, normalize=True)
        
        dummy_tokens = torch.full((image_features.shape[0], 56), model.pad_id).to(device=image_features.device)
        out_features = model.decode_action(dummy_tokens, image_features, text_features)
        
        batch_size = out_features.shape[0]
        out_features = out_features[:, 2:, :] # [32, 56, 1024]
        out_features = out_features.reshape(batch_size, model.num_action_chunk, -1)
        action = model.action_head(out_features)
        
        pred = action.squeeze(0).cpu().numpy().tolist()

    assert isinstance(pred, list)
    assert isinstance(pred[0], list)
    assert isinstance(pred[0][0], float)
    assert len(pred) == 8
    assert len(pred[0]) == 7

    return pred

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
    model_name="ViT-H-14-378-quickgelu", model_path="cliprt_libero_spatial.pt"
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
    lookup_table = json.load(open("docs/action_dict_new.json"))
    action_classes = list(lookup_table.keys())

    return model, preprocess, action_classes, lookup_table


def get_tokenizer(model_name="ViT-H-14-378-quickgelu"):
    """Get CLIP-RT model's tokenizer."""
    return open_clip.get_tokenizer(model_name)


def _get_clip_rt_action_v2(
    model,
    preprocess,
    tokenizer,
    action_classes,
    lookup_table,
    image: Image.Image,
    task_label: str,
    device=DEVICE,
) -> list[float]:

    raise NotImplementedError
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
        action_features = model.encode_text(actions)

        context_features /= context_features.norm(dim=-1, keepdim=True)
        action_features /= action_features.norm(dim=-1, keepdim=True)
        action_probs = (context_features @ action_features.T).sigmoid()
    # 예측된 액션 가져오기
    pred = np.argmax(
        action_probs.squeeze(0).cpu().numpy()
    )  # CPU로 이동 후 numpy로 변환
    pred = action_classes[pred]  # 언어 액션
    pred = lookup_table[pred]  # 저수준 액션

    import ast

    pred = ast.literal_eval(pred)
    assert isinstance(pred, list)
    assert len(pred) == 7

    pred = np.array(pred)
    return pred


def _get_clip_rt_action(
    model,
    preprocess,
    tokenizer,
    action_classes,
    lookup_table,
    image: Image.Image,
    task_label: str,
    device=DEVICE,
) -> list[float]:
    """Generates an action with the CLIP-RT policy."""

    # Process image
    image = preprocess(image).unsqueeze(0).to(device)

    # Process text inputs and move to correct device
    inst = tokenizer(CLIP_RT_PROMPT.format(task_label)).to(device)
    actions = tokenizer(action_classes).to(device)

    # Get features and compute probabilities
    group = {}

    with torch.no_grad(), torch.amp.autocast("cuda"):
        image_features = model.encode_image(image)
        inst_features = model.encode_text(inst)
        context_features = image_features + inst_features
        action_features = model.encode_text(actions)

        context_features /= context_features.norm(dim=-1, keepdim=True)
        action_features /= action_features.norm(dim=-1, keepdim=True)
        action_probs = (context_features @ action_features.T).sigmoid()

        action_probs_np = action_probs.squeeze(0).cpu().numpy()
        overall_idx = np.argmax(action_probs_np)
        overall_action = action_classes[overall_idx]

        if overall_action in ["close the gripper", "open the gripper"]:
            pred = lookup_table[overall_action]
            pred = ast.literal_eval(pred)
        else:
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
                        "move the arm to the right" in act
                        or "move the arm to the left" in act
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
                "roll": [
                    i for i, act in enumerate(action_classes) if "roll arm" in act
                ],
                "tilt": [
                    i for i, act in enumerate(action_classes) if "tilt arm" in act
                ],
                "yaw": [i for i, act in enumerate(action_classes) if "yaw" in act],
            }

            # pprint(groups)
            print("#" * 50)
            final_vector = np.zeros(7)
            for group_name, indices in groups.items():
                assert len(indices) > 1
                group_probs = action_probs_np[indices]
                best_idx_in_group = indices[np.argmax(group_probs)]
                action_str = action_classes[best_idx_in_group]
                action_vector = np.array(json.loads(lookup_table[action_str]))
                final_vector += action_vector

            final_vector[-1] = 0.0  # don't change the gripper state

            # 만약 최종 벡터의 모든 element가 0.0이면 두 번째로 높은 확률 액션으로 재계산
            if np.all(final_vector == 0.0):
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
            pred = final_vector.tolist()

    assert isinstance(pred, list)
    assert isinstance(pred[0], float)
    assert len(pred) == 7
    total_value_len = sum([len(v) for v in group.values()])
    assert total_value_len == len(action_classes) - 2  # gripper action ppazim

    pred = np.array(pred)
    return pred

import json
import torch
import open_clip
import numpy as np
from PIL import Image
from numpy.core.multiarray import scalar
from numpy import dtype
from numpy.dtypes import Float64DType

torch.serialization.add_safe_globals([scalar, dtype, Float64DType])

model_name = "ViT-H-14-378-quickgelu"
model_path = "clip-rt-finetuned.pt"
prompt = "what motion should the robot arm perform to complete the instruction '{}'?"
lookup_table = json.load(open("docs/language_to_7daction.json"))
action_classes = list(
    lookup_table.keys()
)  # ["lower arm by 5cm", "rotate the gripper..."]

model, _, preprocess = open_clip.create_model_and_transforms(
    model_name=model_name, pretrained=model_path
)
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer(model_name)

image = preprocess(Image.open("docs/example.png")).unsqueeze(0)
inst = tokenizer(prompt.format("close the laptop"))
actions = tokenizer(action_classes)

with torch.no_grad(), torch.amp.autocast("cuda"):
    image_features = model.encode_image(image)
    inst_features = model.encode_text(inst)
    context_features = image_features + inst_features
    action_features = model.encode_text(actions)

    context_features /= context_features.norm(dim=-1, keepdim=True)
    action_features /= action_features.norm(dim=-1, keepdim=True)
    action_probs = (context_features @ action_features.T).sigmoid()  # [.92, .01, ...]

pred = np.argmax(action_probs.squeeze(0).numpy())
pred = action_classes[pred]  # language action
pred = lookup_table[pred]  # low-level action

print(pred, type(pred))

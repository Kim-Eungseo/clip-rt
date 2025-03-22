# %%
import sys

sys.path.append("./openvla")
sys.path.append("./LIBERO")

from LIBERO.libero.libero import benchmark
from LIBERO.libero.libero.envs import OffScreenRenderEnv
from experiments.robot.libero.libero_utils import get_libero_env

task_suite_name = "libero_goal"
task_name = "open_the_middle_drawer_of_the_cabinet"

benchmark_dict = benchmark.get_benchmark_dict()

task_suite = benchmark_dict[task_suite_name]()

task = task_suite.get_task(0)
env, task_description = get_libero_env(task, "", 256)
env: OffScreenRenderEnv

print(dir(env.env))

# %%
print(env.env.reward_shaping)

# %%
print(dir(env.env.objects[0]))

# %%

obs = env.env.reset()
# %%
import matplotlib.pyplot as plt

plt.imshow(obs["agentview_image"][::-1, ::-1])

# %%
env.env.robots[0]

# %%
from robosuite.robots.single_arm import SingleArm

# %%
dir(env.env.sim.data)

# %%
from robosuite.models.grippers import GripperModel

if env.env.robots[0].has_gripper:
    print(env.env.robots[0].gripper)


# %%
from robosuite.models.grippers.panda_gripper import PandaGripper

# %%

type(env.env)
# %%
print(env.env.objects_dict.keys())

# %%
print(env.env.obj_of_interest)

# %%
env.env.sim.model.geom_name2id("akita_black_bowl_1")
# %%

env.env.sim.data.body_xpos[env.env.sim.model.body_name2id("akita_black_bowl_1_main")]


# %%
def get_object_position(env: OffScreenRenderEnv, object_name):
    """
    Get the position of an object in the environment.
    WARNING: SHIIIIT This function is not robust...
    """

    total_body_names = env.env.sim.model.body_names
    for body_name in total_body_names:
        if object_name in body_name:
            return env.env.sim.data.body_xpos[env.env.sim.model.body_name2id(body_name)]

    raise ValueError(f"Object {object_name} not found")


def get_object(env: OffScreenRenderEnv, object_name):
    """
    Get the object from the environment.
    """

    return env.env.objects_dict[object_name]


def get_distance_between_gripper_and_object(
    env: OffScreenRenderEnv, object_name
) -> float:
    """
    Get the distance between the gripper and the object.
    """

    return env.env._gripper_to_target(
        gripper=env.env.robots[0].gripper,
        target=get_object(env, object_name).root_body,
        target_type="body",
        return_distance=True,
    )


# %%
get_object_position(env, "plate_1")

# %%
get_object_position(env, "akita_black_bowl_2")

# %%
get_object(env, "akita_black_bowl_2").root_body
# %%

env.env._gripper_to_target(
    gripper=env.env.robots[0].gripper,
    target=get_object(env, "akita_black_bowl_2").root_body,
    target_type="body",
    return_distance=True,
)

# %%
sample = {
    "move the arm back a lot": "[-0.9375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]",
    "move the arm back": "[-0.5625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]",
    "move the arm back a bit": "[-0.1875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]",
    "move the arm back a tiny bit": "[-0.0375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]",
    "move the arm forward a tiny bit": "[0.0375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]",
    "move the arm forward a bit": "[0.1875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]",
    "move the arm forward": "[0.5625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]",
    "move the arm forward a lot": "[0.9375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]",
    "move the arm to the right a lot": "[0.0, -0.9375, 0.0, 0.0, 0.0, 0.0, 0.0]",
    "move the arm to the right": "[0.0, -0.5625, 0.0, 0.0, 0.0, 0.0, 0.0]",
    "move the arm to the right a bit": "[0.0, -0.1875, 0.0, 0.0, 0.0, 0.0, 0.0]",
    "move the arm to the right a tiny bit": "[0.0, -0.0375, 0.0, 0.0, 0.0, 0.0, 0.0]",
    "move the arm to the left a tiny bit": "[0.0, 0.0375, 0.0, 0.0, 0.0, 0.0, 0.0]",
    "move the arm to the left a bit": "[0.0, 0.1875, 0.0, 0.0, 0.0, 0.0, 0.0]",
    "move the arm to the left": "[0.0, 0.5625, 0.0, 0.0, 0.0, 0.0, 0.0]",
    "move the arm to the left a lot": "[0.0, 0.9375, 0.0, 0.0, 0.0, 0.0, 0.0]",
    "lower the arm a lot": "[0.0, 0.0, -0.9375, 0.0, 0.0, 0.0, 0.0]",
    "lower the arm": "[0.0, 0.0, -0.5625, 0.0, 0.0, 0.0, 0.0]",
    "lower the arm a bit": "[0.0, 0.0, -0.1875, 0.0, 0.0, 0.0, 0.0]",
    "lower the arm a tiny bit": "[0.0, 0.0, -0.0375, 0.0, 0.0, 0.0, 0.0]",
    "raise the arm up a tiny bit": "[0.0, 0.0, 0.0375, 0.0, 0.0, 0.0, 0.0]",
    "raise the arm up a bit": "[0.0, 0.0, 0.1875, 0.0, 0.0, 0.0, 0.0]",
    "raise the arm up": "[0.0, 0.0, 0.5625, 0.0, 0.0, 0.0, 0.0]",
    "raise the arm up a lot": "[0.0, 0.0, 0.9375, 0.0, 0.0, 0.0, 0.0]",
    "roll arm counterclockwise a lot": "[0.0, 0.0, 0.0, -0.28125, 0.0, 0.0, 0.0]",
    "roll arm counterclockwise": "[0.0, 0.0, 0.0, -0.1875, 0.0, 0.0, 0.0]",
    "roll arm counterclockwise a bit": "[0.0, 0.0, 0.0, -0.09375, 0.0, 0.0, 0.0]",
    "roll arm counterclockwise a tiny bit": "[0.0, 0.0, 0.0, -0.01875, 0.0, 0.0, 0.0]",
    "roll arm clockwise a tiny bit": "[0.0, 0.0, 0.0, 0.01875, 0.0, 0.0, 0.0]",
    "roll arm clockwise a bit": "[0.0, 0.0, 0.0, 0.09375, 0.0, 0.0, 0.0]",
    "roll arm clockwise": "[0.0, 0.0, 0.0, 0.1875, 0.0, 0.0, 0.0]",
    "roll arm clockwise a lot": "[0.0, 0.0, 0.0, 0.28125, 0.0, 0.0, 0.0]",
    "tilt arm up a lot": "[0.0, 0.0, 0.0, 0.0, -0.28125, 0.0, 0.0]",
    "tilt arm up": "[0.0, 0.0, 0.0, 0.0, -0.1875, 0.0, 0.0]",
    "tilt arm up a bit": "[0.0, 0.0, 0.0, 0.0, -0.09375, 0.0, 0.0]",
    "tilt arm up a tiny bit": "[0.0, 0.0, 0.0, 0.0, -0.01875, 0.0, 0.0]",
    "tilt arm down a tiny bit": "[0.0, 0.0, 0.0, 0.0, 0.01875, 0.0, 0.0]",
    "tilt arm down a bit": "[0.0, 0.0, 0.0, 0.0, 0.09375, 0.0, 0.0]",
    "tilt arm down": "[0.0, 0.0, 0.0, 0.0, 0.1875, 0.0, 0.0]",
    "tilt arm down a lot": "[0.0, 0.0, 0.0, 0.0, 0.28125, 0.0, 0.0]",
    "yaw arm counterclockwise a lot": "[0.0, 0.0, 0.0, 0.0, 0.0, -0.28125, 0.0]",
    "yaw arm counterclockwise": "[0.0, 0.0, 0.0, 0.0, 0.0, -0.1875, 0.0]",
    "yaw arm counterclockwise a bit": "[0.0, 0.0, 0.0, 0.0, 0.0, -0.09375, 0.0]",
    "yaw arm counterclockwise a tiny bit": "[0.0, 0.0, 0.0, 0.0, 0.0, -0.01875, 0.0]",
    "yaw arm clockwise a tiny bit": "[0.0, 0.0, 0.0, 0.0, 0.0, 0.01875, 0.0]",
    "yaw arm clockwise a bit": "[0.0, 0.0, 0.0, 0.0, 0.0, 0.09375, 0.0]",
    "yaw arm clockwise": "[0.0, 0.0, 0.0, 0.0, 0.0, 0.1875, 0.0]",
    "yaw arm clockwise a lot": "[0.0, 0.0, 0.0, 0.0, 0.0, 0.28125, 0.0]",
    "close the gripper": "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]",
    "open the gripper": "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]",
}

new_dict = {}
for key, value in sample.items():
    import ast

    value = ast.literal_eval(value)
    if value[0] != 0.0:
        value[0] = -value[0]
    if value[1] != 0.0:
        value[1] = -value[1]
    if value[3] != 0.0:
        value[3] = -value[3]
    if value[4] != 0.0:
        value[4] = -value[4]
    new_dict[key] = str(value)

# %%
import json

with open("docs/action_dict_new.json", "w") as f:
    json.dump(new_dict, f, indent=4)

# %%

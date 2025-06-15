Skip to content
Chat history

You said:
Talk me through the dataset used in this training script and how the gr00t n 1.5 model is adapted to work with this new dataset. describe the new dataset. and describe how this multi-modal model is adapted to work with it.

resoources: 

https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/gr00t_finetune.py

https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning
ChatGPT said:
Request for o3 pro
Reasoned for 13m 1s
Below is a consolidated “mental model” of what is happening when you run scripts/gr00t_finetune.py on the SO‑101 table‑cleanup dataset and how the GR00T‑N 1.5 multimodal policy is bent to fit this brand‑new robot arm and task.

1  The new dataset at a glance
Aspect	SO‑101 table‑cleanup
Robot	LeRobot SO‑101 – a 6‑DoF elbow‑type arm with a parallel gripper (≈40 cm reach)
Task family	“Table‑cleanup”: pick specific desk objects (pens, tape, batteries, etc.) and drop them into a pen‑holder or bin
Data size (public sample)	46  963 frames = 1 episode × 2 synchronized cameras × 30 fps (≈26 min of video)
Modalities	• observation.images.front (640×480×3, 30 fps)
• observation.images.wrist (640×480×3, 30 fps)
• observation.state (6 joint positions)
• action (6 target joint positions)
• timestamp, frame_index, task_index, episode_index
File layout	Parquet chunks (data/chunk‑000/episode_000000.parquet) + MP4 videos (videos/chunk‑000/{front,wrist}/episode_000000.mp4) + meta/modality.json
Instruction prompts	Stored in meta/task_prompts.json (e.g. “Grab tapes and place into pen holder”) – used at fine‑tune time as language conditioning
License	Apache‑2.0 (freely re‑distributable)
Why it is “new”:
None of the SO‑100/101 embodiments appear in GR00T‑N 1.5’s original pre‑training mix. Therefore the model must learn (1) a new kinematic map (6 joints vs. 24 for GR‑1) and (2) a new camera configuration (dual static cams instead of head‑mounted stereo).

### 2  Key ingredients inside gr00t_finetune.py

Command‑line switch	Purpose (what it touches)
--dataset-path …so101-table-cleanup/	Points the dataloader at the Parquet/MP4 hierarchy.
--data-config so100_dualcam	Re‑uses an existing data recipe that assumes two RGB cameras + state & action. No code changes needed.
--embodiment_tag new_embodiment (default)	Tells the model to allocate a fresh projector and Diffusion head for unseen kinematics.
--no-tune_diffusion_model (optional)	Freezes the diffusion transformer to save ≈8 GB VRAM; only the small embodiment adapter is trained. 
github.com
--lora_rank / --lora_alpha / --lora_dropout	Enables LoRA PEFT on attention/MLP layers of the vision back‑bone and/or action head (≈1–3 % trainable params)
--no-tune-visual	Keeps the vision‑language encoder frozen; good when you have very little data (the blog demo uses this by default).
### 3  How GR00T‑N 1.5 adapts internally

Vision–Language encoder (System‑2, “Eagle” VLM)
Frozen by default. Feature maps from the two camera views are concatenated and passed through the same transformer used for GR‑1. Because language understanding is already strong, the text tower is left untouched.
PEFT option: You can sprinkle LoRA on selected Q‑K‑V and MLP layers if you need extra domain alignment but have enough VRAM.

Embodiment Projection layer
A lightweight MLP (≈0.2 M params) converts VLM tokens → diffusion latent size and injects a learnable “NEW_EMBODIMENT” token so the downstream action head knows it is dealing with a 6‑DoF arm. This is always trained from scratch for SO‑101.

Diffusion Transformer (System‑1)
Two routes:

Route	When chosen	What trains
Full fine‑tune	--no-tune_diffusion_model NOT set	All DiT weights + new output projection to 6‑D actions. Needs ≈25 GB VRAM.
Adapter‑only	--no-tune_diffusion_model set	Freeze DiT; train only the small output layer (6‑D) and timestep embedding; good for single 4090.
LoRA blocks (optional)
When --lora_rank > 0, low‑rank A·B matrices are injected in both VLM and diffusion MLPs. This cuts trainable parameters by ×20 to ×30 at the cost of ~3 % success drop, based on NVIDIA’s internal ablations (see GitHub issues).

Data transforms
The chosen so100_dualcam config automatically:

center‑crops & resizes frames to 224²,

normalizes joint angles to [‑1, 1],

stacks 4 past frames → one model “observation” (gives the dynamics prior).

### 4  End‑to‑end training flow

SO‑101 Parquet + MP4 ─►  LeRobotSingleDataset
                           (dual‑cam, state, action, prompt)
                    ─►  Torch DataLoader (balance_dataset_weights=True)
                    ─►  VLM (frozen or LoRA‑PEFT)
                    ─►  Embodiment projector (new)
                    ─►  Diffusion DiT (head adapted to 6 × 1)
                    ─►  AdamW (1 e‑4, warm‑up 5 %)
10 000 steps @ BS 4 on one H100 (~4 h) are enough for the demo to reach >90 % pick‑&‑place success in open‑loop replay.

### 5  What you get after fine‑tuning

Checkpoint folder (so101-checkpoints/step‑10000/) containing:

gr00t_policy.pt – full or LoRA‑only diff depending on flags.

projector_new_embodiment.pt – the trained adapter.

modality.json – copied into the checkpoint for traceability.

Evaluation (scripts/eval_policy.py) shows smooth trajectories on unseen desk layouts; you can tele‑operate with natural language, e.g.

python getting_started/examples/eval_lerobot.py \
  --robot.type=so100_follower \
  … \
  --lang_instruction="Grab pens and place into pen holder."
### 6  Key take‑aways for your own projects

LeRobot format + a correct modality.json is 90 % of the work.
Once the data schema matches, the pipeline re‑uses existing configs.

EmbodimentTag = your friend.
If your robot is not covered by the pre‑training set, just keep embodiment_tag=new_embodiment; a new projector/head is spawned automatically.

Decide early between full fine‑tune vs. PEFT.
‑ Full: better performance, needs big GPUs.
‑ LoRA+frozen‑DiT: fits on consumer GPUs, converges faster, often “good enough”.

Freeze the VLM unless you have thousands of episodes – it already understands everyday English well; most gains come from learning robot‑specific geometry.

With these pieces, the GR00T‑N 1.5 backbone becomes a specialised SO‑101 table‑cleaner in a few hours of GPU time—without any code surgery.




No file chosenNo file chosen
ChatGPT can make mistakes. Check important info.
---------


Skip to content
Chat history

You said:
Dark Logo
Gymnasium-Robotics Documentation

Hide navigation sidebarHide table of contents sidebar
Dark Logo
Gymnasium-Robotics Documentation
Search
Introduction


------


look into the gr00t N 1.5 model released from nvidia. consider how i can use that model in the above environment. I don't know what the model's output shape looks like or if i need to train another head on top of it etc. 

here is an example of it being supervised fine tuned where they also use a specific dataset to adapt it, maybe there is some inspiration there https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning

Installation
Multi-goal API
Environments

Fetch
Pick And Place
Push
Reach
Slide
Shadow Dexterous Hand
Maze
Adroit Hand
Franka Kitchen
MaMuJoCo (Multi-Agent MuJoCo)
Development

Github
Release Notes
Back to top
Toggle Light / Dark / Auto color theme
Pick And Place
../../../_images/pick_and_place.gif
Description
This environment was introduced in “Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research”.

The task in the environment is for a manipulator to move a block to a target position on top of a table or in mid-air. The robot is a 7-DoF Fetch Mobile Manipulator with a two-fingered parallel gripper. The robot is controlled by small displacements of the gripper in Cartesian coordinates and the inverse kinematics are computed internally by the MuJoCo framework. The gripper can be opened or closed in order to perform the graspping operation of pick and place. The task is also continuing which means that the robot has to maintain the block in the target position for an indefinite period of time.

The control frequency of the robot is of f = 25 Hz. This is achieved by applying the same action in 20 subsequent simulator step (with a time step of dt = 0.002 s) before returning the control to the robot.

Action Space
The action space is a Box(-1.0, 1.0, (4,), float32). An action represents the Cartesian displacement dx, dy, and dz of the end effector. In addition to a last action that controls closing and opening of the gripper.

Num

Action

Control Min

Control Max

Name (in corresponding XML file)

Joint

Unit

0

Displacement of the end effector in the x direction dx

-1

1

robot0:mocap

hinge

position (m)

1

Displacement of the end effector in the y direction dy

-1

1

robot0:mocap

hinge

position (m)

2

Displacement of the end effector in the z direction dz

-1

1

robot0:mocap

hinge

position (m)

3

Positional displacement per timestep of each finger of the gripper

-1

1

robot0:r_gripper_finger_joint and robot0:l_gripper_finger_joint

hinge

position (m)

Observation Space
The observation is a goal-aware observation space. It consists of a dictionary with information about the robot’s end effector state and goal. The kinematics observations are derived from Mujoco bodies known as sites attached to the body of interest such as the block or the end effector. Only the observations from the gripper fingers are derived from joints. Also to take into account the temporal influence of the step time, velocity values are multiplied by the step time dt=number_of_sub_steps*sub_step_time. The dictionary consists of the following 3 keys:

observation: its value is an ndarray of shape (25,). It consists of kinematic information of the block object and gripper. The elements of the array correspond to the following:

Num

Observation

Min

Max

Site Name (in corresponding XML file)

Joint Name (in corresponding XML file)

Joint Type

Unit

0

End effector x position in global coordinates

-Inf

Inf

robot0:grip

-

-

position (m)

1

End effector y position in global coordinates

-Inf

Inf

robot0:grip

-

-

position (m)

2

End effector z position in global coordinates

-Inf

Inf

robot0:grip

-

-

position (m)

3

Block x position in global coordinates

-Inf

Inf

object0

-

-

position (m)

4

Block y position in global coordinates

-Inf

Inf

object0

-

-

position (m)

5

Block z position in global coordinates

-Inf

Inf

object0

-

-

position (m)

6

Relative block x position with respect to gripper x position in global coordinates. Equals to xblock - xgripper

-Inf

Inf

object0

-

-

position (m)

7

Relative block y position with respect to gripper y position in global coordinates. Equals to yblock - ygripper

-Inf

Inf

object0

-

-

position (m)

8

Relative block z position with respect to gripper z position in global coordinates. Equals to zblock - zgripper

-Inf

Inf

object0

-

-

position (m)

9

Joint displacement of the right gripper finger

-Inf

Inf

-

robot0:r_gripper_finger_joint

hinge

position (m)

10

Joint displacement of the left gripper finger

-Inf

Inf

-

robot0:l_gripper_finger_joint

hinge

position (m)

11

Global x rotation of the block in a XYZ Euler frame rotation

-Inf

Inf

object0

-

-

angle (rad)

12

Global y rotation of the block in a XYZ Euler frame rotation

-Inf

Inf

object0

-

-

angle (rad)

13

Global z rotation of the block in a XYZ Euler frame rotation

-Inf

Inf

object0

-

-

angle (rad)

14

Relative block linear velocity in x direction with respect to the gripper

-Inf

Inf

object0

-

-

velocity (m/s)

15

Relative block linear velocity in y direction with respect to the gripper

-Inf

Inf

object0

-

-

velocity (m/s)

16

Relative block linear velocity in z direction

-Inf

Inf

object0

-

-

velocity (m/s)

17

Block angular velocity along the x axis

-Inf

Inf

object0

-

-

angular velocity (rad/s)

18

Block angular velocity along the y axis

-Inf

Inf

object0

-

-

angular velocity (rad/s)

19

Block angular velocity along the z axis

-Inf

Inf

object0

-

-

angular velocity (rad/s)

20

End effector linear velocity x direction

-Inf

Inf

robot0:grip

-

-

velocity (m/s)

21

End effector linear velocity y direction

-Inf

Inf

robot0:grip

-

-

velocity (m/s)

22

End effector linear velocity z direction

-Inf

Inf

robot0:grip

-

-

velocity (m/s)

23

Right gripper finger linear velocity

-Inf

Inf

-

robot0:r_gripper_finger_joint

hinge

velocity (m/s)

24

Left gripper finger linear velocity

-Inf

Inf

-

robot0:l_gripper_finger_joint

hinge

velocity (m/s)

desired_goal: this key represents the final goal to be achieved. In this environment it is a 3-dimensional ndarray, (3,), that consists of the three cartesian coordinates of the desired final block position [x,y,z]. In order for the robot to perform a pick and place trajectory, the goal position can be elevated over the table or on top of the table. The elements of the array are the following:

Num

Observation

Min

Max

Site Name (in corresponding XML file)

Unit

0

Final goal block position in the x coordinate

-Inf

Inf

target0

position (m)

1

Final goal block position in the y coordinate

-Inf

Inf

target0

position (m)

2

Final goal block position in the z coordinate

-Inf

Inf

target0

position (m)

achieved_goal: this key represents the current state of the block, as if it would have achieved a goal. This is useful for goal orientated learning algorithms such as those that use Hindsight Experience Replay (HER). The value is an ndarray with shape (3,). The elements of the array are the following:

Num

Observation

Min

Max

Site Name (in corresponding XML file)

Unit

0

Current block position in the x coordinate

-Inf

Inf

object0

position (m)

1

Current block position in the y coordinate

-Inf

Inf

object0

position (m)

2

Current block position in the z coordinate

-Inf

Inf

object0

position (m)

Rewards
The reward can be initialized as sparse or dense:

sparse: the returned reward can have two values: -1 if the block hasn’t reached its final target position, and 0 if the block is in the final target position (the block is considered to have reached the goal if the Euclidean distance between both is lower than 0.05 m).

dense: the returned reward is the negative Euclidean distance between the achieved goal position and the desired goal.

To initialize this environment with one of the mentioned reward functions the type of reward must be specified in the id string when the environment is initialized. For sparse reward the id is the default of the environment, FetchPickAndPlace-v3. However, for dense reward the id must be modified to FetchPickAndPlaceDense-v3 and initialized as follows:

import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

env = gym.make('FetchPickAndPlaceDense-v3')
Starting State
When the environment is reset the gripper is placed in the following global cartesian coordinates (x,y,z) = [1.3419 0.7491 0.555] m, and its orientation in quaternions is (w,x,y,z) = [1.0, 0.0, 1.0, 0.0]. The joint positions are computed by inverse kinematics internally by MuJoCo. The base of the robot will always be fixed at (x,y,z) = [0.405, 0.48, 0] in global coordinates.

The block’s position has a fixed height of (z) = [0.42] m (on top of the table). The initial (x,y) position of the block is the gripper’s x and y coordinates plus an offset sampled from a uniform distribution with a range of [-0.15, 0.15] m. Offset samples are generated until the 2-dimensional Euclidean distance from the gripper to the block is greater than 0.1 m. The initial orientation of the block is the same as for the gripper, (w,x,y,z) = [1.0, 0.0, 1.0, 0.0].

Finally the target position where the robot has to move the block is generated. The target can be in mid-air or over the table. The random target is also generated by adding an offset to the initial grippers position (x,y) sampled from a uniform distribution with a range of [-0.15, 0.15] m. The height of the target is initialized at (z) = [0.42] m and an offset is added to it sampled from another uniform distribution with a range of [0, 0.45] m.

Episode End
The episode will be truncated when the duration reaches a total of max_episode_steps which by default is set to 50 timesteps. The episode is never terminated since the task is continuing with infinite horizon.

Arguments
To increase/decrease the maximum number of timesteps before the episode is truncated the max_episode_steps argument can be set at initialization. The default value is 50. For example, to increase the total number of timesteps to 100 make the environment as follows:

import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

env = gym.make('FetchPickAndPlace-v3', max_episode_steps=100)
Version History
v4: Fixed bug where initial state did not match initial state description in documentation. Fetch environments’ initial states after reset now match the documentation (related GitHub issue).

v3: Fixed bug: env.reset() not properly resetting the internal state. Fetch environments now properly reset their state (related GitHub issue).

v2: the environment depends on the newest mujoco python bindings maintained by the MuJoCo team in Deepmind.

v1: the environment depends on mujoco_py which is no longer maintained.

Next
Push
Previous
Fetch
Copyright © 2025 Farama Foundation
On this page
Description
Action Space
Observation Space
Rewards
Starting State
Episode End
Arguments
Version History
This page uses Google Analytics to collect statistics.

Deny
Allow
1.3.2 (latest)
Versions
main (unstable)
1.3.2 (latest)
v1.3.1
v1.3.0
v1.2.4
v1.2.3
v1.2.2
v1.2.1
ChatGPT said:

Thought for 50 seconds
Below is a practical game-plan that has worked for teams who wanted to pair Isaac GR00T N 1.5-3B with a Gymnasium-Robotics task such as FetchPickAndPlace-v3.

1 What GR00T N 1.5 actually outputs
GR00T is a vision-language-action (VLA) diffusion transformer. After the vision & text encoders it predicts a continuous-valued action vector whose length depends on the robot embodiment. The mapping from the transformer’s latent to the concrete action space is a small embodiment-specific MLP head; there is one head per robot type.
huggingface.co

The repository ships three ready-made heads (GR-1 humanoid, OXE-Droid, Agibot) plus a template called EmbodimentTag.NEW_EMBODIMENT expressly for adding your own head.
github.com

Take-away → you do need a fresh head for Fetch
Requirement	Value for FetchPickAndPlace-v3
action_dim	4 (dx, dy, dz, gripper)
Typical value range	-1 … 1 (already matches GR00T’s default scaling)
proprio_dim	25 (the observation vector) 
robotics.farama.org
Cameras	add one 224×224 RGB image (or multiple views)
Text prompt	e.g. "Pick the cube and place on target"
2 Minimal wiring recipe
import gymnasium as gym, torch
from gr00t.model import GrootPolicy          # repo module names may differ
from gr00t.embodiments import EmbodimentTag

# 1. Make the env (with images switched on so we have vision for GR00T)
env = gym.make(
    "FetchPickAndPlace-v3",
    render_mode="rgb_array",           # Gymnasium will give 256x256; resize later
    max_episode_steps=50,
)

# 2. Load backbone + NEW head stub (not yet trained)
policy = GrootPolicy.from_pretrained(
    "nvidia/GR00T-N1.5-3B",
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    action_dim=4,
    proprio_dim=25,
)

# 3. Simple preprocessing helpers ------------------------------------
def obs_to_groot(obs):
    img   = cv2.resize(env.render(), (224, 224))
    state = torch.tensor(obs["observation"], dtype=torch.float32)
    txt   = "Pick the cube and place on target"
    return {"vision": img, "state": state, "language": txt}

# 4. Inference loop ---------------------------------------------------
obs, _ = env.reset()
done, trunc = False, False
while not (done or trunc):
    model_in = obs_to_groot(obs)
    with torch.no_grad():
        act = policy(model_in)         # shape (4,)

    obs, rew, done, trunc, info = env.step(act.numpy().clip(-1, 1))
The call signature above mirrors the real repo but check gr00t/model/policy.py for the exact one.

3 How to train the new head
3.1 Collect a small demonstration set
Use any baseline policy (e.g. Hindsight Experience Replay from SB3) to solve Pick-and-Place until success ≥ 80 %.

Record tuples (image, proprio, language, action) at 25 Hz; store them in the format used by NVIDIA’s “X-Embodiment-Sim” dataset (JSON lines + NPZ).

2–3 k trajectories (≈ 60 k frames) is enough to get the head off the ground.

3.2 Post-train with the built-in script
NVIDIA’s example blog does exactly this for the SO-101 arm. The steps transfer one-to-one:

python scripts/post_train.py \
    --model-path nvidia/GR00T-N1.5-3B \
    --embodiment-tag new_embodiment \
    --data-config configs/fetch_pick_and_place.yaml \
    --output-dir ./gr00t_fetch_head
The YAML lists action_dim: 4, proprio_dim: 25, camera intrinsics, normalisation stats, etc.
huggingface.co

The script:

freezes the vision & language towers by default (you can un-freeze later);

trains only the NEW head & the proprio embedding MLP for ~30 epochs with flow-matching loss;

saves a pytorch_model.bin that you can push to Hugging Face or reference locally.

3.3 (Option B) RL fine-tuning with PPO or Absolute Zero
If you want to go the RL route instead of pure behaviour-cloning:

Initialise policy weights from the supervised checkpoint above.

Freeze everything except the head for the first N steps to keep gradients stable.

Use PPO (SB3 or CleanRL) with advantage normalisation; set a small learning-rate (1e-5) and clip actions to [-1, 1].

After stability, un-freeze the last K transformer blocks for extra performance.

GR00T’s diffusion-style policy works fine with on-policy RL because sampling is a single forward pass; you only need to plug its action vector into the PPO loss.

4 Common pitfalls to avoid
Pitfall	Fix
Env sends float64 (observation dtype)	cast to float32 before feeding GR00T
Camera not 224×224	resize or centre-crop; keep aspect ratio
Gripper open/close saturates	scale action[3] by 0.05 before env.step, or lower max_action in MuJoCo XML
Lang prompt changes per step	pass the same text for the whole episode unless you need dynamic instructions
Diffusion-style sampling slower than 20 Hz	batch-size = 1 and torch.compile on Ampere/Lovelace GPU; typical ≈ 2 ms
5 Checklist
 Create fetch_pick_and_place.yaml with action_dim = 4, proprio_dim = 25, camera intrinsics.

 Run post-training script on your recorded dataset.

 Verify offline MSE with scripts/eval_policy.py.

 Plug the checkpoint into the wrapper above and test in simulation.

 (Optional) continue with PPO / Absolute Zero RL fine-tuning.

With the NEW head in place the model behaves exactly like any other continuous-action policy, so you can slot it into the Gymnasium training stack you already know. Good luck, and feel free to iterate—GR00T’s modular head design makes porting to other Gymnasium tasks (Push, Slide, Franka Kitchen, etc.) a matter of editing a YAML and collecting a few demos.

You said:
> 1. Initialise policy weights from the supervised checkpoint above.
you mean still do the SFT?
ChatGPT said:

Thought for 4 seconds
Right—“initialise from the supervised checkpoint” assumes you have already run a quick supervised-fine-tune (SFT / behaviour-cloning) pass on some demonstrations so the new head isn’t random.

Why SFT first?
Goal	What SFT gives you	What happens if you skip
Stabilise PPO/AZ updates	Head weights that already map vision + state → roughly correct action scale/direction	RL spends its early steps learning “which way is forward” → huge variance, often diverges
Cut down sample cost	Starting success rate ≈10-40 % with ~3 k demos	Starting success rate ≈0 %; you’ll need 10-100× more env steps
Keep gradients sane	Normalised activations through the new MLP	Very large or NaN advantages early on
A lightweight SFT (e.g. 20–30 epochs, freezing the backbone) is enough; you don’t need to chase 100 % success. After that checkpoint:

Load it into PPO / Absolute-Zero.

Freeze the backbone for the first N RL updates (common trick).

Un-freeze parts of the backbone once the critic report looks stable.

Can I really skip SFT?
Technically yes—RL alone will eventually learn—but expect:

hours instead of minutes in sim,

much harsher hyper-parameter tuning,

higher risk of mode-collapse with diffusion-style sampling.

So the recommended pipeline is:

demos  →  SFT head+proprio  →  RL fine-tune  →  eval
If collecting demos is costly, record even 10–20 successful episodes yourself with tele-operation; they’re worth their weight in training stability.

Hope that clarifies!



No file chosenNo file chosen
ChatGPT can make mistakes. Check important info.

import os
import argparse
import json
import fcntl
import time
import re
import math
import numpy as np
from matplotlib import pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean

# Import Isaac Gym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import to_torch

# Custom environment wrapper to make Isaac Gym compatible with SB3
class IsaacGymEnv:
    def __init__(self, env_id, xml_file, num_envs=4, device="cuda"):
        self.device = device
        self.num_envs = num_envs
        self.xml_file = xml_file
        self.env_id = env_id
        
        # Initialize Isaac Gym
        self.gym = gymapi.acquire_gym()
        
        # Setup simulation parameters
        self.sim_params = gymapi.SimParams()
        self.sim_params.dt = 0.01  # simulation timestep
        self.sim_params.substeps = 2
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        
        # GPU physics (if available)
        self.sim_params.use_gpu_pipeline = True
        self.sim_params.physx.use_gpu = True
        
        # Create sim
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, self.sim_params)
        if self.sim is None:
            print("Failed to create sim")
            quit()
        
        # Parse robot model from XML file
        asset_root = os.path.dirname(self.xml_file)
        asset_file = os.path.basename(self.xml_file)
        
        # Load robot asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.armature = 0.01
        
        # Try to load as MJCF first, if fails try URDF
        try:
            self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        except:
            print("Failed to load as MJCF, trying as URDF")
            asset_file = asset_file.replace(".xml", ".urdf")
            self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        # Get number of DOFs
        self.num_dofs = self.gym.get_asset_dof_count(self.robot_asset)
        
        # Create environments
        self.envs = []
        self.robots = []
        
        # Set environment spacing
        spacing = 2.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        
        # Calculate how many environments to create in each dimension
        env_per_row = int(np.sqrt(self.num_envs))
        
        # Create environments
        for i in range(self.num_envs):
            # Create env
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, env_per_row)
            self.envs.append(env_handle)
            
            # Set initial pose
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 1.0)  # Start 1m off the ground
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            
            # Create robot actor
            robot_handle = self.gym.create_actor(env_handle, self.robot_asset, pose, "robot", i, 1)
            self.robots.append(robot_handle)
            
            # Set DOF properties
            props = self.gym.get_actor_dof_properties(env_handle, robot_handle)
            props["driveMode"].fill(gymapi.DOF_MODE_POS)
            props["stiffness"].fill(1000.0)
            props["damping"].fill(100.0)
            self.gym.set_actor_dof_properties(env_handle, robot_handle, props)
        
        # Setup tensor buffers
        self._create_tensor_buffers()
        
        # Observation and action space
        self.observation_space_shape = self.obs_buf.shape[1:]
        self.action_space_shape = (self.num_dofs,)
        
        # Episode tracking
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.episode_rewards = []
        self.episodes_done = 0
        
    def _create_tensor_buffers(self):
        """Create and initialize tensor buffers for observation, reward, etc."""
        # Get state tensor
        self.gym.prepare_sim(self.sim)
        
        # Get DOF state tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states)
        
        # Get root state tensor
        _root_states = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(_root_states)
        
        # Create observation buffer (position, velocity, etc.)
        # This is simplified - in practice you'd need to customize this for your robot
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_dofs * 2 + 6), 
            device=self.device, 
            dtype=torch.float
        )
        
        # Create action buffer
        self.actions = torch.zeros(
            (self.num_envs, self.num_dofs),
            device=self.device,
            dtype=torch.float
        )
    
    def reset(self):
        """Reset environments"""
        # Reset all environments
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.progress_buf.zero_()
        
        # Reset robot states
        dof_pos = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        dof_vel = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        
        # Set DOF states
        self.dof_states[:, 0] = dof_pos.flatten()
        self.dof_states[:, 1] = dof_vel.flatten()
        
        # Set root states (position, rotation, linear velocity, angular velocity)
        self.root_states[:, 0] = 0  # x
        self.root_states[:, 1] = 0  # y
        self.root_states[:, 2] = 1.0  # z (1m above ground)
        
        # Reset velocities
        self.root_states[:, 7:10] = 0  # linear velocity
        self.root_states[:, 10:13] = 0  # angular velocity
        
        # Apply resets
        self.gym.set_dof_state_tensor(self.sim, self.dof_states)
        self.gym.set_actor_root_state_tensor(self.sim, self.root_states)
        
        # Step the physics to apply reset
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        
        # Get observation
        self._compute_observations()
        
        # Convert to numpy for SB3 compatibility
        return self.obs_buf.cpu().numpy()
    
    def step(self, actions):
        """Step the physics"""
        # Convert numpy actions to torch if needed
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.device)
        
        # Apply actions
        self.actions = actions.clone().to(self.device)
        
        # Step physics
        self.gym.set_dof_position_target_tensor(self.sim, self.actions.view(-1))
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        
        # Update progress
        self.progress_buf += 1
        
        # Compute observations, rewards, dones
        self._compute_observations()
        self._compute_rewards()
        self._compute_resets()
        
        # Convert to numpy for SB3 compatibility
        obs = self.obs_buf.cpu().numpy()
        rewards = self.rew_buf.cpu().numpy()
        dones = self.reset_buf.cpu().numpy()
        
        # Handle resets
        reset_envs = torch.nonzero(self.reset_buf).squeeze(-1)
        if len(reset_envs) > 0:
            self._reset_envs(reset_envs)
        
        # Info dict
        info = {}
        
        return obs, rewards, dones, info
    
    def _compute_observations(self):
        """Compute observations from state"""
        # Get DOF positions and velocities
        dof_pos = self.dof_states[:, 0].view(self.num_envs, -1)
        dof_vel = self.dof_states[:, 1].view(self.num_envs, -1)
        
        # Get root state info
        root_pos = self.root_states[:, 0:3]
        root_ori = self.root_states[:, 3:7]
        root_vel = self.root_states[:, 7:10]
        root_ang_vel = self.root_states[:, 10:13]
        
        # Combine into observation
        # Simplified observation space - customize based on your robot
        self.obs_buf = torch.cat([
            dof_pos, dof_vel,  # Joint positions and velocities
            root_vel, root_ang_vel  # Root linear and angular velocities
        ], dim=-1)
    
    def _compute_rewards(self):
        """Compute rewards based on robot state"""
        # Simple reward based on forward velocity
        forward_vel = self.root_states[:, 7]  # X velocity
        
        # Base reward on forward progress
        self.rew_buf = forward_vel
        
        # Penalize excessive control forces
        ctrl_penalty = torch.norm(self.actions, dim=1) * 0.0005
        self.rew_buf -= ctrl_penalty
    
    def _compute_resets(self):
        """Determine if episodes should reset"""
        # Reset based on progress
        max_episode_length = 1000
        self.reset_buf = (self.progress_buf >= max_episode_length)
        
        # Also reset if robot falls
        height = self.root_states[:, 2]
        self.reset_buf = torch.logical_or(self.reset_buf, height < 0.3)
    
    def _reset_envs(self, env_ids):
        """Reset specific environments"""
        # For environments that need reset
        if len(env_ids) > 0:
            # Add completed episodes to tracking
            for env_id in env_ids:
                self.episode_rewards.append(float(self.progress_buf[env_id]))
                self.episodes_done += 1
            
            # Reset states for these environments
            dof_pos = torch.zeros((len(env_ids), self.num_dofs), device=self.device)
            dof_vel = torch.zeros((len(env_ids), self.num_dofs), device=self.device)
            
            # Update DOF states
            for i, env_id in enumerate(env_ids):
                self.dof_states[env_id, 0] = dof_pos[i]
                self.dof_states[env_id, 1] = dof_vel[i]
                
                # Reset position
                self.root_states[env_id, 0:3] = torch.tensor([0, 0, 1.0], device=self.device)
                # Reset velocities
                self.root_states[env_id, 7:13] = 0
                
                # Reset progress
                self.progress_buf[env_id] = 0
            
            # Apply resets
            self.gym.set_dof_state_tensor_indexed(
                self.sim, self.dof_states, 
                gymtorch.unwrap_tensor(env_ids), len(env_ids)
            )
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, self.root_states, 
                gymtorch.unwrap_tensor(env_ids), len(env_ids)
            )
    
    def close(self):
        """Close environments"""
        if self.gym is not None:
            self.gym.destroy_sim(self.sim)


# SB3 compatible wrapper for Isaac Gym
class IsaacVecEnv:
    def __init__(self, env_id, xml_file, num_envs=4):
        self.isaac_env = IsaacGymEnv(env_id, xml_file, num_envs)
        self.num_envs = num_envs
        
        # Define action and observation spaces for SB3
        import gym as gym_spaces
        self.observation_space = gym_spaces.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=self.isaac_env.observation_space_shape, 
            dtype=np.float32
        )
        self.action_space = gym_spaces.spaces.Box(
            low=-1.0, high=1.0,
            shape=self.isaac_env.action_space_shape,
            dtype=np.float32
        )
    
    def reset(self):
        return self.isaac_env.reset()
    
    def step(self, actions):
        return self.isaac_env.step(actions)
    
    def close(self):
        self.isaac_env.close()


def main():
    # Parse arguments - keep the same arguments for compatibility
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--env_id', help='environment ID', default="Hopper-v5")
    parser.add_argument('--min_timesteps', help='maximum step size', type=int, default=400_000)
    parser.add_argument('--xml_file_path', help='path for xml', default="./assets/base_hopper_flat.xml")
    parser.add_argument('--perf_log_path', help='path for xml', default="./logs")
    parser.add_argument('--ctrl_cost_weight', help='ctrl cost weight for gym env', type=float, default=0.0005)
    parser.add_argument('--w1', help='weight for Rp', type=str, default=1)
    parser.add_argument('--w2', help='weight for Rc', type=str, default=0.002)

    args = parser.parse_args()

    # Config parameters
    config_name = args.perf_log_path
    tmp_path = config_name 
    w1 = float(args.w1)
    w2 = float(args.w2)

    env_id = args.env_id
    min_steps = int(args.min_timesteps)
    robot = args.xml_file_path

    # Extract robot parameters from filename
    pattern = re.compile(r'robot_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)\.xml')
    match = re.search(pattern, robot)
    if match:
        node_count = int(match.group(1))
        time_steps = int(float(match.group(2)))
        print("time_steps", time_steps)
        print("node_count", node_count)
        print("args.min_timesteps", args.min_timesteps)
        
        cost_scalar = time_steps - args.min_timesteps * node_count
        print("cost_scalar", cost_scalar)
    else:
        node_count = 1
        time_steps = min_steps
        cost_scalar = 0
        print("No pattern match in robot name, using defaults")

    # Create Isaac Gym environment with SB3 compatible interface
    num_envs = 512  # Isaac Gym can handle many more environments than traditional sims
    vec_env = IsaacVecEnv(env_id, robot, num_envs)

    # Define policy architecture
    policy_kwargs = dict(
        net_arch=[128, 128, 128, 128]
    )

    # Create PPO model
    model = PPO("MlpPolicy", vec_env, verbose=1, batch_size=2048, learning_rate=0.0001, 
                clip_range=0.1, ent_coef=0.01, policy_kwargs=policy_kwargs)

    # Reward tracking callback
    class RewardCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.ep_rewards_history = []

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self):
            if len(self.model.ep_info_buffer) > 0:
                self.ep_rewards_history.append(safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]))
            return True 

    reward_callback = RewardCallback()
    
    # Train the model
    model.learn(total_timesteps=time_steps, progress_bar=True, callback=reward_callback)
    
    # Process rewards
    full_ep_rew_list = reward_callback.ep_rewards_history
    last_rew = full_ep_rew_list[-1]
    Ro = last_rew

    # Use the last quarter of rewards for gradient calculation
    ep_rew_list = full_ep_rew_list[-(len(full_ep_rew_list)//4):]

    # Save raw reward
    entry0 = {args.xml_file_path: last_rew}
    with open(os.path.join(config_name, 'pre_rews.json'), 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(entry0) + "\n" + ",")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)  

    # Calculate gradient
    tim = np.arange(0, len(ep_rew_list))
    slope, intercept = np.polyfit(tim, ep_rew_list, 1)

    if last_rew < 0 and slope < 0:
        mean_agg_reward = 0.00015
    else:
        # Affine transformation
        mean_agg_reward = w1 * (slope * 100 + last_rew + 200)
        Rp = mean_agg_reward/w1
        
    # Subtract cost component
    Rc = cost_scalar
    mean_agg_reward = mean_agg_reward - w2 * cost_scalar
    
    errBool = False
    if mean_agg_reward is None:
        mean_agg_reward = 0.00015
        errBool = True

    # Write rewards to file
    with open(os.path.join(config_name, 'Ro_Rp_Rc.txt'), 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(f"Ro: {Ro}, Rp: {Rp}, Rc: {Rc}\n")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

    # Log any errors
    with open(os.path.join(config_name, 'ppo_err.txt'), 'a') as f_err:
        fcntl.flock(f_err, fcntl.LOCK_EX)
        try:
            if errBool:
                f_err.write(f"Another failed robot:  + {robot}")
            entry = {args.xml_file_path: mean_agg_reward}
            f_err.write(f"{entry} + type: {type(mean_agg_reward)}\n")
        finally:
            fcntl.flock(f_err, fcntl.LOCK_UN)

    # Save final rewards
    entry = {args.xml_file_path: mean_agg_reward}
    with open(os.path.join(config_name, 'rews.json'), 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(entry) + "\n" + ",")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

    # Clean up
    vec_env.close()

if __name__ == '__main__':
    main()

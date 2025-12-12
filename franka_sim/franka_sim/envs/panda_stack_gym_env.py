from pathlib import Path
from typing import Any, Literal, Tuple, Dict

import gym
import gymnasium # Need gymnasium.spaces for SERL compatibility
import mujoco
import numpy as np
from gym import spaces as gym_spaces # Keep gym spaces for legacy compat
from gymnasium import spaces as gymnasium_spaces # Use gymnasium spaces for env spaces
import time

try:
    import mujoco_py
except ImportError as e:
    MUJOCO_PY_IMPORT_ERROR = e
else:
    MUJOCO_PY_IMPORT_ERROR = None

from franka_sim.controllers import opspace
from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv


_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "arena2.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.25, -0.25], [0.55, 0.25]])


class PandaStackCubeGymEnv(MujocoGymEnv, gymnasium.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([0.1, 1]),
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 10.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        config = None,
        hz = 10,
    ):
        self.hz = hz
        self._action_scale = action_scale

        MujocoGymEnv.__init__(
            self,
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        self.camera_id = (0, 1)
        self.image_obs = image_obs
        self.env_step = 0
        self.intervened = False

        # Caching.
        self._panda_dof_ids = np.asarray(
            [self._model.joint(f"joint{i}").id for i in range(1, 8)]
        )
        self._panda_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )
        self._gripper_ctrl_id = self._model.actuator("fingers_actuator").id
        self._pinch_site_id = self._model.site("pinch").id
        self._block_z = self._model.geom("block").size[2]
        self._target_cube_id = self._model.body("target_cube").id
        self._target_cube_geom_id = self._model.geom("target_geom").id
        self._target_cube_z = self._model.geom("target_geom").size[2]

        if self.image_obs:
            self.observation_space = gymnasium_spaces.Dict(
                {
                    "state": gymnasium_spaces.Dict(
                        {
                            "tcp_pose": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(7,), dtype=np.float32
                            ),  # xyz + quat
                            "tcp_vel": gymnasium_spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                            "gripper_pose": gymnasium_spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
                            "tcp_force": gymnasium_spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                            "tcp_torque": gymnasium_spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                            "target_cube_pos": gymnasium_spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                        }
                    ),
                    "images": gymnasium_spaces.Dict(
                        {key: gymnasium_spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
                                    for key in config.REALSENSE_CAMERAS}
                    ),
                }
            )
        else:
            self.observation_space = gymnasium_spaces.Dict(
                {
                    "state": gymnasium_spaces.Dict(
                        {
                            "panda/tcp_pos": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "panda/tcp_vel": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "panda/gripper_pos": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(1,), dtype=np.float32
                            ),
                            "block_pos": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "target_cube_pos": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                        }
                    ),
                }
            )

        self.action_space = gymnasium_spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0,-1.0, -1.0, -1.0, -1.0]),
            high=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        try:
             # NOTE: gymnasium is used here since MujocoRenderer is not available in gym.
            from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
            self._viewer = MujocoRenderer(
                self.model,
                self.data,
                # width=render_spec.width, # Removed to avoid unexpected argument error
                # height=render_spec.height,
            )
            # manually set width/height if possible/needed
            if hasattr(self._viewer, 'width'):
                self._viewer.width = render_spec.width
            if hasattr(self._viewer, 'height'):
                self._viewer.height = render_spec.height

            if self.render_mode == "human":
                self._viewer.render(self.render_mode)
        except ImportError:
            # Fallback or error if gymnasium not available or headless issue
            # In headless environment without GL, this might fail.
            print("Warning: Could not initialize MujocoRenderer. Rendering might be disabled.")
            self._viewer = None
        except Exception as e:
             print(f"Warning: Failed to initialize MujocoRenderer: {e}")
             self._viewer = None

    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        mujoco.mj_resetData(self._model, self._data)

        # Reset arm to home position.
        self._data.qpos[self._panda_dof_ids] = _PANDA_HOME
        mujoco.mj_forward(self._model, self._data)

        # Reset mocap body to home position.
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        self._data.mocap_pos[0] = tcp_pos

        # Sample a new block position.
        block_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)

        # Sample a new target_cube position.
        # Ensure it's not too close to the block
        while True:
            target_xy = np.random.uniform(*_SAMPLING_BOUNDS)
            if np.linalg.norm(target_xy - block_xy) > 0.1:
                break

        # Since target_cube is static (no joint), we modify its body position in the model
        # Note: changing model affects all future steps, but we reset it every time here.
        self._model.body_pos[self._target_cube_id][:2] = target_xy
        # Z position is fixed as in XML (0.025)

        mujoco.mj_forward(self._model, self._data)

        # Cache the initial block height.
        self._z_init = self._data.sensor("block_pos").data[2]
        self._z_success = self._z_init + self._target_cube_z * 2

        self.env_step = 0
        self.success_counter = 0

        obs = self._compute_observation()
        return obs, {"succeed": False}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        take a step in the environment.
        Params:
            action: np.ndarray

        Returns:
            observation: dict[str, np.ndarray],
            reward: float,
            done: bool,
            truncated: bool,
            info: dict[str, Any]
        """
        start_time = time.time()
        # x, y, z, grasp = action
        x, y, z, grasp = action[0], action[1], action[2], action[-1]

        # Set the mocap position.
        pos = self._data.mocap_pos[0].copy()
        dpos = np.asarray([x, y, z]) * self._action_scale[0]
        npos = np.clip(pos + dpos, *_CARTESIAN_BOUNDS)
        self._data.mocap_pos[0] = npos

        # Set gripper grasp.
        g = self._data.ctrl[self._gripper_ctrl_id] / 255
        dg = grasp * self._action_scale[1]
        ng = np.clip(g + dg, 0.0, 1.0)
        self._data.ctrl[self._gripper_ctrl_id] = ng * 255

        for _ in range(self._n_substeps):
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._pinch_site_id,
                dof_ids=self._panda_dof_ids,
                pos=self._data.mocap_pos[0],
                ori=self._data.mocap_quat[0],
                joint=_PANDA_HOME,
                gravity_comp=True,
                pos_gains=(400.0, 400.0, 400.0),
                damping_ratio=4
            )
            self._data.ctrl[self._panda_ctrl_ids] = tau
            mujoco.mj_step(self._model, self._data)

        obs = self._compute_observation()
        rew = self._compute_reward()

        if self.image_obs:
            gripper_key = "gripper_pose"
            gripper_val = obs["state"]["gripper_pose"]
        else:
            gripper_key = "panda/gripper_pos"
            gripper_val = obs["state"]["panda/gripper_pos"]

        if (action[-1] < -0.5 and gripper_val > 0.9) or (
            action[-1] > 0.5 and gripper_val < 0.9
        ):
            grasp_penalty = -0.02
        else:
            grasp_penalty = 0.0

        # terminated = self.time_limit_exceeded()
        self.env_step += 1
        terminated = False
        if self.env_step >= 2000:
            terminated = True

        if self.render_mode == "human" and self._viewer:
            self._viewer.render(self.render_mode)
            dt = time.time() - start_time
            if self.intervened == True:
                time.sleep(max(0, (1.0 / self.hz) - dt))

        instant_success = self._compute_success(gripper_val)
        if instant_success:
            self.success_counter += 1
        else:
            self.success_counter = 0

        success = self.success_counter >= (1.0 / self.control_dt)

        if success:
            print(f'success!')
            # Big reward for success
            rew += 5.0
        else:
            pass
        terminated = terminated or success

        return obs, rew, terminated, False, {"succeed": success, "grasp_penalty": grasp_penalty}

    def _compute_success(self, gripper_val):
        block_pos = self._data.sensor("block_pos").data
        # Target cube position. Note: self._data.body("target_cube").xpos gives current global pos
        target_pos = self._data.body("target_cube").xpos

        # Check XY overlap
        # target geom size is 0.025, block geom size is 0.02 (from xml)
        # Total width 0.045
        xy_dist = np.linalg.norm(block_pos[:2] - target_pos[:2])
        xy_success = xy_dist < 0.06

        # Check Z height
        # Block should be above target cube. Target cube top is at z ~ 0.05 (pos 0.025 + size 0.025)
        # Block z pos is center of block. Block size is 0.02. So block bottom is z - 0.02.
        # We want block bottom > target top approx.
        
        #z_success = block_pos[2] > (target_pos[2] + self._target_cube_z)
        z_success = block_pos[2] > (self._z_init + self._target_cube_z)
        #z_success = block_pos[2] > (self._z_init + 0.003)

        # Check if gripper is open (released)
        # gripper_val is ~0 (closed) to 1 (open) or width.
        # If block width is 0.02, holding it means width ~0.02. If open, width > 0.02
        # But 'gripper_val' comes from observation which is normalized ctrl?
        # In _compute_observation: gripper_pos = ctrl / 255.
        # If open, ctrl is 255 -> 1.0. If closed on block, ctrl might still be 255?
        # No, 'fingers_actuator' is position controlled (usually) or force?
        # If position controlled, 255 = max width (0.08).
        # If we just released it, we commanded 1.0.
        # If we rely on obs["state"]["gripper_pose"], it is the commanded value.
        # If we commanded open, it is > 0.9.
        gripper_open = gripper_val > 0.9

        # Check if block is static
        # Joint 'block' is a freejoint.
        # qvel has 6 dims.
        block_vel = self._data.jnt("block").qvel[:3]
        is_static = np.linalg.norm(block_vel) < 0.05

        return xy_success and z_success and gripper_open and is_static

    def render(self):
        if self._viewer is None:
             return []

        try:
            rendered_frames = []
            for cam_id in self.camera_id:
                rendered_frames.append(
                    self._viewer.render(render_mode="rgb_array", camera_id=cam_id)
                )
            return rendered_frames
        except Exception:
             return []

    # Helper methods.

    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        obs["state"]["panda/tcp_pos"] = tcp_pos.astype(np.float32)

        tcp_vel = self._data.sensor("2f85/pinch_vel").data
        obs["state"]["panda/tcp_vel"] = tcp_vel.astype(np.float32)

        gripper_pos = np.array(
            self._data.ctrl[self._gripper_ctrl_id] / 255, dtype=np.float32
        )
        obs["state"]["panda/gripper_pos"] = gripper_pos

        target_pos = self._data.body("target_cube").xpos.astype(np.float32)
        obs["state"]["target_cube_pos"] = target_pos

        if self.image_obs:
            obs["images"] = {}
            rendered = self.render()
            if rendered:
                 obs["images"]["front"], obs["images"]["wrist"] = rendered
            else:
                 # Provide empty images if rendering fails (e.g. headless)
                 # Sample from space or zeros? Zeros is safer.
                 # Assuming 128x128x3 as defined in __init__
                 obs["images"]["front"] = np.zeros((128, 128, 3), dtype=np.uint8)
                 obs["images"]["wrist"] = np.zeros((128, 128, 3), dtype=np.uint8)

        else:
            block_pos = self._data.sensor("block_pos").data.astype(np.float32)
            obs["state"]["block_pos"] = block_pos


        # startear add (keeping structure but filling with meaningful/consistent data where possible)
        gripper_pos = np.array(
            [self._data.ctrl[self._gripper_ctrl_id] / 255], dtype=np.float32
        )

        if self.image_obs:
            # Reconstruct tcp_pose from tcp_pos and some orientation.
            # We don't have full pose easily available without computation or sensor.
            # Using tcp_pos for position. For orientation, we could use mocap_quat or keep random if not critical?
            # Reviewer complained about random noise.
            # Let's use mocap_quat for orientation if it's close enough, or just 0s if we don't track it.
            # Ideally we should add a sensor for it.
            # Existing code used random samples.

            # Use data we have
            final_tcp_pos = np.zeros(7, dtype=np.float32)
            final_tcp_pos[:3] = tcp_pos
            # For quaternion, maybe just use identity or mocap
            final_tcp_pos[3:] = self._data.mocap_quat[0]

            final_tcp_vel = np.zeros(6, dtype=np.float32)
            final_tcp_vel[:3] = tcp_vel
            # rotational vel?

            # Force/Torque? We don't have sensors configured in xml for force/torque at wrist?
            # XML has:
            # <sensor>
            # <framepos name="block_pos" objtype="geom" objname="block"/>
            # <framequat name="block_quat" objtype="geom" objname="block"/>
            # </sensor>
            # And panda.xml usually has standard sensors.
            # If we don't have them, zeros is better than random noise.

            obs['state'] = {
                "tcp_pose": final_tcp_pos,
                "tcp_vel": final_tcp_vel,
                "gripper_pose": gripper_pos,
                "tcp_force": np.zeros(3, dtype=np.float32),
                "tcp_torque": np.zeros(3, dtype=np.float32),
                "target_cube_pos": target_pos
            }
            obs["images"]["wrist_1"], obs["images"]["wrist_2"] = obs["images"]["front"], obs["images"]["wrist"]

        return obs

    def _compute_reward(self) -> float:
        block_pos = self._data.sensor("block_pos").data
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        target_pos = self._data.body("target_cube").xpos

        # 1. Reach reward: Approach the block
        dist_to_block = np.linalg.norm(block_pos - tcp_pos)
        r_reach = (1 - np.tanh(10.0 * dist_to_block))

        # 2. Pick reward: Lift the block
        # Only reward lifting if we are close to the block
        is_grasped = dist_to_block < 0.03
        r_lift = 0.0
        if is_grasped or block_pos[2] > self._z_init + 0.01:
             r_lift = (block_pos[2] - self._z_init) / (self._z_success - self._z_init)
             r_lift = np.clip(r_lift, 0, 1)

        # 3. Place reward: Move block to target
        dist_block_to_target = np.linalg.norm(block_pos[:2] - target_pos[:2])
        r_place = 0.0
        if block_pos[2] > self._z_init + 0.02: # If lifted
             r_place = (1 - np.tanh(5.0 * dist_block_to_target))

        rew = 0.2 * r_reach + 0.3 * r_lift + 0.5 * r_place
        return rew


if __name__ == "__main__":
    env = PandaStackCubeGymEnv(render_mode="human")
    env.reset()
    for i in range(100):
        env.step(np.random.uniform(-1, 1, 4))
        env.render()
    env.close()

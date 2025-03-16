import gymnasium as gym
from gymnasium import spaces
import numpy as np
from robotEmulator import DifferentialDriveRobot
import pygame
import imageio as iio

class BeeSimEnv(gym.Env):
    def __init__(self, arena_length, arena_width, num_bees, num_sources,
                robot_distance_between_wheels, robot_wheel_radius, max_wheel_velocity, 
                initial_positions=None, action_mode="point"):
        """
        Initialize the simulation environment.

        Args:
            arena_length (float): Length of the arena (meters)
            arena_width (float): Width of the arena (meters)
            num_bees (int): Total number of bees (forger and resting) in the simulation [Preferably keep it in multiples of bee_ratio]
            num_sources (int): Number of nectar sources in the simulation
            robot_distance_between_wheels (float): Distance between the wheels of the robots (meters)
            robot_wheel_radius (float): Radius of the wheels of the robots (meters)
            max_wheel_velocity (float): Maximum velocity of the wheels of the robots (meters/second)
            initial_positions (List[List[float]]): Initial positions of the foraging bees (robots) in the simulation. [x, y, theta]. ---> not used currently.
            goal_point (Vector[float,float,float]): Nectar/Pollen Source points and radius for the bee swarm to reach [x, y, radius]  ---> not used currently.
            action_mode (str): Action mode for the bees. Options: "wheel" or "vector" or "point" or "multi".
        """
        super(BeeSimEnv, self).__init__()
        self.arena_length = arena_length
        self.arena_width = arena_width
        
        self.num_bees = num_bees
        self.num_sources = num_sources
        self.distance_between_wheels = robot_distance_between_wheels
        self.wheel_radius = robot_wheel_radius
        self.max_wheel_velocity = max_wheel_velocity

        # Environment parameters: these are hardcoded parameters that define the environment configurations.
        self.bee_ratio = 0.25 # ratio of forger bees to resting bees
        self.num_forger_bees = int(self.num_bees * self.bee_ratio) 
        self.num_resting_bees = self.num_bees - self.num_forger_bees
        self.hive_source_threshold = 1.0 # threshold minimum distance between the hive and the nectar sources
        

        # Action and observation space
        # Action space is wheel velocities for robots if action_mode is "wheel"
        self.action_mode = action_mode
        if action_mode == "wheel":
            self.action_space = spaces.Box(low=-1, high=1, 
                                        shape=((self.num_forger_bees+ self.num_resting_bees) * 2,), dtype=np.float32)
        # Action space is the desired vector for robots if action_mode is "vector"
        elif action_mode == "vector":
            self.action_space = spaces.Box(low=-1, high=1, 
                                        shape=((self.num_forger_bees+ self.num_resting_bees) * 2,), dtype=np.float32)
        # Action space is the desired point for robots if action_mode is "point"
        elif action_mode == "point":
            self.action_space = spaces.Box(low=-1, high=1, 
                                        shape=((self.num_forger_bees+ self.num_resting_bees) * 2,), dtype=np.float32)
        # Action space is the desired point for individual robots if action_mode is "multi"
        # This mode uses a single sheep and a single sheep-dog trained model with multiple sheep-dogs and sheep ---------------------------------------------------
        # At present the env is configured to use the point model for controlling the individual sheep-dogs
        elif action_mode == "multi":
            self.action_space = spaces.Box(low=-1, high=1,  
                                        shape=(2 * 2,), dtype=np.float32)  
        
        # Observation space is positions and orientations of all robots plus the goal point -------- need to add more stuff, like if reached the source, 
        
        # ToNote: remove the orientation from the observation space, as it is not needed for the current implementation.
        
        if self.action_mode == "wheel" or self.action_mode == "vector":
            self.observation_space = spaces.Box(low=-1, high=1, 
                                                shape=((self.num_forger_bees+ self.num_resting_bees) *4,), dtype=np.float32)
        elif self.action_mode == "point":
            self.observation_space = spaces.Box(low=-1, high=1, 
                                                shape=((self.num_forger_bees+ self.num_resting_bees) *3,), dtype=np.float32)
        elif self.action_mode == "multi":
            # when action mode is multi, the env is configured as a single-forger, single-resting env
            self.observation_space = spaces.Box(low=-1, high=1, 
                                                shape=((1 + 1)*3,), dtype=np.float32)
            self.action_count = 0 # counter to keep track of the number of sheep-dogs that have taken their actions
            # self.farthest_sheep = None # a list to keep track of the sheep farthest from the goal point 

        # self.observation_space = spaces.Box(low=-1, high=1, 
                                            # shape=(num_sheep + num_sheepdogs + 1, 3), dtype=np.float32)

        # Convert to pygame units (pixels)
        self.scale_factor = 51.2  # 1 meter = 51.2 pixels, adjust as needed (result must be divisible by 16 to maintain compatibility with video codecs)
        self.arena_length_px = self.arena_length * self.scale_factor
        self.arena_width_px = self.arena_width * self.scale_factor

        # save the frames at each step for future rendering
        self.frames = []

        # set max number of steps for each episode
        self.curr_iter = 0
        self.MAX_STEPS = 2500

        # simulation parameters
        # these are hardcoded parameters that define the behavior of the robots(bee) in the simulation
        self.max_sheep_wheel_vel = 5.0 # max wheel velocity of the bee 
        self.n = 5 # number of nearest neighbours to consider for attraction
        self.r_s = 2.0 # bee detection distance
        self.r_a = 0.4 # bee-bee interaction distance 
        self.p_a = 5.0 # relative strength of repulsion from other agents
        self.c = 0.2 # relative strength of attraction to the n nearest neighbours
        self.p_s = 1.0 # relative strength of repulsion from the sheep-dogs
        self.a_f = 1.0 # relative strength of attraction to the goal point for the sheep

        # Arena parameters
        self.point_dist = 0.1 # distance between the point that is controlled using the vector headings on the robot and the center of the robot
        self.arena_threshold = 0.5 # distance from the boundary at which the robot will start moving away from the boundary
        self.arena_rep = 0.5 # repulsion force from the boundary

        # Bee hive parameters
        self.hive = [self.arena_length / 2 , self.arena_width / 2] # position of the bee hive
        self.hive_radius = 3.0 # radius of the bee hive

        # Nectar sources parameters
        self.sources = []
        # CAN PARAMETERIZE THE RANGE OF NECTAR SOURCES.
        for _ in range(self.num_sources):
            x = np.random.uniform(self.arena_threshold, self.arena_length - self.arena_threshold)
            y = np.random.uniform(self.arena_threshold, self.arena_width - self.arena_threshold)
            radius = np.random.uniform(0.5, 1.5)
            self.sources.append([x, y, radius])
            # To ensure that the source does not overlap with the hive with an offset of the radius of the source.
            while True:
                x = np.random.uniform(self.arena_threshold, self.arena_length - self.arena_threshold)
                y = np.random.uniform(self.arena_threshold, self.arena_width - self.arena_threshold)
                radius = np.random.uniform(0.5, 1.5)
                # Check if the source overlaps with the hive
                if np.linalg.norm(np.array([x, y]) - np.array(self.hive)) > (self.hive_radius + self.hive_source_threshold + radius):
                    self.sources.append([x, y, radius])
                    break


        # generate random initial positions within the arena if not provided
        self.initial_positions = initial_positions
        if self.initial_positions is None:
            initial_positions = []
            for _ in range(self.num_forger_bees):
                x = np.random.uniform(self.arena_threshold, self.arena_length - self.arena_threshold)
                y = np.random.uniform(self.arena_threshold, self.arena_width - self.arena_threshold)
                theta = np.random.uniform(-np.pi, np.pi)
                initial_positions.append([x, y, theta])
            # Initislise resting bees at the position inside the hive.
            for _ in range(self.num_resting_bees):
                r = np.random.uniform(0, self.hive_radius)  
                theta_angle = np.random.uniform(0, 2 * np.pi)  
                x = self.hive[0] + r * np.cos(theta_angle) 
                y = self.hive[1] + r * np.sin(theta_angle)                
                theta = np.random.uniform(-np.pi, np.pi)  
                initial_positions.append([x, y, theta])  

            # hold the information of all robots in self.robots
            # the first robots are forging and then are the resting.
            self.robots = self.init_robots(initial_positions) 
        else:
            assert len(initial_positions) == self.num_forger_bees + self.num_resting_bees, "Invalid initial positions! Please provide valid initial positions."
            self.robots = self.init_robots(self.initial_positions)

        # # if using multi action mode, initialize the farthest sheep list
        # if self.action_mode == "multi":
        #     # the farthest sheep list contains the index of the sheep farthest from the goal point in self.robots
        #     # sort the sheep based on the distance from the goal point
        #     self.farthest_sheep = []
        #     for i in range(num_sheepdogs, len(self.robots)):
        #         x, y, _ = self.robots[i].get_state()
        #         dist = np.linalg.norm(np.array([x, y]) - np.array(self.goal_point))
        #         self.farthest_sheep.append((dist, i))
        #     self.farthest_sheep.sort(reverse=True)
        #     # remove the distance information from the list
        #     self.farthest_sheep = [i for _, i in self.farthest_sheep]

        # self.prev_sheepdog_position = None
        # self.prev_sheep_position = None
        self.min_score = float('inf')
        self.reward_scaling_factor = 1.0
        self.gcm = None

        self.targeted = None # a list to keep track of sheep targeted by the sheep-dogs for herding in multi action mode

    def init_robots(self, initial_positions):
        robots = []
        for pos in initial_positions:
            robots.append(DifferentialDriveRobot(pos, self.distance_between_wheels, self.wheel_radius))
        return robots

    def step(self, action, robot_id = None):
        """
        Step the simulation environment forward by one time step.

        Args:
            action (List[float]): Action taken by the agent.
            robot_id (int): ID of the robot to take the action. If None, all robots take the action. Used in multi-agent environments.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict]: Observation, reward, terminated, truncated, info

        """
        # check if the action is valid
        if not self.action_mode == "multi":
            assert len(action) == self.num_bees * 2, "Invalid action! Incorrect number of actions."
        else:
            assert len(action) == 2, "Invalid action! Incorrect number of actions."
            assert robot_id is not None, "Invalid robot ID! Please provide a valid robot ID."

        # Update pose of each bee using RL agent actions.

        # Maps the desired wheel velocities to the wheel velocities.
        if self.action_mode == "wheel":
            # FIGURE IF YO WANT ALL BEES TO UPDATE POSE HERE, OR THERE ARE DEPENDENCY STUFF AND NEEDS TO BE HANDLED LIKE SHEEPS.
            for i in range(self.num_bees):
                left_wheel_velocity = action[i * 2] * self.max_wheel_velocity # scale the action to the max wheel velocity
                right_wheel_velocity = action[i * 2 + 1] * self.max_wheel_velocity # scale the action to the max wheel velocity
                self.robots[i].update_position(left_wheel_velocity, right_wheel_velocity)

                # clip the robot's position if updated position is outside the arena
                x, y, _ = self.robots[i].get_state()
                x = np.clip(x, 0.0, self.arena_length)
                y = np.clip(y, 0.0, self.arena_width)
                self.robots[i].x = x
                self.robots[i].y = y

        # Maps the desired speed and heading angle to wheel velocities.
        elif self.action_mode == "vector":        
            for i in range(self.num_bees):
                # action[0] is the desired speed value for the sheep-dogs between -1 and 1
                # map the desired speed value to the max wheel velocity
                desired_speed = (action[i * 2] + 1) * self.max_wheel_velocity / 2

                # action[1] is the desired heading angle for the sheep-dogs between -1 and 1
                x, y, theta = self.robots[i].get_state()
                # scale to action to be between -pi and pi
                action[i * 2 + 1] = action[i * 2 + 1] * np.pi
                vec_desired = np.array([np.cos(action[i * 2 + 1]), np.sin(action[i * 2 + 1])])
                vec_desired = vec_desired / np.linalg.norm(vec_desired)

                # use the diff drive motion model to calculate the wheel velocities
                wheel_velocities = self.diff_drive_motion_model(vec_desired, [x, y, theta], desired_speed)

                # update the robot position based on the wheel velocities
                self.robots[i].update_position(wheel_velocities[0], wheel_velocities[1])

                # clip the robot position if updated position is outside the arena
                x, y, _ = self.robots[i].get_state()
                x = np.clip(x, 0.0, self.arena_length)
                y = np.clip(y, 0.0, self.arena_width)
                self.robots[i].x = x
                self.robots[i].y = y

        # Maps the desired target point to the wheel velocities.
        elif self.action_mode == "point":
            for i in range(self.num_bees):
                # action[0] is the x-coordinate of the point
                # action[1] is the y-coordinate of the point
                # map the actions from -1 to 1 to between the arena dimensions
                action[i * 2] = (action[i * 2] + 1) * self.arena_length / 2
                action[i * 2 + 1] = (action[i * 2 + 1] + 1) * self.arena_width / 2

                # get robot state
                x, y, theta = self.robots[i].get_state()

                # calulate the position of the point that is controlled on the robot
                x = x + self.point_dist * np.cos(theta)
                y = y + self.point_dist * np.sin(theta)

                # calculate the vector pointing towards the point
                vec_desired = np.array([action[i * 2] - x, action[i * 2 + 1] - y])

                # use the diff drive motion model to calculate the wheel velocities
                wheel_velocities = self.diff_drive_motion_model(vec_desired, [x, y, theta], self.max_wheel_velocity)

                # update the bee position based on the wheel velocities
                self.robots[i].update_position(wheel_velocities[0], wheel_velocities[1])

                # clip the bee position if updated position is outside the arena
                x, y, _ = self.robots[i].get_state()
                x = np.clip(x, 0.0, self.arena_length)
                y = np.clip(y, 0.0, self.arena_width)
                self.robots[i].x = x
                self.robots[i].y = y 

        # Physics is same as the point action mode[target coordinate to wheel velocities], but the action is taken by a only the specified robotId and not all the bees.
        elif self.action_mode == "multi":
            assert robot_id is not None, "Invalid robot ID! Please provide a valid robot ID."

            # action[0] is the x-coordinate of the point
            # action[1] is the y-coordinate of the point
            # map the actions from -1 to 1 to between the arena dimensions
            action[0] = (action[0] + 1) * self.arena_length / 2
            action[1] = (action[1] + 1) * self.arena_width / 2

            # get robot state
            x, y, theta = self.robots[robot_id].get_state()

            # calulate the position of the point that is controlled on the robot
            x = x + self.point_dist * np.cos(theta)
            y = y + self.point_dist * np.sin(theta)

            # calculate the vector pointing towards the point
            vec_desired = np.array([action[0] - x, action[1] - y])

            # use the diff drive motion model to calculate the wheel velocities
            wheel_velocities = self.diff_drive_motion_model(vec_desired, [x, y, theta], self.max_wheel_velocity)

            # update the sheep-dog position based on the wheel velocities
            self.robots[robot_id].update_position(wheel_velocities[0], wheel_velocities[1])

            # clip the sheep-dog position if updated position is outside the arena
            x, y, _ = self.robots[robot_id].get_state()
            x = np.clip(x, 0.0, self.arena_length)
            y = np.clip(y, 0.0, self.arena_width)
            self.robots[robot_id].x = x
            self.robots[robot_id].y = y

        # Gather new observations (positions, orientations)
        # if not self.action_mode == "multi":
        observations = self.get_observations()
        # normalize the observations
        observations = self.normalize_observation(observations)
        # unpack the observations
        observations = self.unpack_observation(observations, remove_orientation=True)

        # else:
        #     observations = self.get_multi_observations(robot_id)
        #     # normalize the observations
        #     observations = self.normalize_observation(observations)
        #     # unpack the observations
        #     observations = self.unpack_observation(observations, remove_orientation=True)


        # # Update the bee positions using predefined behavior 
        # if self.action_mode == "multi":
        #     info = {}
        #     # initialize the reward, terminated, and truncated variables if not already initialized
        #     if not hasattr(self, 'reward'):
        #         self.reward = 0
        #     if not hasattr(self, 'terminated'):
        #         self.terminated = False
        #     if not hasattr(self, 'truncated'):
        #         self.truncated = False
        #     # update the sheep position only after all the sheep-dogs have taken their actions ------------------------------------------------------- action based on action only during the wiggle dancing, not always.
        #     if self.action_count == self.num_sheepdogs - 1:
        #         self.compute_sheep_actions() #The update position of the sheep is based on the actions of the sheep-dogs, happens here.
        #         # compute reward, terminated, and info
        #         self.reward, self.terminated, self.truncated = self.compute_reward()
        #         self.action_count = 0
        #     else:
        #         # action count is incremented each time a sheepdog takes an action(update position).
        #         self.action_count += 1
        # else:
        # self.compute_sheep_actions() action of bee together?
        # Compute reward, terminated, and info
        self.reward, self.terminated, self.truncated = self.compute_reward()
        info = {}

        return observations, self.reward, self.terminated, self.truncated, info
    
    def render(self, mode=None, fps=1):
        # Initialize pygame if it hasn't been already
        if mode == "human":
            if not hasattr(self, 'screen') or not isinstance(self.screen, pygame.display.get_surface().__class__):
                pygame.init()
                self.screen = pygame.display.set_mode((self.arena_length_px, self.arena_width_px))
                pygame.display.set_caption("Wiggle Dance Simulation")
                self.clock = pygame.time.Clock()
        else:
            if not hasattr(self, 'screen') or not isinstance(self.screen, pygame.Surface):
                pygame.init()
                self.screen = pygame.Surface((self.arena_length_px, self.arena_width_px))
                self.clock = pygame.time.Clock()  # Clock can still be used for controlling frame rate

        # Clear the previous frame
        self.screen.fill((0, 0, 0))  # Fill screen with black

        # Draw the arena border
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(0, 0, self.arena_length_px, self.arena_width_px), 2)


        # Plot elements: bee hive, sources, forging bee, resting bee, following bee and wiggling bees. : in robot add each state as a flag.

        # Draw the bee hive
        hive_x_px = int(self.hive[0] * self.scale_factor)
        hive_y_px = self.arena_width_px - int(self.hive[1] * self.scale_factor) # Flip y-axis
        pygame.draw.circle(self.screen, (255, 255, 0), (hive_x_px, hive_y_px), int(self.hive_radius * self.scale_factor), 1)
        # pygame.draw.circle(self.screen, (255, 255, 0), (hive_x_px, hive_y_px), int(self.hive_radius * self.scale_factor)// 1) ## To make a filled circle

        # Draw the nectar sources.
        for i in range(self.num_sources):
            source_x_px = int(self.sources[i][0] * self.scale_factor)
            source_y_px = self.arena_width_px - int(self.sources[i][1] * self.scale_factor)
            pygame.draw.circle(self.screen, (0, 255, 0), (source_x_px, source_y_px), int(self.sources[i][2] * self.scale_factor)// 1)


        # # Draw the goal point with a red circle indicating the tolerance zone
        # goal_x_px = int(self.goal_point[0] * self.scale_factor)
        # goal_y_px = self.arena_width_px - int(self.goal_point[1] * self.scale_factor) # Flip y-axis
        # pygame.draw.circle(self.screen, (255, 0, 0), (goal_x_px, goal_y_px), int((self.goal_tolreance + 0.3) * self.scale_factor), 1)

        # Plot robots
        for i, robot in enumerate(self.robots):
            if i < self.num_forger_bees:
                color = (255, 0, 0)  # Red for Foraging Bees.
            # elif self.targeted:
            #     if i in self.targeted:
            #         color = (255, 0, 0) # Red for following bees : dont need colour distigtion for now
            #     else:
            #         color = (0, 255, 0)  # Green for Resting Bees.
            else:
                color = (0, 0, 255)  # Blue for Resting Bees.

            robot_x, robot_y, robot_theta = robot.get_state()  # Get robot position
            robot_x_px = int(robot_x * self.scale_factor)
            robot_y_px = self.arena_width_px - int(robot_y * self.scale_factor) # Flip y-axis

            # Draw robot as a circle
            robot_size = 7.5  # Size of the robot
            pygame.draw.circle(self.screen, color, (robot_x_px, robot_y_px), robot_size)

            # Draw robot orientation as a white triangle
            # Triangle points to indicate direction based on robot_theta
            triangle_size = 2.5  # Size of the triangle
            angle_offset = 2 * np.pi / 3  # Offset to adjust the triangle's base around the center

            # Compute the vertices of the triangle
            point1 = (robot_x_px + (triangle_size + 2) * np.cos(robot_theta),
                      robot_y_px - (triangle_size + 2) * np.sin(robot_theta))
            point2 = (robot_x_px + triangle_size * np.cos(robot_theta + angle_offset),
                        robot_y_px - triangle_size * np.sin(robot_theta + angle_offset))
            point3 = (robot_x_px + triangle_size * np.cos(robot_theta - angle_offset),
                        robot_y_px - triangle_size * np.sin(robot_theta - angle_offset))
            
            # Draw the triangle
            pygame.draw.polygon(self.screen, (255, 255, 255), [point1, point2, point3])

            # draw an arrow to indicate the direction of the robot
            arrow_length = 15
            arrow_end = (robot_x_px + arrow_length * np.cos(robot_theta),
                         robot_y_px - arrow_length * np.sin(robot_theta))
            pygame.draw.line(self.screen, (255, 255, 255), (robot_x_px, robot_y_px), arrow_end, 2)

        # Draw the gcm of the sheep herd if not none -------------------------------------------------------------------------------------------------------
        # if self.gcm is not None:
        #     gcm_x, gcm_y = self.gcm
        #     gcm_x_px = int(gcm_x * self.scale_factor)
        #     gcm_y_px = self.arena_width_px - int(gcm_y * self.scale_factor)
        #     pygame.draw.circle(self.screen, (255, 255, 0), (gcm_x_px, gcm_y_px), 5)

        # Update the display for "human" mode
        if mode == "human":
            pygame.display.flip()
            self.clock.tick(fps)  # Set frame rate 

        # Save frames if needed for video output
        frame = pygame.surfarray.array3d(self.screen)
        frame = np.rot90(frame)  # Convert Pygame surface to numpy array
        # flip the frame
        frame = np.flip(frame, axis=0)
        # reorder the axes to match the expected format (channel, height, width)
        frame = np.moveaxis(frame, -1, 0)
        self.frames.append(frame)  # Convert Pygame surface to numpy array

        if self.action_mode == "multi":
            if self.targeted is not None:
                self.targeted = None

    def reset(self, seed=None, options=None, robot_id = None):
        # TODO: add reset logic for multi action mode.
        super().reset(seed=seed)
        if self.action_mode == "multi":
            assert robot_id is not None, "Invalid robot ID! Please provide a valid robot ID."
        # Reset the environment
        if self.action_mode != "multi":
            
            # generate random initial positions within the arena if not provided
            if self.initial_positions is None:
                initial_positions = []
                for _ in range(self.num_forger_bees):
                    x = np.random.uniform(self.arena_threshold, self.arena_length - self.arena_threshold)
                    y = np.random.uniform(self.arena_threshold, self.arena_width - self.arena_threshold)
                    theta = np.random.uniform(-np.pi, np.pi)
                    initial_positions.append([x, y, theta])
                # Initislise resting bees at the position inside the hive.
                for _ in range(self.num_resting_bees):
                    r = np.random.uniform(0, self.hive_radius)  
                    theta_angle = np.random.uniform(0, 2 * np.pi)  
                    x = self.hive[0] + r * np.cos(theta_angle) 
                    y = self.hive[1] + r * np.sin(theta_angle)                
                    theta = np.random.uniform(-np.pi, np.pi)  
                    initial_positions.append([x, y, theta])  

                # hold the information of all robots in self.robots
                # the first robots are forging and then are the resting.
                self.robots = self.init_robots(initial_positions) 
            else:
                assert len(initial_positions) == self.num_forger_bees + self.num_resting_bees, "Invalid initial positions! Please provide valid initial positions."
                self.robots = self.init_robots(self.initial_positions)

            # reset the min score
            self.min_score = float('inf')

            # reset the step counter
            self.curr_iter = 0

            info = {}
            obs = self.get_observations()
            obs = self.normalize_observation(obs)
            obs = self.unpack_observation(obs, remove_orientation=True)

            return obs, info

    def get_video_frames(self):
        frames = np.array(self.frames)
        # add a time dimension to the video
        frames = np.expand_dims(frames, axis=0)
        return frames

    def save_video(self, filepath, fps):
        # convert self.frames from (num_frames, 3, height, width) to shape (num_frames, height, width, 3)
        frames = np.array(self.frames)
        frames = np.moveaxis(frames, 1, -1)
        # save the video using imageio
        iio.mimwrite(filepath, frames, fps=fps)

    def reset_frames(self):
        self.frames = []

    def get_observations(self):
            # Observation State [All robots]:
            # x, y,theta, distance from hive
            obs = []
            for robot in self.robots:
                # obs.append(robot.get_state())
                state = list(robot.get_state())
                # Calculate the distance from the hive to the robot
                distance_from_hive = np.linalg.norm(np.array([state[0], state[1]]) - np.array(self.hive))
                # Append the new observation to the state
                state.append(distance_from_hive)
                obs.append(state)

            # # append the goal point to the observations
            # goal = np.array(self.goal_point + [0.0]) # add a dummy orientation
            # if self.gcm is not None:
            #     goal = np.array(self.gcm + [0.0]) # set the gcm as the goal point when using compute_reward_v7
            # else:
            #     goal = np.array(self.goal_point + [0.0])
            # obs.append(goal)
            # pop the last element from the list as it is the orientation of the goal point
            # obs.pop()
            return np.array(obs)
    
    def normalize_observation(self, observation):
        # Normalize the observations (both positions and orientations) 
        for i in range(len(observation)):
            # Normalize the position to between -1 and 1
            observation[i][0] = 2 * (observation[i][0] / self.arena_length) - 1
            observation[i][1] = 2 * (observation[i][1] / self.arena_width) - 1
            # Normalize the orientation (between -pi and pi to between -1 and 1)
            observation[i][2] = observation[i][2] / np.pi
            
        return observation

    def unpack_observation(self, observation, remove_orientation=False):
        # Flattens the observation sublists into one list, and optionally removes theta value from obs state.
        obs = []
        num_features = len(observation[0]) if len(observation) > 0 else 0  # Get feature count dynamically

        for agent_obs in observation:
            if remove_orientation and num_features >= 3:
                # Remove the 3rd element (theta), keeping all other features
                filtered_obs = [val for idx, val in enumerate(agent_obs) if idx != 2]
            else:
                filtered_obs = agent_obs  # Keep all features if remove_orientation is False

            obs.extend(filtered_obs)  # Flatten into the final obs list
        
        if not remove_orientation:
            # pop the last element from the list as it is the orientation of the goal point
            obs.pop()

        return np.array(obs, dtype=np.float32)
    
    def compute_reward(self):
        """
        Compute the reward based on score improvements from the minimum achieved score.
        Score is calculated based on sheep distance from goal, with:
        - Positive rewards for improving the minimum score achieved
        - Time penalty per step
        - Large terminal reward for success
        - No penalty for score reduction

        Returns:
            float: Reward value
        """

        reward = 0.0
    
        # Check termination conditions first
        terminated = self.check_terminated()
        truncated = self.check_truncated()
        
        # Give large reward for successful herding
        if terminated:
            reward += 50000.0
            return reward, terminated, truncated
            
        # Apply time penalty if truncated
        if truncated:
            reward += -5.0
            return reward, terminated, truncated

        # add negative reward for each time step
        reward += -5.0

        # # Calculate score based on sheep distance from goal
        # score = 0.0
        # for i in range(self.num_sheepdogs, len(self.robots)):
        #     x, y, _ = self.robots[i].get_state()
        #     dist = np.linalg.norm(np.array([x, y]) - np.array(self.goal_point))
        #     # give positive reward if the sheep are within the goal region
        #     if dist <= self.goal_tolreance:
        #         reward += 500.0
        #     score += dist

        # # Update minimum score achieved
        # if score < self.min_score:
        #     reward += 50.0
        #     self.min_score = score
        # else:
        #     reward -= 25.0

        return reward, terminated, truncated

    def check_terminated(self):
        terminated = False
        self.curr_iter += 1
        if self.curr_iter >= self.MAX_STEPS:
            return True
        # # check if the sheep herd has reached the goal point
        # for i in range(self.num_sheepdogs, len(self.robots)):
        #     x, y, _ = self.robots[i].get_state()
        #     dist = np.linalg.norm(np.array([x, y]) - np.array(self.goal_point))
        #     if dist > self.goal_tolreance:
        #         terminated = False
        return terminated

    def check_truncated(self):
        # check if the episode is truncated
        self.curr_iter += 1
        if self.curr_iter >= self.MAX_STEPS:
            return True
        
        return False

    def diff_drive_motion_model(self, vec_desired, pose, max_vel) -> np.array:
        """
        Compute the wheel velocities for the bees based on the desired and current heading vectors.

        Args:
            vec_desired (np.array): Desired heading vector
            pose (np.array): Current position and orientation of the robot

        Returns:
            np.array: Wheel velocities for the robot
        """

        # calculate the angle for the desired heading vector
        if vec_desired[0] or vec_desired[1]:
            vec_desired = vec_desired / np.linalg.norm(vec_desired) # normalize the vector
        des_angle = np.arctan2(vec_desired[1], vec_desired[0])

        # calculate the angle difference
        angle_diff = des_angle - pose[2]

        # normalize the angle difference
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

        # calculate the body frame forward velocity 
        v_b = np.cos(angle_diff)

        # calculate the body frame angular velocity
        w_b = np.sin(angle_diff) / self.point_dist

        # calculate the wheel velocities
        left_wheel_velocity = (2 * v_b - w_b * self.distance_between_wheels) / (2 * self.wheel_radius)
        right_wheel_velocity = (2 * v_b + w_b * self.distance_between_wheels) / (2 * self.wheel_radius)

        wheel_velocities = np.array([left_wheel_velocity, right_wheel_velocity])

        # normalize and scale the wheel velocities
        max_wheel_velocity = max(abs(left_wheel_velocity), abs(right_wheel_velocity))
        wheel_velocities = wheel_velocities / max_wheel_velocity * max_vel 

        return wheel_velocities
    
#  To test the simulator
# import argparse
# def parse_args():
#     parser = argparse.ArgumentParser(description="Evaluate a trained RL model for the herding task.")
#     parser.add_argument("--num_sheep", type=int, default=1, help="Number of sheep in the simulation.")
#     parser.add_argument("--num_shepherds", type=int, default=1, help="Number of shepherds in the simulation.")
#     parser.add_argument("--model_path", type=str, required=True, help="Path to the trained RL model.")
#     parser.add_argument("--save_video", type=str, default="False", help="Save videos of simulations (True/False).")
#     parser.add_argument("--num_sims", type=int, default=10, help="Number of simulations to run.")
#     parser.add_argument("--render_mode", type=str, default="human", choices=["human", "offscreen"], help="Render mode for the environment.")
#     return parser.parse_args()

if __name__ == "__main__":
    # args = parse_args()
    # Environment parameters
    arena_length = 20
    arena_width = 20
    robot_wheel_radius = 0.1
    robot_distance_between_wheels = 0.2
    max_wheel_velocity = 10.0
    num_bees=4
    num_sources=1
    render_mode = "human"


    env = BeeSimEnv(
            arena_length=arena_length,
            arena_width=arena_width,
            num_bees=num_bees,  ### always keep multiple of rati0 ie 4.
            num_sources=num_sources,
            robot_distance_between_wheels=robot_distance_between_wheels,
            robot_wheel_radius=robot_wheel_radius,
            max_wheel_velocity=max_wheel_velocity,
            action_mode="point",
        )
    
    # for i in range(num_shepherds):
    #     env.reset(robot_id=i)

    # for i in range(num_bees):
    #     action, _ = models[i].predict(observations[i], deterministic=False)
    #     observations[i], reward, terminated, truncated, _ = env.step(action, robot_id=i)

        # if render_mode == "human":
    
    # Reset environment and get initial observations
    obs, info = env.reset()
    print("Initial Observations:", obs)
    
    while(True):
        env.render(mode="human", fps=60)

    #save video and reeset frames.
    # env.close()
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from robotEmulator import DifferentialDriveRobot
import pygame
import imageio as iio
import math
import time

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
        # self.step_counter = 0
        # Tollerance distance to consider the bee reached its destination.
        self.goal_reaching_tollerance = 0.4



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
                                        shape=((self.num_forger_bees+ self.num_resting_bees) * 3,), dtype=np.float32)
        # Action space is the desired vector for robots if action_mode is "vector"
        elif action_mode == "vector":
            self.action_space = spaces.Box(low=-1, high=1, 
                                        shape=((self.num_forger_bees+ self.num_resting_bees) * 3,), dtype=np.float32)
        # Action space is the desired point for robots if action_mode is "point"
        elif action_mode == "point":
            self.action_space = spaces.Box(low=-1, high=1, 
                                        shape=((self.num_forger_bees+ self.num_resting_bees) * 3,), dtype=np.float32)
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
                                                shape=((self.num_forger_bees+ self.num_resting_bees) *9,), dtype=np.float32)
        elif self.action_mode == "point":
            self.observation_space = spaces.Box(low=-1, high=1, 
                                                shape=((self.num_forger_bees+ self.num_resting_bees) *8,), dtype=np.float32)
        elif self.action_mode == "multi":
            # when action mode is multi, the env is configured as a single-forger, single-resting env
            self.observation_space = spaces.Box(low=-1, high=1, 
                                                shape=((1 + 1)*9,), dtype=np.float32)
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
            # Here action is of all agents
            assert len(action) == self.num_bees * 3, "Invalid action! Incorrect number of actions."
        else:
            # Here action is of individual agent
            assert len(action) == 4, "Invalid action! Incorrect number of actions."
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
                x, y, _, _  = self.robots[i].get_state()
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
                x, y, theta, _  = self.robots[i].get_state()
                # scale to action to be between -pi and pi
                action[i * 2 + 1] = action[i * 2 + 1] * np.pi
                vec_desired = np.array([np.cos(action[i * 2 + 1]), np.sin(action[i * 2 + 1])])
                vec_desired = vec_desired / np.linalg.norm(vec_desired)

                # use the diff drive motion model to calculate the wheel velocities
                wheel_velocities = self.diff_drive_motion_model(vec_desired, [x, y, theta], desired_speed)

                # update the robot position based on the wheel velocities
                self.robots[i].update_position(wheel_velocities[0], wheel_velocities[1])

                # clip the robot position if updated position is outside the arena
                x, y, _, _  = self.robots[i].get_state()
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


                # what it was before- idk why though !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
                # action[i * 2] = (action[i * 2] + 1) * self.arena_length / 2
                # action[i * 2 + 1] = (action[i * 2 + 1] + 1) * self.arena_width / 2

                target_x = (action[i * 3] + 1) * self.arena_length / 2
                target_y = (action[i * 3 + 1] + 1) * self.arena_width / 2
                dance_intensity = action[i * 3 + 2] 

                # if dancing, dance(state orientation, which source, intensity) and then move to the target point



                # get robot state
                x, y, theta,_,_,_,_ = self.robots[i].get_state()

                # calulate the position of the point that is controlled on the robot
                x = x + self.point_dist * np.cos(theta)
                y = y + self.point_dist * np.sin(theta)

                # calculate the vector pointing towards the point
                vec_desired = np.array([target_x - x, target_y - y])

                #TODO: if any robot is dancing true state of action, then do this here
                self.wiggle_dance(robot_id=0, theta=theta, point_dist=self.point_dist, dance_intensity=dance_intensity)
                
                # use the diff drive motion model to calculate the wheel velocities
                wheel_velocities = self.diff_drive_motion_model(vec_desired, [x, y, theta], self.max_wheel_velocity)

                # update the bee position based on the wheel velocities
                if (i != 0):
                    self.robots[i].update_position(wheel_velocities[0], wheel_velocities[1])

                # Clip the bee position if updated position is outside the arena
                x, y, _,_,_,_,_  = self.robots[i].get_state()
                x = np.clip(x, 0.0, self.arena_length)
                y = np.clip(y, 0.0, self.arena_width)
                self.robots[i].x = x
                self.robots[i].y = y 

        # Physics is same as the point action mode[target coordinate to wheel velocities], but the action is taken by a only the specified robotId and not all the bees.
        elif self.action_mode == "multi":
            assert robot_id is not None, "Invalid robot ID! Please provide a valid robot ID."

            # This is the action space defined for MARL Model.
            # action[0] is the x-coordinate of the point
            # action[1] is the y-coordinate of the point
            # map the actions from -1 to 1 to between the arena dimensions
            # action[2] : dancing_action = 0: no dance, 1: dance
            # action[3] : waggle_comm_action = 0: not observing, 1: observing

            action[0] = (action[0] + 1) * self.arena_length / 2
            action[1] = (action[1] + 1) * self.arena_width / 2
            dancing_action = action[2]
            waggle_comm_action = action[3]             
            dance_intensity = 1


            # These are the actual actions as seen in the simulator.
            # Action 0: Move randomly
            # Action 1: Move towards the target point: listning to the action of the bee
            # Action 2: Wiggle dance
            # Action 4: Gather nectar if near a nectar source
            # Action 5: Return to the hive
            # Action 6: Observe Waggle Dance


            # Get robot state
            x, y, theta, dancing, carrying_nectar, energy_level, waggle_comm, returning_hive = self.robots[robot_id].get_state()


            # Wiggle dance.
            if (dancing_action == 1):
                self.wiggle_dance(robot_id=robot_id, theta=theta, point_dist=self.point_dist, dance_intensity=dance_intensity)
                energy_level -= 1

                
            # A bee is observing another bee dance.
            # waggle_comm is used to manage action states via code.
            # waggle_comm_action is the action state managed by the RL model.
            elif (waggle_comm_action == 1):
                waggle_comm = 1
                energy_level -= 1

                # Find which bee is dancing now. (Assuming: only one bee is dancing at a time)
                observed_robot_id = None
                for other_id, bee in enumerate(self.robots):
                    if other_id != robot_id and bee.dancing:
                        observed_robot_id = other_id
                        break
                    
                # Guided exploration (listening to wiggle dance)
                if (observed_robot_id != None):
                    # desitination calculated from distance, and orientation of the bee that is dancing.
                    print("[ROBOT STATE] Robot ID: ", robot_id, " is observing the wiggle dance of the bee ID: ", observed_robot_id)

                    desination_x, desination_y = self.read_wiggle(observed_robot_id)

                    # calulate the position of the point that is controlled on the robot
                    x = x + self.point_dist * np.cos(theta)
                    y = y + self.point_dist * np.sin(theta)
                    # calculate the vector pointing towards the point
                    vec_desired = np.array([desination_x - x, desination_y - y])
                    # use the diff drive motion model to calculate the wheel velocities
                    wheel_velocities = self.diff_drive_motion_model(vec_desired, [x, y, theta], self.max_wheel_velocity)
                    # update the sheep-dog position based on the wheel velocities
                    self.robots[robot_id].update_position(wheel_velocities[0], wheel_velocities[1])

                    # If the bee reached the destination as mentioned by the dance.
                    dist = np.linalg.norm(np.array([x, y]) - np.array([desination_x, desination_y]))
                    if (dist < self.goal_reaching_tollerance):
                        print("[ROBOT STATE] Robot ID: ", robot_id, " Reached nector source from dance observation!")
                        energy_level += 100
                        carrying_nectar == 1 
                        waggle_comm = 0   # Communication is done.
                        returning_hive = 1

                else:
                        print("[ROBOT STATE] Robot ID: ", robot_id, "No Dancing Bee to be observed by the robot id: ")
                

            # # Reached nectar source and Returning to hive.
            # elif (carrying_nectar == 1 and returning_hive != 1):
            #     energy_level -= 1
            #     # calulate the position of the point that is controlled on the robot.
            #     x = x + self.point_dist * np.cos(theta)
            #     y = y + self.point_dist * np.sin(theta)
            #     # calculate the vector pointing towards the point
            #     vec_desired = np.array([self.hive[0] - x, self.hive[1] - y])
            #     # use the diff drive motion model to calculate the wheel velocities
            #     wheel_velocities = self.diff_drive_motion_model(vec_desired, [x, y, theta], self.max_wheel_velocity)
            #     # update the sheep-dog position based on the wheel velocities
            #     self.robots[robot_id].update_position(wheel_velocities[0], wheel_velocities[1])
            #     dist = np.linalg.norm(np.array([x, y]) - np.array([self.hive[0], self.hive[1]]))
                
            #     if (dist < self.goal_reaching_tollerance):
            #         returning_hive = 1
            #         print("[ROBOT STATE] Robot ID: ", robot_id, " Reached nector source, transitioning state here")


            else:
            # Gathering nectar
                
                # Check if the bee is near a nectar source.
                for i in range(self.num_sources):
                    source_x, source_y, source_radius = self.sources[i]
                    # calculate the distance between the bee and the nectar source
                    dist = np.linalg.norm(np.array([x, y]) - np.array([source_x, source_y]))
                    if dist <= source_radius + self.goal_reaching_tollerance:
                        # Inside the nectar source, so gather nectar
                        print("[ROBOT STATE] Robot ID: ", robot_id, " Reached nector source, is gathering nectar from source: ", i)
                        carrying_nectar = 1
                        # TODO: Make it based on nectar source size.
                        energy_level += 100
                        returning_hive = 1
                        # record which source this bee found
                        self.robots[robot_id].last_source_pose = (source_x, source_y)
                        break
                
                # Check if the bee is near the hive
                hive_dist = np.linalg.norm(np.array([x, y]) - np.array(self.hive))                
                if hive_dist <= self.hive_radius + self.goal_reaching_tollerance:
                    # Reached hive. Stop exploring.
                    carrying_nectar = 0
                    energy_level -= 1
                    returning_hive = 0
                    print("[ROBOT STATE] Robot ID: ", robot_id, " Reached Hive with nectar!", i)
                
                elif (returning_hive == 1):
                    energy_level -= 1   
                    print("[ROBOT STATE] Robot ID: ", robot_id, " Returning to Hive!", i)
                    # calulate the position of the point that is controlled on the robot
                    x = x + self.point_dist * np.cos(theta)
                    y = y + self.point_dist * np.sin(theta)
                    # calculate the vector pointing towards the point
                    vec_desired = np.array([self.hive[0] - x, self.hive[1] - y])
                    # use the diff drive motion model to calculate the wheel velocities
                    wheel_velocities = self.diff_drive_motion_model(vec_desired, [x, y, theta], self.max_wheel_velocity)
                    # update the sheep-dog position based on the wheel velocities
                    self.robots[robot_id].update_position(wheel_velocities[0], wheel_velocities[1])

                # Bee is exploring   
                # Not inside any nectar source, so move towards the target point
                else:
                    energy_level -= 1   
                    print("[ROBOT STATE] Robot ID: ", robot_id, " Exploring!", i)
                    # calulate the position of the point that is controlled on the robot
                    x = x + self.point_dist * np.cos(theta)
                    y = y + self.point_dist * np.sin(theta)
                    # calculate the vector pointing towards the point
                    vec_desired = np.array([action[0] - x, action[1] - y])
                    # use the diff drive motion model to calculate the wheel velocities
                    wheel_velocities = self.diff_drive_motion_model(vec_desired, [x, y, theta], self.max_wheel_velocity)
                    # update the sheep-dog position based on the wheel velocities
                    self.robots[robot_id].update_position(wheel_velocities[0], wheel_velocities[1])


            # Clip the bee position if updated position is outside the arena.
            x, y,_,dancing,_,_,_ ,_= self.robots[robot_id].get_state()
            x = np.clip(x, 0.0, self.arena_length)
            y = np.clip(y, 0.0, self.arena_width)
            self.robots[robot_id].x = x
            self.robots[robot_id].y = y


            self.robots[robot_id].update_state(dancing=dancing, carrying_nectar=carrying_nectar, energy_level=energy_level, waggle_comm=waggle_comm, returning_hive=returning_hive)



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

    def wiggle_dance(self, robot_id =0, theta=None, point_dist=None, dance_intensity=None):
        self.robots[robot_id].dancing = True
        if(self.robots[robot_id].dancing == True):
            dt = 0.1
            steps = int(30.0/ dt)
            # # Orienting for the dance.
            # self.theta = orientation
            # for i in range(steps):
            self.robots[robot_id].wiggle_dance(step_counter=self.robots[robot_id].dance_step_counter, orientation=theta, distance=self.point_dist, intensity=dance_intensity)
            # self.robots[robot_id].dance_step_counter = self.robots[robot_id].dance_step_counter + 1
            
            ############### may need to change this here.
            if hasattr(self.robots[robot_id], 'last_source_pose'):
                src_x, src_y = self.robots[robot_id].last_source_pose
            else:
                # If the robot is dancing, without being to a source, give the position value as -100, -100.
                src_x, src_y = -100, -100

            last_dance_info = {
                'dancer': robot_id,
                'dest_x': src_x,
                'dest_y': src_y,
                'intensity': dance_intensity,
                ########## For testing Step 3 Demo, can set the count to 1.
                'count' : 3  # This is the max robots that can read the dance: stopping condition.
            }
            self.robots[robot_id].last_dance_info = last_dance_info

            # if self.robots[robot_id].dance_step_counter >= steps:  # Timer based dance duration.
            if hasattr(self.robots[robot_id], 'last_dance_info'):
                max_count = self.robots[robot_id].last_dance_info['count']
                if self.robots[robot_id].dance_step_counter >= max_count: 
                    # dance_state = "completed"
                    self.robots[robot_id].dance_step_counter = 0
                    self.robots[robot_id].dancing = False
                    self.robots[robot_id].last_dance_info = last_dance_info
                    print("Dance completed")

    def read_wiggle(self, observed_robot_id):
        if hasattr(self.robots[observed_robot_id], 'dance_step_counter') > 0:
            # Here observed robot id is the id of the bee is being observed.
            self.robots[observed_robot_id].dance_step_counter = self.robots[observed_robot_id].dance_step_counter + 1
            return self.robots[observed_robot_id].last_dance_info['dest_x'], self.robots[observed_robot_id].last_dance_info['dest_y']
            # else:
            #     return None, None
        else:
            x, y, theta, dancing, carrying_nectar, energy_level, waggle_comm, returning_hive = self.robots[observed_robot_id].get_state()
            dancing = 0 
            print("[ROBOT STATE] Robot ID: ", observed_robot_id, " This bee is tired of dancing!")
            self.robots[observed_robot_id].update_state(dancing=dancing, carrying_nectar=carrying_nectar, energy_level=energy_level, waggle_comm=waggle_comm, returning_hive=returning_hive)
            return None, None


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
            robot_x, robot_y, robot_theta, robot_dancing ,_,_,_ = robot.get_state()  # Get robot position

            print(f"Robot State {i}: x = {robot_x}, y = {robot_y}, theta = {robot_theta}, dancing = {robot_dancing}")
            
            color = (0,0,255)  # Blue for All Bees.

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
                    r = np.random.uniform(0, self.hive_raself.hivedius)  
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

        else:
            # Reseting in multi-robot cases.
            info = {}
            obs = self.get_multi_observations(robot_id)
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
            obs = []
            for robot in self.robots:
                # obs.append(robot.get_state())
                state = list(robot.get_state())
                
                # Environmantal obs.
                # Calculate the distance from the hive to the robot
                distance_from_hive = np.linalg.norm(np.array([state[0], state[1]]) - np.array(self.hive))
                direction_to_hive = math.atan(state[1]/state[0])  # Get the angle to the hive
                # Append the new observation to the state
                state.append(distance_from_hive)
                state.append(direction_to_hive)  

                # Waggle dance signal
                # TODO: other should read, adn if reading should be a learnt behaviour. but if it reads, then what it reads is encoded here(fixed).

                waggle_comm = state[6]
                # if (waggle_comm)
                # {
                #     # Read the data.
                #     # 
                # }
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

    def get_multi_observations(self, robot_id):
        # Returns the observation of the robot with the given robot_id
        # Observation State [All robots]:
        obs = []
        robot = self.robots[robot_id]
        # obs.append(robot.get_state())
        state = list(robot.get_state())
        # Environmantal obs.
        # Calculate the distance from the hive to the robot
        distance_from_hive = np.linalg.norm(np.array([state[0], state[1]]) - np.array(self.hive))
        direction_to_hive = math.atan(state[1]/state[0])  # Get the angle to the hive
        # Append the new observation to the state
        state.append(distance_from_hive)
        state.append(direction_to_hive)
        # Waggle dance signal
        # TODO: other should read, adn if reading should be a learnt behaviour. but if it reads, then what it reads is encoded here(fixed).

        waggle_comm = state[6]
        # if (waggle_comm)
        # {
        #     # Read the data.
        #     #
        # }
        obs.append(state)
    
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
        #     x, y, _ , _ = self.robots[i].get_state()
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

    # Convert save_video to boolean
    save_video = "true" 

    # create a video directory if it does not exist
    if save_video:
        import os
        if not os.path.exists("videos"):
            os.makedirs("videos")


    env = BeeSimEnv(
            arena_length=arena_length,
            arena_width=arena_width,
            num_bees=num_bees,  ### always keep multiple of rati0 ie 4.
            num_sources=num_sources,
            robot_distance_between_wheels=robot_distance_between_wheels,
            robot_wheel_radius=robot_wheel_radius,
            max_wheel_velocity=max_wheel_velocity,
            action_mode="multi",
        )



#######################    DETERMINISTIC SCENARIO-BASED TESTING OF COMPLETE BEE STATE CYCLE    ###################################

if __name__ == "__main__":
    # Environment setup
    env = BeeSimEnv(
        arena_length=20,
        arena_width=20,
        num_bees=4,
        num_sources=1,
        robot_distance_between_wheels=0.2,
        robot_wheel_radius=0.1,
        max_wheel_velocity=10.0,
        action_mode="multi",
    )

    # Place one nectar source manually
    nectar_source = (6.0, 5.0, 1.0)  # x, y, radius
    env.sources = [nectar_source]
    env.hive = (1.0, 1.0)
    env.hive_radius = 1.5

    # Bee 0 will find nectar
    robot_id_0 = 0
    robot_id_1 = 1

    env.robots[robot_id_0].x = 6.0
    env.robots[robot_id_0].y = 5.0
    env.robots[robot_id_1].x = 1.0
    env.robots[robot_id_1].y = 1.0
    print()
    print("======== STEP 1: BEE 0 FINDS NECTAR =========")
    action_find_nectar = [0.0, 0.0, 0, 0]  # No move needed — already at nectar
    obs0, reward, _, _, _ = env.step(action_find_nectar, robot_id=robot_id_0)
    print("Robot 0 State:", env.robots[robot_id_0].get_state())
    assert env.robots[robot_id_0].carrying_nectar == 1, "Bee 0 should be carrying nectar."
    print("=========================================================")
    print()


    print("======== STEP 2: BEE 0 RETURNS TO HIVE =========")
    env.robots[robot_id_0].x = 1.0
    env.robots[robot_id_0].y = 1.0
    action_return_hive = [0.0, 0.0, 0, 0]  # Simply move into hive area
    obs0, reward, _, _, _ = env.step(action_return_hive, robot_id=robot_id_0)
    print("Robot 0 State:", env.robots[robot_id_0].get_state())
    print("=========================================================")
    print()


    print("======== STEP 3: BEE 0 PERFORMS WIGGLE DANCE =========")
    action_dance = [0.0, 0.0, 1, 0]  # Dance = 1
    for _ in range(5):  # Run multiple steps to simulate full dance completion
        obs0, reward, _, _, _ = env.step(action_dance, robot_id=robot_id_0)
        print("Dance Info:", env.robots[robot_id_0].last_dance_info)

    assert hasattr(env.robots[robot_id_0], 'last_dance_info'), "Dance info not stored."
    print("Dance Info:", env.robots[robot_id_0].last_dance_info)
    print("=========================================================")
    print()
    
    
    print("======== STEP 4: BEE 1 SUCCESSFULLY OBSERVES WIGGLE DANCE =========")
    action_observe = [0.0, 0.0, 0, 1]  # Observe = 1
    obs1, reward, _, _, _ = env.step(action_observe, robot_id=robot_id_1)
    print("Robot 1 State:", env.robots[robot_id_1].get_state())
    print("=========================================================")
    print()

    # print("======== STEP 5: BEE 1 SUCCESSFULLY OBSERVES WIGGLE DANCE  =========")
    # action_dance = [0.0, 0.0, 1, 0]  # Dance = 1
    # obs0, reward, _, _, _ = env.step(action_dance, robot_id=robot_id_0)
    # action_observe = [0.0, 0.0, 0, 1]  # Observe = 1
    # obs1, reward, _, _, _ = env.step(action_observe, robot_id=robot_id_1)
    # print("Robot 1 State:", env.robots[robot_id_1].get_state())
    # print("=========================================================")
    # print()
        

    print("======== STEP 6: BEE 1 STARTS MOVING TO SOURCE =========")
    for _ in range(75):
        obs1, reward, _, _, _ = env.step(action_observe, robot_id=robot_id_1)
        # print("Robot 1 Pos:", (env.robots[robot_id_1].x, env.robots[robot_id_1].y))

    final_pos = np.array([env.robots[robot_id_1].x, env.robots[robot_id_1].y])
    target = np.array([nectar_source[0], nectar_source[1]])
    dist = np.linalg.norm(final_pos - target)
    print("Distance to nectar:", dist)

    assert dist < 0.4, "Bee 1 should be near the nectar source."

    print("✅ Test Passed: Full state loop for bee swarm works.")


# Tested the loop for STEP 4: BEE 1 OBSERVES WIGGLE DANCE: BUT NO DANCING BEE PRESENT 





############################################################  TEST ENV #########################################################

# # Using sample action in-place of predicted action from the model.

#     # Reset environment and get initial observations
#     observations = {}
#     for id in range(num_bees):
#         # env.reset(robot_id=id)
#         observations[id], info = env.reset(robot_id=id)
#         terminated = False
#         truncated = False
#         episode_reward = 0
#         episode_length = 0

#         print("Initial Observations:", observations[id], " of robot id: ", id)

#         # Create a sample action vector for each bee
#         # Here, we simply sample a random action from the defined action_space.
#         sample_action = env.action_space.sample()
#         print("Sample Action:", sample_action, " of robot id: ", id)

#         # Take a step in the environment using the sample action
#         new_obs, reward, terminated, truncated, info = env.step(sample_action, robot_id=id)
#         print("New Observations:", new_obs, " of robot id: ", id)
#         print("Reward:", reward, " of robot id: ", id)

#     # # Single agent
#     # obs, info = env.reset()
#     # print("Initial Observations:", obs)

#     # # Create a sample action vector for each bee (3 values per bee: x, y, dance intensity)
#     # # Here, we simply sample a random action from the defined action_space.
#     # sample_action = env.action_space.sample()
#     # print("Sample Action:", sample_action)

#     # # Take a step in the environment using the sample action
#     # new_obs, reward, terminated, truncated, info = env.step(sample_action)
#     # print("New Observations:", new_obs)
#     # print("Reward:", reward)
    

#     ############################################################  TEST ENV #########################################################
    
#     # while(True):
#     #     # env.render(mode="human", fps=60)
#     #     sample_action = env.action_space.sample()
#     #     new_obs, reward, terminated, truncated, info = env.step(sample_action)
#     #     env.render(mode="human", fps=60)
#     #     # time.sleep(1)

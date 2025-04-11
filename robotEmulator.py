import numpy as np
import time

class DifferentialDriveRobot:
    def __init__(self, initial_position, distance_between_wheels, wheel_radius):
        self.x, self.y, self.theta = initial_position # theta is limited to [-pi, pi]
        self.distance_between_wheels = distance_between_wheels
        self.wheel_radius = wheel_radius
        self.prev_velocity = np.array([0.0, 0.0])
        self.max_acceleration = float("inf") # m/s^2
        self.omega_acceleration = 7.5 # rad/s^2

        # State parameters.
        self.dancing = False
        self.carrying_nectar = False
        self.returning_hive = False # [State managed by state machine not RL Model ]
        self.energy_level = 1000
        # It reads if the waggle dance is being read or not by the others.
        self.waggle_comm = False


        # Dance based parameters
        self.last_source_pose = {-100,-100}
        self.dance_step_counter = 0
        self.last_dance_info = None

    def update_position(self, left_wheel_velocity, right_wheel_velocity, dt=0.1):
        # Differential drive kinematics
        v = self.wheel_radius * (left_wheel_velocity + right_wheel_velocity) / 2.0
        omega = self.wheel_radius * (right_wheel_velocity - left_wheel_velocity) / self.distance_between_wheels

        # Apply artificial acceleration limit on velocity changes
        v = np.clip(v, self.prev_velocity[0] - self.max_acceleration * dt, self.prev_velocity[0] + self.max_acceleration * dt)        
        omega = np.clip(omega, self.prev_velocity[1] - self.omega_acceleration * dt, self.prev_velocity[1] + self.omega_acceleration * dt)

        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += omega * dt

        # Limit theta to [-pi, pi]
        if self.theta > np.pi:
            self.theta -= 2*np.pi
        if self.theta < -np.pi:
            self.theta += 2*np.pi

        # update previous velocity
        self.prev_velocity = np.array([v, omega])

    def reset_robot(self):
        # not used right now, as at reset the robots are made again from sratch.
        self.dancing = False
        self.carrying_nectar = False
        self.energy_level = 50
        self.waggle_comm = False

    def get_state(self):
        return np.array([self.x, self.y, self.theta, self.dancing, self.carrying_nectar, self.energy_level, self.waggle_comm, self.returning_hive])
    
    def wiggle_dance(self, step_counter, distance, orientation, intensity, dt=0.1):
        # will call updateposition with the appropriate wheel velocities
        # manage the timing of the dance
        # code for "Dance" action.

        # write me the code for update position of the robot in a way that makes a 8 like structure, the 8 should be along an AXIS. 
        # The axis / line is defined by the distance and orientation value and start point of the line is the current pose of the robot.

        # # figure 8
        # for i in range(steps):
            print("Dancing values :" , self.dancing)
            dt = 0.1
            steps = int(30.0/ dt)
            # Calculate the velocities to create a figure-8 pattern
            left_wheel_velocity = intensity * np.sin(2 * np.pi * step_counter / steps)
            right_wheel_velocity = intensity * np.cos(2 * np.pi * step_counter / steps)
            
            # Update the robot's position
            self.update_position(left_wheel_velocity, right_wheel_velocity, dt)
            step_counter += 1

        # # Vibration as dance
        # # for i in range(steps):
        #     # Make the wheels alternate directions quickly to create vibration
        #     vibration_amplitude = intensity * 0.5  # Reduce intensity for smaller movements
        #     # left_wheel_velocity = (-1) ** step_counter * vibration_amplitude  # Alternates between + and -
        #     # right_wheel_velocity = (-1) ** step_counter * vibration_amplitude  # Alternates between + and -
        #     # step_counter = 2
        #     print("sTEP COUNTER", step_counter)
        #     left_wheel_velocity = (-1) ** step_counter *   5  # Alternates between + and -
        #     right_wheel_velocity = (-1) ** step_counter * 5  # Alternates between + and -

            

            # Update the robot's position
            self.update_position(left_wheel_velocity, right_wheel_velocity, dt)

    def update_state(self, dancing, carrying_nectar, energy_level, waggle_comm, returning_hive):
        self.dancing = dancing
        self.carrying_nectar = carrying_nectar
        self.energy_level = energy_level
        self.waggle_comm = waggle_comm
        self.returning_hive = returning_hive
        
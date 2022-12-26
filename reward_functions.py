import numpy as np
from CarlaEnv.wrappers import angle_diff, vector
import math
import datetime

low_speed_timer = 0
acceleration = 0
max_distance    = 3.0  # Max distance from center before terminating
target_speed    = 60 # kmh

def create_reward_fn(reward_fn, max_speed=-1):
    """
        Wraps input reward function in a function that adds the
        custom termination logic used in these experiments

        reward_fn (function(CarlaEnv)):
            A function that calculates the agent's reward given
            the current state of the environment. 
        max_speed:
            Optional termination criteria that will terminate the
            agent when it surpasses this speed.
            (If training with reward_kendal, set this to 20)
    """
    def func(env):
        terminal_reason = "Running..."

        # Stop if speed is less than 1.0 km/h after the first 5s of an episode
        global low_speed_timer
        global acceleration

        low_speed_timer += 1.0 / env.fps
        speed = env.vehicle.get_speed()

        if low_speed_timer > 5.0 and speed < 1.0 / 3.6:
            env.terminal_state = True
            terminal_reason = "Vehicle stopped"

        # Stop if distance from center > max distance
        if env.distance_from_center > max_distance:
            env.terminal_state = True
            terminal_reason = "Off-track"

        # Stop if speed is too high
        if max_speed > 0 and speed_kmh > max_speed:
            env.terminal_state = True
            terminal_reason = "Too fast"
        # TOO FAST AIN'T THAT BAD?

        # Collision Penalty

    
        # Calculate reward
        reward = 0
        if not env.terminal_state:
            new_reward, acceleration = reward_fn(env, acceleration)
            reward += new_reward
        else:
            low_speed_timer = 0.0
            reward -= 10

        if env.terminal_state:
            env.extra_info.extend([
                terminal_reason,
                ""
            ])
        return reward
    return func

#---------------------------------------------------
# Create reward functions dict
#---------------------------------------------------

reward_functions = {}

# Kendall's (Learn to Drive in a Day) reward function
def reward_kendall(env):
    speed_kmh = 3.6 * env.vehicle.get_speed()
    return speed_kmh

reward_functions["reward_kendall"] = create_reward_fn(reward_kendall)

# Our reward function (additive)
def reward_speed_centering_angle_add(env):
    """
        reward = Positive speed reward for being close to target speed,
                 however, quick decline in reward beyond target speed
               + centering factor (1 when centered, 0 when not)
               + angle factor (1 when aligned with the road, 0 when more than 20 degress off)
    """

    min_speed = 15.0 # km/h
    max_speed = 80.0 # km/h

    # Get angle difference between closest waypoint and vehicle forward vector
    fwd    = vector(env.vehicle.get_velocity())
    wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
    angle  = angle_diff(fwd, wp_fwd)

    speed_kmh = 3.6 * env.vehicle.get_speed()
    if speed_kmh < min_speed:                     # When speed is in [0, min_speed] range
        speed_reward = speed_kmh / min_speed      # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > target_speed:                # When speed is in [target_speed, inf]
                                                  # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
        speed_reward = 1.0 - (speed_kmh-target_speed) / (max_speed-target_speed)
    else:                                         # Otherwise
        speed_reward = 1.0                        # Return 1 for speeds in range [min_speed, target_speed]

    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

    # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
    angle_factor = max(1.0 - abs(angle / np.deg2rad(20)), 0.0)

    # Final reward
    reward = speed_reward + centering_factor + angle_factor

    return reward

reward_functions["reward_speed_centering_angle_add"] = create_reward_fn(reward_speed_centering_angle_add)

# Our reward function (multiplicative)
def reward_speed_centering_angle_multiply(env):
    """
        reward = Positive speed reward for being close to target speed,
                 however, quick decline in reward beyond target speed
               * centering factor (1 when centered, 0 when not)
               * angle factor (1 when aligned with the road, 0 when more than 20 degress off)
    """

    min_speed = 15.0 # km/h
    max_speed = 105.0 # km/h

    # Get angle difference between closest waypoint and vehicle forward vector
    fwd    = vector(env.vehicle.get_velocity())
    wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
    angle  = angle_diff(fwd, wp_fwd)

    speed_kmh = 3.6 * env.vehicle.get_speed()
    if speed_kmh < min_speed:                     # When speed is in [0, min_speed] range
        speed_reward = speed_kmh / min_speed      # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > target_speed:                # When speed is in [target_speed, inf]
                                                  # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
        speed_reward = 1.0 - (speed_kmh-target_speed) / (max_speed-target_speed)
    else:                                         # Otherwise
        speed_reward = 1.0                        # Return 1 for speeds in range [min_speed, target_speed]

    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

    # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
    angle_factor = max(1.0 - abs(angle / np.deg2rad(20)), 0.0)

    # Final reward
    reward = speed_reward * centering_factor * angle_factor

    return reward

reward_functions["reward_speed_centering_angle_multiply"] = create_reward_fn(reward_speed_centering_angle_multiply)



def highway_reward(env, prev_accel = 0):
    min_speed = 15 # km/h
    max_speed = 80 #km/h
    max_speed_cross_penalty_factor = 0.2

    speed = 3.6 * env.vehicle.get_speed()
    if speed <= min_speed: 
        spd_reward = ((speed- min_speed)/ min_speed)
    elif speed > min_speed and speed <= target_speed:
        spd_reward = 1- ((target_speed - speed)/(target_speed - min_speed))
    elif speed > target_speed and speed <= max_speed:
        spd_reward = ((max_speed-speed)/(max_speed-target_speed))
    else: 
        spd_reward = -1 + (-1*(speed - max_speed)*max_speed_cross_penalty_factor) # Reward tapers from -1 to -inf at penalty_factor rate

    fwd    = vector(env.vehicle.get_velocity())
    accel  = vector(env.vehicle.get_acceleration())
    wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
    angle  = angle_diff(fwd, wp_fwd)

    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

    # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
    angle_factor = max(1.0 - abs(angle / np.deg2rad(20)), 0.0)

    # Collision penalty
    collision_intensity = env.vehicle.collision_sensor.get_collision_impulse_intensity()
    collision_penalty = -math.log(max(1,collision_intensity),10)
    collision_penalty = 0

    #Jerk Penalty
    jerk_scale_factor = 1
    jerk = (accel - prev_accel) * env.fps
    jerk = math.sqrt((jerk[0]**2)*0 + jerk[1]**2 + (jerk[2]**2)*0)
    jerk_corrected = np.log10(max(jerk*jerk_scale_factor, 1))/3
    jerk_penalty = min(jerk_corrected, 1)
    tm=str(datetime.datetime.now()).split(' ')
    print(f"[{tm[1]}]----------------REWARD COMPONENTS--------------")
    print(f"Speed Reward: {spd_reward}\nCentering Reward: {centering_factor}\nAngle Reward: {angle_factor}\nJerk Penalty: {jerk_penalty}\nCollision Penalty: {collision_penalty}\n")
    return ((spd_reward + centering_factor + angle_factor) + collision_penalty - (jerk_penalty/5)), accel


reward_functions["highway_reward"] = create_reward_fn(highway_reward)

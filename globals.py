# Class with synchronization or mutex with dictionary. Jose. We have no OS.
# Not used for now

# Actual state values of the car
car_state = {
    "speed": 0.,
    "gear": 0.,
    "rpm": 0.
}

# Target values obtained from agent.get_action()
global agent_target
agent_target = {
    "throttle": 0., # 0.0 to 1.0
    "brake": 0., # 0.0 to 1.0
    "steer": 0., # -1.0 to 1.0
    "clutch": 0., # ? 0.8 I've seen
    # "upgear": 0., # 
    # "downgear": 0.,
    # "gear": 0. # Should be an integer
}


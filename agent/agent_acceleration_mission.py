from agent.agent import Agent

from globals.globals import * # Global variables and constants, as if they were here

class Acceleration_Mission(Agent):
    def __init__(self):
        super().__init__()

    def act_sim(self, cones, sim_client2, simulator_car_controls):
        super().act_sim(cones, sim_client2, simulator_car_controls)

    def act(self, cones):
        '''
        
        '''
        self._get_target(cones)

    # SUPERCHARGED METHODS
    def _get_target(self, cones):
        '''
        Update agent_target, calculated from the cones and car_state.
        '''
        
        # STEER
        def take_x(cone): return cone['coords']['x']
        blues = [cone for cone in cones if (cone['label'] == 'blue_cone')]
        blues.sort(key=take_x)
        yellows = [cone for cone in cones if (cone['label'] == 'yellow_cone')]
        yellows.sort(key=take_x)

        large_oranges = [cone for cone in cones if (cone['label'] == 'large_orange_cone')]
        large_oranges.sort(key=take_x)

        orange = [cone for cone in cones if (cone['label'] == 'orange_cone')]
        orange.sort(key=take_x)


        # if (len(large_oranges) > 2) and (large_oranges[0]['coords']['x']) < 1:
        #       agent_target['steer'] = 1

        brake_condition = (len(orange) >= 6) and (orange[0]['coords']['y'] < 1)

        # SPEED
        if (car_state['speed'] < 5) and (not brake_condition): #si va lento y no ve conos naranjas
            agent_target['acc'] = 0.5
        elif brake_condition: # da igual la velocidad, si ve conos naranjas
            agent_target['steer'] = 0 # -1 left, 1 right, 0 neutral
            agent_target['acc'] = 0.0
            agent_target['brake'] = 1.0
        else: # If it's fast we stop accelerating
            agent_target['acc'] = 0.0
        
        # STEER
        if (len(blues) > 0) and (len(yellows) > 0):
            #I assume they're sorted from closer to further
            center = (blues[0]['coords']['y'] + yellows[0]['coords']['y']) / 2
            # print(f'center:{center}')
            agent_target['steer'] = center * 0.5 # -1 left, 1 right, 0 neutral TODO HACER CON MAS SENTIDO
        elif len(blues) > 0:
            agent_target['steer'] = 1 # -1 left, 1 right, 0 neutral
        elif len(yellows) > 0:
            agent_target['steer'] = -1 # -1 left, 1 right, 0 neutral
        else:
            agent_target['steer'] = 0
from agent.agent import Agent

class Acceleration_Mission(Agent):
    def __init__(self):
        super().__init__()

    def act_sim(self, cones, sim_client2, simulator_car_controls):
        super().act_sim(cones, sim_client2, simulator_car_controls)

    # SUPERCHARGED METHODS
    def get_target(self, cones, car_state, actuation):
        '''
        Update actuation, calculated from the cones and car_state.
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

        brake_condition = (len(orange) >= 6) and (orange[0]['coords']['y'] < 1)

        # SPEED
        if (car_state['speed'] < 5) and (not brake_condition): #si va lento y no ve conos naranjas
            actuation['acc'] = 0.5
        elif brake_condition: # da igual la velocidad, si ve conos naranjas
            actuation['steer'] = 0 # 1 left, -1 right, 0 neutral
            actuation['acc'] = 0.0
            actuation['brake'] = 1.0

            if(car_state['speed'] < 0.25): #Si se ha parado completamente, AS_Finished
                return True
        else: # If it's fast we stop accelerating
            actuation['acc'] = 0.0
        
        # STEER
        if (len(blues) > 0) and (len(yellows) > 0):
            #I assume they're sorted from closer to further
            center = (blues[0]['coords']['y'] + yellows[0]['coords']['y']) / 2
            # print(f'center:{center}')
            actuation['steer'] = center * 0.5 # 1 left, -1 right, 0 neutral TODO HACER CON MAS SENTIDO
        elif len(blues) > 0:
            actuation['steer'] = -1 # 1 left, -1 right, 0 neutral
        elif len(yellows) > 0:
            actuation['steer'] = 1 # 1 left, -1 right, 0 neutral
        else:
            actuation['steer'] = 0

        return False
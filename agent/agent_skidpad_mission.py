from agent.agent import Agent
from globals.globals import * # Global variables and constants, as if they were here

# TODO por que heredar varios agentes en vez de meter metodos en uno.

class Skidpad_Mission(Agent):
    #TODO REWRITE ALL THE METHODS THAT ARE SPECIALIZED FOR THIS MISSION or else:
    def __init__(self):
        super().__init__()
    
    def act_sim(self, cones, sim_client2, simulator_car_controls):
        super().act_sim(cones, sim_client2, simulator_car_controls)

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

        brake_condition = (len(orange) >= 6) and (orange[0]['coords']['y'] < 1) and (len(blues) == 0 and len(yellows) == 0)
        intersection_trigger = (len(large_oranges) == 4) and (large_oranges[0]['coords']['x'] < 1.2)
        # Variable that keeps track of the number of times that the car has cross the intersection
        laps = 0
        
        # Code to be executed when the car crosses the intersection
        def intersection_behaviour(laps):
            if(laps < 3): # The first two laps must be turning to the right
                agent_target['steer'] = 0.75 # -1 left, 1 right, 0 neutral
                agent_target['acc'] = 0.5

            elif(laps > 2 and laps < 5): # The last 2 laps must be to the left
                agent_target['acc'] = 0.5
                agent_target['steer'] = -0.75 # -1 left, 1 right, 0 neutral
                
            else: # If it has accomplished all 4 turns then its time to exit
                agent_target['steer'] = 0 # -1 left, 1 right, 0 neutral
                agent_target['acc'] = 0.5


        # SPEED
        if (car_state['speed'] < 2) and (not brake_condition) and (not intersection_trigger): #si va lento y no ve conos naranjas
            agent_target['acc'] = 0.5
        elif (brake_condition and (not intersection_trigger)): # da igual la velocidad, si ve conos naranjas y no ve la interseccion
            agent_target['acc'] = 0.0
            agent_target['brake'] = 1.0
        elif (intersection_trigger): # si se encuentra en la interseccion
            intersection_behaviour(laps)
            laps += 1 # Add one to the counter
        else: # si va rapido dejamos de acelerar
            agent_target['acc'] = 0.0
           
        
        #STEER

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
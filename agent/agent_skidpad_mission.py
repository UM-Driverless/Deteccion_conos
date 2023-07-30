from agent.agent import Agent
from globals.globals import * # Global variables and constants, as if they were here

# TODO por que heredar varios agentes en vez de meter metodos en uno.

class Skidpad_Mission(Agent):
    
    def __init__(self):
        super().__init__()
        self.hysteresis = 0 # 0 means you can go straight
        self.intersection_trigger = False
        self.laps: int = 0 # Variable that keeps track of the number of times that the car has crossed the intersection
    # SUPERCHARGED METHODS
    def get_target(self, cones, car_state, agent_act):
        '''
        Update agent_act, calculated from the cones and car_state.
        '''
        
        def take_x(cone): return cone['coords']['x']
        
        self.blues = [cone for cone in cones if (cone['label'] == 'blue_cone')]
        self.blues.sort(key=take_x)
        self.yellows = [cone for cone in cones if (cone['label'] == 'yellow_cone')]
        self.yellows.sort(key=take_x)

        self.large_oranges = [cone for cone in cones if (cone['label'] == 'large_orange_cone')]
        self.large_oranges.sort(key=take_x)

        self.oranges = [cone for cone in cones if (cone['label'] == 'orange_cone')]
        self.oranges.sort(key=take_x)
        
        self.brake_condition = (len(self.oranges) >= 6) and (self.oranges[0]['coords']['y'] < 1) and (len(self.blues) == 0 and len(self.yellows) == 0)
        
        # Intersection trigger with hysteresis
        self._intersection_trigger()
        

        # Default behavior, overwritten if turning
        if (len(self.blues) > 0) and (len(self.yellows) > 0):
            #I assume they're sorted from closer to further
            center = (self.blues[0]['coords']['y'] + self.yellows[0]['coords']['y']) / 2
            # print(f'center:{center}')
            agent_act['steer'] = center * 0.5 # 1 left, -1 right, 0 neutral TODO HACER CON MAS SENTIDO
        elif len(self.blues) > 0:
            agent_act['steer'] = -1 # 1 left, -1 right, 0 neutral
        elif len(self.yellows) > 0:
            agent_act['steer'] = 1 # 1 left, -1 right, 0 neutral
        else:
            agent_act['steer'] = 0
        
        # Now check intersection and turn
        if self.intersection_trigger:
            # Found intersection. Turn to one side
            print('Turning to one side...')
            agent_act['acc_normalized'] = 0.05 # TODO REPLACE BY TARGET SPEED
            if(self.laps == 1 or self.laps == 2): # laps 1 and 2 to the right
                agent_act['steer'] = -.2 # 1 left, -1 right, 0 neutral
            elif(self.laps == 3 and self.laps == 4): # The last 2 laps must be to the left
                agent_act['steer'] = .2 # 1 left, -1 right, 0 neutral
            else: # If it has accomplished all 4 turns then its time to exit
                agent_act['steer'] = 0 # 1 left, -1 right, 0 neutral
        elif self.brake_condition: # da igual la velocidad, si ve conos naranjas y no ve la interseccion
            # Brake
            print('Braking...')
            agent_act['acc_normalized'] = 0.0
            agent_act['brake'] = 1.0
        elif (car_state['speed'] < 4): #si va lento y no ve conos naranjas
            # Accelerate
            print('Accelerating...')
            agent_act['acc_normalized'] = 0.5
        else: # si va rapido dejamos de acelerar
            # Too fast
            print('Slowing down...')
            agent_act['acc_normalized'] = 0.0
        
        print(f'lap {self.laps}, steer: {agent_act["steer"]}, ')
    def _intersection_trigger(self):
        if self.hysteresis < 1:
            # Consider if straight or turn
            # self.intersection_trigger = (len(large_oranges) == 4) and (large_oranges[0]['coords']['x'] < 1.2)
            if (len(self.large_oranges) > 1):
                print(f'I see big oranges at {self.large_oranges[0]["coords"]["x"]}')
                self.intersection_trigger = (self.large_oranges[0]['coords']['x'] < 2.5)
            else:
                self.intersection_trigger = False
            
            if self.intersection_trigger:
                self.hysteresis = 10
                print('INTERSECTION TRIGGER ------------------------')
                self.laps += 1 # Add one to the counter
        else:
            # Continue turning
            self.hysteresis -= 1
            print(f'hysteresis (>0): {self.hysteresis}')
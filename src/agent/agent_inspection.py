from agent.agent import Agent

class Acceleration_Mission(Agent):
    def __init__(self):
        super().__init__()

    def act_sim(self, cones, sim_client2, simulator_car_controls):
        super().act_sim(cones, sim_client2, simulator_car_controls)

    # SUPERCHARGED METHODS
    def get_target(self, car_state, agent_act, time): #Solo se ejecutará una vez
        '''
        Update agent_act, calculated from the cones and car_state.
        '''

        # SPEED
        # Slowly spining drivetrain
        if (car_state['speed'] < 2 and time < 25): # Mantener velocidad lenta al menos que llegue a los 25 seg
            agent_act['acc'] = 0.25
        elif time >= 25:
            if(car_state['speed'] < 0.1): #Si se ha parado completamente, AS_Finished
                return True
            else: # If it's fast we stop accelerating 
                agent_act['brake'] = 1.0
        else: 
            agent_act['acc'] = 0.0
        
        # STEER
        if (time < 25):
            if(time % 2 == 0): # Movemos a la izquierda si es un segundo par y a la derecha si es impar
                agent_act['steer'] = 0.75 # 1 left, -1 right, 0 neutral
            else:
                agent_act['steer'] = -0.75 # 1 left, -1 right, 0 neutral
        else:
            agent_act['steer'] = 0

        return False
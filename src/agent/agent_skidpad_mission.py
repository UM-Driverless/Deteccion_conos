from agent.agent import Agent

# TODO USAR COORDS CONOS GRANDES NARANJAS CUANDO CERCA?

class Skidpad_Mission(Agent):
    
    def __init__(self):
        super().__init__()
        self.hysteresis = 0 # Blocks self.intersection_state from changing temporarily. When < 1 allows change.
        self.intersection_state = 0 # 0 = Normal track, 1 = Load trigger, 2 = Turn when stopped seeing large orange cones
        self.laps: int = 0 # Variable that keeps track of the number of times that the car has crossed the intersection
        self.speed_target = 4
    def get_target(self, cones, car_state, actuation):
        '''
        Update actuation, calculated from the cones and car_state.
        '''
        
        self._sort_cones(cones)
        
        # TODO x amount of frames in a row to consider change of state, not just two.
        # TODO USE BIG ORANGE CONES AS TARGET WHEN SEEN. IGNORE THE REST.
        
        self._intersection_state_machine()
        self.steering_control(car_state, actuation)
        self.speed_control(car_state, actuation)
        
        print(f'lap {self.laps}, steer: {actuation["steer"]:6.3f}, acc: {actuation["acc"]:6.3f}, brake: {actuation["brake"]:6.3f}, INTERSECTION STATE: {self.intersection_state}')
    def _intersection_state_machine(self):
        if self.hysteresis < 1:
            # Consider the intersection - Change state
            if self.intersection_state == 0: # No intersection before
                if (self.large_orange_position() > 0 and self.large_orange_position() < 3.0):
                    self.intersection_state = 1 # Intersection now
            elif self.intersection_state == 1: # Intersection before
                if self.large_orange_position() < 0.:
                    self.intersection_state = 2 # Out of intersection now
                    self.hysteresis = 10
                    print('INTERSECTION TRIGGER ------------------------')
                    self.laps += 1 # Add one to the counter
            elif self.intersection_state == 2:
                if self.large_orange_position() < 0:
                    self.intersection_state = 0 # Out of intersection now
        else:
            # Just turn without thinking
            self.hysteresis -= 1
            print(f'hysteresis (>0): {self.hysteresis}')
    def steering_control(self, car_state, actuation):
        '''
        Depending on the values of the state machine, act
        '''
        if self.intersection_state == 2:
            # Found intersection. Turn to one side
            print('Steering intersection...')
            if(self.laps == 1 or self.laps == 2): # laps 1 and 2 to the right
                actuation['steer'] = -.25 # + = left
            elif(self.laps == 3 or self.laps == 4): # The last 2 laps must be to the left
                actuation['steer'] = .25 # + = left
            else: # If it has accomplished all 4 turns, then it's time to exit
                actuation['steer'] = 0 # + = left
        else:
            # Just steer normally
            print('Steering normally...')
            if (len(self.blues) > 0) and (len(self.yellows) > 0):
                #I assume they're sorted from closer to further
                center = (self.blues[0]['coords']['y'] + self.yellows[0]['coords']['y']) / 2
                # print(f'center:{center}')
                actuation['steer'] = center * 0.5 # + = left
            elif len(self.blues) > 0:
                actuation['steer'] = -1.8 # + = left
            elif len(self.yellows) > 0:
                actuation['steer'] = 1.8 # + = left
            else:
                actuation['steer'] = 0.
            
            # FORCE GOOD DIRECTION OF TURN WHEN CLOSE TO THE MESS
            if (self.large_orange_position() > 0. and self.large_orange_position() < 4.):
                if (self.laps == 1 or self.laps == 2):
                    actuation['steer'] = -abs(actuation['steer'])
                elif (self.laps == 3 or self.laps == 4):
                    actuation['steer'] = abs(actuation['steer'])
                elif self.laps >= 5:
                    # Go to braking zone
                    if len(self.oranges) >= 2:
                        actuation['steer'] = 0.5 * (self.oranges[0]['coords']['y'] + self.oranges[1]['coords']['y']) / 2
                    else:
                        actuation['steer'] = 0
        
    def speed_control(self, car_state, actuation):
        '''
        Default speed control, ignoring braking condition.
        '''
        # Adjust target speed
        if self.large_orange_position() > 0. and self.large_orange_position() < 3.5:
            self.speed_target = 4.
        else:
            self.speed_target = 5.
        
        # Speed control
        if (len(self.oranges) >= 6) and (self.oranges[0]['coords']['y'] < 0.5) and len(self.blues) < 2 and len(self.yellows) < 2:
            # Braking region
            print('Braking...')
            actuation['acc'] = 0.0
            actuation['brake'] = 1.0
        else:
            # Accelerate normally
            actuation['acc'] = (self.speed_target - car_state['speed']) * 0.1
        
        # If negative acceleration, brake instead
        if actuation['acc'] < 0:
            actuation['brake'] = -actuation['acc']
            actuation['acc'] = 0
    def _sort_cones(self, cones):
        def take_x(cone): return cone['coords']['x']
        
        self.blues = [cone for cone in cones if (cone['label'] == 'blue_cone')]
        self.blues.sort(key=take_x)
        self.yellows = [cone for cone in cones if (cone['label'] == 'yellow_cone')]
        self.yellows.sort(key=take_x)
 
        self.large_oranges = [cone for cone in cones if (cone['label'] == 'large_orange_cone')]
        self.large_oranges.sort(key=take_x)

        self.oranges = [cone for cone in cones if (cone['label'] == 'orange_cone')]
        self.oranges.sort(key=take_x)
    def large_orange_position(self):
        '''
        Returns the longitudinal position of the closest big orange cone visible, relative to the car camera.
        If there are less than cone_min cones, returns -1.0 (impossible position seen from the camera)
        '''
        cone_min = 2
        if (len(self.large_oranges) > cone_min):
            return self.large_oranges[0]['coords']['x']
        else:
            return -1.0
'''
Here is all the logic of the agent
'''

def testAction(images, speed, throttle, steer, brake):
    ''' This function only makes the car accelerate '''

    return [throttle, 0.0, steer]
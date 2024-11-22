import os



VISION_W = 1  #  One lane on either side visible
VISION_F = 21  # 21 grid spaces in front of the car visible
VISION_B = 14  # 14 grid spaces behind the card visible

# When Training
# VISUALENABLED = False 
# DLAGENTENABLED = True
# DL_IS_TRAINING = True

# # When not training
VISUALENABLED = True # Whether to show GUI or not
DLAGENTENABLED = False  # Whether the subject car is controlled manually or run by DL Agent 
DL_IS_TRAINING = False  # Whether training is being done or not

MAX_SIMULATION_CAR = 60 # Max No. of object cars

CONSTANT_PENALTY = -0.005 # A constant negative score added at evrry frame to prevent total reward from overshooting

EMERGENCY_BRAKE_MAX_SPEED_DIFF = 20 # Difference of Speed with car at front at which emergency break to apply
EMERGENCY_BRAKE_PENALTY = -0.005 # Penalty for the same


MISMATCH_ACTION_PENALTY = -0.0001 # Penalty given if the agent tries to take action it is not allowed to
SWITCHING_LANE_PENALTY = -0.00001 # Penaly given if agent immediately switches back to a lane it just came from

MAX_MEM = 1000000
MAX_EPISODE = 1000
MAX_FRAME_COUNT = 40000

LEARNING_RATE =  0.0001
BATCH_SIZE =  64
EPSILON_GREEDY_START_PROB = 1.0
EPSILON_GREEDY_END_PROB = 0.1
EPSILON_GREEDY_MAX_STATES = 1000000
EPSILON_GREEDY_TEST_PROB = 0.05
TARGET_NETWORK_UPDATE_FREQUENCY = 10000
LEARN_START = 100000

MODEL_NAME = 'DQN__lr={}_input=36-3_conv=2_FC=2_nn=100-5_batch={}'\
    .format(LEARNING_RATE, BATCH_SIZE)

# Configs for GUI
ROAD_VIEW_OFFSET = 300
INPUT_VIEW_OFFSET_X = 100
INPUT_VIEW_OFFSET_Y = 320

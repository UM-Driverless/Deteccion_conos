## IDs de SEN
SEN_ID = {
    # Rueda delantera izquierda
    "IMU_SENFL": int('300', 16),
    "SIG_SENFL": int('301', 16),
    "STATE_SENFL": int('302', 16),
    # Rueda delantera derecha
    "IMU_SENFR": int('303', 16),
    "SIG_SENFR": int('304', 16),
    "STATE_SENFR": int('305', 16),
    # Rueda trasera izquierda
    "IMU_SENRL": int('306', 16),
    "SIG_SENRL": int('307', 16),
    "STATE_SENRL": int('308', 16),
    # Rueda trasera derecha
    "IMU_SENRR": int('309', 16),
    "SIG_SENRR": int('310', 16),
    "STATE_SENRR": int('311', 16)
}

## IDs de TRAJECTORY
TRAJ_ID = {
    'TRAJECTORY_ACT': int('320', 16),
    'TRAJECTORY_GPS': int('321', 16),
    'TRAJECTORY_IMU': int('322', 16),
    'TRAJECTORY_STATE': int('323', 16),
}

## IDs de ASSIS
ASSIS_ID = {
    'ASSIS_C': int('350', 16),
    'ASSIS_R': int('351', 16),
    'ASSIS_L': int('352', 16),
}

## IDs de ASB
ASB_ID = {
    'ASB_ANALOG': int('360', 16),
    'ASB_SIGNALS': int('361', 16),
    'ASB_STATE': int('362', 16)
}

# Arduino ID
ARD_ID = {
    'ARD_ID': int('201', 16)
}
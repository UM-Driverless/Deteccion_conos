## IDs de SEN
SEN_ID = {
# Rueda delantera izquierda
"IMU_SENFL": int("300", 16),
"SIG_SENFL": int("301", 16),
"STATE_SENFL": int("302", 16),
# Rueda delantera derecha
"IMU_SENFR": int("303", 16),
"SIG_SENFR": int("304", 16),
"STATE_SENFR": int("305", 16),
# Rueda trasera izquierda
"IMU_SENRL": int("306", 16),
"SIG_SENRL": int("307", 16),
"STATE_SENRL": int("308", 16),
# Rueda trasera derecha
"IMU_SENRR": int("309", 16),
"SIG_SENRR": int("310", 16),
"STATE_SENRR": int("311", 16),
}

## IDs de TRAJECTORY
TRAJ_ID = {
"TRAJ_ACT": int("320", 16),
"TRAJ_GPS": int("321", 16),
"TRAJ_IMU": int("322", 16),
"TRAJ_STATE": int("323", 16),
}

## IDs STEERING MESSAGES
STEER_ID = {
"STEER_ID": int("601", 16),  # 0x601
# MSG F: Toggle New Position Bit
"MSG_00": int("2B", 16),  # 0x2B Envíar 2 bytes a la controladora
"MSG_01": int("40", 16),  # 0x40 Indice low
"MSG_02": int("60", 16),  # 0x60 Indice High
"MSG_03": int("00", 16),  # 0x00 Subindice
"MSG_04": int("0F", 16),  # 0x0F Datos 0
"MSG_05": int("00", 16),  # 0x00 Datos 1
# MSG D: Posicion objetivo
"MSG_10": int("23", 16),  # 0x23 Envíar 4 bytes a la controladora
"MSG_11": int("7A", 16),  # 0x7A Indice low
"MSG_12": int("60", 16),  # 0x60 Indice High
"MSG_13": int("00", 16),  # 0x00 Subindice
# MSG E: Orden de posicionamiento
"MSG_20": int("2B", 16),  # 0x2B Envíar 2 bytes a la controladora
"MSG_21": int("40", 16),  # 0x40 Indice low
"MSG_22": int("60", 16),  # 0x60 Indice High
"MSG_23": int("00", 16),  # 0x00 Subindice
"MSG_24": int("3F", 16),  # 0x3F Datos 0
"MSG_25": int("00", 16),  # 0x00 Datos 1
}

## IS STEERING inicialización
STEER_INIT_ID = {
# MSG A: Perfil de posición
"MSG_00": int("2F", 16),  # 0x2F Envíar 1 bytes a la controladora
"MSG_01": int("60", 16),  # 0x60 Indice low
"MSG_02": int("60", 16),  # 0x60 Indice High
"MSG_03": int("00", 16),  # 0x00 Subindice
"MSG_04": int("01", 16),  # 0x01 Datos 0
# MSG B: Parámetros
"MSG_10": int("00", 16),  # 0x00 ...
"MSG_11": int("00", 16),  # 0x00 ...
"MSG_12": int("00", 16),  # 0x00 ..
"MSG_13": int("00", 16),  # 0x00 ...
"MSG_14": int("00", 16),  # 0x00 ...
"MSG_15": int("00", 16),  # 0x00 ...
"MSG_16": int("00", 16),  # 0x00 ..
"MSG_17": int("00", 16),  # 0x00 ...
# MSG C: Habilitar
"MSG_20": int("2B", 16),  # 0x2B Envíar 2 bytes a la controladora
"MSG_21": int("40", 16),  # 0x40 Indice low
"MSG_22": int("60", 16),  # 0x60 Indice High
"MSG_23": int("00", 16),  # 0x00 Subindice
"MSG_24": int("0F", 16),  # 0x0F Datos 0
"MSG_25": int("00", 16),  # 0x00 Datos 1
# MSG Ñ: Deshabilitar
"MSG_30": int("2B", 16),  # 0x2B Envíar 2 bytes a la controladora
"MSG_31": int("40", 16),  # 0x40 Indice low
"MSG_32": int("60", 16),  # 0x60 Indice High
"MSG_33": int("00", 16),  # 0x00 Subindice
"MSG_34": int("06", 16),  # 0x06 Datos 0
"MSG_35": int("00", 16),  # 0x00 Datos 1
}

## IDs de ASSIS
ASSIS_ID = {
"ASSIS_C": int("350", 16),
"ASSIS_R": int("351", 16),
"ASSIS_L": int("352", 16),
}

## IDs de ASB
ASB_ID = {
"ASB_ANALOG": int("360", 16),
"ASB_SIGNALS": int("361", 16),
"ASB_STATE": int("362", 16),
}

## ID Arduino
ARD_ID = {
"ID": int("201", 16)
}

CAN_SEND_MSG_TIMEOUT = 0.005
CAN_ACTION_DIMENSION = 100.
CAN_STEER_DIMENSION = 122880.

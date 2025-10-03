### DS300

T_GROUP_TO_COND = {
    1:  {"chamber_pressure": 5,  "injection_pressure": 2200, "control_backpressure": 1},
    2:  {"chamber_pressure": 15, "injection_pressure": 2200, "control_backpressure": 1},
    3:  {"chamber_pressure": 25, "injection_pressure": 2200, "control_backpressure": 1},
    4:  {"chamber_pressure": 35, "injection_pressure": 2200, "control_backpressure": 1},
    5:  {"chamber_pressure": 5,  "injection_pressure": 1400, "control_backpressure": 1},
    6:  {"chamber_pressure": 15, "injection_pressure": 1400, "control_backpressure": 1},
    7:  {"chamber_pressure": 35, "injection_pressure": 1400, "control_backpressure": 1},
    8:  {"chamber_pressure": 5,  "injection_pressure": 2200, "control_backpressure": 4},
    9:  {"chamber_pressure": 15, "injection_pressure": 2200, "control_backpressure": 4},
    10: {"chamber_pressure": 35, "injection_pressure": 2200, "control_backpressure": 4},
    11: {"chamber_pressure": 5,  "injection_pressure": 1600, "control_backpressure": 1},
    12: {"chamber_pressure": 35, "injection_pressure": 1600, "control_backpressure": 1},
}
 
def cine_to_injection_duration_us(cine_number: int) -> float:
    # 1..5 -> 340; +20 each block up to 91..95 -> 700
    # 96..100 -> 750; then +50 per block up to 141..145 -> 1200
    cine_number = max(1, min(145, int(cine_number)))
    block = (cine_number - 1) // 5
    if block <= 18:
        return 340 + 20 * block
    else:
        return 750 + 50 * (block - 19)
    


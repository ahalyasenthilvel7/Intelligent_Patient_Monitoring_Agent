
import random
from datetime import datetime

def generate_reading(mode="Auto"):

    if mode == "Normal":
        heart_rate = random.randint(65, 95)
        spo2 = random.randint(96, 100)
        temperature = round(random.uniform(36.2, 37.2), 1)
        systolic_bp = random.randint(105, 130)
        diastolic_bp = random.randint(65, 85)
        respiratory_rate = random.randint(12, 20)

    elif mode == "Unusual":
        heart_rate = random.randint(125, 145)
        spo2 = random.randint(85, 91)
        temperature = round(random.uniform(38.5, 39.8), 1)
        systolic_bp = random.randint(165, 190)
        diastolic_bp = random.randint(100, 115)
        respiratory_rate = random.randint(27, 35)

    else:
        if random.random() < 0.15:
            return generate_reading("Unusual")
        return generate_reading("Normal")

    return {
        "timestamp": datetime.now(),
        "heart_rate": heart_rate,
        "spo2": spo2,
        "temperature": temperature,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "respiratory_rate": respiratory_rate
    }

import time, statistics, requests
from concurrent.futures import ThreadPoolExecutor
B = 'https://aci606-cloud-fleet-tik7.onrender.com'
S = {'robot_id': 'ENSAYO', 'timestamp': 1780000000.0, 'x': 1.0, 'y': 1.0, 'speed': 0.7, 'battery': 70.0, 'motor_temp': 55.0, 'vibration': 0.6, 'motor_current': 4.0, 'obstacle_distance': 0.5}
def one(_):
    t = time.perf_counter()
    r = requests.post(B + '/predict', json=S, timeout=60)
    return (time.perf_counter() - t) * 1000, r.status_code
one(0)
lat = [one(0)[0] for _ in range(30)]
print('secuencial P50 / P95 (ms):', round(statistics.median(lat)), round(statistics.quantiles(lat, n=100)[94]))
with ThreadPoolExecutor(16) as ex:
    res = list(ex.map(one, range(160)))
print('concurrente, errores:', sum(c != 200 for _, c in res), 'de', len(res))
bad = dict(S, battery=150.0)
print('dato invalido ->', requests.post(B + '/predict', json=bad, timeout=60).status_code)
print('health ->', requests.get(B + '/health', timeout=60).status_code)

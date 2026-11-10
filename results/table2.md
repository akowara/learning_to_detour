**Table 2.** Success rate per target (maximum) speed in the Isaac Sim city, with the mean flight speed in m/s in parentheses (10 episodes per cell, mean over all episodes, so a policy that stalls or hovers shows a low speed). Avg speed: mean flight speed over all target speeds. Success: within 5 m of the goal without collision.

| Route | Policy | 4 m/s | 6 m/s | 8 m/s | 10 m/s | 12 m/s | Avg speed |
|---|---|---|---|---|---|---|---|
| Open street | Original policy | 100% (3.7) | 100% (5.2) | 100% (6.8) | 100% (8.1) | 100% (9.2) | 6.6 |
|  | Fine-tuned, no SHAC | 100% (4.8) | 100% (6.0) | 100% (7.1) | 100% (8.3) | 100% (9.4) | 7.1 |
|  | Fine-tuned, SHAC (ours) | **100% (5.4)** | **100% (6.3)** | **100% (7.5)** | **100% (8.6)** | **100% (9.6)** | 7.4 |
| Turn | Original policy | 0% (0.6) | 0% (0.7) | 0% (0.7) | 0% (0.8) | 0% (3.6) | 1.3 |
|  | Fine-tuned, no SHAC | 10% (3.7) | 70% (2.5) | 90% (3.1) | 60% (3.0) | 80% (3.6) | 3.2 |
|  | Fine-tuned, SHAC (ours) | **100% (3.2)** | **100% (3.3)** | **100% (3.7)** | **100% (3.1)** | **80% (2.3)** | 3.1 |
| Detour | Original policy | 0% (1.0) | 0% (1.0) | 0% (1.1) | 0% (6.9) | 0% (7.1) | 3.4 |
|  | Fine-tuned, no SHAC | 0% (3.2) | 0% (3.9) | 0% (3.0) | 0% (4.4) | 0% (4.6) | 3.8 |
|  | Fine-tuned, SHAC (ours) | **100% (4.1)** | **100% (4.5)** | **100% (4.6)** | **100% (4.4)** | **100% (4.6)** | 4.4 |

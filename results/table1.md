**Table 1.** Goal-reaching in an unseen photorealistic city (Isaac Sim), per route and policy. Each cell pools 5 maximum speeds (4, 6, 8, 10, 12 m/s) × 10 episodes. Success = within 5 m of the goal (horizontal) without collision; timeout = neither within 140 s. Time to goal and path ratio (flown length / shortest collision-free path to the goal radius) are over successful episodes.

| Route | Policy | Success | Collision | Timeout | Time to goal (s) | Path ratio |
|---|---|---|---|---|---|---|
| Open street (112 m straight, 112 m shortest) | Original policy | 100% | 0% | 0% | 18.0 ± 6.3 | 0.99 ± 0.01 |
|  | Fine-tuned, no SHAC | 100% | 0% | 0% | 15.9 ± 3.8 | 0.99 ± 0.01 |
|  | Fine-tuned, SHAC (ours) | **100%** | 0% | 0% | 15.0 ± 3.2 | 1.00 ± 0.01 |
| Turn (103 m straight, 116 m shortest) | Original policy | 0% | 68% | 32% | -- | -- |
|  | Fine-tuned, no SHAC | 62% | 4% | 34% | 48.4 ± 11.8 | 1.36 ± 0.22 |
|  | Fine-tuned, SHAC (ours) | **96%** | 0% | 4% | 45.5 ± 7.8 | 1.29 ± 0.10 |
| Detour (165 m straight, 310 m shortest) | Original policy | 0% | 42% | 58% | -- | -- |
|  | Fine-tuned, no SHAC | 0% | 40% | 60% | -- | -- |
|  | Fine-tuned, SHAC (ours) | **100%** | 0% | 0% | 87.2 ± 4.7 | 1.26 ± 0.01 |

Eval settings (all runs): altitude 6 m, goal radius 5 m (vertical tolerance 5 m), tilt limit 35°, 3 s speed ramp, ±3 m start jitter, attitude gains kr 3.5 / kw 0.5.

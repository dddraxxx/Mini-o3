# Qwen3.5 Base Raw-Crop RL Validation Tool Usage

Run: `save/qwen35_9b_official_tool_h200_rl_from_base_rawcrop_t12_100step_20260531_220325`

This report summarizes validation generations by checkpoint step and VisualProbe subset. `img_idx=0` means the model cropped the original image; `img_idx>0` means it cropped a previous zoom observation. This run was started before the new `clip/exceed/format/invalid` logging change, so per-sample failure fields still use the old JSON names where present.

## Per-Step Subset Summary

| step | subset | n | reward | final | format | tools mean | tools min-max | original-call % | previous-crop-call % | cases using previous crop % | requested img_idx counts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 10 | easy | 32 | 0.5000 | 0.9688 | 0.9375 | 2.31 | 0-8 | 98.63% | 1.37% | 3.12% | `0:72, 1:1` |
| 10 | medium | 32 | 0.4062 | 0.9375 | 0.9375 | 2.41 | 0-11 | 93.42% | 6.58% | 9.38% | `0:71, 1:3, 2:2` |
| 10 | hard | 32 | 0.1875 | 0.7500 | 0.7500 | 3.81 | 1-11 | 99.17% | 0.00% | 0.00% | `0:120, None:1` |
| 20 | easy | 32 | 0.5312 | 0.9688 | 0.9688 | 2.38 | 0-7 | 100.00% | 0.00% | 0.00% | `0:76` |
| 20 | medium | 32 | 0.4375 | 0.9062 | 0.9062 | 2.41 | 0-10 | 98.70% | 1.30% | 3.12% | `0:76, 1:1` |
| 20 | hard | 32 | 0.3438 | 0.9062 | 0.9062 | 3.38 | 0-11 | 97.20% | 2.80% | 6.25% | `0:104, 1:3` |
| 30 | easy | 32 | 0.4062 | 0.9375 | 0.9062 | 2.47 | 0-9 | 100.00% | 0.00% | 0.00% | `0:79` |
| 30 | medium | 32 | 0.5000 | 0.9375 | 0.9062 | 1.97 | 0-6 | 98.41% | 1.59% | 3.12% | `0:62, 1:1` |
| 30 | hard | 32 | 0.2188 | 0.8438 | 0.8125 | 3.50 | 0-10 | 100.00% | 0.00% | 0.00% | `0:112` |
| 40 | easy | 32 | 0.5312 | 0.9375 | 0.9375 | 2.28 | 0-5 | 98.63% | 1.37% | 3.12% | `0:72, 1:1` |
| 40 | medium | 32 | 0.3438 | 0.9062 | 0.9062 | 2.66 | 0-11 | 100.00% | 0.00% | 0.00% | `0:84` |
| 40 | hard | 32 | 0.1250 | 0.8438 | 0.8438 | 3.56 | 0-11 | 100.00% | 0.00% | 0.00% | `0:112` |
| 50 | easy | 32 | 0.5625 | 0.9062 | 0.9062 | 2.94 | 1-8 | 93.62% | 6.38% | 6.25% | `0:88, 1:1, 2:1, 3:4` |
| 50 | medium | 32 | 0.4062 | 0.9062 | 0.9062 | 2.47 | 0-11 | 100.00% | 0.00% | 0.00% | `0:78` |
| 50 | hard | 32 | 0.2188 | 0.7812 | 0.7812 | 3.19 | 0-11 | 100.00% | 0.00% | 0.00% | `0:101` |
| 60 | easy | 32 | 0.5312 | 0.9062 | 0.8750 | 3.88 | 0-11 | 100.00% | 0.00% | 0.00% | `0:122` |
| 60 | medium | 32 | 0.4375 | 0.8750 | 0.8750 | 3.66 | 0-11 | 100.00% | 0.00% | 0.00% | `0:113` |
| 60 | hard | 32 | 0.1875 | 0.7812 | 0.7812 | 3.06 | 0-11 | 95.83% | 4.17% | 9.38% | `0:92, 1:3, 2:1` |
| 70 | easy | 32 | 0.4375 | 0.9375 | 0.9375 | 2.56 | 0-9 | 100.00% | 0.00% | 0.00% | `0:82` |
| 70 | medium | 32 | 0.5312 | 0.8750 | 0.8750 | 3.22 | 1-11 | 98.04% | 0.98% | 3.12% | `0:100, 1:1, None:1` |
| 70 | hard | 32 | 0.1875 | 0.8125 | 0.8125 | 3.69 | 0-11 | 89.38% | 10.62% | 12.50% | `0:101, 1:3, 2:1, 3:1, 4:7` |

## Requested Image Index By Call Turn

### Step 10

#### Easy

| call turn | calls | img_idx=0 | img_idx=1 | img_idx=2 | img_idx>=3 | original-call % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 30 | 30 | 0 | 0 | 0 | 100.00% |
| 2 | 24 | 23 | 1 | 0 | 0 | 95.83% |
| 3 | 9 | 9 | 0 | 0 | 0 | 100.00% |
| 4 | 3 | 3 | 0 | 0 | 0 | 100.00% |
| 5 | 2 | 2 | 0 | 0 | 0 | 100.00% |
| 6 | 2 | 2 | 0 | 0 | 0 | 100.00% |
| 7 | 2 | 2 | 0 | 0 | 0 | 100.00% |
| 8 | 1 | 1 | 0 | 0 | 0 | 100.00% |

#### Medium

| call turn | calls | img_idx=0 | img_idx=1 | img_idx=2 | img_idx>=3 | original-call % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 31 | 31 | 0 | 0 | 0 | 100.00% |
| 2 | 23 | 21 | 2 | 0 | 0 | 91.30% |
| 3 | 9 | 6 | 1 | 2 | 0 | 66.67% |
| 4 | 4 | 4 | 0 | 0 | 0 | 100.00% |
| 5 | 2 | 2 | 0 | 0 | 0 | 100.00% |
| 6 | 2 | 2 | 0 | 0 | 0 | 100.00% |
| 7 | 1 | 1 | 0 | 0 | 0 | 100.00% |
| 8 | 1 | 1 | 0 | 0 | 0 | 100.00% |
| 9 | 1 | 1 | 0 | 0 | 0 | 100.00% |
| 10 | 1 | 1 | 0 | 0 | 0 | 100.00% |
| 11 | 1 | 1 | 0 | 0 | 0 | 100.00% |

#### Hard

| call turn | calls | img_idx=0 | img_idx=1 | img_idx=2 | img_idx>=3 | original-call % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 32 | 0 | 0 | 0 | 100.00% |
| 2 | 29 | 28 | 0 | 0 | 0 | 96.55% |
| 3 | 17 | 17 | 0 | 0 | 0 | 100.00% |
| 4 | 11 | 11 | 0 | 0 | 0 | 100.00% |
| 5 | 7 | 7 | 0 | 0 | 0 | 100.00% |
| 6 | 6 | 6 | 0 | 0 | 0 | 100.00% |
| 7 | 5 | 5 | 0 | 0 | 0 | 100.00% |
| 8 | 5 | 5 | 0 | 0 | 0 | 100.00% |
| 9 | 5 | 5 | 0 | 0 | 0 | 100.00% |
| 10 | 2 | 2 | 0 | 0 | 0 | 100.00% |
| 11 | 2 | 2 | 0 | 0 | 0 | 100.00% |

### Step 20

#### Easy

| call turn | calls | img_idx=0 | img_idx=1 | img_idx=2 | img_idx>=3 | original-call % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 30 | 30 | 0 | 0 | 0 | 100.00% |
| 2 | 21 | 21 | 0 | 0 | 0 | 100.00% |
| 3 | 12 | 12 | 0 | 0 | 0 | 100.00% |
| 4 | 6 | 6 | 0 | 0 | 0 | 100.00% |
| 5 | 4 | 4 | 0 | 0 | 0 | 100.00% |
| 6 | 2 | 2 | 0 | 0 | 0 | 100.00% |
| 7 | 1 | 1 | 0 | 0 | 0 | 100.00% |

#### Medium

| call turn | calls | img_idx=0 | img_idx=1 | img_idx=2 | img_idx>=3 | original-call % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 31 | 31 | 0 | 0 | 0 | 100.00% |
| 2 | 22 | 21 | 1 | 0 | 0 | 95.45% |
| 3 | 11 | 11 | 0 | 0 | 0 | 100.00% |
| 4 | 4 | 4 | 0 | 0 | 0 | 100.00% |
| 5 | 2 | 2 | 0 | 0 | 0 | 100.00% |
| 6 | 2 | 2 | 0 | 0 | 0 | 100.00% |
| 7 | 2 | 2 | 0 | 0 | 0 | 100.00% |
| 8 | 1 | 1 | 0 | 0 | 0 | 100.00% |
| 9 | 1 | 1 | 0 | 0 | 0 | 100.00% |
| 10 | 1 | 1 | 0 | 0 | 0 | 100.00% |

#### Hard

| call turn | calls | img_idx=0 | img_idx=1 | img_idx=2 | img_idx>=3 | original-call % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 31 | 31 | 0 | 0 | 0 | 100.00% |
| 2 | 25 | 24 | 1 | 0 | 0 | 96.00% |
| 3 | 18 | 17 | 1 | 0 | 0 | 94.44% |
| 4 | 12 | 11 | 1 | 0 | 0 | 91.67% |
| 5 | 5 | 5 | 0 | 0 | 0 | 100.00% |
| 6 | 4 | 4 | 0 | 0 | 0 | 100.00% |
| 7 | 3 | 3 | 0 | 0 | 0 | 100.00% |
| 8 | 3 | 3 | 0 | 0 | 0 | 100.00% |
| 9 | 3 | 3 | 0 | 0 | 0 | 100.00% |
| 10 | 2 | 2 | 0 | 0 | 0 | 100.00% |
| 11 | 1 | 1 | 0 | 0 | 0 | 100.00% |

### Step 30

#### Easy

| call turn | calls | img_idx=0 | img_idx=1 | img_idx=2 | img_idx>=3 | original-call % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 30 | 30 | 0 | 0 | 0 | 100.00% |
| 2 | 21 | 21 | 0 | 0 | 0 | 100.00% |
| 3 | 11 | 11 | 0 | 0 | 0 | 100.00% |
| 4 | 7 | 7 | 0 | 0 | 0 | 100.00% |
| 5 | 4 | 4 | 0 | 0 | 0 | 100.00% |
| 6 | 3 | 3 | 0 | 0 | 0 | 100.00% |
| 7 | 1 | 1 | 0 | 0 | 0 | 100.00% |
| 8 | 1 | 1 | 0 | 0 | 0 | 100.00% |
| 9 | 1 | 1 | 0 | 0 | 0 | 100.00% |

#### Medium

| call turn | calls | img_idx=0 | img_idx=1 | img_idx=2 | img_idx>=3 | original-call % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 29 | 28 | 1 | 0 | 0 | 96.55% |
| 2 | 21 | 21 | 0 | 0 | 0 | 100.00% |
| 3 | 7 | 7 | 0 | 0 | 0 | 100.00% |
| 4 | 4 | 4 | 0 | 0 | 0 | 100.00% |
| 5 | 1 | 1 | 0 | 0 | 0 | 100.00% |
| 6 | 1 | 1 | 0 | 0 | 0 | 100.00% |

#### Hard

| call turn | calls | img_idx=0 | img_idx=1 | img_idx=2 | img_idx>=3 | original-call % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 31 | 31 | 0 | 0 | 0 | 100.00% |
| 2 | 27 | 27 | 0 | 0 | 0 | 100.00% |
| 3 | 18 | 18 | 0 | 0 | 0 | 100.00% |
| 4 | 13 | 13 | 0 | 0 | 0 | 100.00% |
| 5 | 9 | 9 | 0 | 0 | 0 | 100.00% |
| 6 | 5 | 5 | 0 | 0 | 0 | 100.00% |
| 7 | 3 | 3 | 0 | 0 | 0 | 100.00% |
| 8 | 3 | 3 | 0 | 0 | 0 | 100.00% |
| 9 | 2 | 2 | 0 | 0 | 0 | 100.00% |
| 10 | 1 | 1 | 0 | 0 | 0 | 100.00% |

### Step 40

#### Easy

| call turn | calls | img_idx=0 | img_idx=1 | img_idx=2 | img_idx>=3 | original-call % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 30 | 30 | 0 | 0 | 0 | 100.00% |
| 2 | 25 | 24 | 1 | 0 | 0 | 96.00% |
| 3 | 12 | 12 | 0 | 0 | 0 | 100.00% |
| 4 | 4 | 4 | 0 | 0 | 0 | 100.00% |
| 5 | 2 | 2 | 0 | 0 | 0 | 100.00% |

#### Medium

| call turn | calls | img_idx=0 | img_idx=1 | img_idx=2 | img_idx>=3 | original-call % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 31 | 31 | 0 | 0 | 0 | 100.00% |
| 2 | 23 | 23 | 0 | 0 | 0 | 100.00% |
| 3 | 11 | 11 | 0 | 0 | 0 | 100.00% |
| 4 | 5 | 5 | 0 | 0 | 0 | 100.00% |
| 5 | 4 | 4 | 0 | 0 | 0 | 100.00% |
| 6 | 3 | 3 | 0 | 0 | 0 | 100.00% |
| 7 | 2 | 2 | 0 | 0 | 0 | 100.00% |
| 8 | 2 | 2 | 0 | 0 | 0 | 100.00% |
| 9 | 1 | 1 | 0 | 0 | 0 | 100.00% |
| 10 | 1 | 1 | 0 | 0 | 0 | 100.00% |
| 11 | 1 | 1 | 0 | 0 | 0 | 100.00% |

#### Hard

| call turn | calls | img_idx=0 | img_idx=1 | img_idx=2 | img_idx>=3 | original-call % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 31 | 31 | 0 | 0 | 0 | 100.00% |
| 2 | 22 | 22 | 0 | 0 | 0 | 100.00% |
| 3 | 15 | 15 | 0 | 0 | 0 | 100.00% |
| 4 | 10 | 10 | 0 | 0 | 0 | 100.00% |
| 5 | 9 | 9 | 0 | 0 | 0 | 100.00% |
| 6 | 6 | 6 | 0 | 0 | 0 | 100.00% |
| 7 | 5 | 5 | 0 | 0 | 0 | 100.00% |
| 8 | 5 | 5 | 0 | 0 | 0 | 100.00% |
| 9 | 4 | 4 | 0 | 0 | 0 | 100.00% |
| 10 | 3 | 3 | 0 | 0 | 0 | 100.00% |
| 11 | 2 | 2 | 0 | 0 | 0 | 100.00% |

### Step 50

#### Easy

| call turn | calls | img_idx=0 | img_idx=1 | img_idx=2 | img_idx>=3 | original-call % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 32 | 0 | 0 | 0 | 100.00% |
| 2 | 26 | 25 | 0 | 1 | 0 | 96.15% |
| 3 | 16 | 16 | 0 | 0 | 0 | 100.00% |
| 4 | 8 | 6 | 1 | 0 | 1 | 75.00% |
| 5 | 4 | 3 | 0 | 0 | 1 | 75.00% |
| 6 | 4 | 3 | 0 | 0 | 1 | 75.00% |
| 7 | 3 | 2 | 0 | 0 | 1 | 66.67% |
| 8 | 1 | 1 | 0 | 0 | 0 | 100.00% |

#### Medium

| call turn | calls | img_idx=0 | img_idx=1 | img_idx=2 | img_idx>=3 | original-call % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 28 | 28 | 0 | 0 | 0 | 100.00% |
| 2 | 15 | 15 | 0 | 0 | 0 | 100.00% |
| 3 | 12 | 12 | 0 | 0 | 0 | 100.00% |
| 4 | 9 | 9 | 0 | 0 | 0 | 100.00% |
| 5 | 5 | 5 | 0 | 0 | 0 | 100.00% |
| 6 | 4 | 4 | 0 | 0 | 0 | 100.00% |
| 7 | 1 | 1 | 0 | 0 | 0 | 100.00% |
| 8 | 1 | 1 | 0 | 0 | 0 | 100.00% |
| 9 | 1 | 1 | 0 | 0 | 0 | 100.00% |
| 10 | 1 | 1 | 0 | 0 | 0 | 100.00% |
| 11 | 1 | 1 | 0 | 0 | 0 | 100.00% |

#### Hard

| call turn | calls | img_idx=0 | img_idx=1 | img_idx=2 | img_idx>=3 | original-call % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 29 | 29 | 0 | 0 | 0 | 100.00% |
| 2 | 24 | 24 | 0 | 0 | 0 | 100.00% |
| 3 | 18 | 18 | 0 | 0 | 0 | 100.00% |
| 4 | 11 | 11 | 0 | 0 | 0 | 100.00% |
| 5 | 5 | 5 | 0 | 0 | 0 | 100.00% |
| 6 | 4 | 4 | 0 | 0 | 0 | 100.00% |
| 7 | 3 | 3 | 0 | 0 | 0 | 100.00% |
| 8 | 3 | 3 | 0 | 0 | 0 | 100.00% |
| 9 | 2 | 2 | 0 | 0 | 0 | 100.00% |
| 10 | 1 | 1 | 0 | 0 | 0 | 100.00% |
| 11 | 1 | 1 | 0 | 0 | 0 | 100.00% |

### Step 60

#### Easy

| call turn | calls | img_idx=0 | img_idx=1 | img_idx=2 | img_idx>=3 | original-call % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 29 | 29 | 0 | 0 | 0 | 100.00% |
| 2 | 26 | 26 | 0 | 0 | 0 | 100.00% |
| 3 | 20 | 20 | 0 | 0 | 0 | 100.00% |
| 4 | 13 | 13 | 0 | 0 | 0 | 100.00% |
| 5 | 10 | 10 | 0 | 0 | 0 | 100.00% |
| 6 | 7 | 7 | 0 | 0 | 0 | 100.00% |
| 7 | 6 | 6 | 0 | 0 | 0 | 100.00% |
| 8 | 5 | 5 | 0 | 0 | 0 | 100.00% |
| 9 | 2 | 2 | 0 | 0 | 0 | 100.00% |
| 10 | 2 | 2 | 0 | 0 | 0 | 100.00% |
| 11 | 2 | 2 | 0 | 0 | 0 | 100.00% |

#### Medium

| call turn | calls | img_idx=0 | img_idx=1 | img_idx=2 | img_idx>=3 | original-call % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 29 | 29 | 0 | 0 | 0 | 100.00% |
| 2 | 22 | 22 | 0 | 0 | 0 | 100.00% |
| 3 | 14 | 14 | 0 | 0 | 0 | 100.00% |
| 4 | 12 | 12 | 0 | 0 | 0 | 100.00% |
| 5 | 9 | 9 | 0 | 0 | 0 | 100.00% |
| 6 | 7 | 7 | 0 | 0 | 0 | 100.00% |
| 7 | 4 | 4 | 0 | 0 | 0 | 100.00% |
| 8 | 4 | 4 | 0 | 0 | 0 | 100.00% |
| 9 | 4 | 4 | 0 | 0 | 0 | 100.00% |
| 10 | 4 | 4 | 0 | 0 | 0 | 100.00% |
| 11 | 4 | 4 | 0 | 0 | 0 | 100.00% |

#### Hard

| call turn | calls | img_idx=0 | img_idx=1 | img_idx=2 | img_idx>=3 | original-call % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 28 | 28 | 0 | 0 | 0 | 100.00% |
| 2 | 22 | 21 | 1 | 0 | 0 | 95.45% |
| 3 | 15 | 14 | 1 | 0 | 0 | 93.33% |
| 4 | 10 | 9 | 1 | 0 | 0 | 90.00% |
| 5 | 7 | 6 | 0 | 1 | 0 | 85.71% |
| 6 | 3 | 3 | 0 | 0 | 0 | 100.00% |
| 7 | 3 | 3 | 0 | 0 | 0 | 100.00% |
| 8 | 2 | 2 | 0 | 0 | 0 | 100.00% |
| 9 | 2 | 2 | 0 | 0 | 0 | 100.00% |
| 10 | 2 | 2 | 0 | 0 | 0 | 100.00% |
| 11 | 2 | 2 | 0 | 0 | 0 | 100.00% |

### Step 70

#### Easy

| call turn | calls | img_idx=0 | img_idx=1 | img_idx=2 | img_idx>=3 | original-call % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 27 | 27 | 0 | 0 | 0 | 100.00% |
| 2 | 24 | 24 | 0 | 0 | 0 | 100.00% |
| 3 | 13 | 13 | 0 | 0 | 0 | 100.00% |
| 4 | 8 | 8 | 0 | 0 | 0 | 100.00% |
| 5 | 5 | 5 | 0 | 0 | 0 | 100.00% |
| 6 | 2 | 2 | 0 | 0 | 0 | 100.00% |
| 7 | 1 | 1 | 0 | 0 | 0 | 100.00% |
| 8 | 1 | 1 | 0 | 0 | 0 | 100.00% |
| 9 | 1 | 1 | 0 | 0 | 0 | 100.00% |

#### Medium

| call turn | calls | img_idx=0 | img_idx=1 | img_idx=2 | img_idx>=3 | original-call % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 32 | 0 | 0 | 0 | 100.00% |
| 2 | 18 | 18 | 0 | 0 | 0 | 100.00% |
| 3 | 12 | 11 | 1 | 0 | 0 | 91.67% |
| 4 | 10 | 10 | 0 | 0 | 0 | 100.00% |
| 5 | 8 | 8 | 0 | 0 | 0 | 100.00% |
| 6 | 6 | 5 | 0 | 0 | 0 | 83.33% |
| 7 | 5 | 5 | 0 | 0 | 0 | 100.00% |
| 8 | 4 | 4 | 0 | 0 | 0 | 100.00% |
| 9 | 3 | 3 | 0 | 0 | 0 | 100.00% |
| 10 | 2 | 2 | 0 | 0 | 0 | 100.00% |
| 11 | 2 | 2 | 0 | 0 | 0 | 100.00% |

#### Hard

| call turn | calls | img_idx=0 | img_idx=1 | img_idx=2 | img_idx>=3 | original-call % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 31 | 31 | 0 | 0 | 0 | 100.00% |
| 2 | 23 | 20 | 3 | 0 | 0 | 86.96% |
| 3 | 12 | 11 | 0 | 1 | 0 | 91.67% |
| 4 | 10 | 9 | 0 | 0 | 1 | 90.00% |
| 5 | 7 | 6 | 0 | 0 | 1 | 85.71% |
| 6 | 5 | 4 | 0 | 0 | 1 | 80.00% |
| 7 | 5 | 4 | 0 | 0 | 1 | 80.00% |
| 8 | 5 | 4 | 0 | 0 | 1 | 80.00% |
| 9 | 5 | 4 | 0 | 0 | 1 | 80.00% |
| 10 | 5 | 4 | 0 | 0 | 1 | 80.00% |
| 11 | 5 | 4 | 0 | 0 | 1 | 80.00% |

## Invalid Reason Breakdown

This run was launched before the new `clip/exceed/format/invalid` code landed,
so this table maps old fields into the new intent: `turn_limit` comes from
`exceed_reason=*turn_limit*`; `response_length` comes from
`exceed_reason=*response_length*`; `length_stop` would come from
`void_reason=length`; `missing_final_answer` comes from old
`void_reason=missing_final_answer`.

| step | subset | n | invalid | turn limit | response length | length stop | missing final | invalid % | turn-limit % | length % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | easy | 32 | 1 | 0 | 0 | 0 | 1 | 3.12% | 0.00% | 0.00% |
| 10 | medium | 32 | 2 | 1 | 0 | 0 | 1 | 6.25% | 3.12% | 0.00% |
| 10 | hard | 32 | 8 | 2 | 0 | 0 | 6 | 25.00% | 6.25% | 0.00% |
| 20 | easy | 32 | 1 | 0 | 0 | 0 | 1 | 3.12% | 0.00% | 0.00% |
| 20 | medium | 32 | 2 | 0 | 0 | 0 | 2 | 6.25% | 0.00% | 0.00% |
| 20 | hard | 32 | 2 | 1 | 0 | 0 | 1 | 6.25% | 3.12% | 0.00% |
| 30 | easy | 32 | 2 | 0 | 0 | 0 | 2 | 6.25% | 0.00% | 0.00% |
| 30 | medium | 32 | 1 | 0 | 0 | 0 | 1 | 3.12% | 0.00% | 0.00% |
| 30 | hard | 32 | 5 | 0 | 1 | 0 | 4 | 15.62% | 0.00% | 3.12% |
| 40 | easy | 32 | 1 | 0 | 0 | 0 | 1 | 3.12% | 0.00% | 0.00% |
| 40 | medium | 32 | 2 | 1 | 0 | 0 | 1 | 6.25% | 3.12% | 0.00% |
| 40 | hard | 32 | 5 | 2 | 0 | 0 | 3 | 15.62% | 6.25% | 0.00% |
| 50 | easy | 32 | 2 | 0 | 1 | 0 | 1 | 6.25% | 0.00% | 3.12% |
| 50 | medium | 32 | 3 | 1 | 0 | 0 | 2 | 9.38% | 3.12% | 0.00% |
| 50 | hard | 32 | 4 | 0 | 0 | 0 | 4 | 12.50% | 0.00% | 0.00% |
| 60 | easy | 32 | 3 | 2 | 0 | 0 | 1 | 9.38% | 6.25% | 0.00% |
| 60 | medium | 32 | 4 | 4 | 0 | 0 | 0 | 12.50% | 12.50% | 0.00% |
| 60 | hard | 32 | 7 | 2 | 0 | 0 | 5 | 21.88% | 6.25% | 0.00% |
| 70 | easy | 32 | 2 | 0 | 0 | 0 | 2 | 6.25% | 0.00% | 0.00% |
| 70 | medium | 32 | 4 | 1 | 0 | 0 | 3 | 12.50% | 3.12% | 0.00% |
| 70 | hard | 32 | 6 | 5 | 0 | 0 | 1 | 18.75% | 15.62% | 0.00% |

Overall across all validation dumps:

- total rows: `672`
- invalid-equivalent rows: `67` (`9.97%`)
- turn-limit rows: `22` (`3.27%`)
- response-length rows: `2` (`0.30%`)
- length-stop rows: `0` (`0.00%`)
- total length-related rows: `2` (`0.30%`)
- missing-final rows: `43` (`6.40%`)

The dominant invalid causes are missing final answers and turn limit. Pure
response-length failures are rare in validation: only two rows across these
seven validation dumps, and no `void_reason=length` rows were observed.

## Observations

- Easy: best reward is step 50 (0.5625); latest available step 70 is 0.4375.
- Medium: best reward is step 70 (0.5312); latest available step 70 is 0.5312.
- Hard: best reward is step 20 (0.3438); latest available step 70 is 0.1875.
- Across subsets, later checkpoints use more tool calls than early checkpoints, especially at step 60/70. This does not correlate with better reward.
- `img_idx=0` remains dominant for most calls. The model sometimes crops previous observations, but the previous-crop ratio is usually modest and unstable by subset/step.
- Hard examples tend to have lower reward despite comparable or higher tool usage, suggesting more zooming alone is not solving the hard subset.
- Across the 21 step/subset points, reward vs mean tool-call count has correlation about `-0.52`. More calls are more often a sign of difficulty or wandering than improvement.
- Previous-crop use is rare overall: easy has 8 previous-crop calls out of 599, medium has 8 out of 593 plus one missing `img_idx`, and hard has 19 out of 762 plus one missing `img_idx`.
- Requested `img_idx` matches the tool response `minio3_crop/source_index` for every call where both are present. All tool responses report `minio3_crop/source_space=raw`, so the raw-image-bank path is active in these validation dumps.
- The two missing `img_idx` calls are isolated: one in step 10 hard and one in step 70 medium. They should be inspected as formatting/tool-call quality issues rather than crop routing bugs.

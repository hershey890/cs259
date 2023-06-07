+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.65.01    Driver Version: 515.65.01    CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA TITAN V      On   | 00000000:AF:00.0 Off |                  N/A |
| 28%   32C    P8    23W / 250W |      0MiB / 12288MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+

+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 530.47                 Driver Version: 531.68       CUDA Version: 12.1     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                  Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf            Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3070         On | 00000000:2B:00.0  On |                  N/A |
|  0%   53C    P8               16W / 220W|   1029MiB /  8192MiB |     18%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A        22      G   /Xwayland                                 N/A      |
+---------------------------------------------------------------------------------------+

# Important parameters for the TitanV/V100 
## Possible Parametrization Options
- Clock Speed: 1200 MHz (base) to 1455 MHz (boost)
- Memory Clock: 848 MHz 
### L1 data 
- Size                     32...128 KiB
- Line size                32 B
- Hit latency              28
- Number of sets           4 
- Load granularity         32 B 
- Update granularity       128 B 
- Update policy            non-LRU
- Physical address indexed no
### L2 data
- Size                     6,144 KiB
- Line size                64 B
- Hit latency              ∼193
- Populated by cudaMemcpy  yes
- Physical address indexed yes
### Arithmetic throughput TFLOPS through PCIe
- Half precision   83.03
- Single precision 14.03
- Double precision 7.07
### L1 Data Cache throughput
- 108.3 to 256 Bytes/Cycle (per SM)
NIC: Stands for “Network Interface Controller.” It’s just the card or port (e.g. mlx5_0) that your server uses to send and receive data over the network (like NFS traffic).

NV12 (or “NV#”): Means two GPUs are connected by NVLink lanes. The number (e.g. 12) is how many NVLink links they share. NVLink is a super-fast bridge that lets GPUs talk directly to each other without going through the CPU.

PXB: Short for “PCIe eXtended Bridge.” It means the GPU and the NIC (or another GPU) are on the same PCIe switch. Data only passes through one bridge chip instead of going all the way up to the CPU. That path is faster than going through the CPU but not as fast as NVLink.

SYS: Means “goes through the System (CPU) root complex.” In other words, if GPU → NIC is labeled SYS, that traffic climbs up into the CPU’s PCIe controller, then back down to the NIC. It’s a bit more indirect and slightly higher-latency than PXB.

PIX: “PCIe eXtended One-hop.” Similar to PXB but indicates a single PCIe link (one hop) with no extra bridges in between. It’s the shortest PCIe path.

NV# (e.g. NV2, NV4, etc.): Same idea as NV12 but indicates fewer NVLink lanes bonding two devices (common when GPUs have fewer direct NVLink connections).

In very simple terms:

NIC: the network port (how data comes in/out of the server).

PXB: GPU ↔ device goes through one PCIe switch (short path, bypasses CPU).

SYS: GPU ↔ device goes up through the CPU first (longer path).

NV12 / NV#: GPUs talking over NVLink with that many lanes (fastest, direct GPU-GPU link).



1. Commands run & their meanings

df -Th /rsrch1/ip/msalehjahromi/
• Shows that /rsrch1/ip is an NFS mount (Type = nfs4).

mount | grep '/rsrch1/ip'
• Confirms NFS server (10.113.115.68) and protocol details (vers=4.1 over TCP).

iostat -x 1 /dev/nvme2n1 1
• Takes exactly one extended I/O snapshot for device /dev/nvme2n1.
• Fields:
  – r/s, w/s = read/write requests per second
  – rkB/s, wkB/s = kB read/write per second
  – rrqm/s, %rrqm = merged read requests per second & merge rate
  – r_await, w_await = avg latency (ms) of reads/writes
  – %util = % of time device was servicing I/O (if ~100 %, it’s saturated)
– In your sample: %rrqm = 63.58 means 63.58 % of read requests were merged.

nvidia-smi topo -m
• Prints full GPU↔GPU/NIC connectivity matrix (NVLink, PCIe paths).

nvidia-smi topo -m | grep -E 'GPU5|GPU6|NIC0|NIC1'
• Filters just lines for GPU5, GPU6, NIC0 (mlx5_0), NIC1 (mlx5_1).
                GPU5   GPU6   NIC0    NIC1
            ─────────────────────────────────
        GPU5   │   X     NV12   SYS     SYS
        GPU6   │  NV12    X     SYS     PXB
        NIC0   │   SYS   SYS     X      PXB
        NIC1   │   SYS   PXB    PXB     X
        GPU5 ↔ GPU6 = NVLink (12 lanes).

        GPU5 → NIC0/1 = SYS (i.e. traffic goes up through CPU’s PCIe root).

        GPU6 → NIC0 = SYS, but GPU6 → NIC1 = PXB (one PCIe bridge hop).


lspci | grep -i nvme & lspci | grep -i nvidia
• Lists PCI bus IDs for NVMe controllers and NVIDIA GPUs.

iostat -x /dev/nvme2n1 1
• Another way to get a single-sample extended I/O report (same fields as above).

nvidia-smi dmon -s u -i 5,6 1
• Shows per-second samples of GPU 5 & 6 utilization + PCIe Rx/Tx MB/s.

nvidia-smi --query-gpu=index,pci.rx_throughput,pci.tx_throughput --format=csv -l 1 | grep -E '^5|^6'
• Prints a CSV per-second for GPUs 5 & 6: their PCIe Rx/Tx throughput in MiB/s.
    CSV query also yields lines like 5, 12.30 MiB/s, 45.20 MiB/s → GPU index, Rx, Tx.

(Suggested) sar -n DEV 1 | grep -E 'enp226s0|mlx5_0'
• If ifstat is missing, use sar to show NFS NIC (e.g. enp226s0 or mlx5_0) RX/TX kB/s.

(Suggested) ip -s link 1
• Shows per-interface packet/byte counters each second (subtract to get rate).

(Suggested) PyTorch snippet to measure Host↔Device speed (≈ GB/s over PCIe):


import torch, time
size = (1024,1024,128)            # ~512 MiB
cpu = torch.empty(size, device="cpu"); torch.cuda.synchronize()
_ = cpu.to("cuda"); torch.cuda.synchronize()
t0 = time.time(); dev = cpu.to("cuda"); torch.cuda.synchronize()
print("H→D:", cpu.numel()*4/(time.time()-t0)/1024**3, "GB/s")
t0 = time.time(); _ = dev.to("cpu"); torch.cuda.synchronize()
print("D→H:", dev.numel()*4/(time.time()-t0)/1024**3, "GB/s")


3. How your GPUs are connected

GPU5 → NIC0 (mlx5_0) = SYS (path goes through CPU root complex)

GPU5 → NIC1 (mlx5_1) = SYS (also via CPU)

GPU6 → NIC0 = SYS (via CPU)

GPU6 → NIC1 = PXB (one PCIe bridge hop, bypassing the CPU root)

This means if you stream data from NFS (over mlx5_1) to GPU6, it has a shorter PCIe path (PXB) than GPU5 (which uses SYS). For NIC0 (mlx5_0), both GPUs route via the CPU root.


msalehjahromi@1mcprddgx05:~$ nvidia-smi topo -m
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    NIC0    NIC1    NIC2    NIC3    NIC4    NIC5    NIC6    NIC7    NIC8    NIC9    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NV12    NV12    NV12    NV12    NV12    NV12    NV12    PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     48-63,176-191   3               N/A
GPU1    NV12     X      NV12    NV12    NV12    NV12    NV12    NV12    PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     48-63,176-191   3               N/A
GPU2    NV12    NV12     X      NV12    NV12    NV12    NV12    NV12    SYS     SYS     PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     16-31,144-159   1               N/A
GPU3    NV12    NV12    NV12     X      NV12    NV12    NV12    NV12    SYS     SYS     PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     16-31,144-159   1               N/A
GPU4    NV12    NV12    NV12    NV12     X      NV12    NV12    NV12    SYS     SYS     SYS     SYS     PXB     PXB     SYS     SYS     SYS     SYS     112-127,240-255 7               N/A
GPU5    NV12    NV12    NV12    NV12    NV12     X      NV12    NV12    SYS     SYS     SYS     SYS     PXB     PXB     SYS     SYS     SYS     SYS     112-127,240-255 7               N/A
GPU6    NV12    NV12    NV12    NV12    NV12    NV12     X      NV12    SYS     SYS     SYS     SYS     SYS     SYS     PXB     PXB     SYS     SYS     80-95,208-223   5               N/A
GPU7    NV12    NV12    NV12    NV12    NV12    NV12    NV12     X      SYS     SYS     SYS     SYS     SYS     SYS     PXB     PXB     SYS     SYS     80-95,208-223   5               N/A
NIC0    PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS      X      PXB     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS
NIC1    PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     PXB      X      SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS
NIC2    SYS     SYS     PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS      X      PXB     SYS     SYS     SYS     SYS     SYS     SYS
NIC3    SYS     SYS     PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     PXB      X      SYS     SYS     SYS     SYS     SYS     SYS
NIC4    SYS     SYS     SYS     SYS     PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS      X      PXB     SYS     SYS     SYS     SYS
NIC5    SYS     SYS     SYS     SYS     PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     PXB      X      SYS     SYS     SYS     SYS
NIC6    SYS     SYS     SYS     SYS     SYS     SYS     PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS      X      PXB     SYS     SYS
NIC7    SYS     SYS     SYS     SYS     SYS     SYS     PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     PXB      X      SYS     SYS
NIC8    SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS      X      PIX
NIC9    SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     PIX      X 

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
  NIC2: mlx5_2
  NIC3: mlx5_3
  NIC4: mlx5_4
  NIC5: mlx5_5
  NIC6: mlx5_6
  NIC7: mlx5_7
  NIC8: mlx5_8
  NIC9: mlx5_9


iostat -m
Linux 5.4.0-200-generic (1mcprddgx05)   06/03/2025      _x86_64_        (256 CPU)

avg-cpu:  %user   %nice %system %iowait  %steal   %idle
           7.66    1.94    1.40    0.53    0.00   88.47

Device             tps    MB_read/s    MB_wrtn/s    MB_dscd/s    MB_read    MB_wrtn    MB_dscd
loop0             0.03         0.00         0.00         0.00        261          0          0
loop1             0.00         0.00         0.00         0.00          9          0          0
loop2             0.00         0.00         0.00         0.00         10          0          0
loop3             0.00         0.00         0.00         0.00          6          0          0
loop4             0.00         0.00         0.00         0.00         17          0          0
loop5             0.00         0.00         0.00         0.00         24          0          0
loop6             0.01         0.00         0.00         0.00         69          0          0
loop7             0.00         0.00         0.00         0.00         17          0          0
loop8             0.00         0.00         0.00         0.00          6          0          0
loop9             0.00         0.00         0.00         0.00          2          0          0
md0              38.14         0.08         1.62         0.57     700938   14567326    5082319
md1               1.42         0.00         0.00         1.58      44577      17690   14179267
nvme0n1           0.60         0.00         0.00         0.39      23943       6626    3543937
nvme1n1          39.23         0.87         1.62         0.44    7833981   14605301    3954616
nvme2n1          38.04         0.84         1.62         0.44    7521212   14605301    3962189
nvme3n1           0.56         0.00         0.00         0.39      22006       5143    3544932
nvme4n1           0.40         0.00         0.00         0.39      15332       2810    3545289
nvme5n1           0.47         0.00         0.00         0.39      17908       3109    3545107





- df -Th /rsrch1/ip/msalehjahromi/

Filesystem               Type  Size  Used Avail Use% Mounted on                                                                   
10.113.115.68:/rsrch1/ip nfs4   15T  4.9T   11T  33% /rsrch1/ip                                                                   


- lsblk /dev/nvme2n1p1

NAME      MAJ:MIN RM  SIZE RO TYPE MOUNTPOINT                                                                                     
nvme2n1p1 259:8    0  512M  0 part /boot/efi    


# Find the PCI bus ID for nvme2n1
lspci | grep -i nvme
09:00.0 Non-Volatile memory controller: Samsung Electronics Co Ltd NVMe SSD Controller PM173X                                     
22:00.0 Non-Volatile memory controller: Samsung Electronics Co Ltd NVMe SSD Controller SM981/PM981/PM983                          
23:00.0 Non-Volatile memory controller: Samsung Electronics Co Ltd NVMe SSD Controller SM981/PM981/PM983                          
52:00.0 Non-Volatile memory controller: Samsung Electronics Co Ltd NVMe SSD Controller PM173X                                     
8a:00.0 Non-Volatile memory controller: Samsung Electronics Co Ltd NVMe SSD Controller PM173X                                     
ca:00.0 Non-Volatile memory controller: Samsung Electronics Co Ltd NVMe SSD Controller PM173X                                     


# Find the PCI bus ID for each A100
lspci | grep -i nvidia                                              
07:00.0 3D controller: NVIDIA Corporation Device 20b0 (rev a1)                                                                      
0f:00.0 3D controller: NVIDIA Corporation Device 20b0 (rev a1)                                                                    
47:00.0 3D controller: NVIDIA Corporation Device 20b0 (rev a1)                                                                    
4e:00.0 3D controller: NVIDIA Corporation Device 20b0 (rev a1)                                                                    
87:00.0 3D controller: NVIDIA Corporation Device 20b0 (rev a1)                                                                      
90:00.0 3D controller: NVIDIA Corporation Device 20b0 (rev a1)                                                                    
b7:00.0 3D controller: NVIDIA Corporation Device 20b0 (rev a1)                                                                    
bd:00.0 3D controller: NVIDIA Corporation Device 20b0 (rev a1)                                                                    
c4:00.0 Bridge: NVIDIA Corporation Device 1af1 (rev a1)                                                                           
c5:00.0 Bridge: NVIDIA Corporation Device 1af1 (rev a1)                                                                           
c6:00.0 Bridge: NVIDIA Corporation Device 1af1 (rev a1)                                                                             
c7:00.0 Bridge: NVIDIA Corporation Device 1af1 (rev a1)                                                                           
c8:00.0 Bridge: NVIDIA Corporation Device 1af1 (rev a1)                                                                           
c9:00.0 Bridge: NVIDIA Corporation Device 1af1 (rev a1) 


- dd if=/rsrch1/ip/msalehjahromi/ bs=1G count=1 of=/dev/null
ogress                                                                                                                            │····························
dd: error reading '/rsrch1/ip/msalehjahromi/': Is a directory                                                                     │····························
0+0 records in                                                                                                                    │····························
0+0 records out                                                                                                                   │····························
0 bytes copied, 0.000156046 s, 0.0 kB/s 


- lspci | grep -i nvidia

07:00.0 3D controller: NVIDIA Corporation Device 20b0 (rev a1)                                                                    


mount | grep '/rsrch1/ip'                                           │··························································
d1prpcnfs.mdanderson.edu:/rsrch1/ip on /rsrch1/ip type nfs (rw,relatime,vers=3,rsize=1048576,wsize=262144,namlen=255,hard,proto=t│··························································
cp,timeo=600,retrans=2,sec=sys,mountaddr=10.113.115.69,mountvers=3,mountport=50258,mountproto=udp,local_lock=none,addr=10.113.115.│··························································
69)                                                                                                                               │··························································
/etc/auto.share on /rsrch1/ip type autofs (rw,relatime,fd=6,pgrp=75333,timeout=300,minproto=5,maxproto=5,direct,pipe_ino=536773)  │··························································
10.113.115.68:/rsrch1/ip on /rsrch1/ip type nfs4 (rw,relatime,vers=4.1,rsize=32768,wsize=262144,namlen=255,soft,proto=tcp,timeo=60│··························································
0,retrans=2,sec=sys,clientaddr=10.113.120.155,local_lock=none,addr=10.113.115.68) 

## Command Explanations

### **`df -Th /rsrch1/ip/msalehjahromi/`**
- **What**: Shows disk space usage for the path
- **Result**: `/rsrch1/ip` is a **15TB NFS filesystem**, 33% full (4.9TB used, 11TB free)

### **`lsblk /dev/nvme2n1p1`** 
- **What**: Shows block device info for NVMe partition
- **Result**: 512MB partition mounted at `/boot/efi` (boot partition)

### **`lspci | grep -i nvme`**
- **What**: Lists all NVMe storage controllers
- **Result**: **6 Samsung NVMe SSDs** available (PM173X and SM981/PM981/PM983 models)
  - Multiple high-speed local storage options available

### **`dd if=/rsrch1/ip/msalehjahromi/ bs=1G count=1 of=/dev/null`**
- **What**: Attempts to test read throughput (read 1GB, discard output)
- **Result**: **Failed** - tried to read a directory instead of a file

### **`lspci | grep -i nvidia`**
- **What**: Lists NVIDIA GPU hardware
- **Result**: **8 NVIDIA A100 GPUs** (Device 20b0) + NVIDIA bridges
  - Full 8-GPU A100 node setup
  - Additional NVIDIA bridges for GPU interconnect

### **`mount | grep '/rsrch1/ip'`**
- **What**: Shows how `/rsrch1/ip` is mounted
- **Result**: **NFS4 network filesystem** mounted from server `10.113.115.68`
  - Read size: 32KB, Write size: 262KB
  - Uses TCP protocol with soft mount (timeouts allowed)

## System Architecture

```
NFS server (10.113.115.68)
           └───────────┘
                 ↓ (NFS over Ethernet/InfiniBand)
           ┌───────────┐
           │  NIC (e.g. mlx5_0 or enp226s0) │
           └───────────┘
                 ↓ (DMA into RAM via CPU)
           ┌───────────┐
           │   System RAM  │
           └───────────┘
                 ↓ (PCIe copy)
           ┌───────────┐
           │   8x A100 GPUs   │
           └───────────┘
```

## Key Insights

**Hardware**: **8x A100 GPUs** with **6x NVMe SSDs** - High-end multi-GPU training node

**Storage Bottleneck**: Data stored on **remote NFS** instead of fast local NVMe storage
- **Network I/O**: Limited by NFS read/write sizes (32KB/262KB)
- **Local Storage**: 6x Samsung NVMe SSDs available but not utilized for training data

**Recommendation**: Consider copying training data to local NVMe for faster I/O during multi-GPU training


          nvidia-smi topo -m | grep -E 'GPU5|GPU6|NIC0|NIC1'



NFS protocol version
NFS can run in multiple versions (v3, v4.0, v4.1, etc.). Each version handles caching, read/write sizes, and locking differently. For example, NFS v3 often uses smaller I/O block sizes and fewer optimizations, whereas NFS v4.1 introduced parallel NFS (pNFS) and larger I/O buffers. If the server fell back to NFS v3 instead of v4.1, you might see much lower throughput. Verifying which version the client negotiated (via mount or /proc/mounts) is important.

Driver versions
The NIC and GPU both depend on kernel modules/drivers (e.g., Mellanox OFED for InfiniBand/Ethernet, NVIDIA drivers for A100). If the network driver on the A100 node is outdated or misconfigured, DMA transfers might underperform. Similarly, an older NVIDIA driver can limit PCIe bandwidth or not expose the optimal “PXB” connection to the GPU. Ensuring both the NIC’s firmware/driver and the NVIDIA driver are current (and match the cluster’s recommended versions) can eliminate driver‐related bottlenecks.

NUMA pinning
Modern servers have multiple NUMA nodes (separate CPU sockets with their own memory). If your processes (the DataLoader threads or the CUDA context) are pinned to CPUs on one NUMA node but the GPU’s PCIe root lives on another, data must cross the inter‐socket interconnect (e.g., QPI/UPI) before reaching GPU memory. That extra hop cuts effective bandwidth in half. Checking lscpu for CPU/GPU NUMA affinities and ensuring your job’s CPU threads and CUDA context are bound to the same NUMA node as the GPU can restore full PCIe bandwidth.

One‐sentence definitions

NFS: A protocol that lets you treat a remote directory (like /rsrch1/ip) as if it were local storage.

NIC: The server’s network card (e.g. mlx5_1) that receives NFS data from the storage server.

GPU: The A100 accelerator that needs to load training data into its memory.

PXB: A direct, one‐bridge PCIe route between a NIC and a GPU—avoids going through the CPU.

SYS: A “system” route that forces data to first go through the CPU’s PCIe root complex, then back down to the GPU.

Simplified explanation of the two paths

NFS (Network File System): what you use to mount /rsrch1/ip over the network.

NIC (Network Interface Controller): the physical network card (e.g. mlx5_1, mlx5_2) that carries NFS traffic into the server.

GPU (Graphics Processing Unit): here, an A100 that loads data to do your training.

PXB (PCIe eXtended Bridge): a “one‐hop” PCIe link directly from the NIC into a GPU’s PCIe fabric—data does not have to go through the CPU first.

SYS (System path): means “go up into the CPU’s PCIe root complex, then come back down into the GPU.” In other words, two hops: NIC → CPU → GPU.

Fast path (1 hop):

Mount NFS on NIC2, which is wired PXB → GPU0.

So data goes:

scss
Copy
Edit
NFS NIC2 → (PXB) → GPU0 memory
Only one PCIe hop, bypassing the CPU root—maximizes bandwidth (~24 GB/s).

Slow path (2 hops):

Mount NFS on NIC1, which is wired SYS → GPU2/3.

So data goes:

scss
Copy
Edit
NFS NIC1 → (PCIe up into CPU root) → (PCIe down to GPU2/3)
Two hops through the CPU → lower effective bandwidth (~8–12 GB/s).
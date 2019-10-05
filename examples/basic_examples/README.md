# Basic Examples   
Use these examples to test how lightning works.   

#### Test on CPU  
```bash
python cpu_template.py
```

---   
#### Train on a single GPU
```bash
python gpu_template.py --gpus 1
```   

---    
#### DataParallel (dp)   
Train on multiple GPUs using DataParallel.

```bash
python gpu_template.py --gpus 2 --distributed_backend dp
```   

---
#### DistributedDataParallel (ddp)    

Train on multiple GPUs using DistributedDataParallel   
```bash
python gpu_template.py --gpus 2 --distributed_backend ddp
```

---
#### DistributedDataParallel+DP (ddp2)    

Train on multiple GPUs using DistributedDataParallel + dataparallel.
On a single node, uses all GPUs for 1 model. Then shares gradient information
across nodes.   
```bash
python gpu_template.py --gpus 2 --distributed_backend ddp2
```
# GPPRL
Massively Parallel PPRL implementation using CUDA C and Python

1. Install CUDA: https://developer.nvidia.com/cuda-downloads

2. Compile the CUDA executable:
```
nvcc dice-gpu-optimized.cu -o dice-gpu-optimized
```

3. Run the python script
```
match-cuda.py <Zip 1 path> <Zip 2 path> <Threshold>
python match-cuda.py "E:\path\to\sources\x.zip" "E:\path\to\sources\y.zip" 0.85
```

import base64
import csv
from zipfile import ZipFile
import json
import subprocess
import numpy
import sys
import csv
import time
start = time.time()

threshold = 0.85

results = dict()
z1 = ZipFile("E:/NACHC PPRL Test Datasets/ymca.zip")
z2 = ZipFile("E:/NACHC PPRL Test Datasets/ncceh.zip")
print(z1.namelist())

for path in z1.namelist():

    with z1.open(path) as f:
        j1 = json.load(f)

    with z2.open(path) as f:
        j2 = json.load(f)


    bytes_list = [base64.b64decode(x) for x in j1["clks"]]
    with open("dataset1.bin", "wb") as f:
        for clk in bytes_list:
            f.write(clk)



    bytes_list = [base64.b64decode(x) for x in j2["clks"]]
    with open("dataset2.bin", "wb") as f:
        for clk in bytes_list:
            f.write(clk)

    subprocess.run(["./dice-gpu-optimized.exe", str(threshold)], capture_output=True)

    with open("matches.csv", "r") as f:
        r = csv.reader(f)
        for line in r:
            if line[0] == "2736": print(line[1])
            if line[1] != "-1":
                results[int(line[0])] = int(line[1])

print(results)

with open("results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset 1", "dataset 2"])
    w.writerows(sorted(results.items(), key=lambda item: item[0]))
end = time.time()

print(end - start)
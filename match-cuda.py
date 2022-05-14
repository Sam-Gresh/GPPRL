import base64
from zipfile import ZipFile
import json
import subprocess
import csv
import time
import argparse
import os

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("zip1", metavar='<Zip 1 path>', type=str,
                    help='The first zipfile containing hashes')
parser.add_argument("zip2", metavar='<Zip 2 path>', type=str,
                    help='The second zipfile containing hashes')
parser.add_argument("threshold", metavar='<Threshold>', type=float,
                    help='The matching threshold')

args = parser.parse_args()



start = time.time()

threshold = args.threshold

results = dict()
z1 = ZipFile(args.zip1)
z2 = ZipFile(args.zip2)
print(f"Matching on schemas: {z1.namelist()}")

for path in z1.namelist():
    print(f"Starting Schema {path}...")
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


    result = subprocess.run(["./dice-gpu-optimized.exe", str(threshold)], cwd=os.getcwd(), capture_output=True, text=True)
    print(result.stdout)

    with open("./matches.csv", "r") as f:
        r = csv.reader(f)
        for line in r:
            if line[1] != "-1":
                results[int(line[0])] = int(line[1])

with open("results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset 1", "dataset 2"])
    w.writerows(sorted(results.items(), key=lambda item: item[0]))
end = time.time()

os.remove("./matches.csv")
os.remove("./dataset1.bin")
os.remove("./dataset2.bin")
print("Results located in results.csv")
print(f"Finished matching in: {end - start}")
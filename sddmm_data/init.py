

import os
import shutil
import asyncio
import pathlib

# copy folder infrastructure to c drive
path = pathlib.Path(__file__)
parent_path = path.parent
dst_path = pathlib.Path(parent_path.drive + "/" + parent_path.parts[-1])
if not dst_path.exists():
    print("Copy template [" + str(parent_path) + "] to destination [" + str(dst_path) + "]")
    shutil.copytree(src=str(parent_path), dst=str(dst_path))

build_folder = str(parent_path.parent) + str(pathlib.Path("/build/x64-Release"))

# copy all builds into infrastructure
create_img = "create_img.exe"
gpu_sddm_benchmarks = "GPU_SDDMMBenchmarks.exe"
data_gen_mat_market_companion = "data_gen_mat_market_companion.exe"
data_gen = "data_gen.exe"
data_gen_mat_market = "data_gen_mat_market.exe"

exe_dst_path = str(parent_path.drive) + str(pathlib.Path("/sddmm_data")) + str(pathlib.Path("/"))
exe_src_path = build_folder + str(pathlib.Path("/")) 

shutil.copy(exe_src_path + create_img,                    exe_dst_path + create_img)
shutil.copy(exe_src_path + gpu_sddm_benchmarks,           exe_dst_path + gpu_sddm_benchmarks)
shutil.copy(exe_src_path + data_gen_mat_market_companion, exe_dst_path + data_gen_mat_market_companion)
shutil.copy(exe_src_path + data_gen,                      exe_dst_path + data_gen)
shutil.copy(exe_src_path + data_gen_mat_market,           exe_dst_path + data_gen_mat_market)

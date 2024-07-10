# input a folder and split its subfolders to running on different gpus
import os
import sys
import subprocess
import multiprocessing as mp
from concurrent import futures

GPUS=[0, 1, 2, 3, 4, 5, 6, 7]

def run(root_path, start_frame, log_file, exp_name):
    cur_proc = mp.current_process()
    print("PROCESS", cur_proc.name, cur_proc._identity)
    worker_id = cur_proc._identity[0] - 1  # 1-indexed processes
    gpu = GPUS[worker_id % len(GPUS)]
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu} "
        f"python mini_dust3r/api/inference.py --root_path {root_path} --start_frame {start_frame} --exp_name {exp_name}"
    )
    # cmd = (
    #     f"CUDA_VISIBLE_DEVICES={gpu} "
    #     f"python scripts/save_action_figure.py {pkl_file}"
    # )

    print(f"LOGGING TO {log_file}")
    cmd = f"{cmd} > {log_file} 2>&1"
    print(cmd)
    subprocess.call(cmd, shell=True)

def main(root_folder, exp_name):
    file_list = os.listdir(root_folder)
    log_dir = f"logs/{exp_name}"
    os.makedirs(log_dir, exist_ok=True)
    with futures.ProcessPoolExecutor(max_workers=8) as exe:
        for file in file_list:
            root_path = os.path.join(root_folder, file)
            sequence_id = root_path.split("/")[-1][:2]
            num_frames = len(os.listdir(root_path+"/images"))
            start_frame = 0
            while num_frames - start_frame > 100:
                log_file = f"{log_dir}/{sequence_id}_{start_frame}.log"
                exe.submit(
                    run,
                    root_path,
                    start_frame,
                    log_file,
                    exp_name
                )
                start_frame += 100


if __name__ == "__main__":
    root_folder = "data/EMDB2"
    exp_name = sys.argv[1]
    main(root_folder, exp_name)

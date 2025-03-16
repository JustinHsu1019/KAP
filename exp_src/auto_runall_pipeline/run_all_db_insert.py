import subprocess
import concurrent.futures

def run_task(command, log_file):
    with open(log_file, 'w') as f:
        process = subprocess.Popen(command, stdout=f, stderr=f, shell=True)
        return process

commands = [
    ("python3 exp_src/db_insert.py --task Tess", "logs/tess.log"),
    ("python3 exp_src/db_insert.py --task Ourswomllm", "logs/ourswomllm.log"),
    ("python3 exp_src/db_insert.py --task Ourswoocr", "logs/ourswoocr.log"),
    ("python3 exp_src/db_insert.py --task Oursworewrite", "logs/oursworewrite.log"),
    ("python3 exp_src/db_insert.py --task Ours", "logs/ours.log"),
]

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(run_task, cmd, log) for cmd, log in commands]

    concurrent.futures.wait(futures)

print("All tasks completed.")

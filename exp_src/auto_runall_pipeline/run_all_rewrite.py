import subprocess

def run_task(command, log_file):
    with open(log_file, 'w') as f:
        process = subprocess.Popen(command, stdout=f, stderr=f, shell=True)
    return process

task2 = run_task("python3 src/rewrite.py --task Ourswomllm", "logs/ourswomllm.log")
task3 = run_task("python3 src/rewrite.py --task Ourswoocr", "logs/ourswoocr.log")

task2.wait()
task3.wait()

task4 = run_task("python3 src/rewrite.py --task Oursworewrite", "logs/oursworewrite.log")
final_task = run_task("python3 src/rewrite.py --task Ours", "logs/ours.log")

task4.wait()
final_task.wait()

print("All tasks completed.")

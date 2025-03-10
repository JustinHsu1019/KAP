import subprocess

def run_ir_autotest(alpha_list):
    for alpha in alpha_list:
        command = ["python3", "src/ir_autotest_hybrid.py", "--alpha", str(alpha)]
        print(f"Running: {' '.join(command)}")
        subprocess.run(command, check=True)

if __name__ == "__main__":
    alpha_list = [1.0, 0.5, 0.0]
    run_ir_autotest(alpha_list)

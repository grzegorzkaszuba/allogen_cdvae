import sys
import subprocess
import os

def run_script(script_path, script_args, retries):
    command = ['python', script_path] + script_args
    for _ in range(retries):
        try:
            result = subprocess.run(command, text=True)
            if result.returncode == 0:
                print("Script ran successfully.")
                return result.returncode
            elif result.returncode == 123:
                print("Non-deterministic numerical error. Retrying...")
            else:
                print(f"Script {script_path} exited with error code: {result.returncode}")
                return result.returncode
        except Exception as e:
            print(f"An exception occurred: {e}")
            return -1
    print("Max retries reached. Exiting.")
    return 123

script_args = sys.argv[1:]  # grab the args for the wrapper
script_path = os.path.join(os.getcwd(), 'run.py')
retries = 5  # adjust this for the number of retries you want

exit_code = run_script(script_path, script_args, retries)
print(f'Exit code: {exit_code}')

import subprocess
import os

def run_shell_script(sh_script_path, env):
    try:
        subprocess.run(["sh", sh_script_path], check=True, env=env)
        print("Shell script executed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running the script: {e}")


if __name__ == "__main__":
    sh_script_path = "models/stylegan3/run_train.sh"
    env = os.environ.copy()
    
    run_shell_script(sh_script_path, env)

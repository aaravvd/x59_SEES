import os
import shutil
import subprocess
import math
import sys

# --- CONFIGURATION ---
BASE_DIR = os.getcwd()
TARGET_FILE = os.path.join("0", "U")
PLACEHOLDER = "TARGET_VECTOR"

# --- MAX LIMITS TO PREVENT HANGING OR DIVERGENCE ---
# Max execution time allowed per single trial (in seconds)
MAX_TRIAL_TIMEOUT = 1200  # 20 minutes
# Divergence keywords to watch for in the log output
DIVERGENCE_KEYWORDS = ["NaN", "FatalError", "divergence", "sigFpe"]

def get_user_inputs():
    """Prompts the user for trial setup parameters."""
    print("=== OpenFOAM Angle Sweep Configuration ===")
    try:
        num_trials = int(input("Enter total number of trials (e.g., 150): "))
        angle_step = float(input("Enter angle increment per trial in degrees (e.g., 0.5): "))
        base_velocity = float(input("Enter base air velocity magnitude (e.g., 10.0 m/s): "))
        start_angle = float(input("Enter starting angle in degrees (e.g., 0.0): "))
        return num_trials, angle_step, base_velocity, start_angle
    except ValueError:
        print("\n[ERROR] Invalid input. Please enter numbers only.")
        sys.exit(1)

def generate_trials(num_trials, angle_step, base_velocity, start_angle):
    """Calculates the Ux and Uy vector components for each angle step."""
    trials = []
    for i in range(num_trials):
        current_angle = start_angle + (i * angle_step)
        rad = math.radians(current_angle)
        
        ux = base_velocity * math.cos(rad)
        uy = base_velocity * math.sin(rad)
        uz = 0.0  
        
        vector_tuple = (round(ux, 5), round(uy, 5), round(uz, 5))
        
        trials.append({
            "name": f"trial_angle_{current_angle:.1f}",
            "vector": vector_tuple,
            "angle": current_angle
        })
    return trials

def modify_vector_file(file_path, new_vector):
    """Replaces the placeholder string with the new formatted vector."""
    vector_string = f"({new_vector[0]} {new_vector[1]} {new_vector[2]})"
    
    with open(file_path, "r") as f:
        content = f.read()
    
    if PLACEHOLDER not in content:
        raise ValueError(f"Placeholder '{PLACEHOLDER}' not found in {file_path}.")
        
    modified_content = content.replace(PLACEHOLDER, vector_string)
    
    with open(file_path, "w") as f:
        f.write(modified_content)

def run_openfoam_trial(trial_name, vector_value, angle_val):
    """Creates trial directory, updates velocity, and runs solver with an active break condition."""
    trial_dir = os.path.join(BASE_DIR, trial_name)
    print(f"\n" + "="*50)
    print(f"Running: {trial_name} | Angle: {angle_val}° | Vector: {vector_value}")
    print("="*50)

    # 1. Duplicate case template
    if os.path.exists(trial_dir):
        shutil.rmtree(trial_dir)
    os.makedirs(trial_dir)
    
    for folder in ["0", "constant", "system"]:
        src = os.path.join(BASE_DIR, folder)
        dst = os.path.join(trial_dir, folder)
        shutil.copytree(src, dst)

    # 2. Inject calculated vector
    trial_target_file = os.path.join(trial_dir, TARGET_FILE)
    modify_vector_file(trial_target_file, vector_value)

    # 3. Execute mesh
    try:
        print(f"[{trial_name}] Meshing...")
        subprocess.run(["blockMesh"], cwd=trial_dir, check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print(f"[ERROR] Meshing failed for {trial_name}. Skipping trial.")
        return False

    # 4. Run solver with live logs and break condition tracking
    print(f"[{trial_name}] Running solver (monitoring divergence)...")
    log_file_path = os.path.join(trial_dir, f"log.{trial_name}")
    diverged = False

    # Use simpleFoam as default solver (swap with pimpleFoam, icoFoam, etc., if needed)
    solver_cmd = ["simpleFoam"]

    try:
        # Popen allows us to read stdout line-by-line while the process runs
        with subprocess.Popen(solver_cmd, cwd=trial_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process, \
             open(log_file_path, "w") as log_file:
            
            for line in process.stdout:
                # Write to standard log file
                log_file.write(line)
                log_file.flush()
                
                # Check for bad keywords (e.g., NaN in residuals)
                if any(kw in line for kw in DIVERGENCE_KEYWORDS):
                    diverged = True
                    print(f"\n[ALERT] Divergence detected in {trial_name}! Breaking trial...")
                    process.terminate()  # Kill the solver process immediately
                    break
            
            # Wait for process to fully cleanup or hit safety timeout
            process.wait(timeout=MAX_TRIAL_TIMEOUT)

        if diverged:
            print(f"[{trial_name}] Trial aborted due to divergence.")
            return False
        elif process.returncode != 0:
            print(f"[{trial_name}] Trial finished with an unexpected exit code: {process.returncode}")
            return False
        else:
            print(f"[{trial_name}] Success. Log saved.")
            return True

    except subprocess.TimeoutExpired:
        print(f"\n[ALERT] Trial {trial_name} exceeded max execution timeout of {MAX_TRIAL_TIMEOUT}s. Breaking...")
        return False
    except Exception as e:
        print(f"[ERROR] Run failed due to an error: {e}")
        return False

if __name__ == "__main__":
    num_trials, angle_step, base_velocity, start_angle = get_user_inputs()
    trial_list = generate_trials(num_trials, angle_step, base_velocity, start_angle)
    
    print(f"\nGenerated {len(trial_list)} trials. Starting execution loop...")
    
    success_count = 0
    failed_count = 0
    
    for trial in trial_list:
        success = run_openfoam_trial(trial["name"], trial["vector"], trial["angle"])
        if success:
            success_count += 1
        else:
            failed_count += 1
            
    print("\n" + "="*50)
    print("ALL automated trials completed.")
    print(f"Successful Runs: {success_count}")
    print(f"Aborted / Failed Runs: {failed_count}")
    print("="*50)

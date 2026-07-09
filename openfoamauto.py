import os
import glob
import shutil
import subprocess
import math
import sys
import re


# --- CONFIGURATION ---
BASE_DIR = os.getcwd()
TARGET_FILE = os.path.join("0", "U")
CONTROL_DICT_FILE = os.path.join("system", "controlDict")

# --- MAX LIMITS TO PREVENT HANGING OR DIVERGENCE ---
# Max execution time allowed per single trial (in seconds, wall-clock)
MAX_TRIAL_TIMEOUT = 1200  # 20 minutes
# Divergence keywords to watch for in the log output
DIVERGENCE_KEYWORDS = ["NaN", "FatalError", "divergence", "sigFpe"]
# How many lines of the log to show when a trial fails, so you can see
# the actual OpenFOAM error instead of just an exit code.
LOG_TAIL_LINES = 40
# Number of MPI subdomains per trial. On a 10-core machine, using 8 leaves
# 2 cores free for the OS/background processes rather than saturating all
# cores, which tends to hurt more than it helps for decomposePar overhead.
NUM_SUBDOMAINS = 8


def clean_previous_trials(base_dir=BASE_DIR, pattern="trial_angle_*"):
    """
    Removes all previously generated trial_angle_* folders in base_dir.
    Does NOT touch the original 0/, constant/, system/ folders, since those
    are only ever read (not written) by this script.
    """
    matches = sorted(glob.glob(os.path.join(base_dir, pattern)))
    if not matches:
        print("[CLEAN] sonion cant find ts.")
        return

    print(f"[CLEAN] Removing {len(matches)} previous trial folder(s) u larper...")
    for path in matches:
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"  - removed {os.path.basename(path)}")
    print("[CLEAN] done boi")


def get_user_inputs():
    """Prompts the user for trial setup parameters."""
    print("=== OpenFOAM Angle Sweep Configuration by Kenny the ligma sigma ===")
    try:
        num_trials = int(input("Enter total number of trials (e.g., 67): "))
        angle_step = float(input("Enter angle increment per trial in degrees (e.g., 0.5): "))
        base_velocity = float(input(
            "Enter base air velocity magnitude in m/s "
            "(e.g., ~340 m/s for Mach 1 at sea level, "
            "~295-300 m/s for Mach ~1.4 at ~55,000 ft, adjust for your altitude/temperature ohio): "
        ))
        start_angle = float(input("Enter starting angle in degrees (e.g., 0.0): "))
        end_time = float(input("Enter simulation end time in skibidi seconds (e.g., 0.3-0.4): "))
        return num_trials, angle_step, base_velocity, start_angle, end_time
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
            "name": f"trial_angle_{current_angle:.2f}",
            "vector": vector_tuple,
            "angle": current_angle
        })
    return trials


def modify_vector_file(file_path, new_vector):
    """
    Replace the internalField velocity vector while keeping the rest
    of the OpenFOAM file unchanged.
    """
    vector_string = f"({new_vector[0]} {new_vector[1]} {new_vector[2]})"

    with open(file_path, "r") as f:
        content = f.read()

    pattern = r"(internalField\s+uniform\s+)\([^)]+\)"

    if not re.search(pattern, content):
        raise RuntimeError(
            "Could not find 'internalField uniform (...)' in 0/U"
        )

    content = re.sub(pattern, rf"\1{vector_string}", content)

    with open(file_path, "w") as f:
        f.write(content)


def modify_end_time(file_path, end_time):
    """
    Replace the endTime entry in system/controlDict while keeping the
    rest of the file unchanged.
    """
    with open(file_path, "r") as f:
        content = f.read()

    pattern = r"(^\s*endTime\s+)[^;]+;"

    if not re.search(pattern, content, flags=re.MULTILINE):
        raise RuntimeError(
            "Could not find 'endTime' entry in system/controlDict"
        )

    content = re.sub(pattern, rf"\g<1>{end_time};", content, flags=re.MULTILINE)

    with open(file_path, "w") as f:
        f.write(content)


def write_decompose_par_dict(trial_dir, num_subdomains):
    """
    Writes a system/decomposeParDict configured for the given number of
    subdomains, using the scotch method (no geometry input required).
    Overwrites any existing decomposeParDict copied from the base case.
    """
    path = os.path.join(trial_dir, "system", "decomposeParDict")
    content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}}

numberOfSubdomains  {num_subdomains};

method          scotch;

// ************************************************************************* //
"""
    with open(path, "w") as f:
        f.write(content)


def print_log_tail(log_file_path, num_lines=LOG_TAIL_LINES):
    """Prints the last num_lines of a log file so the real solver error is visible."""
    try:
        with open(log_file_path, "r") as f:
            lines = f.readlines()
        tail = lines[-num_lines:] if len(lines) > num_lines else lines
        print(f"--- Last {len(tail)} lines of {os.path.basename(log_file_path)} ---")
        for line in tail:
            print(line.rstrip())
        print("--- end of log excerpt ---")
    except Exception as e:
        print(f"[WARN] Could not read log file for diagnostics: {e}")


def run_openfoam_trial(trial_name, vector_value, angle_val, end_time):
    """Creates trial directory, updates velocity/end time, and runs solver with an active break condition."""
    trial_dir = os.path.join(BASE_DIR, trial_name)
    print(f"\n" + "=" * 50)
    print(f"Running: {trial_name} | Angle: {angle_val}° | Vector: {vector_value} | End time: {end_time}s")
    print("=" * 50)

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

    # 3. Inject end time
    trial_control_dict = os.path.join(trial_dir, CONTROL_DICT_FILE)
    modify_end_time(trial_control_dict, end_time)

    # 4. Execute mesh: blockMesh builds the background block, then
    # snappyHexMesh cuts the body geometry (constant/triSurface/*.stl) out
    # of it using system/snappyHexMeshDict. Skipping snappyHexMesh means
    # the body is never actually present in the mesh.
    mesh_log_path = os.path.join(trial_dir, f"log.blockMesh.{trial_name}")
    try:
        print(f"[{trial_name}] Meshing background block...")
        with open(mesh_log_path, "w") as mesh_log:
            subprocess.run(
                ["blockMesh"],
                cwd=trial_dir,
                stdout=mesh_log,
                stderr=subprocess.STDOUT,
                check=True
            )
    except subprocess.CalledProcessError:
        print(f"[ERROR] blockMesh failed for {trial_name}. Skipping trial.")
        print_log_tail(mesh_log_path)
        return False

    snappy_log_path = os.path.join(trial_dir, f"log.snappyHexMesh.{trial_name}")
    try:
        print(f"[{trial_name}] Cutting body geometry with snappyHexMesh...")
        with open(snappy_log_path, "w") as snappy_log:
            subprocess.run(
                ["snappyHexMesh", "-overwrite"],
                cwd=trial_dir,
                stdout=snappy_log,
                stderr=subprocess.STDOUT,
                check=True
            )
    except subprocess.CalledProcessError:
        print(f"[ERROR] snappyHexMesh failed for {trial_name}. Skipping trial.")
        print_log_tail(snappy_log_path)
        return False

    # 4b. Verify mesh quality before spending time decomposing/solving on it
    checkmesh_log_path = os.path.join(trial_dir, f"log.checkMesh.{trial_name}")
    try:
        print(f"[{trial_name}] Verifying mesh quality (checkMesh)...")
        with open(checkmesh_log_path, "w") as checkmesh_log:
            result = subprocess.run(
                ["checkMesh"],
                cwd=trial_dir,
                stdout=checkmesh_log,
                stderr=subprocess.STDOUT
            )
        with open(checkmesh_log_path, "r") as f:
            checkmesh_output = f.read()

        failed_match = re.search(r"Failed (\d+) mesh checks?", checkmesh_output)
        num_failed = int(failed_match.group(1)) if failed_match else 0

        if result.returncode != 0 or num_failed > 0:
            print(f"[ERROR] checkMesh reports {num_failed} failed check(s) for {trial_name}. Skipping trial.")
            print_log_tail(checkmesh_log_path)
            return False
        else:
            print(f"[{trial_name}] Mesh OK.")
    except Exception as e:
        print(f"[ERROR] checkMesh could not be run for {trial_name}: {e}")
        return False

    # 5. Decompose domain for parallel run
    write_decompose_par_dict(trial_dir, NUM_SUBDOMAINS)
    decompose_log_path = os.path.join(trial_dir, f"log.decomposePar.{trial_name}")
    try:
        print(f"[{trial_name}] Decomposing domain into {NUM_SUBDOMAINS} subdomains...")
        with open(decompose_log_path, "w") as decompose_log:
            subprocess.run(
                ["decomposePar"],
                cwd=trial_dir,
                stdout=decompose_log,
                stderr=subprocess.STDOUT,
                check=True
            )
    except subprocess.CalledProcessError:
        print(f"[ERROR] decomposePar failed for {trial_name}. Skipping trial.")
        print_log_tail(decompose_log_path)
        return False

    # 6. Run solver in parallel with live logs and break condition tracking
    print(f"[{trial_name}] Running solver on {NUM_SUBDOMAINS} perfomative male cores (monitoring divergence)...")
    log_file_path = os.path.join(trial_dir, f"log.{trial_name}")
    diverged = False

    # rhoCentralFoam: density-based compressible solver, appropriate for
    # high-speed / transonic / supersonic flow (e.g. sonic boom studies).
    # Run via mpirun across NUM_SUBDOMAINS cores.
    solver_cmd = ["mpirun", "-np", str(NUM_SUBDOMAINS), "rhoCentralFoam", "-parallel"]

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
            print_log_tail(log_file_path)
            return False
        elif process.returncode != 0:
            print(f"[{trial_name}] Trial finished with an unexpected exit code: {process.returncode}")
            print_log_tail(log_file_path)
            return False
        else:
            print(f"[{trial_name}] Solver finished with 67. Reconstructing parallel results...")
            reconstruct_log_path = os.path.join(trial_dir, f"log.reconstructPar.{trial_name}")
            try:
                with open(reconstruct_log_path, "w") as reconstruct_log:
                    subprocess.run(
                        ["reconstructPar"],
                        cwd=trial_dir,
                        stdout=reconstruct_log,
                        stderr=subprocess.STDOUT,
                        check=True
                    )
            except subprocess.CalledProcessError:
                print(f"[WARN] reconstructPar failed for {trial_name}. "
                      f"Raw processor*/ folders are retained in {trial_dir} for manual reconstruction.")
                print_log_tail(reconstruct_log_path)
                return False

            # Clean up per-processor folders now that results are merged
            for proc_dir in glob.glob(os.path.join(trial_dir, "processor*")):
                shutil.rmtree(proc_dir)

            # Create an empty .foam marker file so this trial can be opened
            # directly in ParaView (File > Open > select this file, then
            # choose the OpenFOAM reader). No conversion needed since
            # reconstructPar already left standard time directories.
            foam_marker_path = os.path.join(trial_dir, f"{trial_name}.foam")
            open(foam_marker_path, "w").close()

            print(f"[{trial_name}] Success. ts frying me Log saved. ParaView file: {foam_marker_path}")
            return True

    except subprocess.TimeoutExpired:
        print(f"\n[ALERT] Trial {trial_name} exceeded max execution timeout of {MAX_TRIAL_TIMEOUT}s. Breaking...")
        print_log_tail(log_file_path)
        return False
    except Exception as e:
        print(f"[ERROR] Run failed due to an error: {e}")
        return False


if __name__ == "__main__":
    clean_choice = input("Clean up previous trial_angle_* folders before starting? (y/n) HAHA YN: ").strip().lower()
    if clean_choice == "y":
        clean_previous_trials()

    num_trials, angle_step, base_velocity, start_angle, end_time = get_user_inputs()
    trial_list = generate_trials(num_trials, angle_step, base_velocity, start_angle)

    print(f"\nGenerated {len(trial_list)} trials.(ideally be 67) Starting execution loop...")

    success_count = 0
    failed_count = 0

    for trial in trial_list:
        success = run_openfoam_trial(trial["name"], trial["vector"], trial["angle"], end_time)
        if success:
            success_count += 1
        else:
            failed_count += 1

    print("\n" + "=" * 50)
    print("ALL automated trials completed.")
    print(f"Successful Runs tuff: {success_count}")
    print(f"Aborted / Failed Runs bruv: {failed_count}")
    print("=" * 50)
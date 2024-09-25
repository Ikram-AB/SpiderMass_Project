import subprocess
import os

def convert_raw_to_mzml(raw_file_path, output_dir):
    command = [
        "msconvert", raw_file_path,
        "--filter", "lockmassRefiner mz=524 tol=1.0",
        "-o", output_dir, "--mzML"
    ]
    with open(os.devnull, 'w') as devnull:
        subprocess.run(command, check=True, stdout=devnull, stderr=devnull)

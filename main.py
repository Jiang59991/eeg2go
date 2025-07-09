import subprocess
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

module_scripts = [
    "database.init_db",
    "database.import_harvard_demo",
    "database.default_pipelines",
    "database.default_fxdefs",
    "database.default_featuresets",
    "setup_age_correlation"
]

for module in module_scripts:
    print(f"Running {module}...")
    result = subprocess.run(["python", "-m", module], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Errors in {module}:\n{result.stderr}")
    else:
        print(f"{module} completed successfully.\n")

print("All setup scripts executed.")

import subprocess
import sys
import os
import shutil

def run_script(script_name):
    print(f"\nRunning {script_name}...")
    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        if result.returncode == 0:
            print(f"✓ Successfully completed {script_name}")
        else:
            print(f"✗ Failed to run {script_name}")
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {script_name}: {str(e)}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"✗ Could not find {script_name}")
        sys.exit(1)

def main():
    # Check if "all_data" folder exists and delete it if it does
    if os.path.exists("all_data"):
        print("Deleting existing 'all_data' folder...")
        try:
            shutil.rmtree("all_data")
            print("✓ 'all_data' folder deleted successfully.")
        except Exception as e:
            print(f"✗ Error deleting 'all_data' folder: {str(e)}")
            sys.exit(1)
    
    # List of scripts to run in order
    scripts = [
        "generate_images.py",
        "split.py"
    ]
    
    # Check if all scripts exist
    missing_scripts = [script for script in scripts if not os.path.exists(script)]
    if missing_scripts:
        print("Error: The following scripts are missing:")
        for script in missing_scripts:
            print(f"- {script}")
        sys.exit(1)
    
    print("Starting OCR pipeline...")
    
    # Run each script in sequence
    for script in scripts:
        run_script(script)
    
    print("\n✓ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
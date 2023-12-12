import subprocess


# Estimated 15 Minute Run Time, Assuming 8 Processors Available
def main():
    # Run EDA Script
    subprocess.run(["python", "scripts/Exploratory_Analysis.py"])

    # Run Pre-processing Script
    subprocess.run(["python", "scripts/Pre-Processing.py"])

    # Run data Setup Script
    subprocess.run(["python", "scripts/data_setup.py"])

    # Run Model 1 Script
    subprocess.run(["python", "scripts/Model1.py"])

    # Run Model 2 Script
    subprocess.run(["python", "scripts/Model2.py"])

    # Run Model Comparison Script
    subprocess.run(["python", "scripts/Model_Comparison.py"])

    # Run Predictive Attributes Script
    subprocess.run(["python", "scripts/Predictive_Attributes.py"])


if __name__ == "__main__":
    main()

import subprocess

# Estimated 15 Minute Run Time, Assuming 8 Processors Available
def main():
    # Run EDA Script
    subprocess.run(["python", "Exploratory_Analysis.py"])

    # Run Pre-processing Script
    subprocess.run(["python", "Pre-Processing.py"])

    # Run Data Setup Script
    subprocess.run(["python", "data_setup.py"])

    # Run Model 1 Script
    subprocess.run(["python", "Model1.py"])

    # Run Model 2 Script
    subprocess.run(["python", "Model2.py"])

    # Run Model Comparison Script
    subprocess.run(["python", "Model_Comparison.py"])

    # Run Predictive Attributes Script
    subprocess.run(["python", "Predictive_Attributes.py"])


if __name__ == "__main__":
    main()

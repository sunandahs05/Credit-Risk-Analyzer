import os
import sys

# Ensure current directory is in path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

import data_acquisition
import feature_engineering
import model_training

def main():
    print("Starting pipeline...")
    # Absolute paths relative to this script
    data_raw_path = os.path.abspath(os.path.join(script_dir, "../../data/raw"))
    data_processed_path = os.path.abspath(os.path.join(script_dir, "../data/processed"))
    
    print(f"Raw data path: {data_raw_path}")
    print(f"Processed data path: {data_processed_path}")
    
    data_acquisition.run_data_acquisition(data_raw_path=data_raw_path, data_processed_path=data_processed_path)
    feature_engineering.run_feature_engineering(data_processed_path=data_processed_path)
    model_training.run_model_training(data_processed_path=data_processed_path)
    
    print("Pipeline complete!")

if __name__ == "__main__":
    main()

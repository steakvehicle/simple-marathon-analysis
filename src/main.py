import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processor import load_data, clean_data, calculate_statistics, calculate_pace


def main():
    raw_data_path = "data/raw/results_2025.xml"
    processed_data_path = "data/processed/marathon_cleaned.csv"

    print("Loading marathon data...")
    df = load_data(raw_data_path)

    print("Cleaning data...")
    df_clean = clean_data(df)

    print("Calculating pace...")
    df_clean = calculate_pace(df_clean)

    print("Calculating statistics...")
    stats = calculate_statistics(df_clean)
    print("\nMarathon Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    df_clean.to_csv(processed_data_path, index=False)
    print(f"\nCleaned data saved to {processed_data_path}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

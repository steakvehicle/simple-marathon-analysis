# Marathon Data Analysis

A Python-based data analysis project for marathon runner performance using pandas, matplotlib, and seaborn.

## Project Structure

```
simple_marathon_analysis/
├── src/
│   ├── __init__.py
│   ├── data_processor.py    # Core data processing functions
│   └── main.py              # Entry point
├── notebooks/               # Jupyter notebooks
├── tests/                   # Unit tests
├── requirements.txt         # Python dependencies
└── README.md
```

## Data Processing

### `parse_xml(file_path)`
Parses marathon results from an XML file. Extracts competitor information including name, club, class, distance, finish time, and status. Derives gender and age group from the class code. Empty club values are replaced with `'no_club'`. Rows with VAC classes (`vvac`, `pvac`, `tvac`) are dropped.

### `load_data(file_path)`
Loads data from XML or CSV files. For CSV files, missing club values are replaced with `'no_club'`.

### `clean_data(df)`
Removes duplicate entries based on `start_number` and `class` columns, then resets the index.

### `calculate_statistics(df)`
Computes summary statistics including mean, median, and standard deviation of finish times, identifies best and slowest finishers, and counts participants by gender, age group, and distance.

### `calculate_pace(df)`
Calculates average pace per kilometer for each runner. Negative finish times are clipped to zero.

## Runner Status Codes

| Code | Description    |
|------|----------------|
| F    | Finished       |
| DNS  | Did Not Start  |
| DNF  | Did Not Finish |

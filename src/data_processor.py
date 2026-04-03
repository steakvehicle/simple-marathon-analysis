import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET


def parse_xml(file_path: str) -> pd.DataFrame:
    """Parse marathon results from an XML file.

    Extracts competitor information including name, club, class, distance,
    finish time, and status. Derives gender and age group from the class
    code. Empty club values are replaced with 'no_club'. Rows with VAC
    classes (vvac, pvac, tvac) are dropped.

    Args:
        file_path: Path to the XML results file.

    Returns:
        DataFrame with columns: 'start_number', 'family_name', 'given_name',
        'club', 'class', 'distance_km', 'finish_time_hms', 'finish_time_sec',
        'status', 'full_name', 'gender', and 'age_group'.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    event_name = root.find('EventName').text if root.find('EventName') is not None else 'Unknown'
    rows = []

    for event_class in root.findall('EventClass'):
        class_name = event_class.find('ClassName').text
        class_dist_str = event_class.find('ClassDist').text
        class_dist_km = float(class_dist_str.replace(',', '.'))

        for competitor in event_class.findall('Competitor'):
            name_el = competitor.find('Name')
            family = name_el.find('Family').text if name_el is not None and name_el.find('Family') is not None else ''
            given = name_el.find('Given').text if name_el is not None and name_el.find('Given') is not None else ''
            status_el = competitor.find('Status')
            status = status_el.text if status_el is not None and status_el.text else 'F'
            tsecs_el = competitor.find('TSecs')
            tsecs = int(tsecs_el.text) if tsecs_el is not None and tsecs_el.text else None
            
            rows.append({
                'start_number': int(competitor.find('StartNumber').text),
                'family_name': family,
                'given_name': given,
                'club': competitor.find('ClubName').text if competitor.find('ClubName') is not None and competitor.find('ClubName').text else 'no_club',
                'class': class_name,
                'distance_km': class_dist_km,
                'finish_time_hms': competitor.find('Time').text if competitor.find('Time') is not None else '',
                'finish_time_sec': tsecs,
                'rank': int(competitor.find('Rank').text) if competitor.find('Rank') is not None and competitor.find('Rank').text else None,
                'status': status,
            })
    
    df = pd.DataFrame(rows)

    df['full_name'] = df['given_name'] + ' ' + df['family_name']
    df['gender'] = df['class'].str[1].map({'M': 'M', 'N': 'F', 'P': 'M', 'T': 'F'})
    df['gender'] = df['gender'].fillna('Unknown')
    age_map = {'YL': 'open', '30': '30-39', '40': '40-49', '50': '50-59', '60': '60+',
               '17': 'U17', '15': 'U15', 'VAC': 'open'}
    df['age_group'] = df['class'].str[2:].map(age_map).fillna('open')
    df = df[~df['class'].str.lower().isin(['vvac', 'pvac', 'tvac'])]
    df['club'] = df['club'].replace('', 'no_club')
    df = df.drop(columns=['rank', 'order'], errors='ignore')
    print(f"Loaded {len(df)} rows from {event_name}")

    return df


def load_data(file_path: str) -> pd.DataFrame:
    """Load marathon data from a file.

    Supports XML and CSV formats. The file extension determines
    the parsing method used. For CSV files, empty club values are
    replaced with 'no_club'.

    Args:
        file_path: Path to the data file (.xml or .csv).

    Returns:
        DataFrame containing the raw marathon data.

    Raises:
        ValueError: If the file format is not supported.
    """
    if file_path.endswith('.xml'):
        return parse_xml(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        if 'club' in df.columns:
            df['club'] = df['club'].fillna('no_club')
        return df
    else:
        raise ValueError("Unsupported file format. Use XML or CSV")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean marathon data by removing duplicate entries.

    Removes rows with duplicate start_number and class combinations,
    then resets the index.

    Args:
        df: DataFrame containing raw marathon data with 'start_number'
            and 'class' columns.

    Returns:
        Cleaned DataFrame with duplicates removed and index reset.
    """
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates(subset=['start_number', 'class'])
    df_clean = df_clean.reset_index(drop=True)
    print(f"Cleaned data: {len(df_clean)} runners")

    return df_clean


def calculate_statistics(df: pd.DataFrame) -> dict:
    """Calculate summary statistics for marathon results.

    Computes mean, median, and standard deviation of finish times,
    identifies best and slowest finishers, and counts participants
    by gender, age group, and distance.

    Args:
        df: DataFrame containing marathon results with columns such as
            'finish_time_sec', 'full_name', 'finish_time_hms', 'gender',
            'age_group', and 'distance_km'.

    Returns:
        Dictionary with keys: 'mean_finish_time_min', 'median_finish_time_min',
        'std_finish_time_min', 'best_finish', 'slowest_finish',
        'participants_by_gender', 'participants_by_age_group',
        'participants_by_distance', and 'total_finishers'.
    """
    stats = {}
    if 'finish_time_sec' in df.columns:
        stats['mean_finish_time_min'] = df['finish_time_sec'].mean() / 60
        stats['median_finish_time_min'] = df['finish_time_sec'].median() / 60
        stats['std_finish_time_min'] = df['finish_time_sec'].std() / 60
        df_finished = df.dropna(subset=['finish_time_sec'])
        
        if len(df_finished) > 0:
            best_idx = df_finished['finish_time_sec'].idxmin()
            worst_idx = df_finished['finish_time_sec'].idxmax()
            stats['best_finish'] = f"{df.loc[best_idx, 'full_name']} ({df.loc[best_idx, 'finish_time_hms']})"
            stats['slowest_finish'] = f"{df.loc[worst_idx, 'full_name']} ({df.loc[worst_idx, 'finish_time_hms']})"

    if 'gender' in df.columns:
        stats['participants_by_gender'] = df['gender'].value_counts().to_dict()
    if 'age_group' in df.columns:
        stats['participants_by_age_group'] = df['age_group'].value_counts().to_dict()
    if 'distance_km' in df.columns:
        stats['participants_by_distance'] = df.groupby('distance_km').size().to_dict()

    stats['total_finishers'] = len(df.dropna(subset=['finish_time_sec']))

    return stats


def calculate_pace(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the average pace per kilometer for each runner.

    Negative finish times are clipped to zero to prevent negative pace values.

    Args:
        df: DataFrame containing 'finish_time_sec' and 'distance_km' columns.

    Returns:
        Copy of the input DataFrame with an additional 'pace_min_per_km' column.
    """
    df_with_pace = df.copy()
    df_with_pace['pace_min_per_km'] = (df_with_pace['finish_time_sec'].clip(lower=0) / 60) / df_with_pace['distance_km']
    return df_with_pace

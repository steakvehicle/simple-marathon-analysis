import unittest
import pandas as pd
import numpy as np
import os
import tempfile

from src.data_processor import calculate_pace, calculate_statistics, clean_data, load_data, parse_xml


class TestCalculatePace(unittest.TestCase):

    def test_pace_calculation(self):
        df = pd.DataFrame({
            'finish_time_sec': [3600, 1800, 7200],
            'distance_km': [10.0, 5.0, 21.0975],
        })
        result = calculate_pace(df)
        expected = [6.0, 6.0, 7200 / 60 / 21.0975]
        np.testing.assert_almost_equal(
            result['pace_min_per_km'].tolist(), expected, decimal=4
        )

    def test_does_not_modify_original(self):
        df = pd.DataFrame({
            'finish_time_sec': [3600],
            'distance_km': [10.0],
        })
        _ = calculate_pace(df)
        self.assertNotIn('pace_min_per_km', df.columns)

    def test_returns_copy(self):
        df = pd.DataFrame({
            'finish_time_sec': [3600],
            'distance_km': [10.0],
        })
        result = calculate_pace(df)
        self.assertIsNot(result, df)

    def test_preserves_existing_columns(self):
        df = pd.DataFrame({
            'finish_time_sec': [3600],
            'distance_km': [10.0],
            'full_name': ['Test Runner'],
        })
        result = calculate_pace(df)
        self.assertIn('full_name', result.columns)
        self.assertEqual(result.loc[0, 'full_name'], 'Test Runner')

    def test_nan_pace_for_missing_time(self):
        df = pd.DataFrame({
            'finish_time_sec': [3600, np.nan],
            'distance_km': [10.0, 10.0],
        })
        result = calculate_pace(df)
        self.assertAlmostEqual(result.loc[0, 'pace_min_per_km'], 6.0)
        self.assertTrue(np.isnan(result.loc[1, 'pace_min_per_km']))

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=['finish_time_sec', 'distance_km'])
        result = calculate_pace(df)
        self.assertIn('pace_min_per_km', result.columns)
        self.assertEqual(len(result), 0)

    def test_pace_not_negative(self):
        df = pd.DataFrame({
            'finish_time_sec': [3600, -100, 0],
            'distance_km': [10.0, 10.0, 10.0],
        })
        result = calculate_pace(df)
        self.assertTrue((result['pace_min_per_km'] >= 0).all())


class TestCalculateStatistics(unittest.TestCase):

    def test_mean_finish_time(self):
        df = pd.DataFrame({
            'finish_time_sec': [3600, 7200],
            'full_name': ['A', 'B'],
            'finish_time_hms': ['1:00:00', '2:00:00'],
        })
        stats = calculate_statistics(df)
        self.assertAlmostEqual(stats['mean_finish_time_min'], 90.0)

    def test_median_finish_time(self):
        df = pd.DataFrame({
            'finish_time_sec': [3600, 5400, 7200],
            'full_name': ['A', 'B', 'C'],
            'finish_time_hms': ['1:00:00', '1:30:00', '2:00:00'],
        })
        stats = calculate_statistics(df)
        self.assertAlmostEqual(stats['median_finish_time_min'], 90.0)

    def test_best_and_slowest_finisher(self):
        df = pd.DataFrame({
            'finish_time_sec': [3600, 7200],
            'full_name': ['Fast Runner', 'Slow Runner'],
            'finish_time_hms': ['1:00:00', '2:00:00'],
        })
        stats = calculate_statistics(df)
        self.assertIn('Fast Runner', stats['best_finish'])
        self.assertIn('Slow Runner', stats['slowest_finish'])

    def test_participants_by_gender(self):
        df = pd.DataFrame({
            'finish_time_sec': [3600, 7200, 5400],
            'full_name': ['A', 'B', 'C'],
            'finish_time_hms': ['1:00:00', '2:00:00', '1:30:00'],
            'gender': ['M', 'F', 'M'],
        })
        stats = calculate_statistics(df)
        self.assertEqual(stats['participants_by_gender'], {'M': 2, 'F': 1})

    def test_participants_by_age_group(self):
        df = pd.DataFrame({
            'finish_time_sec': [3600, 7200],
            'full_name': ['A', 'B'],
            'finish_time_hms': ['1:00:00', '2:00:00'],
            'age_group': ['open', '30-39'],
        })
        stats = calculate_statistics(df)
        self.assertEqual(stats['participants_by_age_group'], {'open': 1, '30-39': 1})

    def test_participants_by_distance(self):
        df = pd.DataFrame({
            'finish_time_sec': [3600, 3600, 7200],
            'full_name': ['A', 'B', 'C'],
            'finish_time_hms': ['1:00:00', '1:00:00', '2:00:00'],
            'distance_km': [10.0, 10.0, 21.0975],
        })
        stats = calculate_statistics(df)
        self.assertEqual(stats['participants_by_distance'], {10.0: 2, 21.0975: 1})

    def test_total_finishers(self):
        df = pd.DataFrame({
            'finish_time_sec': [3600, 7200, np.nan],
            'full_name': ['A', 'B', 'C'],
            'finish_time_hms': ['1:00:00', '2:00:00', ''],
        })
        stats = calculate_statistics(df)
        self.assertEqual(stats['total_finishers'], 2)

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=['finish_time_sec'])
        stats = calculate_statistics(df)
        self.assertEqual(stats['total_finishers'], 0)
        self.assertTrue(np.isnan(stats['mean_finish_time_min']))

    def test_all_nan_finish_times(self):
        df = pd.DataFrame({
            'finish_time_sec': [np.nan, np.nan],
            'full_name': ['A', 'B'],
            'finish_time_hms': ['', ''],
        })
        stats = calculate_statistics(df)
        self.assertEqual(stats['total_finishers'], 0)
        self.assertNotIn('best_finish', stats)
        self.assertNotIn('slowest_finish', stats)

    def test_returns_dict(self):
        df = pd.DataFrame({
            'finish_time_sec': [3600],
            'full_name': ['A'],
            'finish_time_hms': ['1:00:00'],
            'gender': ['M'],
            'age_group': ['open'],
            'distance_km': [10.0],
        })
        stats = calculate_statistics(df)
        self.assertIsInstance(stats, dict)
        self.assertIn('total_finishers', stats)


class TestCleanData(unittest.TestCase):

    def test_removes_duplicates(self):
        df = pd.DataFrame({
            'start_number': [1, 1, 2],
            'class': ['M30', 'M30', 'M30'],
            'full_name': ['A', 'A_dup', 'B'],
        })
        result = clean_data(df)
        self.assertEqual(len(result), 2)

    def test_resets_index(self):
        df = pd.DataFrame({
            'start_number': [1, 1, 2],
            'class': ['M30', 'M30', 'M30'],
        })
        result = clean_data(df)
        self.assertEqual(result.index.tolist(), [0, 1])

    def test_does_not_modify_original(self):
        df = pd.DataFrame({
            'start_number': [1, 1],
            'class': ['M30', 'M30'],
        })
        original_len = len(df)
        _ = clean_data(df)
        self.assertEqual(len(df), original_len)

    def test_returns_copy(self):
        df = pd.DataFrame({
            'start_number': [1],
            'class': ['M30'],
        })
        result = clean_data(df)
        self.assertIsNot(result, df)

    def test_no_duplicates_unchanged(self):
        df = pd.DataFrame({
            'start_number': [1, 2, 3],
            'class': ['M30', 'M40', 'M50'],
            'full_name': ['A', 'B', 'C'],
        })
        result = clean_data(df)
        self.assertEqual(len(result), 3)

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=['start_number', 'class'])
        result = clean_data(df)
        self.assertEqual(len(result), 0)

    def test_same_start_number_different_class(self):
        df = pd.DataFrame({
            'start_number': [1, 1],
            'class': ['M30', 'M40'],
            'full_name': ['A', 'A'],
        })
        result = clean_data(df)
        self.assertEqual(len(result), 2)


class TestLoadData(unittest.TestCase):

    def test_load_csv(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('start_number,family_name,given_name,club,class,distance_km,finish_time_hms,finish_time_sec,status,full_name,gender,age_group\n')
            f.write('1,Smith,John,Club A,M30,10.0,1:00:00,3600,F,John Smith,M,30-39\n')
            f.write('2,Doe,Jane,Club B,F40,10.0,1:10:00,4200,F,Jane Doe,F,40-49\n')
            temp_path = f.name

        try:
            df = load_data(temp_path)
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(len(df), 2)
            self.assertIn('start_number', df.columns)
        finally:
            os.unlink(temp_path)

    def test_load_csv_empty_club(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('start_number,family_name,given_name,club,class,distance_km,finish_time_hms,finish_time_sec,status,full_name,gender,age_group\n')
            f.write('1,Smith,John,,M30,10.0,1:00:00,3600,F,John Smith,M,30-39\n')
            temp_path = f.name

        try:
            df = load_data(temp_path)
            self.assertEqual(df.loc[0, 'club'], 'no_club')
        finally:
            os.unlink(temp_path)

    def test_unsupported_format_raises(self):
        with self.assertRaises(ValueError):
            load_data('data.txt')

    def test_unsupported_format_json_raises(self):
        with self.assertRaises(ValueError):
            load_data('data.json')

    def test_load_csv_returns_dataframe(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('a,b\n1,2\n')
            temp_path = f.name

        try:
            df = load_data(temp_path)
            self.assertIsInstance(df, pd.DataFrame)
        finally:
            os.unlink(temp_path)


class TestParseXml(unittest.TestCase):

    VALID_XML = """<?xml version="1.0"?>
    <Results>
        <EventName>Test Event</EventName>
        <EventClass>
            <ClassName>AM30</ClassName>
            <ClassDist>10,0</ClassDist>
            <Competitor>
                <StartNumber>1</StartNumber>
                <Name><Family>Smith</Family><Given>John</Given></Name>
                <ClubName>Club A</ClubName>
                <Time>1:00:00</Time>
                <TSecs>3600</TSecs>
                <Rank>1</Rank>
                <Status></Status>
            </Competitor>
            <Competitor>
                <StartNumber>2</StartNumber>
                <Name><Family>Doe</Family><Given>Jane</Given></Name>
                <ClubName>Club B</ClubName>
                <Time>1:10:00</Time>
                <TSecs>4200</TSecs>
                <Rank>2</Rank>
                <Status></Status>
            </Competitor>
        </EventClass>
    </Results>
    """

    def _write_xml(self, content):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(content)
            return f.name

    def test_returns_dataframe(self):
        path = self._write_xml(self.VALID_XML)
        try:
            df = parse_xml(path)
            self.assertIsInstance(df, pd.DataFrame)
        finally:
            os.unlink(path)

    def test_correct_row_count(self):
        path = self._write_xml(self.VALID_XML)
        try:
            df = parse_xml(path)
            self.assertEqual(len(df), 2)
        finally:
            os.unlink(path)

    def test_expected_columns(self):
        path = self._write_xml(self.VALID_XML)
        try:
            df = parse_xml(path)
            expected_cols = [
                'start_number', 'family_name', 'given_name', 'club', 'class',
                'distance_km', 'finish_time_hms', 'finish_time_sec', 'status',
                'full_name', 'gender', 'age_group',
            ]
            for col in expected_cols:
                self.assertIn(col, df.columns)
        finally:
            os.unlink(path)

    def test_full_name_construction(self):
        path = self._write_xml(self.VALID_XML)
        try:
            df = parse_xml(path)
            self.assertEqual(df.loc[0, 'full_name'], 'John Smith')
            self.assertEqual(df.loc[1, 'full_name'], 'Jane Doe')
        finally:
            os.unlink(path)

    def test_gender_derived_from_class(self):
        path = self._write_xml(self.VALID_XML)
        try:
            df = parse_xml(path)
            self.assertEqual(df.loc[0, 'gender'], 'M')
        finally:
            os.unlink(path)

    def test_age_group_derived_from_class(self):
        path = self._write_xml(self.VALID_XML)
        try:
            df = parse_xml(path)
            self.assertEqual(df.loc[0, 'age_group'], '30-39')
        finally:
            os.unlink(path)

    def test_empty_status_defaults_to_finished(self):
        path = self._write_xml(self.VALID_XML)
        try:
            df = parse_xml(path)
            self.assertEqual(df.loc[0, 'status'], 'F')
        finally:
            os.unlink(path)

    def test_empty_club_replaced_with_no_club(self):
        xml = """<?xml version="1.0"?>
        <Results>
            <EventName>Test</EventName>
            <EventClass>
                <ClassName>AM30</ClassName>
                <ClassDist>10,0</ClassDist>
                <Competitor>
                    <StartNumber>1</StartNumber>
                    <Name><Family>Smith</Family><Given>John</Given></Name>
                    <ClubName></ClubName>
                    <Time>1:00:00</Time>
                    <TSecs>3600</TSecs>
                    <Rank>1</Rank>
                    <Status></Status>
                </Competitor>
            </EventClass>
        </Results>
        """
        path = self._write_xml(xml)
        try:
            df = parse_xml(path)
            self.assertEqual(df.loc[0, 'club'], 'no_club')
        finally:
            os.unlink(path)

    def test_missing_club_element_replaced_with_no_club(self):
        xml = """<?xml version="1.0"?>
        <Results>
            <EventName>Test</EventName>
            <EventClass>
                <ClassName>AM30</ClassName>
                <ClassDist>10,0</ClassDist>
                <Competitor>
                    <StartNumber>1</StartNumber>
                    <Name><Family>Smith</Family><Given>John</Given></Name>
                    <Time>1:00:00</Time>
                    <TSecs>3600</TSecs>
                    <Rank>1</Rank>
                    <Status></Status>
                </Competitor>
            </EventClass>
        </Results>
        """
        path = self._write_xml(xml)
        try:
            df = parse_xml(path)
            self.assertEqual(df.loc[0, 'club'], 'no_club')
        finally:
            os.unlink(path)

    def test_vac_classes_dropped(self):
        xml = """<?xml version="1.0"?>
        <Results>
            <EventName>Test</EventName>
            <EventClass>
                <ClassName>M30</ClassName>
                <ClassDist>10,0</ClassDist>
                <Competitor>
                    <StartNumber>1</StartNumber>
                    <Name><Family>Smith</Family><Given>John</Given></Name>
                    <ClubName>Club A</ClubName>
                    <Time>1:00:00</Time>
                    <TSecs>3600</TSecs>
                    <Rank>1</Rank>
                    <Status></Status>
                </Competitor>
            </EventClass>
            <EventClass>
                <ClassName>pvac</ClassName>
                <ClassDist>10,0</ClassDist>
                <Competitor>
                    <StartNumber>99</StartNumber>
                    <Name><Family>Vac</Family><Given>Runner</Given></Name>
                    <ClubName>Club B</ClubName>
                    <Time>1:00:00</Time>
                    <TSecs>3600</TSecs>
                    <Rank>1</Rank>
                    <Status></Status>
                </Competitor>
            </EventClass>
        </Results>
        """
        path = self._write_xml(xml)
        try:
            df = parse_xml(path)
            self.assertEqual(len(df), 1)
            self.assertNotIn('pvac', df['class'].values)
        finally:
            os.unlink(path)

    def test_distance_km_with_comma_decimal(self):
        path = self._write_xml(self.VALID_XML)
        try:
            df = parse_xml(path)
            self.assertAlmostEqual(df.loc[0, 'distance_km'], 10.0)
        finally:
            os.unlink(path)

    def test_start_number_parsed_correctly(self):
        path = self._write_xml(self.VALID_XML)
        try:
            df = parse_xml(path)
            self.assertEqual(df.loc[0, 'start_number'], 1)
            self.assertEqual(df.loc[1, 'start_number'], 2)
        finally:
            os.unlink(path)

    def test_finish_time_sec_parsed(self):
        path = self._write_xml(self.VALID_XML)
        try:
            df = parse_xml(path)
            self.assertEqual(df.loc[0, 'finish_time_sec'], 3600)
            self.assertEqual(df.loc[1, 'finish_time_sec'], 4200)
        finally:
            os.unlink(path)

    def test_multiple_event_classes(self):
        xml = """<?xml version="1.0"?>
        <Results>
            <EventName>Test</EventName>
            <EventClass>
                <ClassName>M30</ClassName>
                <ClassDist>10,0</ClassDist>
                <Competitor>
                    <StartNumber>1</StartNumber>
                    <Name><Family>A</Family><Given>B</Given></Name>
                    <ClubName>C</ClubName>
                    <Time>1:00:00</Time>
                    <TSecs>3600</TSecs>
                    <Rank>1</Rank>
                    <Status></Status>
                </Competitor>
            </EventClass>
            <EventClass>
                <ClassName>F40</ClassName>
                <ClassDist>21,1</ClassDist>
                <Competitor>
                    <StartNumber>2</StartNumber>
                    <Name><Family>X</Family><Given>Y</Given></Name>
                    <ClubName>Z</ClubName>
                    <Time>2:00:00</Time>
                    <TSecs>7200</TSecs>
                    <Rank>1</Rank>
                    <Status></Status>
                </Competitor>
            </EventClass>
        </Results>
        """
        path = self._write_xml(xml)
        try:
            df = parse_xml(path)
            self.assertEqual(len(df), 2)
            self.assertIn('M30', df['class'].values)
            self.assertIn('F40', df['class'].values)
        finally:
            os.unlink(path)


if __name__ == '__main__':
    unittest.main()

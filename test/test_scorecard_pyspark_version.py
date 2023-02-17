
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
import pytest
import warnings
from scorecard_pyspark_version import *
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
import pytest
from pyspark.sql import SparkSession,Row
from pyspark.ml.feature import Bucketizer
from pyspark.sql.functions import lit, col, array
import re

@pytest.fixture(scope="module")
def spark():
    spark = SparkSession.builder.appName("check_empty_bins_test").getOrCreate()
    yield spark
    spark.stop()

@pytest.fixture
def example_data():
    
    yield dtm

def test_miv_01():
    # Create a test dataframe
    good = pd.Series([10, 20, 30, 40])
    bad = pd.Series([2, 3, 6, 8])

    # Calculate the expected result
    expected_result = ((bad/sum(bad)) - (good/sum(good))) * np.log((bad/sum(bad)) / (good/sum(good)))
    #expected_result = expected_result.sum()

    # Call the miv_01 function
    result = miv_01(good, bad)

    # Assert that the result is equal to the expected result
    assert np.allclose(result,expected_result,rtol=0.00001)

def test_woe_01():
    # Create a test dataframe
    good = pd.Series([10, 20, 30, 40])
    bad = pd.Series([2, 3, 6, 8])

    # Calculate the expected result
    expected_result = np.log((bad/sum(bad)) / (good/sum(good)))
    #expected_result = expected_result.sum()

    # Call the miv_01 function
    result = woe_01(good, bad)

    # Assert that the result is equal to the expected result
    assert np.allclose(result,expected_result,rtol=0.00001)

import numpy as np

def test_pretty():
    # Test for default values
    result = pretty(0, 10, 6)
    expected = np.array([0., 2., 4., 6., 8., 10.])
    assert np.allclose(result, expected,rtol=0.001), f'Expected {expected}, but got {result}'
    
    # Test for custom values
    result = pretty(3, 8, 6)
    expected = np.array([3., 4., 5., 6., 7., 8.])
    assert np.allclose(result, expected,rtol=0.001), f'Expected {expected}, but got {result}'
    
    # Test for edge cases
    result = pretty(0, 10, 2)
    expected = np.array([0., 10.])
    assert np.allclose(result, expected,rtol=0.001), f'Expected {expected}, but got {result}'
    
    result = pretty(0.1, 0.5, 5)
    expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert np.allclose(result, expected,rtol=0.001), f'Expected {expected}, but got {result}'

def test_check_empty_bins(spark):
    # setup data
    data = [(1.0,0,"M97","[1,2)"),(1.5,1,"M97","[1,2)"),(5.6,0,"M97","[5,6)"),(3.4,1,"M97","[3,4)"),(7.3,1,"M97","[7,8)"),(9.6,0,"M97","[9,10)")]
    dtm = spark.createDataFrame(data, ["value","y","variable","bin"])

    data_1 = [("[1,2)",1,1),("[1,2)",1,1),("[3,4)",0,1),("[5,6)",1,0),("[7,8)",0,1),("[9,10)",1,0)]
    binning = spark.createDataFrame(data_1, ["bin","0","1"])
    

    # check result
    result = check_empty_bins(dtm, binning)

    assert result.count() == 5
    assert result.columns == ['bins','0','1']
    assert result.agg({"0": "sum", "1": "sum"}).collect()[0] == (3, 3)

def test_binning_format():
    # Test input data
    binning_data = {
        'variable': ['var1', 'var2', 'var3'],
        'bin': ['[0,1)', '[1,2)', '[2,3)'],
        'good': [10, 20, 30],
        'bad': [5, 10, 15]
    }
    binning = pd.DataFrame(binning_data)
    
    # Calculate expected output
    binning['count'] = binning['good'] + binning['bad']
    binning['count_distr'] = binning['count'] / binning['count'].sum()
    binning['badprob'] = binning['bad'] / binning['count']
    binning['woe'] = woe_01(binning['good'], binning['bad'])
    binning['bin_iv'] = miv_01(binning['good'], binning['bad'])
    binning['total_iv'] = binning['bin_iv'].sum()
    binning['breaks'] = binning['bin']
    if any([r'[' in str(i) for i in binning['bin']]):
        def re_extract_all(x): 
            gp23 = re.match(r"^\[(.*), *(.*)\)((%,%missing)*)", x)
            breaks_string = x if gp23 is None else gp23.group(2)+gp23.group(3)
            return breaks_string
        binning['breaks'] = [re_extract_all(i) for i in binning['bin']]
    # is_sv    
    binning['is_sv'] = False
    expected_output = binning[['variable', 'bin', 'count', 'count_distr', 'good', 'bad', 'badprob', 'woe', 'bin_iv', 'total_iv', 'breaks', 'is_sv']]
    expected_output.rename(columns={'is_sv':'is_special_values'},inplace=True)
    # Call the function under test
    result = binning_format(binning)
    
    # Assert that the output matches the expected output
    assert result.equals(expected_output)

def test_remove_dots_in_column_names(spark):
    # Create a sample dataframe with columns containing dots
    schema = StructType([
        StructField("col.1", IntegerType(), True),
        StructField("col.2", StringType(), True),
        StructField("col_3", StringType(), True)
    ])
    data = [
        (1, "value_1", "value_3"),
        (2, "value_2", "value_4"),
        (3, "value_3", "value_5")
    ]
    df = spark.createDataFrame(data, schema=schema)

    # Test the remove_dots_in_column_names function
    result = remove_dots_in_column_names(df)

    # Assert that the function replaces the dots with underscores in column names
    assert "col.1" not in result.columns
    assert "col.2" not in result.columns
    assert "col_1" in result.columns
    assert "col_2" in result.columns
    assert "col_3" in result.columns
    
def test_x_variable():
    # Create a sample dataframe
    data = {
        'y1': [1, 2, 3, 4, 5],
        'x1': [11, 12, 13, 14, 15],
        'x2': [16, 17, 18, 19, 20],
        'x3': [21, 22, 23, 24, 25]
    }
    df = pd.DataFrame(data)

    # Test the x_variable function with x set to None
    with warnings.catch_warnings(record=True) as w:
        result = x_variable(df, 'y1', None)
        assert set(result) == set(['x1','x2', 'x3'])
        assert len(w) == 0

    # Test the x_variable function with x set to a list
    with warnings.catch_warnings(record=True) as w:
        result = x_variable(df, 'y1', 'x1')
        assert result == ['x1']
        assert len(w) == 0

    # Test the x_variable function with x containing incorrect inputs
    with warnings.catch_warnings(record=True) as w:
        result = x_variable(df, 'y1', x=['x1','x2','x3','x4'])
        assert set(result) == set(['x1', 'x2', 'x3'])
        assert len(w) == 1
        assert str(w[0].message) == 'Incorrect inputs; there are 1 x variables are not exist in input data, which are removed from x. \n(x4)'

    # Test the x_variable function with var_skip set to a list
    with warnings.catch_warnings(record=True) as w:
        result = x_variable(df, 'y1', 'x1', 'x2')
        assert result == ['x1']
        assert len(w) == 0

def test_woebin2_init_bin():
    spark = SparkSession.builder.appName("woebin-test").getOrCreate()

    # create test data
    data = pd.DataFrame({'variable': "M97",
                         'value': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
                         'y': [0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]})

    dtm = spark.createDataFrame(data)

    # test the function
    result = woebin2_init_bin(dtm, 0.1,breaks=None,spl_val=None)

    # check the output
    assert set(result.keys()) == {'initial_binning'}
    assert len(result['initial_binning']) == 6
    assert result['initial_binning']['variable'][0] == 'M97'
    assert result['initial_binning']['bin'].tolist() == ['[-inf,0.6000000000000001)', '[0.6000000000000001,1.0)', '[1.0,1.2000000000000002)', '[1.2000000000000002,1.6)','[1.6,1.8)', '[1.8,inf)']
    assert np.allclose(result['initial_binning']['good'].tolist(), [4,2,2,2,1,2], atol=1e-6)
    assert np.allclose(result['initial_binning']['bad'].tolist(), [2,1,1,1,1,1], atol=1e-6)
    assert np.allclose(result['initial_binning']['badprob'].tolist(), [0.3333,0.3333,0.3333,0.3333,0.5,0.3333], atol=1e-3)

def test_str_to_list_with_string():
    assert str_to_list("hello") == ["hello"]

def test_str_to_list_with_list():
    assert str_to_list([1, 2, 3]) == [1, 2, 3]

def test_str_to_list_with_none():
    assert str_to_list(None) == None

def test_str_to_list_with_non_string():
    assert str_to_list(123) == 123

def test_str_to_list_with_empty_string():
    assert str_to_list("") == [""]

def test_str_to_list_with_boolean():
    assert str_to_list(True) == True

def test_str_to_list_with_float():
    assert str_to_list(3.14) == 3.14

def test_woebin2_chimerge(spark):
    init_count_distr = 0.02
    count_distr_limit = 0.05
    stop_limit = 0.1
    bin_num_limit = 8
    breaks = None
    spl_val = None 
    # prepare input data
    data = pd.read_csv('sample_py_df.csv',usecols=['M97','def_trig'])
    data.rename(columns={'M97':'value','def_trig':'y'},inplace=True)
    data['variable'] = 'M97'
    dtm = spark.createDataFrame(data)
    expected_result = pd.read_csv('chimerged_output_for_test.csv')

    result = woebin2_chimerge(dtm, init_count_distr=init_count_distr, 
                              count_distr_limit=count_distr_limit, stop_limit=stop_limit, 
                              bin_num_limit=bin_num_limit, breaks=breaks, spl_val=spl_val)
    expected_result.drop('variable',axis=1,inplace=True)
    result['binning'].drop('variable',axis=1,inplace=True)
    assert_frame_equal(expected_result, result['binning'],check_dtype=False)
    
def test_woebin2_breaks():
    spark = SparkSession.builder.appName("woebin-test").getOrCreate()

    # create test data
    data = pd.DataFrame({'variable': "M97",
                         'value': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
                         'y': [0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]})

    dtm = spark.createDataFrame(data)

    # test the function
    result = woebin2_breaks(dtm, brk=[0.5,1.0,1.5,2.0],spl_val=None)

    # check the output
    assert set(result.keys()) == {'initial_binning'}
    assert len(result['initial_binning']) == 5
    assert result['initial_binning']['variable'][0] == 'M97'
    assert result['initial_binning']['bin'].tolist() == ['[-inf,0.5)', '[0.5,1.0)', '[1.0,1.5)', '[1.5,2.0)','[2.0,inf)']
    assert np.allclose(result['initial_binning']['good'].tolist(), [3,3,3,4,0], atol=1e-6)
    assert np.allclose(result['initial_binning']['bad'].tolist(), [1,2,2,1,1], atol=1e-6)

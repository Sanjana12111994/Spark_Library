import unittest
from src.main.imputation.imputer_params import *


class TestGetterSetters(unittest.TestCase):

    def test_get_set_strategy(self):
        strat = HasStrategy()
        # normal behaviour
        strat.strategy = "median"
        new_strategy = strat.strategy
        self.assertEqual(new_strategy, "median")

        # strategy is not a string
        with self.assertRaises(TypeError) as e:
            strat.strategy = 123

        self.assertIn("must be a string", str(e.exception))

        # invalid strategy
        with self.assertRaises(ValueError) as e:
            strat.strategy = "invalid"

        self.assertIn("valid strategies are", str(e.exception))

    def test_get_set_input_cols(self):
        in_cols = HasInputOutputCols(input_cols=["in"])
        # normal behaviour
        in_cols.input_cols = ["col_in1", "col_in2"]
        input_cols = in_cols.input_cols
        self.assertEqual(input_cols, ["col_in1", "col_in2"])

        # input columns not a list
        with self.assertRaises(TypeError) as e:
            in_cols.input_cols = ("tuple_col1")

        self.assertIn("must be in a list", str(e.exception))

        # input column length 0
        with self.assertRaises(ValueError) as e:
            in_cols.input_cols = []

        self.assertIn("cannot be length 0", str(e.exception))

        # column names are not type str
        with self.assertRaises(TypeError) as e:
            in_cols.input_cols = [1, 2, 3]

        self.assertIn("must be strings", str(e.exception))

    def test_get_set_output_cols(self):
        out_cols = HasInputOutputCols(input_cols=["in"], output_cols=["out"])
        # normal behaviour
        out_cols.output_cols = ["col_out1", "col_out2"]
        output_cols = out_cols.output_cols
        self.assertEqual(output_cols, ["col_out1", "col_out2"])

        # no output columns specified
        out_cols.output_cols = None
        self.assertEqual(out_cols.output_cols, out_cols.input_cols)

        # output columns not a list
        with self.assertRaises(TypeError) as e:
            out_cols.output_cols = ("tuple_col2")

        self.assertIn("must be in a list", str(e.exception))

        # output column length 0
        with self.assertRaises(ValueError) as e:
            out_cols.output_cols = []

        self.assertIn("cannot be length 0", str(e.exception))

        # output column names are not type str
        with self.assertRaises(TypeError) as e:
            out_cols.output_cols = [1, 2, 3]

        self.assertIn("must be strings", str(e.exception))

    def test_get_set_missing_value(self):
        miss_val = HasMissingValue()
        # default behaviour
        miss_val.missing_value = float("nan")
        missing_val = miss_val.missing_value
        self.assertNotEqual(missing_val, missing_val)  # NaN has property that NaN != NaN

        # normal behaviour
        miss_val.missing_value = 0.25
        missing_val = miss_val.missing_value
        self.assertEqual(missing_val, 0.25)

        # setting missing value to None
        with self.assertRaises(ValueError) as e:
            miss_val.missing_value = None

        self.assertIn("cannot be None", str(e.exception))

    def test_get_set_relative_error(self):
        rel_err = HasRelativeError()
        # normal behaviour
        rel_err.relative_error = 0.1
        rel_error = rel_err.relative_error
        self.assertEqual(rel_error, 0.1)

        # setting missing value to None
        with self.assertRaises(ValueError) as e:
            rel_err.relative_error = None

        self.assertIn("cannot be None", str(e.exception))

    def test_get_set_order_by_cols(self):
        order = HasOrderByCols(order_by_cols=["order"])
        # normal behaviour
        order.order_by_cols = ["col"]
        order_by_cols = order.order_by_cols
        self.assertEqual(order_by_cols, ["col"])

        # multiple columns
        order.order_by_cols = ["1", "2"]
        order_by_cols = order.order_by_cols
        self.assertEqual(order_by_cols, ["1", "2"])

        # order by columns not a list
        with self.assertRaises(TypeError) as e:
            order.order_by_cols = ("tuple_col2")

        self.assertIn("must be in a list", str(e.exception))

        # output column length 0
        with self.assertRaises(ValueError) as e:
            order.order_by_cols = []

        self.assertIn("cannot be length 0", str(e.exception))

        # output column names are not type str
        with self.assertRaises(TypeError) as e:
            order.order_by_cols = [1, 2, 3]

        self.assertIn("must be strings", str(e.exception))

    def test_get_set_list_group_by_cols(self):
        grp_by = HasListGroupByCols()

        # default
        self.assertTrue(grp_by.list_group_by_cols is None)

        # setting
        grp_by.list_group_by_cols = [["col"], ["col"]]
        # getting
        list_group_by_cols = grp_by.list_group_by_cols
        self.assertEqual(list_group_by_cols, [["col"], ["col"]])

        # not a list
        with self.assertRaises(TypeError) as e:
            grp_by.list_group_by_cols = (["col"], ["col"])

        self.assertIn("must be a List of List of strings", str(e.exception))

        # group_by_cols not all lists
        with self.assertRaises(TypeError) as e:
            grp_by.list_group_by_cols = [["col"], ("col",)]

        self.assertIn("group by cols must be a List of strings.", str(e.exception))

        # not all columns are strings
        with self.assertRaises(TypeError) as e:
            grp_by.list_group_by_cols = [["col"], [1]]

        self.assertIn("Columns must be a string", str(e.exception))

        # empty string as column
        with self.assertRaises(ValueError) as e:
            grp_by.list_group_by_cols = [["col"], [""]]

        self.assertIn("Column names cannot be empty string", str(e.exception))

    def test_get_set_list_partition_by_cols(self):
        part_by = HasListPartitionByCols()

        # default
        self.assertTrue(part_by.list_partition_by_cols is None)

        # setting
        part_by.list_partition_by_cols = [["col"], ["col"]]
        # getting
        list_partition_by_cols = part_by.list_partition_by_cols
        self.assertEqual(list_partition_by_cols, [["col"], ["col"]])

        # not a list
        with self.assertRaises(TypeError) as e:
            part_by.list_partition_by_cols = (["col"], ["col"])

        self.assertIn("must be a List of List of strings", str(e.exception))

        # partition_by_cols not all lists
        with self.assertRaises(TypeError) as e:
            part_by.list_partition_by_cols = [["col"], ("col",)]

        self.assertIn("partition by cols must be a List of strings.", str(e.exception))

        # not all columns are strings
        with self.assertRaises(TypeError) as e:
            part_by.list_partition_by_cols = [["col"], [10]]

        self.assertIn("Columns must be a string", str(e.exception))

        # empty string as column
        with self.assertRaises(ValueError) as e:
            part_by.list_partition_by_cols = [["col"], [""]]

        self.assertIn("Column names cannot be empty string", str(e.exception))






"""Exclusive spectral filtering, including validation before mutation."""
import unittest

from fancy_plotting import xnotice


class _PHA:
    channel = (1, 2, 3)
    units = "energy"


class _Session:
    def __init__(self):
        self.data = {1: _PHA(), 2: _PHA()}
        self.calls = []

    def get_default_id(self):
        return 1

    def get_data(self, key):
        return self.data[key]

    def ignore_id(self, key):
        self.calls.append(("ignore", key))

    def notice_id(self, key, lo, hi):
        self.calls.append(("notice", key, lo, hi))


class ExclusiveFilterTests(unittest.TestCase):
    def test_default_and_multiple_ids_use_exclusive_sherpa_calls(self):
        session = _Session()
        xnotice(lo=1, hi=4, session=session)
        self.assertEqual(session.calls, [("ignore", 1), ("notice", 1, 1., 4.)])
        session.calls.clear()
        xnotice([2, 1], 2, 5, session=session)
        self.assertEqual(session.calls, [("ignore", 2), ("notice", 2, 2., 5.),
                                         ("ignore", 1), ("notice", 1, 2., 5.)])

    def test_bad_bounds_and_unknown_ids_leave_all_filters_untouched(self):
        session = _Session()
        for args in ((1, None, 3), (1, 3, None), (1, 3, 3),
                     (1, 4, 2), (1, float("inf"), 5), (1, 1, float("nan"))):
            with self.subTest(args=args), self.assertRaises(ValueError):
                xnotice(*args, session=session)
        for ids in ([], [1, 1], [1, ""], True):
            with self.subTest(ids=ids), self.assertRaises((TypeError, ValueError)):
                xnotice(ids, 1, 3, session=session)
        with self.assertRaises(KeyError):
            xnotice([1, 99], 1, 3, session=session)
        self.assertEqual(session.calls, [])


if __name__ == "__main__":
    unittest.main()

# Running the tests

Install the package's test dependencies and run the test suite from this
project directory:

```bash
python -m pip install -e '.[test]'
python -m pytest
```

Core tests run without Sherpa. Sherpa/CIAO integration tests are skipped when
Sherpa is unavailable. The [combined-fit benchmark](BENCHMARKING.md) compares
combined-fit predictions and source-call counts; it has no timing threshold.

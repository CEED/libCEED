#!/bin/bash -ex

ceed_dir=${1:-.}
echo "Using CEED_DIR=${ceed_dir}"

#$PYBIN/pip install --upgrade -r requirements-test.txt
#$PYBIN/pip install libceed --no-index --find-links wheelhouse/
pushd ${ceed_dir}/python/tests
python setup-qfunctions.py build
python -m pytest test-*.py --ceed /cpu/self/opt/blocked -vv
popd

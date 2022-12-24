#!/bin/bash -ex

export PYTHONUSERBASE=/io/wheel-user-base

for PYBIN in /opt/python/cp3{7,8,9,10,11}*/bin; do
    $PYBIN/pip install --user --upgrade -r requirements.txt
    $PYBIN/pip wheel . -w wheelhouse/
done

for wheel in wheelhouse/libceed*-linux*.whl; do
    auditwheel repair "$wheel" --plat ${WHEEL_PLAT} -w wheelhouse/
done

for PYBIN in /opt/python/cp3{7,8,9,10,11}*/bin; do
    $PYBIN/pip install --user --upgrade -r requirements-test.txt
    $PYBIN/pip install --user libceed --no-index --find-links wheelhouse/
    pushd python/tests
    $PYBIN/python setup-qfunctions.py build
    $PYBIN/python -m pytest test-*.py --ceed /cpu/self/opt/blocked -vv
    popd
done

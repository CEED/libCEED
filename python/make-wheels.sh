#!/bin/bash -ex

export PYTHONUSERBASE=$(realpath wheel-user-base)

for PYBIN in /opt/python/*/bin; do
    $PYBIN/pip install --user -r requirements.txt
    $PYBIN/pip wheel . -w wheelhouse/
done

for wheel in wheelhouse/libceed*-linux*.whl; do
    auditwheel repair "$wheel" --plat manylinux2014_x86_64 -w wheelhouse/
done

for PYBIN in /opt/python/*/bin; do
    $PYBIN/pip install --user -r requirements-test.txt
    $PYBIN/pip install --user libceed --upgrade --no-index --find-links wheelhouse/
    pushd tests/python
    $PYBIN/python setup-qfunctions.py build
    $PYBIN/python -m pytest test-*.py --ceed /cpu/self/opt/blocked -vv
    popd
done

ACTIVATE=venv/bin/activate

venv: $(ACTIVATE)
$(ACTIVATE): requirements.txt requirements_dev.txt
	test -d venv || virtualenv venv
	. $(ACTIVATE); pip install -r requirements_dev.txt

.PHONY : dist
dist:
	python setup.py sdist bdist_wheel

.PHONY : clean
clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	rm -rf build
	rm -rf dist

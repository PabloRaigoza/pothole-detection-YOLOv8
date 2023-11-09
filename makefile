.PHONY: test venv

run:
	venv/bin/python3 learn.py

train:
	venv/bin/python3 train.py

build:
	sudo apt-get install python3.6
	sudo apt install python3-pip
	sudo apt-get install unzip
	mkdir v3
	curl -L "https://app.roboflow.com/ds/4mKx3G4LtV?key=aFojErQra4" > v3/roboflow.zip; unzip v3/roboflow.zip;
	python3 -m venv venv
	venv/bin/pip3 install -r requirements.txt


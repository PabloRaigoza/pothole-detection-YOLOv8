.PHONY: test venv

run:
	venv/bin/python3 learn.py

train:
	venv/bin/python3 train.py

build:
	sudo apt-get install python3.6
	sudo apt install python3-pip
	sudo apt-get install unzip
	sudo apt install nvidia-cuda-toolkit  
	mkdir v3;
	curl -L "https://app.roboflow.com/ds/h8uohoV1Fo?key=y4yt1YrRNl" > v3/roboflow.zip;
	unzip v3/roboflow.zip -d v3/;
	python3 -m venv venv
	venv/bin/pip3 install -r requirements.txt


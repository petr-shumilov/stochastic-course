FROM python:3.6

WORKDIR /homework
VOLUME /homework/task1
VOLUME /homework/task2
VOLUME /homework/task3
VOLUME /homework/task4


RUN apt-get update \
	&& apt-get install python3-setuptools -y \
	&& pip install --upgrade pip setuptools wheel 

#ENTRYPOINT echo " ============== TASK 1 ============== " && cd task1 && python setup.py install && echo " ============== TASK 2 ============== " && cd ../task2 && python setup.py install && echo " ============== TASK 3 ============== " && cd ../task3 && python setup.py install && /bin/bash
ENTRYPOINT echo " ============== TASK 4 ============== " && cd ./task4 && python setup.py install && /bin/bash
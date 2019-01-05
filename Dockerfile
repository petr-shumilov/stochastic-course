FROM python:3.6

WORKDIR /homework

RUN apt-get update \
	&& apt-get install python3-setuptools git -y \
	&& pip install --upgrade pip setuptools wheel \
	&& git clone https://github.com/petr-shumilov/stochastic-course.git

WORKDIR /homework/stochastic-course

ENTRYPOINT echo " ============== TASK 1 ============== " && cd task1 && python setup.py install && echo " ============== TASK 2 ============== " && cd ../task2 && python setup.py install && echo " ============== TASK 3 ============== " && cd ../task3 && python setup.py install && echo " ============== TASK 4 ============== " && cd ../task4 && python setup.py install && cd .. && /bin/bash

# stochastic-course
## Strcuture
1. `./task1` - linear regression 
2. `./task2` - SVM
3. `./task3` - K-means
4. `./task4` - validation of K-means 

Code of each tasks located in `~/task${N}/${PACKAGE_NAME}/main.py` where `PACKAGE_NAME` package name for each task accordingly
## How to start
My apps are supportoing two methods of reliable checking -- downloading and running `Docker` image (recommended) or manual performing the scripts (you must have all dependencies).
### Docker
For this case you need have only `Docker`. Copy this line to bash-like tty and press enter for downloading and building tasks:
```bash
docker run --name stochastic -it shumilov/study:stochastic
```
Wait few minutes until the packages installed. As a result you will get access for container's tty. For starting specify task you can perform `python -m ${PACKAGE_NAME}`. For ex:
```bash
python -m linear_regression
# or 
python -m svm
# or 
python -m k-means
# or 
python -m validation 
```
For watching the results you can copy files from container:
```bash
# at other terminal 
docker cp stochastic:/homework/stochastic-course/task1/result.pdf /your/path/
# or
docker cp stochastic:/homework/stochastic-course/task3/results /your/path/
```
Results corresponding the tasks will be available in file `result.pdf` or in folder `results`
### Manual (unrecommended)
_WARNING:_  you must have following Python-packages
* NumPy
* Matplotlib
* Pillow
* Sklearn 

and you need fix input and output paths.  
Go to the specific directory and run the corresponding script:
```bash
# downloading the sources 
git clone https://github.com/petr-shumilov/stochastic-course.git
cp stochastic-course

# for ex. if you wanna task1 
cd task1
python3.6 linear_regression/main.py 
```



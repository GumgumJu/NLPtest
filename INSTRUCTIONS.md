# Senoee - Technical test 
<center>Zhenyu ZHU <center>

***

## How to install and use the project

- It is recommended that you run this program in a virtual environment
- Open a terminal or command prompt and go to the root directory of the project.
- All required dependencies are stored in requirement.txt

```
# creat a virtual environment
python -m venv venv

# activate the virtual environment (Windows)
cd venv\Scripts\activate
# activate the virtual environment (Unix)
source venv/bin/activate

pip install --upgrade pip

# Install all dependencies required for the project
pip install -r requirements.txt
```
- Make sure you go back to the original directory
- If you want to run the project directly, you can just type 

```
python predict.py
```
Since the model has been pre-processed and saved, there is no need to re-fit the model

- Enter the name and press enter, the output will returns a list of tuple (predicted name, confidence) or None if the name is not recognized.


- If you have updated training data, please place the file in **. /data ** folder, and change the **PATH** in **model.py**
- Run model.py first, then run predict.py

```
python model.py
python predict.py
```


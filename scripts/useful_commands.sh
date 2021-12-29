# run all experiments!
conda activate /home/tomhe/soccer3.8
KPLUS_SCALER=2 nohup python3.8 ./run_all_soccer_paper_experiments.py &


# read the python output mid-run
tail -f nohup.out

# see which python processes are running
ps -ef | grep python

# delete all experiment logs+csvs
rm *.csv *.log

# edit config.py
nano soccer/config.py

# revert your changes to config.py
git checkout soccer/config.py
# git reset --hard HEAD



# download a file from the server
# run on your LOCAL MACHINE:
scp tomhe@sgesabatos.cs.bgu.ac.il:Desktop/distributed_k_means/README.md .


# kill all the running experiments on the machine
pkill -f python3.8


### install faiss
#1. install conda... ... ...
#2. then run:
conda create --name soccer python=3.8
conda activate soccer
pip install -r requirements.txt
conda install -c pytorch faiss-cpu


# follow this - https://docs.github.com/en/github/authenticating-to-github/keeping-your-account-and-data-secure/creating-a-personal-access-token
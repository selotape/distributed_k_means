

pushd "$(git rev-parse --show-cdup)"

### COVTYPE
pushd data_samples/covtype && wget https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz && gunzip covtype.data.gz && popd

### KDDCUP99
pushd data_samples/kddcup99 && wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz && gunzip kddcup.data.gz && popd

### SKIN
mkdir -p data_samples/skin && pushd data_samples/skin && wget https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt && popd

### BIGCROSS
mkdir -p data_samples/bigcross && pushd data_samples/bigcross && wget https://s3.amazonaws.com/h2o-training/clustering/BigCross.data.gz && gunzip BigCross.data.gz && popd

### POWER CONSUMPTION
mkdir -p data_samples/power && pushd data_samples/power && wget https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip && unzip household_power_consumption.zip && popd

### Activity Recognition
mkdir -p data_samples/activity && pushd data_samples/activity && wget https://archive.ics.uci.edu/ml/machine-learning-databases/00344/Activity%20recognition%20exp.zip && mv Activity\ recognition\ exp.zip Activity_recognition_exp.zip && unzip Activity_recognition_exp.zip && popd

### HIGGS
mkdir -p data_samples/higgs && pushd data_samples/higgs && wget https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz && gunzip HIGGS.csv.gz && popd

### CENSUS1990
mkdir -p data_samples/census1990 && pushd data_samples/census1990 && wget https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt && popd

popd

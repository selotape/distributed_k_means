

pushd "$(git rev-parse --show-cdup)"

### KDDCUP99
mkdir -p data_samples/kddcup99 && pushd data_samples/kddcup99 && wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz && gunzip kddcup.data.gz && popd

### BIGCROSS
mkdir -p data_samples/bigcross && pushd data_samples/bigcross && wget https://s3.amazonaws.com/h2o-training/clustering/BigCross.data.gz && gunzip BigCross.data.gz && popd

### HIGGS
mkdir -p data_samples/higgs && pushd data_samples/higgs && wget https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz && gunzip HIGGS.csv.gz && popd

### CENSUS1990
mkdir -p data_samples/census1990 && pushd data_samples/census1990 && wget https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt && popd

popd

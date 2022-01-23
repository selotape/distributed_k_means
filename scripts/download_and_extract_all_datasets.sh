

pushd "$(git rev-parse --show-cdup)"

### KDDCUP99
mkdir -p datasets/kddcup99 && pushd datasets/kddcup99 && wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz && gunzip kddcup.data.gz && popd

### BIGCROSS
mkdir -p datasets/bigcross && pushd datasets/bigcross && wget https://s3.amazonaws.com/h2o-training/clustering/BigCross.data.gz && gunzip BigCross.data.gz && popd

### HIGGS
mkdir -p datasets/higgs && pushd datasets/higgs && wget https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz && gunzip HIGGS.csv.gz && popd

### CENSUS1990
mkdir -p datasets/census1990 && pushd datasets/census1990 && wget https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt && popd

popd

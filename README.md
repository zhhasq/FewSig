# FewSig
## Data
The data with computed distance matrix can be downloaded on our website: https://sites.google.com/view/onlinefewshot/home<br />

## Reference
The code used for computing DTW distance is from https://www.cs.ucr.edu/~eamonn/UCRsuite.html <br />
The code used for computing Euclidean distance is based on https://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html <br />
The detail is available at  https://github.com/zhhasq/MASS3_FFTW


## Prerequisite 
1. python 3.9
2. packages in requirements.txt
3. Compile code used for computing DTW distance: <br />
  3.1 Go to FewSig/utils/dtw <br />
  3.2 on linux run: <br />
   > `c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) UCR_DTW_py.cpp -o dtw_ucr$(python3-config --extension-suffix)` <br />
   > `c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) dtw_basic_search_py2.cpp -o dtw_basic_search2$(python3-config --extension-suffix)`

## Run FewSig
The main.py contains instructions of how to config parameters and run FewSig.

## Normalization
We do normalization in two places:<br />
1. Each time series sub-sequence is z-normalized before computing the similarity search distance. 
2. The NCA transformed feature vectors are z-normalized prior to the softmax.
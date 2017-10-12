# CMF
This is a Python implementation of https://www.mathworks.com/matlabcentral/fileexchange/34126-fast-continuous-max-flow-algorithm-to-2D-3D-image-segmentation .

## Repository structure
* `matlab` contains the original code. I kept it for comparison purposes
* `data` contains the two original examples data
* `python` contains my actual Python implementation

## Status
Both 2D and 3D CMF have been implemented. 2D produces the exact same results, 3D has not been thoroughly tested

## Requirements
* Python 3+
* `NumPy` and `SciPy`
* `Matplotlib`
* `Nibabel` to load the 3D example

## Use
You can run `test.py` in the python folder for a quick test. It will run by default the 2D example, but you can select between 2D and 3D:
```bash
./test.py 2 // for 2D
./test.py 3 // for 3D
```

The actual functions are in `cmf.py`. They can be imported easily (as long it is in your PYTHONPATH):
```Python
from cmf import CMF_2D, CMF_3D
```

`plot.py` contains two plot functions for the examples. The 3D plot does not currently show the 3D segmentation.
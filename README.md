# HAMP Retrieval for IWV and first T, Q profiles applicable to HALO-(AC)³
This is the package that performs the regression-retrieval analysis to retrieve IWV from HAMP during HALO-(AC)3. 
The read-me file is currently under process and needs to be updated.

The generation of the training dataset using ERA5 fields and therefrom calculating brightness temperatures using the forward simulator PAMTRA are excecuted with:
```python
    run_ERA5PAMTRA.ipynb
```
To perform the IWV retrieval use:
```python
    Column_HAMP_PAMTRA_retrieval.ipynb
```


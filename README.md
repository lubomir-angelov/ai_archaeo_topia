# ai_archaeo_topia
A repository to track the work on the AI ArchaeoTopia project.  https://archaeotopia.naim.bg/

# GDAL
If you see errors such as:
no module named _gdal_array

You need to re-install GDAL in your python virtual env:

```bash
pip uninstall gdal

# ensure numpy is installed prior to installing gdal
pip install numpy

# ensure setuptools and wheel are installed to do the build in your current environment
pip install setuptools wheel

# install gdal (note the version might be different on your machine!)
pip install --no-build-isolation --no-cache-dir --force-reinstall gdal==3.4.1
```

# ai_archaeo_topia
A repository to track the work on the AI ArchaeoTopia project.  https://archaeotopia.naim.bg/

# GDAL
If you see errors such as:
no module named _gdal_array

You need to re-install GDAL in your python virtual env:
(From: https://gis.stackexchange.com/questions/153199/import-error-no-module-named-gdal-array)

```bash
pip uninstall gdal

# ensure numpy is installed prior to installing gdal
pip install numpy

# ensure setuptools and wheel are installed to do the build in your current environment
pip install setuptools wheel

# install gdal (note the version might be different on your machine!)
pip install --no-build-isolation --no-cache-dir --force-reinstall gdal==3.4.1
```

# CVAT and SAM2 MCP integration
Next steps to run the integration
## 1. Pre-flight smoke test (no CVAT needed)
```bash
make cvat-mcp-smoke
```
# 2. Set credentials in .env or export them
```bash
export CVAT_SAM2_CVAT_USERNAME=your_cvat_user
export CVAT_SAM2_CVAT_PASSWORD=your_cvat_pass
```
# 3. Health check (requires CVAT running)
```bash
make cvat-mcp-health
```
# 4. Dry-run annotation pipeline (preview only)
```bash
make annotation-dry-run INPUT_DIR=/path/to/map_images
```
# 5. Full annotation run
```bash
make annotation-run INPUT_DIR=/path/to/map_images
```
# 6. Start MCP server for opencode (stdio mode)
```bash
make cvat-mcp-run
```
The opencode.json already has the cvat-sam2 MCP config. Set:
```bash 
CVAT_SAM2_CVAT_USERNAME
CVAT_SAM2_CVAT_PASSWORD
```
before opencode can authenticate to CVAT.

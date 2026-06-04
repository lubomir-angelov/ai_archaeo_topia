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

# SAM2 modes
```bash
Mode A: local/internal mock
- fastest tests
- no HTTP services
- useful for unit tests only

Mode B: architecture mock
- real sam2_mcp process
- real sam2_backend process
- backend in SAM2_BACKEND_MODE=mock
- required for milestone acceptance
```

## Architecture acceptance path
```bash
# Terminal 1
SAM2_BACKEND_MODE=mock make sam2-backend-run

# Terminal 2
export SAM2_MCP_BACKEND_URL="http://127.0.0.1:8181"
make sam2-mcp-run

# Terminal 3
export PDF_PATH="path/to/input.pdf"
export OUTPUT_ROOT="data/annotations/runs"
export RUN_ID="manual_pdf_seed_001"
export SAM2_MCP_URL="http://127.0.0.1:8181"
make annotation-pdf-sam2-mock-dry-run
```
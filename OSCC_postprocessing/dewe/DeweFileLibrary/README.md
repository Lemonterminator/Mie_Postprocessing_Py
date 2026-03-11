# DeweFileLibrary

This directory is reserved for native Dewesoft DWDataReader binaries.

The Python package should not ship these `.dll` / `.so` files directly when
published to PyPI. Instead, download the official archive and extract the
required binaries here locally.

Helper script:

```powershell
python -m OSCC_postprocessing.dewe.download_dwdatareader
```

Default download source:

- `https://downloads.dewesoft.com/developers/dwdatareader/DWDataReader_v5_0_4.zip`

Expected Windows files:

- `DWDataReaderLib64.dll`
- `DWDataReaderLib.dll`

Optional Linux files from the same archive can also be placed here if needed.

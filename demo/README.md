# Raglite demo kit

![Demo animation placeholder](demo.gif)

Run the offline two-minute demo end-to-end:

```bash
./demo/run_demo.sh
```

Or on Windows PowerShell:

```powershell
./demo/run_demo.ps1
```

The scripts create a virtual environment, install Raglite with the server extra, build a
database from `demo/mini_corpus`, run a sample query, and start the API server on
`http://127.0.0.1:8080` with helpful `curl` examples.

You can also reproduce the self-test in isolation:

```bash
raglite self-test
```

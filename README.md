# ImageNet
Single layer handwriting digit recognition neural network

## Build/Run Specifications

### Virtual Environment
This project can be run in a virtual environment. Ensure all requirements are installed from the `requirements.txt` file.
Then use `source env/bin/activate` to activate the Python virtual environment.

### Ghostscript Setup
Ghostscript, an interpreter for PostScript and PDF, is used to convert Python Tkinter Canvas drawings
to PIL images. Run `brew install ghostscript` to ensure the latest version is downloaded

### Run Configurations
When the virtual environment is activated, run
```
python3 main.py <mode> [<image directory> <output file>]
```
to run the image recognition. A mode of 0 will start the GUI to draw digits, while
a mode of 1 will search for images in the provided directory and print labels to the given filepath.

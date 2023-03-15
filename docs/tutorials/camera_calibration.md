# Camera Calibration

Camera calibration is an important first step in using most of `HAMS` modules. To do so, first print the [calibration board present in this file](https://github.com/abakisita/camera_calibration/blob/master/aruco_marker_board.pdf).

Click a couple of images of the printed board and place them in a folder named `calibration_images`. Additionally, **measure** the `length` of the markers and the `separation` between two adjacent markers `in cm`.

To run the camera calibration module from the command line,

```bash
mapping-cli.exe generate-calib <PHONE_MODEL_NAME>  <MARKER_LENGTH> <MARKER_SEPARATION> <OUTPUT_FOLDER>
```

```{button-link} ./camera_calibration_notebook.ipynb
:color: primary
:shadow:

Running the camera calibration module on the collected images.
```
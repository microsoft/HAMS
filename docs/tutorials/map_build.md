# Map Building

To setup the track, we first need to build maps of the individual maneuvers.

First, setup the maneuver with the required aruco markers spread in such a manner that when the test taker performs the maneuver, the camera placed inside the test vehicle
is able to see at least 2 markers at all times.

Now, collect images of the maneuver that is being tested - for best results, we recommend taking photos at approximately the level of the markers under sufficient sunlight i.e.
not too bright or not too dark. Take photos from different angles(within the maneuver), although the photos should not capture any other markers that are not part of this map.

## File Requirements

1. Path to the binaries / executable named `mapper_from_images`(see the `Installation` section from the sidebar to get download instructions)
2. Compile all the images of a map into a single folder - let's call this folder `images`
3. Generate the camera calibration file. To generate one, you can check the `Camera Calibration` module from the `Tutorials` section. Let's call this file `calib.yml`
4. Depending on the aruco markers setup on the map, add the dictionary. Typically, we use `TAG16h5`
5. Marker size: Size of the printed aruco markers 


In order to directly run this module, execute the following command in your terminal

```bash
mapping-cli.exe map <EXEC_PATH> <IMAGES> <CALIBRATION_FILE> <ARUCO_DICTIONARY> <MARKER_SIZE> <OUTPUT_DIRECTORY>
```

Follow the instructions below to generate and visualize the map

```{button-link} ./map_building_notebook.ipynb
:color: primary
:shadow:

Running the map building module on track images.
```
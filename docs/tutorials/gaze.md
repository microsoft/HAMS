# Gaze Detection

Here, you'll first need to setup [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace). In order to setup `OpenFace`, follow the instructions on [their wiki](https://github.com/TadasBaltrusaitis/OpenFace/wiki) depending on your Operating system.


<details open>
  <summary><b>seatbelt.yml</b>(click to open/close)</summary>

  ```yaml
  face_landmark_exe_path: "<PATH TO FaceLandmarkVidMulti executable/binary>"
  centre_threshold: 10
  left_threshold: 10
  right_threshold: 10
  maneuver: full
  ```
</details>

In the above config file, copy the path to the `FaceLandmarkVidMulti` landmark executable/binary from the installation. 

````{margin} **Note**
```{note}
If `FaceLandmarkVidMulti` already exists in the PATH variable, you don't need to enter the entire path. Just `FaceLandmarkVidMulti`(linux) or `FaceLandmarkVidMulti.exe`(Windows) is sufficient.
```
````

<details>
  <summary><b>Explanation of the above configuration values</b>(click to open)</summary>

  ```{list-table}
  :header-rows: 1

  * - Parameter
    - Description
    - Example Value
  * - face_landmark_exe_path
    - Path to the `FaceLandmarkVidMulti` executable
    - "D:\\OpenFace_2.2.0_win_x64\\FaceLandmarkVidMulti.exe"
  * - centre_threshold
    - Number of center looking detections(integer)
    - 10
  * - left_threshold
    - Number of minimum left gaze detections(int)
    - 10
  * - right_threshold
    - Number of minimum right gaze detections(int)
    - 10
  * - maneuver
    - Name of the maneuver for which the gaze detection is being run on
    - Traffic
  ```
</details>

```{button-link} ./gaze_notebook.ipynb
:color: primary
:shadow:

Running the gaze detection code on the driver-facing video.
```
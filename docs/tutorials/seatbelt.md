# Seatbelt Detection

## Setup

Given the driver-facing video, the seatbelt detection module uses an object detection pipeline to detect the seatbelt on the driver.

In order to run the seatbelt module on the front facing video(let's call it `front_video.mp4`), first add a file named `seatbelt.yml` to your current folder.

<details open>
  <summary><b>seatbelt.yml</b>(click to open/close)</summary>

  ```yaml
  device: "cpu"
  skip_frames: 25
  classifier_confidence_threshold: 0.75
  detection_percentage: 0.75
  model_path: ["models", "seatbelt_model.pth"]
  ```
</details>

<details>
  <summary><b>Explanation of the above configuration values</b>(click to open)</summary>

  ```{list-table}
  :header-rows: 1

  * - Parameter
    - Description
    - Example Value
  * - device
    - Hardware for pytorch to run the model inference on
    - "cpu" or "cuda:0"
  * - skip_frames
    - Number of frames to skip(integer)
    - 25
  * - classifier_confidence_threshold
    - Threshold to classify seatbelt detection(float)
    - 0.75
  * - detection_percentage
    - Percentage number of detections to consider the test as pass(float)
    - 0.75
  * - model_path
    - path to the saved model. Format: ['directory', 'file_name']
    - ["models", "seatbelt_model.pth"]
  ```
</details>

Now, run the following command:

```bash
python main.py --seat-belt front_video.mp4 --config seatbelt.yml --output-path results/
```

The following notebook has a code-walkthrough to run the Seatbelt module and visualize the results:

```{button-link} ./seatbelt_notebook.ipynb
:color: primary
:shadow:

Running the seatbelt code on driver-facing video.
```
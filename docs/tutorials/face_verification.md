# Face Verification

Face verification module is used to check if the test taker that is registered is the one driving the test vehicle. 

![Azure Portal Face API](../static/face.png)

To run this, you'll need Microsoft's [Cognitive Face Python Library](https://pypi.org/project/cognitive-face/) and an API key which you can setup from [here](https://azure.microsoft.com/en-us/products/cognitive-services/face). On your Azure Portal, head on to the Face API resource and enter the `endpoint` in the `base_url` field below and copy the `KEY 1` to the `subscription_key` below(see the screenshot above for reference).

<details open>
  <summary><b>face_verify.yml</b>(click to open/close)</summary>

  ```yaml
  base_url: "https://southcentralus.api.cognitive.microsoft.com/face/v1.0"
  subscription_key: "<ENTER_YOUR_API_KEY>"
  calib_frame_period: 100
  test_frame_period: 100
  recog_confidence_threshold: 0.75
  video_acceptance_threshold: 0.75
  ```
</details>

<details>
  <summary><b>Explanation of the above configuration values</b>(click to open)</summary>

  ```{list-table}
  :header-rows: 1

  * - Parameter
    - Description
    - Example Value
  * - base_url
    - Base URL for the Cognitive Face API to access
    - "https://southcentralus.api.cognitive.microsoft.com/face/v1.0"
  * - subscription_key
    - API Key from Azure Face API
    - KEY
  * - calib_frame_period
    - Number of seconds of the calibration to read(int)
    - 100
  * - test_frame_period
    - Number of frames to skip in between successive evaluations(int)
    - 100
  * - recog_confidence_threshold
    - Threshold for similarity to consider a match(float)
    - 0.75
  * - video_acceptance_threshold
    - Successful similarity rate to consider face verification a success(float)
    - 0.72
  ```
</details>

Now, run the following command:

```bash
python main.py --face-verify --front-video front_video.mp4 --calib-video calib_video.mp4 --config face_verify.yml --output-path results/
```

```{button-link} ./face_verification_notebook.ipynb
:color: primary
:shadow:

Running the seatbelt code on driver-facing video.
```
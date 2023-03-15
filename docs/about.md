# About ALT

Inadequate driver skills and apathy towards/lack of awareness of safe driving practices are key contributing factors for the lack of road safety. The problem is exacerbated by the fact that the license issuing system is broken in India, with an estimated 59% of licenses issued without a test, making it a significant societal concern. The challenges arise from capacity and cost constraints, and corruption that plagues the driver testing process. While there have been efforts aimed at creating instrumented tracks to automate the license test, these have been stymied by the high cost of the infrastructure (e.g., pole-mounted high-resolution cameras looking down on the tracks) and poor test coverage (e.g., inability to monitor the driver inside the vehicle).

HAMS-based testing offers a compelling alternative. It is a low-cost and affordable system based on a windshield-mounted smartphone, though for reasons of scalability (i.e., handling a large volume of tests), we can offload computation to an onsite server or to the cloud. The view inside the vehicle also helps expand the test coverage. For instance, the test can verify that the driver taking the test is the same as the one who had registered for it (essential for protecting against impersonation), verify that the driver is wearing their seat belt (an essential safety precaution), and check whether the driver scans their mirrors before effecting a maneuver such as a lane change (an example of multimodal sensing, with inertial sensing and camera-based monitoring being employed in tandem).

HAMS-based testing allows the entire testing process to be performed without any human intervention. A test report, together with video evidence (to substantiate the test result in case of a dispute), is produced in an automated manner within minutes of the completion of the test. This manner of testing, with the test taken by the driver alone in the vehicle (i.e., no test inspector) has proved to be a boon in the context of the physical distancing norms arising from the COVID-19 pandemic.

## Deployments

To roll out HAMS-based driver testing, we first partnered with the Government of Uttarakhand and the Institute of Driving and Traffic Research (IDTR), run by Maruti-Suzuki. Testing is conducted on a track and includes a range of parameters including verification of driver identity, checking of the seat belt, fine-grained trajectory tracking during maneuvers such as negotiating a roundabout and performing parallel parking, and checking on mirror scanning during lane changing.

**HAMS-based driver license testing @ Dehradun, Uttarakhand:** [HAMS-based license testing went live at Dehradun Regional Transport Office (RTO)](https://news.microsoft.com/en-in/features/microsoft-ai-automates-drivers-license-test-india/), the capital of Uttarakhand state in July 2019. Till date, 10000+ automated tests (as of 15 Feb 2021) have been conducted, with an accuracy of 98%. The objectivity and transparency of the automated testing process has won the praise of not just the RTO staff but also the majority of the candidates, including many that failed the test.  The thoroughness of HAMS-based testing is underscored by the fact that now the passing rate is only 54% compared to over 90% with the prior manual testing.

**Scaling HAMS deployments across India:**  The success in Dehradun has spurred interest in HAMS-based automated testing across India and also overseas. RFPs issued by several states have called for capabilities such as continuous driver identification, gaze tracking, and mirror scan monitoring, that were not available before HAMS. HAMS-based testing has been rolled out in IDTR Aurangabad, Bihar, and is in process of being implemented at multiple RTOs across the country.

````{eval-rst}
..  youtube:: zFuIP5hI4yU
````    

````{eval-rst}
..  youtube:: XtgpWXM5Hfg
````    
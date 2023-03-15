# Welcome to HAMS's documentation!

![LICENSE](https://img.shields.io/github/license/microsoft/HAMS?style=for-the-badge) [![Dashboard](https://img.shields.io/website?down_message=Dashboard%20Offline&style=for-the-badge&up_color=green&up_message=Dashboard&url=https%3A%2F%2Fhams-dashboard.westus3.cloudapp.azure.com%2F)](https://hams-dashboard.westus3.cloudapp.azure.com) [![Documentation](https://img.shields.io/badge/docs-Documentation-blue?style=for-the-badge&logo=appveyor)](https://microsoft.github.io/HAMS)   

Inadequate driver skills and apathy towards/lack of awareness of safe driving practices are key contributing factors for the lack of road safety. The problem is exacerbated by the fact that the license issuing system is broken in India, with an estimated 59% of licenses issued without a test, making it a significant societal concern. The challenges arise from capacity and cost constraints, and corruption that plagues the driver testing process. While there have been efforts aimed at creating instrumented tracks to automate the license test, these have been stymied by the high cost of the infrastructure (e.g., pole-mounted high-resolution cameras looking down on the tracks) and poor test coverage (e.g., inability to monitor the driver inside the vehicle).

HAMS-based testing offers a compelling alternative. It is a low-cost and affordable system based on a windshield-mounted smartphone, though for reasons of scalability (i.e., handling a large volume of tests), we can offload computation to an onsite server or to the cloud. The view inside the vehicle also helps expand the test coverage. For instance, the test can verify that the driver taking the test is the same as the one who had registered for it (essential for protecting against impersonation), verify that the driver is wearing their seat belt (an essential safety precaution), and check whether the driver scans their mirrors before effecting a maneuver such as a lane change (an example of multimodal sensing, with inertial sensing and camera-based monitoring being employed in tandem).

To cite this repository, please use the following:

```bibtex
@inproceedings{nambi2019alt,
      title={ALT: towards automating driver license testing using smartphones},
      author={Nambi, Akshay Uttama and Mehta, Ishit and Ghosh, Anurag and Lingam, Vijay and Padmanabhan, Venkata N},
      booktitle={Proceedings of the 17th Conference on Embedded Networked Sensor Systems},
      pages={29--42},
      year={2019}
   }
```

To use these code, follow the tutorials [here](tutorials/index.md). To know more about our project, you can explore the documentation using the sidebar or from down below:

````{eval-rst}
.. toctree::
   :maxdepth: 1
   :caption: More about HAMS

   about
   dashboard

.. toctree::
   :maxdepth: 1
   :caption: USAGE

   tutorials/install
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: REFERENCE

   modules
````    

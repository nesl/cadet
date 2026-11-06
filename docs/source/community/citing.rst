Citing CADET
============

If you use CADET in your research, please cite the paper:

.. code-block:: bibtex

   @inproceedings{sharma2026cadet,
     title     = {{CADET}: A Modular Platform for Evaluating Distributed
                  Cooperative Autonomy in Connected Autonomous Vehicles},
     author    = {Sharma, Pragya and Wang, Brian and Srivastava, Mani},
     booktitle = {2026 IEEE International Conference on Robotics and Automation (ICRA)},
     year      = {2026},
     publisher = {IEEE},
     doi       = {10.1109/ICRA57385.2026.11696961},
   }

The paper is available at `arXiv <https://arxiv.org/abs/2606.04072>`_. This
documentation is published at `nesl.github.io/cadet
<https://nesl.github.io/cadet/>`_, and the project page and demo videos are at
`nesl.github.io/cadet-web <https://nesl.github.io/cadet-web>`_.

----

Related publications
--------------------

Earlier studies using this platform:

.. code-block:: bibtex

   @inproceedings{sharma2025cloud,
     title     = {Cloud Is Closer Than It Appears: Revisiting the Tradeoffs of
                  Distributed Real-Time Inference},
     author    = {Sharma, Pragya and Qiu, Hang and Srivastava, Mani},
     booktitle = {2025 34th International Conference on Computer Communications
                  and Networks (ICCCN)},
     year      = {2025},
     doi       = {10.1109/ICCCN65249.2025.11133852},
   }

`IEEE Xplore <https://doi.org/10.1109/ICCCN65249.2025.11133852>`_

.. code-block:: bibtex

   @inproceedings{sharma2025poster,
     title     = {Poster: Towards a Performance-Driven Device-Edge-Cloud
                  Relationship},
     author    = {Sharma, Pragya and Wang, Brian and Ouyang, Xiaomin and
                  Nanayakkara, Rahal and Balaji, Bharathan and Tabuada, Paulo
                  and Srivastava, Mani B.},
     booktitle = {Proceedings of the 26th International Workshop on Mobile
                  Computing Systems and Applications (HotMobile)},
     year      = {2025},
     pages     = {125},
     publisher = {Association for Computing Machinery},
     doi       = {10.1145/3708468.3715681},
   }

`ACM Digital Library <https://doi.org/10.1145/3708468.3715681>`_

----

Acknowledgements
----------------

CADET builds on several open-source projects:

- **Pylot**: CADET's pipeline code is derived from Pylot (Apache-2.0);
  see ``NOTICE``.
- **CARLA**: the open-source vehicle simulator (version 0.9.10).
- **ScenarioRunner**: the CARLA scenario scripting framework.
- **Mininet-WiFi**: the emulated wireless network platform underlying the
  ``netwaggle`` backend.
- **NetWaggle** — the wireless network emulator adapted for CADET's network
  emulation layer.

See ``NOTICE`` in the repository root for per-component licence details.

# Trajectory Planning Optimization
Basic code modules for performance-criteria-based trajectory planning for future implementation in the robot Yu. The repository consists of:

* Robot kinematics and dynamics (originally written in C)
* An auto-editor that converts C codes into python script
* Codes for robot energy calculation and optimization using Kalman algorithm
* Codes for force & momentum-limited optimization
* Optimization results (simulation and robot logs)

# Before running the codes
Robotics Toolbox (RTB) for Python needs to be installed to run the codes with (Python version > 3.6)

```shell script
pip3 install roboticstoolbox-python
```

Some changes were made in the trajectory.py file of RTB for better generation of trapezoidal velocity profiles. Therefore, copy `trajectory.py` from the root directory of this repo to `~\AppData\Local\Programs\Python\Python311\Lib\site-packages\roboticstoolbox\tools`, replace the `trajectory.py` there.

The DH Parameters of Yu+ needs to be included in RTB. To do so, follow the instructions below:

* Find Voraus.py in the root of this repo, copy it to `~\AppData\Local\Programs\Python\Python311\Lib\site-packages\roboticstoolbox\models\DH`
* Edit the file `__init__.py` in the target folder (DH), Add a line:

```python
from roboticstoolbox.models.DH.Stanford import Voraus
```

* Add `'Yu'` to the list that defines `__all__`.

Now the codes should be ready to run.
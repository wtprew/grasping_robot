#!/usr/bin/env python
from hardware.calibrate import Calibration

if __name__ == '__main__':
    calibration = Calibration(cam_id=831612070538, execute=False, calibrate=True)
    calibration.run()

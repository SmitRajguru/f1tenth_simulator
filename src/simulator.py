#!/usr/bin/python3

# import libraries
import numpy as np
import rospy
import rospkg
from car import Car
from map import Map
import math
import numpy as np
from std_msgs.msg import Float64
from std_msgs.msg import Bool

rospack = rospkg.RosPack()
path = rospack.get_path("f1tenth_simulator")

# create a node
rospy.init_node("simulator", anonymous=True)


# create a simulator class
class Simulator:
    def __init__(self) -> None:
        self.simTimePub = rospy.Publisher("/sim_time", Float64, queue_size=1)

        self.resetSub = rospy.Subscriber(
            "/reset_sim", Bool, self.resetSim, queue_size=1
        )

        # initialize the simulator
        self.initSim()

    def initSim(self):
        self.simTime = 0.0

        self.useMap = bool(rospy.get_param("~useMap"))
        self.useLidar = bool(rospy.get_param("~useLidar"))

        # create a list of cars
        self.cars = []

        # set the update rate
        self.simulatorRate = rospy.get_param("~simulatorRate")
        self.simulator_dt = 1.0 / self.simulatorRate
        self.rate = rospy.Rate(self.simulatorRate)

        # load the map
        map = rospy.get_param("~map/path")
        origin = rospy.get_param("~map/origin")
        resolution = rospy.get_param("~map/resolution")
        occupied_thresh = rospy.get_param("~map/occupied_thresh")
        wallBuffer = rospy.get_param("~map/wallBuffer")
        obstacles_timeout = rospy.get_param("~map/obstacles/timeout")
        obstacles_list = rospy.get_param("~map/obstacles/list")
        checkpoints_threshold = rospy.get_param("~map/checkpoints/threshold")
        checkpoints_list = rospy.get_param("~map/checkpoints/list")
        self.map = Map(
            f"{path}/{map}",
            origin,
            resolution,
            occupied_thresh,
            wallBuffer,
            (obstacles_timeout, obstacles_list),
            (checkpoints_threshold, checkpoints_list),
        )
        self.reset = False

    def resetSim(self, msg):
        self.reset = msg.data

    def AddCar(self, car_name):
        # create a car
        self.cars.append(Car(car_name, (self.map, self.useMap), self.useLidar))

    def UpdateCars(self):
        # update all cars
        for car in self.cars:
            car.update(self.simulator_dt, self.simTime)

    def killCars(self):
        for car in self.cars:
            car.kill()

    def run(self):
        while not rospy.is_shutdown():
            if self.reset:
                self.killCars()
                raise Exception("Resetting simulator")

            # update all cars
            self.UpdateCars()

            # update the map
            self.map.update(self.simulator_dt)

            # update the simulation time
            self.simTime += self.simulator_dt
            self.simTimePub.publish(self.simTime)

            # sleep for the remainder of the loop
            self.rate.sleep()


if __name__ == "__main__":
    sim = Simulator()
    while not rospy.is_shutdown():
        try:
            for car in rospy.get_param("~cars"):
                sim.AddCar(car)
            sim.run()
        except Exception as e:
            print(e)
        finally:
            sim.initSim()

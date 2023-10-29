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
        self.simDtimePub = rospy.Publisher("/sim_dtime", Float64, queue_size=1)
        self.rosDtimePub = rospy.Publisher("/ros_dtime", Float64, queue_size=1)

        self.resetSub = rospy.Subscriber(
            "/reset_sim", Bool, self.resetSim, queue_size=1
        )

        # create a list of cars
        self.cars = {}

        # initialize the simulator
        self.initSim()

    def initSim(self):
        self.simTime = 0.0
        self.rostime = rospy.get_time()

        self.useMap = bool(rospy.get_param("~useMap"))
        self.useLidar = bool(rospy.get_param("~useLidar"))

        # set the update rate
        self.simulatorRate = rospy.get_param("~simulatorRate")
        self.simulator_dt = 1.0 / self.simulatorRate
        self.rate = rospy.Rate(self.simulatorRate)

        self.reset = False

    def initMap(self):
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

    def destroyMap(self):
        self.map.remove()

    def resetSim(self, msg):
        self.reset = msg.data

    def UpdateCars(self):
        # update all cars
        for car in self.cars.keys():
            self.cars[car].update(self.simulator_dt, self.simTime)

    def initCars(self):
        car_list = rospy.get_param("~cars")
        remove_cars = []
        add_cars = []
        update_cars = []

        for car in self.cars.keys():
            if car in car_list:
                update_cars.append(car)
            else:
                remove_cars.append(car)

        for car in car_list:
            if car not in update_cars:
                add_cars.append(car)

        for car in remove_cars:
            # remove a car
            self.cars[car].remove()
            del self.cars[car]

        for car in add_cars:
            # create a car
            self.cars[car] = Car(car, (self.map, self.useMap), self.useLidar)
        for car in update_cars:
            self.cars[car].updateParams((self.map, self.useMap), self.useLidar)

    def destroyCars(self):
        for car in self.cars.keys():
            self.cars[car].remove()

    def run(self):
        while not rospy.is_shutdown():
            if self.reset:
                self.destroyMap()
                self.initSim()
                self.initMap()
                self.initCars()
                print("Resetting simulator")

            # publish the delta time
            self.simDtimePub.publish(self.simulator_dt)
            self.rosDtimePub.publish(rospy.get_time() - self.rostime)
            self.rostime = rospy.get_time()

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
    sim.initMap()
    sim.initCars()
    sim.run()
    sim.destroyCars()
    sim.map.remove()

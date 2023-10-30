#!/usr/bin/python3

# import libraries
import numpy as np
from PIL import Image, ImageOps
from nav_msgs.msg import OccupancyGrid
import rospy
import range_libc


class Map:
    def __init__(
        self,
        mapParams,
        obstaclesParams,
        checkpointParams,
    ) -> None:
        self.mapPub = rospy.Publisher("/map", OccupancyGrid, queue_size=1, latch=True)

        (map, origin, resolution, occupied_thresh, wallBuffer, useMap) = mapParams

        # load the map
        self.map = np.asarray(ImageOps.grayscale(Image.open(map))).copy()

        # invert the map so that 0 is free and 255 is occupied
        self.map = 255 - self.map

        # scale the map to be between 0 and 100
        self.map = (self.map / 255) * 100

        # round the map to the nearest integer
        self.map = np.round(self.map).astype(np.uint8)

        # flip the map so that 0,0 is in the bottom left
        # self.map = np.flip(self.map, axis=0)

        # rotate map -90 degrees
        # self.map = np.rot90(self.map, k=1, axes=(0, 1))

        # rotate map 90 degrees
        self.map = np.rot90(self.map, k=1, axes=(1, 0))

        self.useMap = useMap
        self.origin = origin
        self.resolution = resolution

        self.occupied_thresh = int(occupied_thresh * 100)

        self.wallBuffer = wallBuffer

        self.width = self.map.shape[0]
        self.height = self.map.shape[1]

        self.checkpoint_thresh = checkpointParams[0]
        self.checkpoints = checkpointParams[1]

        self.oMap = None
        self.range_methods = {}
        self.grid = OccupancyGrid()
        self.initGrid()

        self.obstacles = []
        self.obstacleTimeout = obstaclesParams[0]
        for obstacle in obstaclesParams[1]:
            self.obstacles.append(
                Obstacle(
                    obstacle[0],
                    obstacle[1],
                    obstacle[2],
                    obstacle[3],
                    self.resolution,
                )
            )

        self.isMapUpdated = True
        self.updateObstacles()

    def initGrid(self):
        self.grid.header.frame_id = "map"
        self.grid.info.resolution = self.resolution
        self.grid.info.width = self.width
        self.grid.info.height = self.height
        self.grid.info.origin.position.x = self.origin[0]
        self.grid.info.origin.position.y = self.origin[1]
        self.grid.info.origin.position.z = 0
        self.grid.info.origin.orientation.x = 0
        self.grid.info.origin.orientation.y = 0
        self.grid.info.origin.orientation.z = 0
        self.grid.info.origin.orientation.w = 1

    def remove(self):
        self.mapPub.unregister()

    def initRangeMethods(self):
        for key, value in self.range_methods.items():
            value["range_method"] = self.getRangeMethod(
                value["max_range"], value["num_thetas"]
            )

    def addCar(self, car, max_range, num_thetas):
        self.range_methods[car] = {
            "range_method": self.getRangeMethod(max_range, num_thetas),
            "max_range": max_range,
            "num_thetas": num_thetas,
        }

    def removeCar(self, car):
        del self.range_methods[car]

    def getRangeMethod(self, max_range, num_thetas):
        return range_libc.PyCDDTCast(
            self.oMap,
            max_range / self.resolution,  # max range in pixels
            num_thetas,  # theta discretization
        )
        # return range_libc.PyBresenhamsLine(
        #     self.oMap,
        #     max_range / self.resolution,  # max range in pixels
        # )

    def update(self, dt):
        # update the obstacles
        for obstacle in self.obstacles:
            if obstacle.update(dt):
                self.isMapUpdated = True

        # update the obstacles
        self.updateObstacles()

        if self.isMapUpdated:
            self.oMap = range_libc.PyOMap(self.grid)
            self.initRangeMethods()
            if self.useMap:
                self.mapPub.publish(self.grid)
            self.isMapUpdated = False

    def updateObstacles(self):
        # reset the map
        self.gridData = self.map.copy()

        # add the obstacles to the map
        for obstacle in self.obstacles:
            if obstacle.isActive:
                # get the points in the obstacle
                points = np.round(obstacle.border / self.resolution).astype(np.int32)
                # add the origin to the points
                points -= np.array(
                    [self.origin[0] / self.resolution, self.origin[1] / self.resolution]
                ).astype(np.int32)

                # set the points and little around in the map to be occupied
                for point in points:
                    self.gridData[
                        int(point[0] - self.wallBuffer / 2 * self.resolution) : int(
                            point[0] + self.wallBuffer / 2 * self.resolution
                        ),
                        int(point[1] - self.wallBuffer / 2 * self.resolution) : int(
                            point[1] + self.wallBuffer / 2 * self.resolution
                        ),
                    ] = 50

        self.grid.data = self.gridData.T.flatten().tolist()

        if self.oMap is None:
            self.oMap = range_libc.PyOMap(self.grid)

    def getRange(self, car, x, y, theta):
        range = self.range_methods[car]["range_method"].calc_range(x, y, theta)
        return range

    def getRanges(self, car, x, y, thetas):
        queries = np.zeros((len(thetas), 3), dtype=np.float32)
        ranges = np.zeros(len(thetas), dtype=np.float32)

        queries[:, 0] = x
        queries[:, 1] = y
        queries[:, 2] = thetas

        self.range_methods[car]["range_method"].calc_range_many(queries, ranges)

        return ranges


class Obstacle:
    def __init__(self, pt1, pt2, pt3, pt4, resolution) -> None:
        self.resolution = resolution

        self.isActive = True
        self.timer = 0
        self.updated = False

        # generate rectangle points
        self.pts = np.array([pt1, pt2, pt3, pt4])

        self.border = []
        self.generateBorderPoints()

    def update(self, dt):
        if not self.isActive:
            self.timer -= dt
            if self.timer <= 0:
                self.isActive = True
                self.updated = True

        if self.updated:
            self.updated = False
            return True

        return False

    def isPointInObstacle(self, pt, thresh, timer):
        # check if the point is in the border
        dist = self.border - pt
        dist = np.sqrt(dist[:, 0] ** 2 + dist[:, 1] ** 2)

        if np.any(dist <= thresh + 2 * self.resolution):
            self.updated = True
            self.isActive = False
            self.timer = timer
            return True
        return False

    def generateBorderPoints(self):
        # get points on the rectangle that are resolution apart
        for i in range(4):
            # get the start and end points
            start = self.pts[i]
            end = self.pts[(i + 1) % 4]

            # get the distance between the points
            dist = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

            # get the number of points to check
            num_points = int(dist / self.resolution)

            # get the x,y of the points to check
            x_list = np.linspace(start[0], end[0], num_points)
            y_list = np.linspace(start[1], end[1], num_points)

            self.border.extend(np.array([x_list, y_list]).T.tolist())

        self.border = np.array(self.border)

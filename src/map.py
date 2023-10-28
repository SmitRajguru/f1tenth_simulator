#!/usr/bin/python3

# import libraries
import numpy as np
from PIL import Image, ImageOps


class Map:
    def __init__(
        self, map, origin, resolution, occupied_thresh, wallBuffer, checkpointParams
    ) -> None:
        # load the map
        self.map = np.asarray(ImageOps.grayscale(Image.open(map))).copy()

        # invert the map so that 0 is free and 255 is occupied
        self.map = 255 - self.map

        # flip the map so that 0,0 is in the bottom left
        self.map = np.flip(self.map, axis=0)

        self.origin = origin
        self.resolution = resolution

        self.occupied_thresh = int(occupied_thresh * 255)

        self.wallBuffer = wallBuffer

        self.width = self.map.shape[0]
        self.height = self.map.shape[1]

        self.checkpoint_thresh = checkpointParams[0]
        self.checkpoints = checkpointParams[1]

        # print(f"Map 0,0: {self.getPixelFromXY(0, 0)}")

    def getPixelFromXY(self, x, y):
        # convert from meters to pixels
        x = int((x - self.origin[0]) / self.resolution)
        y = int((y - self.origin[1]) / self.resolution)

        # check if the pixel is within the map
        if x < 0 or x >= self.height or y < 0 or y >= self.width:
            return 255

        # return the pixel value
        return self.map[y, x]

    def getEdge(self, prev_x, prev_y, x, y):
        # get the distance between the two points
        dist = np.sqrt((prev_x - x) ** 2 + (prev_y - y) ** 2)

        # get the number of points to check
        num_points = int(dist / self.resolution)

        # get the x,y of the points to check
        x_list = np.linspace(prev_x, x, num_points)
        y_list = np.linspace(prev_y, y, num_points)

        # check if any of the points are in collision
        for x, y in zip(x_list, y_list):
            if self.getPixelFromXY(x, y) >= self.occupied_thresh:
                return x, y

        return None

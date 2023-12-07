# import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# load map
map_image = cv2.imread("maps/RAFall23GP.png")
origin = [-21.000000, -3.850000, 0.000000]  # m
resolution = 0.025000

width = map_image.shape[1]
height = map_image.shape[0]

width_m = width * resolution
height_m = height * resolution

plt.ion()

# plot the image
fig = plt.figure()
plt.show()


obstacles = []
pts = []


# get plot click points to define the obstacle
def onclick(event):
    global ix, iy, pts
    ix, iy = event.xdata, event.ydata
    print("x = %d, y = %d" % (ix, iy))
    pts.append([ix, iy])


cid = fig.canvas.mpl_connect("button_press_event", onclick)


# define the obstacles
while True:
    pts = []

    plt.imshow(map_image)
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")

    # draw the obstacles
    for obstacle in obstacles:
        for i in range(len(obstacle) - 1):
            plt.plot(
                [obstacle[i][0], obstacle[i + 1][0]],
                [obstacle[i][1], obstacle[i + 1][1]],
                "r",
            )

    fig.canvas.draw()
    fig.canvas.flush_events()

    inp = input("enter to add obstacle[a] or exit[e]: ")
    if inp == "e":
        break
    elif inp == "a":
        # remove duplicates
        add = []
        for pt in pts:
            if pt not in add:
                add.append(pt)
        obstacles.append(np.array(add))
    else:
        print("Not adding the obstacle")
        continue


# # print obstacles as a yaml list
# for obs in obstacles:
#     s = ""
#     for pt in obs:
#         pt = pt * resolution
#         pt[1] = height_m - pt[1]
#         pt[0] += origin[0]
#         pt[1] += origin[1]

#         s += "[{}, {}],".format(round(pt[0], 2), round(pt[1], 2))
#     s = s[:-1]
#     print("  - [{}]".format(s))


# print checkpoints as a yaml list
for obs in obstacles:
    for pt in obs:
        pt = pt * resolution
        pt[1] = height_m - pt[1]
        pt[0] += origin[0]
        pt[1] += origin[1]

        print("  - [{}, {}]".format(round(pt[0], 2), round(pt[1], 2)))

#!/usr/bin/python3

# x_y points of the obstacle
data = """
1.68	0.64
3.06	0.62
2.94	1.58
1.64	1.62
6.12	0.38
6.26	-0.39
7.62	-0.1
7.5	0.7
4.46	2.36
2.88	2.9
2.6	2.02
4.18	1.42
-1.74	4.32
-1.74	3.52
-0.3	3.48
-0.2	4.38
-9.24	5.4
-9.36	4.6
-8.12	4.32
-7.86	5
-14.04	3.32
-13.16	3.2
-12.72	3.96
-13.32	4.52
-11.96	1.66
-12.54	1.06
-11.74	0.24
-10.92	0.76
-8.9	-0.24
-8.84	-1.04
-7.7	-1
-7.66	-0.2
-3.82	-0.1
-3.94	-0.96
-2.38	-0.98
-2.42	-0.02
"""

# split the data into lines
data = data.split("\n")

points = []
for line in data:
    # split the line into x and y
    line = line.split("\t")
    pt = []
    # convert the strings to floats
    for i in range(len(line)):
        if line[i]:
            pt.append(float(line[i]))
    # add the points to the list
    if len(pt) == 2:
        points.append(pt)


# generate the obstacle points
obstacles = []
for i in range(0, len(points), 4):
    pt1 = points[i]
    pt2 = points[i + 1]
    pt3 = points[i + 2]
    pt4 = points[i + 3]
    obstacles.append([pt1, pt2, pt3, pt4])

# print the obstacles
print(obstacles)

#!/usr/bin/python3


# import libraries
import numpy as np
import rospy
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool, Float64
from tf.transformations import quaternion_from_euler
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
import tf


# create a car class
class Car:
    def __init__(self, car_name, map_params, lidar_params) -> None:
        self.name = car_name
        self.map, self.useMap = map_params
        self.useLidar = lidar_params

        # car parameters
        self.WB = rospy.get_param(f"~{car_name}/WB")
        self.steeringMax = rospy.get_param(f"~{car_name}/steeringMax")
        self.invertSteering = rospy.get_param(f"~{car_name}/invertSteering")

        # lidar parameters
        self.lidarUpdateRate = rospy.get_param(f"~{car_name}/lidar/updateRate")
        self.lidarUpdatedt = 1.0 / self.lidarUpdateRate
        self.lidarAngleMin = rospy.get_param(f"~{car_name}/lidar/angleMin")
        self.lidarAngleMax = rospy.get_param(f"~{car_name}/lidar/angleMax")
        self.lidarAngleIncrement = rospy.get_param(f"~{car_name}/lidar/angleIncrement")
        self.lidarRangeMin = rospy.get_param(f"~{car_name}/lidar/rangeMin")
        self.lidarRangeMax = rospy.get_param(f"~{car_name}/lidar/rangeMax")
        self.lidarCount = (
            int((self.lidarAngleMax - self.lidarAngleMin) / self.lidarAngleIncrement)
            + 1
        )
        self.thetas = np.linspace(
            self.lidarAngleMin, self.lidarAngleMax, self.lidarCount
        )
        self.lastLidarTime = None

        # Lap Parameters
        self.lapCount = 0
        self.lapStartTime = None
        self.checkpointIndex = 0
        self.obstacleCount = 0
        self.isControlled = False

        # car dynamics
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.velocity = 0.0
        self.angular_velocity = 0.0
        self.steering_angle = 0.0

        self.prev_x = 0.0
        self.prev_y = 0.0
        self.prev_yaw = 0.0

        self.path = Path()
        self.last_path_time = rospy.Time.now()
        self.sequenceId = 0

        self.map.addCar(self.name, self.lidarRangeMax, self.lidarCount)

        self.carTF = tf.TransformBroadcaster()

        self.odomPub = rospy.Publisher("/" + car_name + "/odom", Odometry, queue_size=1)

        self.pathPub = rospy.Publisher("/" + car_name + "/path", Path, queue_size=1)

        self.lidarPub = rospy.Publisher(
            "/" + car_name + "/scan", LaserScan, queue_size=1
        )
        self.checkpointPub = rospy.Publisher(
            "/" + car_name + "/checkpoint", Marker, queue_size=1, latch=True
        )
        self.lapPub = rospy.Publisher(
            "/" + car_name + "/lap_count", Float64, queue_size=1, latch=True
        )
        self.laptimePub = rospy.Publisher(
            "/" + car_name + "/laptime", Float64, queue_size=1
        )
        self.obstaclePub = rospy.Publisher(
            "/" + car_name + "/obstacle_count", Float64, queue_size=1, latch=True
        )

        self.commandSub = rospy.Subscriber(
            "/" + car_name + "/command", AckermannDrive, self.commandCallback
        )
        self.resetSub = rospy.Subscriber(
            "/" + car_name + "/reset", Pose, self.resetCallback
        )
        self.resetboolSub = rospy.Subscriber(
            "/" + car_name + "/resetBool", Bool, self.resetBoolCallback
        )

    def commandCallback(self, msg):
        print(f"{self.name} Command: {msg.steering_angle}, {msg.speed}")
        self.isControlled = True

        self.velocity = msg.speed
        msg.steering_angle = np.clip(
            msg.steering_angle, -self.steeringMax, self.steeringMax
        )
        if self.invertSteering:
            self.steering_angle = -msg.steering_angle

    def update(self, dt, simTime):
        # update previous values
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_yaw = self.yaw

        # get some random noise
        noise = np.random.normal(0, 0.0, 5)

        # update the pose
        self.velocity = self.velocity + noise[0]
        self.yaw += (self.velocity / self.WB) * np.tan(
            self.steering_angle
        ) * dt + noise[3]
        self.x += self.velocity * np.cos(self.yaw) * dt + noise[1]
        self.y += self.velocity * np.sin(self.yaw) * dt + noise[2]

        self.angular_velocity = (self.yaw - self.prev_yaw) / dt + noise[4]

        if self.useMap:
            # check if the car new position is inside an obstacle
            self.checkCollision()

            # check if the car has crossed a checkpoint
            self.checkCheckpoint(simTime)

        if (
            self.lastLidarTime is None
            or simTime >= self.lastLidarTime + self.lidarUpdatedt
        ) and self.useLidar:
            self.lastLidarTime = simTime
            self.publishLidar()

        self.publishTF()
        self.publishOdometry()

    def checkCollision(self):
        # get angle from previous point to current point
        theta = np.arctan2(self.y - self.prev_y, self.x - self.prev_x)

        # get the range from previous point to current point
        dist = np.sqrt((self.y - self.prev_y) ** 2 + (self.x - self.prev_x) ** 2)
        lidar = self.map.getRanges(self.name, self.prev_x, self.prev_y, [theta])
        lidar = lidar[0]

        # check if the car is inside an obstacle
        if dist >= lidar - self.map.wallBuffer:
            # get the collision point from previous point theta and range
            collision_point = (
                self.prev_x + lidar * np.cos(theta),
                self.prev_y + lidar * np.sin(theta),
            )

            # update the car to not move
            self.x = collision_point[0] - self.map.wallBuffer * np.cos(theta)
            self.y = collision_point[1] - self.map.wallBuffer * np.sin(theta)

            self.obstacleCount += 1
            self.obstaclePub.publish(self.obstacleCount)

            # check if point is inside an obstacle
            for obstacle in self.map.obstacles:
                if obstacle.isPointInObstacle(
                    collision_point, self.map.obstacles_timeout
                ):
                    self.map.isMapUpdated = True
                    break

    def checkCheckpoint(self, simTime):
        if not self.isControlled:
            return
        isCheckpointUpdate = False
        isStart = False
        if self.lapStartTime is None:
            isCheckpointUpdate = True
            isStart = True
            self.lapStartTime = simTime - (
                self.map.checkpoint_thresh / self.velocity + 0.5
            )
            self.lapCount -= 1 / len(self.map.checkpoints)
            self.obstaclePub.publish(self.obstacleCount)

        # check if car is inside the checkpoint
        dist = np.sqrt(
            (self.x - self.map.checkpoints[self.checkpointIndex][0]) ** 2
            + (self.y - self.map.checkpoints[self.checkpointIndex][1]) ** 2
        )
        if dist < self.map.checkpoint_thresh:
            isCheckpointUpdate = True

        if isCheckpointUpdate:
            self.checkpointIndex += 1
            if self.checkpointIndex >= len(self.map.checkpoints):
                self.checkpointIndex = 0
            self.lapCount += 1 / len(self.map.checkpoints)
            self.lapPub.publish(self.lapCount)
            self.publishCheckpoint()
            if self.checkpointIndex == 1 and not isStart:
                self.laptimePub.publish(simTime - self.lapStartTime)
                self.lapStartTime = simTime

    def publishCheckpoint(self):
        checkpoint = Marker()
        checkpoint.header.frame_id = "world"
        checkpoint.header.stamp = rospy.Time.now()
        checkpoint.id = 0
        checkpoint.type = Marker.SPHERE
        checkpoint.action = Marker.ADD
        checkpoint.pose.position.x = self.map.checkpoints[self.checkpointIndex][0]
        checkpoint.pose.position.y = self.map.checkpoints[self.checkpointIndex][1]
        checkpoint.pose.position.z = 0.0
        checkpoint.pose.orientation.x = 0.0
        checkpoint.pose.orientation.y = 0.0
        checkpoint.pose.orientation.z = 0.0
        checkpoint.pose.orientation.w = 1.0
        checkpoint.scale.x = 0.2
        checkpoint.scale.y = 0.2
        checkpoint.scale.z = 0.2
        checkpoint.color.a = 1.0
        checkpoint.color.r = 0.0
        checkpoint.color.g = 1.0
        checkpoint.color.b = 0.0

        self.checkpointPub.publish(checkpoint)

    def publishLidar(self):
        scan = self.lidarCallback(self.x, self.y, self.yaw)
        scan.header.stamp = rospy.Time.now()

        self.lidarPub.publish(scan)

    def lidarCallback(self, x, y, yaw):
        # get the laser scan
        scan = LaserScan()
        scan.header.frame_id = f"{self.name}/laser"
        scan.angle_min = self.lidarAngleMin
        scan.angle_max = self.lidarAngleMax
        scan.angle_increment = self.lidarAngleIncrement
        scan.range_min = self.lidarRangeMin
        scan.range_max = self.lidarRangeMax

        # get the global slope of the rays
        global_thetas = self.thetas + yaw

        # get the ranges
        ranges = self.map.getRanges(self.name, x, y, global_thetas)

        # set the ranges
        scan.ranges = np.clip(ranges, self.lidarRangeMin, self.lidarRangeMax)

        return scan

    def resetCallback(self, msg):
        print(
            f"{self.name} Resetting pose: {msg.position.x}, {msg.position.y}, {msg.orientation.z}"
        )

        self.x = msg.position.x
        self.y = msg.position.y
        self.yaw = msg.orientation.z
        self.velocity = 0.0
        self.angular_velocity = 0.0
        self.steering_angle = 0.0

        # Lap Parameters
        self.lapCount = 0
        self.lapStartTime = None
        self.checkpointIndex = 0
        self.obstacleCount = 0

        # reset path
        self.path.poses.clear()
        self.pathPub.publish(self.path)

    def resetBoolCallback(self, msg):
        if not msg.data:
            return
        print(f"{self.name} Resetting pose: {msg.data}")

        self.x = 0
        self.y = 0
        self.yaw = 0
        self.velocity = 0.0
        self.angular_velocity = 0.0
        self.steering_angle = 0.0

        # Lap Parameters
        self.lapCount = 0
        self.lapStartTime = None
        self.checkpointIndex = 0
        self.obstacleCount = 0

        # reset path
        self.path.poses.clear()
        self.pathPub.publish(self.path)

    def publishOdometry(self):
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "odom"
        odom.child_frame_id = f"{self.name}/base_link"

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y

        quat = quaternion_from_euler(0, 0, self.yaw)

        odom.pose.pose.orientation.x = quat[0]
        odom.pose.pose.orientation.y = quat[1]
        odom.pose.pose.orientation.z = quat[2]
        odom.pose.pose.orientation.w = quat[3]

        odom.twist.twist.linear.x = self.velocity

        odom.twist.twist.angular.z = self.angular_velocity

        self.odomPub.publish(odom)

        if (rospy.Time.now() - self.last_path_time).to_sec() < 0.1:
            return
        self.last_path_time = rospy.Time.now()

        newpose = PoseStamped()
        newpose.header = odom.header
        newpose.header.seq = self.sequenceId
        newpose.pose = odom.pose.pose

        self.path.header = odom.header
        self.path.header.seq = self.sequenceId
        self.path.header.frame_id = "world"
        self.path.poses.append(newpose)

        self.sequenceId += 1

        self.pathPub.publish(self.path)

    def publishTF(self):
        # publish tf from world to base_link
        self.carTF.sendTransform(
            (self.x, self.y, 0),
            quaternion_from_euler(0, 0, self.yaw),
            rospy.Time.now(),
            f"{self.name}/base_link",
            "world",
        )

        # publish tf from base_link to laser
        self.carTF.sendTransform(
            (0, 0, 0),
            quaternion_from_euler(0, 0, 0),
            rospy.Time.now(),
            f"{self.name}/laser",
            f"{self.name}/base_link",
        )

        # publish tf from base_link to car_name
        self.carTF.sendTransform(
            (0, 0, 0),
            quaternion_from_euler(0, 0, 0),
            rospy.Time.now(),
            f"{self.name}",
            f"{self.name}/base_link",
        )

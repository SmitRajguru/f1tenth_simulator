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
from sensor_msgs.msg import Imu
from visualization_msgs.msg import Marker
import tf
import threading


# create a car class
class Car:
    def __init__(self, car_name, map_params, lidar_params) -> None:
        self.name = car_name
        self.lidarPub = None
        self.lidarThread = None
        self.lidarThread_stop = False

        self.updateParams(map_params, lidar_params)

        self.carTF = tf.TransformBroadcaster()

        if self.amcl:
            self.pathPub = rospy.Publisher(
                "/" + car_name + "/amcl/path", Path, queue_size=1
            )
            self.imuPub = rospy.Publisher(
                "/" + car_name + "/amcl/imu", Imu, queue_size=1
            )
            self.lidarPub = rospy.Publisher(
                "/" + car_name + "/amcl/scan", LaserScan, queue_size=1
            )
        else:
            self.odomPub = rospy.Publisher(
                "/" + car_name + "/odom", Odometry, queue_size=1
            )

            self.imuPub = rospy.Publisher("/" + car_name + "/imu", Imu, queue_size=1)
            self.lidarPub = rospy.Publisher(
                "/" + car_name + "/scan", LaserScan, queue_size=1
            )
        self.pathPub = rospy.Publisher("/" + car_name + "/path", Path, queue_size=1)
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

    def remove(self):
        self.commandSub.unregister()
        self.resetSub.unregister()
        self.resetboolSub.unregister()

        self.lidarThread_stop = True
        if self.lidarThread is not None:
            self.lidarThread.join()

        self.map.removeCar(self.name)

    def updateParams(self, map_params, lidar_params):
        self.lidarThread_stop = True
        if self.lidarThread is not None:
            self.lidarThread.join()

        self.map, self.useMap = map_params
        self.useLidar = lidar_params

        # car parameters
        self.amcl = bool(rospy.get_param(f"~{self.name}/amcl"))
        self.WB = rospy.get_param(f"~{self.name}/WB")
        self.steeringMax = rospy.get_param(f"~{self.name}/steeringMax")
        self.invertSteering = rospy.get_param(f"~{self.name}/invertSteering")

        # lidar parameters
        self.lidarUpdateRate = rospy.get_param(f"~{self.name}/lidar/updateRate")
        self.lidarAngleMin = rospy.get_param(f"~{self.name}/lidar/angleMin")
        self.lidarAngleMax = rospy.get_param(f"~{self.name}/lidar/angleMax")
        self.lidarAngleIncrement = rospy.get_param(f"~{self.name}/lidar/angleIncrement")
        self.lidarRangeMin = rospy.get_param(f"~{self.name}/lidar/rangeMin")
        self.lidarRangeMax = rospy.get_param(f"~{self.name}/lidar/rangeMax")
        self.lidarCount = (
            int((self.lidarAngleMax - self.lidarAngleMin) / self.lidarAngleIncrement)
            + 1
        )
        self.thetas = np.linspace(
            self.lidarAngleMin, self.lidarAngleMax, self.lidarCount
        )
        self.lidarThread_stop = False

        self.map.addCar(self.name, self.lidarRangeMax, self.lidarCount)

        # IMU parameters
        self.linear_acceleration_variance = rospy.get_param(
            f"~{self.name}/imu/linear_acceleration_variance"
        )
        self.angular_velocity_variance = rospy.get_param(
            f"~{self.name}/imu/angular_velocity_variance"
        )
        self.orientation_variance = rospy.get_param(
            f"~{self.name}/imu/orientation_variance"
        )

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

        if self.useLidar:
            self.lidarThread = threading.Thread(
                target=self.lidarThreadCallback,
                daemon=True,
                args=(self.lidarUpdateRate,),
            )
            self.lidarThread.start()
        else:
            self.lidarThread = None

    def commandCallback(self, msg):
        # print(f"{self.name} Command: {msg.steering_angle}, {msg.speed}")
        self.isControlled = True

        self.velocity = msg.speed
        msg.steering_angle = np.clip(
            msg.steering_angle, -self.steeringMax, self.steeringMax
        )
        if self.invertSteering:
            self.steering_angle = -msg.steering_angle

    def kill(self):
        self.isControlled = False
        self.lidarThread_stop = True

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
                    collision_point, self.map.wallBuffer / 2, self.map.obstacleTimeout
                ):
                    self.map.isMapUpdated = True
                    break

    def checkCheckpoint(self, simTime):
        if not self.isControlled or len(self.map.checkpoints) == 0:
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

    def lidarThreadCallback(self, updateRate):
        rate = rospy.Rate(updateRate)
        while not self.lidarThread_stop:
            self.publishLidar()
            rate.sleep()

    def publishLidar(self):
        scan = self.lidarCallback(self.x, self.y, self.yaw)
        scan.header.stamp = rospy.Time.now()

        if self.lidarPub is not None:
            self.lidarPub.publish(scan)

    def lidarCallback(self, x, y, yaw):
        # get the laser scan
        scan = LaserScan()
        if self.amcl:
            scan.header.frame_id = f"{self.name}/amcl/laser"
        else:
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
        self.isControlled = False

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
        self.isControlled = False

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

        if not self.amcl:
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

        base_name = f"{self.name}"
        if self.amcl:
            base_name = f"{self.name}/amcl"

        # generate IMU message
        imu = Imu()
        imu.header.stamp = rospy.Time.now()
        imu.header.frame_id = f"{base_name}/imu"

        imu.angular_velocity.x = 0
        imu.angular_velocity.y = 0
        imu.angular_velocity.z = self.angular_velocity

        imu.linear_acceleration.x = self.velocity * self.angular_velocity
        imu.linear_acceleration.y = 0
        imu.linear_acceleration.z = 0

        # add noise to imu
        imu_orientation = tf.transformations.euler_from_quaternion(
            [quat[0], quat[1], quat[2], quat[3]]
        )
        imu_orientation = np.array(imu_orientation)
        imu_orientation[0] += np.random.normal(0, self.orientation_variance[0])
        imu_orientation[1] += np.random.normal(0, self.orientation_variance[1])
        imu_orientation[2] += np.random.normal(0, self.orientation_variance[2])
        imu_orientation = tf.transformations.quaternion_from_euler(
            imu_orientation[0], imu_orientation[1], imu_orientation[2]
        )

        imu.orientation.x = imu_orientation[0]
        imu.orientation.y = imu_orientation[1]
        imu.orientation.z = imu_orientation[2]
        imu.orientation.w = imu_orientation[3]

        imu.linear_acceleration.x += np.random.normal(
            0, self.linear_acceleration_variance[0]
        )
        imu.linear_acceleration.y += np.random.normal(
            0, self.linear_acceleration_variance[1]
        )
        imu.linear_acceleration.z += np.random.normal(
            0, self.linear_acceleration_variance[2]
        )

        imu.angular_velocity.x += np.random.normal(0, self.angular_velocity_variance[0])
        imu.angular_velocity.y += np.random.normal(0, self.angular_velocity_variance[1])
        imu.angular_velocity.z += np.random.normal(0, self.angular_velocity_variance[2])

        self.imuPub.publish(imu)

    def publishTF(self):
        if not self.amcl:
            # publish tf from world to base_link
            self.carTF.sendTransform(
                (self.x, self.y, 0),
                quaternion_from_euler(0, 0, self.yaw),
                rospy.Time.now(),
                f"{self.name}/base_link",
                "world",
            )

        base_name = f"{self.name}"
        if self.amcl:
            base_name = f"{self.name}/amcl"

        # publish tf from base_link to laser
        self.carTF.sendTransform(
            (0, 0, 0),
            quaternion_from_euler(0, 0, 0),
            rospy.Time.now(),
            f"{base_name}/laser",
            f"{base_name}/base_link",
        )

        # publish tf from base_link to imu
        self.carTF.sendTransform(
            (0, 0, 0),
            quaternion_from_euler(0, 0, 0),
            rospy.Time.now(),
            f"{base_name}/imu",
            f"{base_name}/base_link",
        )

        # publish tf from base_link to car_name
        self.carTF.sendTransform(
            (0, 0, 0),
            quaternion_from_euler(0, 0, 0),
            rospy.Time.now(),
            f"{base_name}",
            f"{base_name}/base_link",
        )

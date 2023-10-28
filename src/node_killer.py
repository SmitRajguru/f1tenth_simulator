#!/usr/bin/python3

# importing libraries
import rospy
import subprocess
from std_msgs.msg import Bool

rospy.init_node("node_killer")


def killer_callback(msg):
    node_list = rospy.get_param("node_list")

    for node in node_list:
        subprocess.Popen(["rosnode", "kill", node])


sub = rospy.Subscriber("/kill_nodes", data_class=Bool, callback=killer_callback)

rospy.spin()

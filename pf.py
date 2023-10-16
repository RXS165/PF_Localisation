from geometry_msgs.msg import Pose, PoseArray, Quaternion
from . pf_base import PFLocaliserBase
import math
import rospy

from . util import rotateQuaternion, getHeading
from random import random
import random
import numpy as np

from sklearn.cluster import DBSCAN

from time import time


class PFLocaliser(PFLocaliserBase):
       
    def __init__(self):
        # ----- Call the superclass constructor
        super(PFLocaliser, self).__init__()
        
        # ----- Set motion model parameters

        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20     # Number of readings to predict
        self.last_odom_x = 0
        self.last_odom_y = 0
        self.last_odom_heading = 0
        self.weights = []
        self.w_fast = 0
        self.w_slow = 0
        self.alpha_fast = 0.7
        self.alpha_slow = 0.01
        self.cluster_eps = 0.3
        self.cluster_min_samples = 10
    def initialise_particle_cloud(self, initialpose):
        """
        Set particle cloud to initialpose plus noise

        Called whenever an initialpose message is received (to change the
        starting location of the robot), or a new occupancy_map is received.
        self.particlecloud can be initialised here. Initial pose of the robot
        is also set here.
        
        :Args:
            | initialpose: the initial pose estimate
        :Return:
            | (geometry_msgs.msg.PoseArray) poses of the particles
        """
        
        rospy.loginfo("Initial particles Generated")
        initialParticles = PoseArray()
        for i in range(400):
            #initialParticles.poses.append(self.genGaussPose(initialpose.pose.pose, 1))
            initialParticles.poses.append(self.genRandomPose())
        return initialParticles
    
    def update_particle_cloud(self, scan):
        """
        This should use the supplied laser scan to update the current
        particle cloud. i.e. self.particlecloud should be updated.
        
        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update

         """
        # systematic resampling 
        
        if (self.prev_odom_x  != self.last_odom_x or self.prev_odom_y != self.last_odom_y  or self.prev_odom_heading != self.last_odom_heading):
            originals = self.particlecloud.poses
            resampled_particles = PoseArray()
            weights = []
            w_total = 0
            N = len(self.particlecloud.poses)

            for p in self.particlecloud.poses:
                w = self.sensor_model.get_weight(scan, p)
                weights.append(w)

            w_total = sum(weights)
            normalized_weights = [w / w_total for w in weights] 
            self.weights = normalized_weights
            cumulative_w = [sum(normalized_weights[:i+1]) for i in range(N)]

            avgWeight = (sum(normalized_weights)/N)*w_total
            self.w_fast += self.alpha_fast * (avgWeight-self.w_fast)
            self.w_slow += self.alpha_slow * (avgWeight - self.w_slow)

            numRandoms = max(int(0.1*N), int(N*max(0, 1-(self.w_fast/self.w_slow))))
            # numResamples = N - numRandoms
            #rospy.loginfo("avgWeight = " + str(avgWeight))
            #rospy.loginfo("prob = " + str(1-(self.w_fast/self.w_slow)))
            # rospy.loginfo("w_fast: " +str(self.w_fast))
            # rospy.loginfo("w_slow: " +str(self.w_slow))
            # rospy.loginfo("w_avg: " +str(avgWeight))
            ##RESAMPLING
            
            index = 0
            start = random.uniform(0.0, 1/N)

            resampled_weights = []

            
                # resampledWeights.append(self.sensor_model.get_weight(scan, p))
            for n in range(N):
                newPose= Pose()

                while start > cumulative_w[index]:
                    index += 1
                oldPose = originals[index]
                
                
                newPose.position.x = oldPose.position.x + random.gauss(0,1) * 0.02
                newPose.position.y = oldPose.position.y + random.gauss(0,1) * 0.02
                newPose.orientation = rotateQuaternion(oldPose.orientation, math.pi*2 * random.gauss(0,1)*0.02)
                resampled_particles.poses.append(newPose)
                resampled_weights.append(self.weights[index]*w_total)
                start += 1/N

            for n in range(numRandoms):
                i = random.randint(0, N-1)
                randomPose = self.genRandomPose()
                resampled_particles.poses[i] = randomPose
                resampled_weights[i] = self.sensor_model.get_weight(scan, randomPose)
            resampled_w_total = sum(resampled_weights)

            self.weights = [w / resampled_w_total for w in resampled_weights]
        # update particlecloud
            self.last_odom_x = self.prev_odom_x
            self.last_odom_y = self.prev_odom_y
            self.last_odom_heading = self.prev_odom_heading
            self.particlecloud = resampled_particles
        




    def estimate_pose(self):
        """
        This should calculate and return an updated robot pose estimate based
        on the particle cloud (self.particlecloud).
        
        Create new estimated pose, given particle cloud
        E.g. just average the location and orientation values of each of
        the particles and return this.
        
        Better approximations could be made by doing some simple clustering,
        e.g. taking the average location of half the particles after 
        throwing away any which are outliers

        :Return:
            | (geometry_msgs.msg.Pose) robot's estimated pose.
         """

        finalPose = Pose()

        clusterSpace = []
        for p in self.particlecloud.poses:
            clusterSpace.append((p.position.x, p.position.y))

        clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.cluster_min_samples).fit(clusterSpace)
        # cluster_weights = [sum(self.weights[i] for i in range(len(self.particlecloud.poses)) if clustering.labels_[i] == cluster_label) for cluster_label in set(clustering.labels_)]
        # best_cluster_index = cluster_weights.index(max(cluster_weights))
        # best_cluster_label = list(set(clustering.labels_))[best_cluster_index]
        clusterSize = {}
        for label in list(set(clustering.labels_)):
            if label != -1:
                if label in clusterSize:
                    clusterSize[label] += 1
                else:
                    clusterSize[label] = 1
        num_clusters = len(clusterSize.keys())
        rospy.loginfo("number of clusters = " + str(num_clusters))



        if(num_clusters > 0):

            best_cluster_label = max(clusterSize, key=clusterSize.get)
            subGroupToUse = [self.particlecloud.poses[i] for i in range(len(self.particlecloud.poses)) if clustering.labels_[i] == best_cluster_label]
            N = len(subGroupToUse)

        elif len(self.weights) > 0:
            bestPose = self.particlecloud.poses[self.weights.index(max(self.weights))]
            finalPose.position.x = bestPose.position.x
            finalPose.position.y = bestPose.position.y
            finalPose.orientation = bestPose.orientation
            return finalPose
        
        else:
            N = len(self.particlecloud.poses)
            subGroupToUse = self.particlecloud.poses

        
        finalHeadingX = 0
        finalHeadingY = 0
        finalHeadingZ = 0
        finalHeadingW = 0
        finalX = 0
        finalY = 0

        # if(len(self.weights)>0):
        # #rospy.loginfo("pose estimated")
        #     for i in range(N):
        #         p = self.particlecloud.poses[i]
        #         finalX += p.position.x * self.weights[i]
        #         finalY += p.position.y * self.weights[i]
        #         finalHeadingX += p.orientation.x * self.weights[i]
        #         finalHeadingY += p.orientation.y * self.weights[i]
        #         finalHeadingZ += p.orientation.z * self.weights[i]
        #         finalHeadingW += p.orientation.w * self.weights[i]
        # else:
        norm = 1/N
        for i in range(N):
            p = subGroupToUse[i]
            finalX += p.position.x * norm
            finalY += p.position.y * norm
            finalHeadingX += p.orientation.x * norm
            finalHeadingY += p.orientation.y * norm
            finalHeadingZ += p.orientation.z * norm
            finalHeadingW += p.orientation.w * norm
        finalPose.position.x = finalX
        finalPose.position.y = finalY
        finalPose.orientation.x = finalHeadingX
        finalPose.orientation.y = finalHeadingY
        finalPose.orientation.z = finalHeadingZ
        finalPose.orientation.w = finalHeadingW

        return finalPose
    
    def genRandomPose(self):
        width = self.occupancy_map.info.width
        height = self.occupancy_map.info.height
        resolution = self.occupancy_map.info.resolution
        origin_x = self.occupancy_map.info.origin.position.x
        origin_y = self.occupancy_map.info.origin.position.y

        n = 0
        while n < 1:
            rndX = random.uniform(0, width * resolution) + origin_x
            rndY = random.uniform(0, height * resolution) + origin_y

            x_index = int((rndX-origin_x)/resolution)
            y_index = int((rndY-origin_y)/resolution)
            rndTheta = random.uniform(0.0, math.pi * 2)

            if self.occupancy_map.data[y_index*width + x_index] == 0:
                samplePose = Pose()
                samplePose.position.x = rndX
                samplePose.position.y = rndY
                samplePose.orientation = rotateQuaternion(Quaternion(w=1.0), rndTheta)
                n+=1
        return samplePose
    
    def genGaussPose(self, initialPose, sdev):
        gaussPose = Pose()
        gaussPose.position.x = initialPose.position.x + random.gauss(0, sdev)
        gaussPose.position.y = initialPose.position.y + random.gauss(0, sdev)
        gaussPose.orientation = rotateQuaternion(initialPose.orientation, random.gauss(0, sdev))
        return gaussPose
        
    
    def returnBest(self):
        finalPose = Pose()
        bestParticle = self.particlecloud.poses[self.weights.index(max(self.weights))]
        finalPose.position.x = bestParticle.position.x
        finalPose.position.y = bestParticle.position.y
        finalPose.orientation.x = bestParticle.orientation.x
        finalPose.orientation.y = bestParticle.orientation.y
        finalPose.orientation.z = bestParticle.orientation.z
        finalPose.orientation.w = bestParticle.orientation.w
        return finalPose
    

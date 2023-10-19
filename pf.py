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
        self.ODOM_ROTATION_NOISE = 0.5	# Odometry model rotation noise
        self.ODOM_TRANSLATION_NOISE = 0.5	# Odometry x axis (forward) noise
        self.ODOM_DRIFT_NOISE = 0.5	# Odometry y axis (side-side) noise
        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20     # Number of readings to predict
        
        # ----- Custom model Parameters
        self.last_odom_x = 0
        self.last_odom_y = 0
        self.last_odom_heading = 0
        self.M = 350
        self.norm_weights = [1/self.M for i in range(self.M)]
        self.w_fast = 1 /self.M
        self.w_slow = 1/self.M
        self.alpha_fast = 0.4
        self.alpha_slow = 0.02
        self.cluster_eps = 0.05
        self.cluster_min_samples = int(self.M*0.02)
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
        for i in range(self.M):
            #initialParticles.poses.append(self.getGaussianRandomPose(initialpose.pose.pose, 1))
            initialParticles.poses.append(self.getUniformRandomPose(1))
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
            resampled_particles = PoseArray()
            weights = []
            w_total = 0

            for p in self.particlecloud.poses:
                w = self.sensor_model.get_weight(scan, p)
                weights.append(w)

            w_total = sum(weights)
            normalized_weights = [w / w_total for w in weights] 
            self.norm_weights = normalized_weights
            cumulative_w = [sum(normalized_weights[:i+1]) for i in range(self.M)]

            avgWeight = w_total / self.M
            self.w_fast += self.alpha_fast * (avgWeight - self.w_fast)
            self.w_slow += self.alpha_slow * (avgWeight - self.w_slow)
            rospy.loginfo("random_prob = " + str(1-(self.w_fast/self.w_slow)))

            numRandoms = int(self.M*min(max(0.04, 1-(self.w_fast/self.w_slow)), 0.5))
            # numResamples = N - numRandoms
            #rospy.loginfo("avgWeight = " + str(avgWeight))
            #rospy.loginfo("prob = " + str(1-(self.w_fast/self.w_slow)))
            # rospy.loginfo("w_fast: " +str(self.w_fast))
            # rospy.loginfo("w_slow: " +str(self.w_slow))
            # rospy.loginfo("w_avg: " +str(avgWeight))
            ##RESAMPLING
            
            index = 0
            start = random.uniform(0.0, 1/self.M)

            resampled_weights = []

            for n in range(self.M):
                newPose= Pose()

                while start > cumulative_w[index]:
                    index += 1
                oldPose = self.particlecloud.poses[index]
                
                
                newPose.position.x = oldPose.position.x + random.gauss(0,1) * 0.02
                newPose.position.y = oldPose.position.y + random.gauss(0,1) * 0.02
                newPose.orientation = rotateQuaternion(oldPose.orientation, math.pi*2 * random.gauss(0,1)*0.02)
                resampled_particles.poses.append(newPose)
                resampled_weights.append(self.norm_weights[index]*w_total)
                start += 1/self.M

            for n in range(numRandoms):
                i = random.randint(0, self.M-1)
                randomPose = self.getUniformRandomPose(1)
                new_w = self.sensor_model.get_weight(scan, randomPose)
                resampled_particles.poses[i] = randomPose
                resampled_weights[i] = new_w
            resampled_w_total = sum(resampled_weights)

            self.norm_weights = [w / resampled_w_total for w in resampled_weights]
        # update particlecloud
            self.last_odom_x = self.prev_odom_x
            self.last_odom_y = self.prev_odom_y
            self.last_odom_heading = self.prev_odom_heading
            self.particlecloud.poses = resampled_particles.poses
        #rospy.loginfo("resampling time = "+str((time()-start_time)*100)+"ms")





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
        #start_time = time()
        finalPose = Pose()
        clusterSpace = []

        for p in self.particlecloud.poses:
            clusterSpace.append((p.position.x, p.position.y))

        clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.cluster_min_samples).fit(clusterSpace)
        clusterScores = {}

        for i in range(len(clustering.labels_)):
            if clustering.labels_[i] != -1:
                if clustering.labels_[i] in clusterScores:
                    clusterScores[clustering.labels_[i]] += self.norm_weights[i]
                else:
                    clusterScores[clustering.labels_[i]] = self.norm_weights[i]

        #rospy.loginfo(clusterScores)

        num_clusters = len(clusterScores.keys())

        if(num_clusters == 0):
            return self.bestParticle()

        else:
            best_cluster_label = max(clusterScores, key=clusterScores.get)
            weight_total = clusterScores[best_cluster_label]

            #rospy.loginfo("best cluster: " + str(best_cluster_label))
            subGroupToUse = [(self.particlecloud.poses[i], self.norm_weights[i]) for i in range(self.M) if clustering.labels_[i] == best_cluster_label]
            N = len(subGroupToUse)

            finalHeadingX = 0
            finalHeadingY = 0
            finalHeadingZ = 0
            finalHeadingW = 0
            finalX = 0
            finalY = 0


            norm = 1/N
            for i in range(N):
                p = subGroupToUse[i][0]
                norm = subGroupToUse[i][1] / weight_total
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

            #rospy.loginfo("pose estimating time = "+str((time()-start_time)*100)+"ms")
            return finalPose
    
    def getUniformRandomPose(self, N):
        width = self.occupancy_map.info.width
        #height = self.occupancy_map.info.height
        resolution = self.occupancy_map.info.resolution
        origin_x = self.occupancy_map.info.origin.position.x
        origin_y = self.occupancy_map.info.origin.position.y

        n = 0
        while n < N:
            rndX = random.uniform(-16, 16)
            rndY = random.uniform(-16, 16)

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
    
    def getGaussianRandomPose(self, initialPose, sdev):
        gaussPose = Pose()
        gaussPose.position.x = initialPose.position.x + random.gauss(0, sdev)
        gaussPose.position.y = initialPose.position.y + random.gauss(0, sdev)
        gaussPose.orientation = rotateQuaternion(initialPose.orientation, random.gauss(0, sdev))
        return gaussPose
        
    
    def bestParticle(self):
        finalPose = Pose()
        bestParticle = self.particlecloud.poses[self.norm_weights.index(max(self.norm_weights))]
        finalPose.position.x = bestParticle.position.x
        finalPose.position.y = bestParticle.position.y
        finalPose.orientation = bestParticle.orientation
        return finalPose
    

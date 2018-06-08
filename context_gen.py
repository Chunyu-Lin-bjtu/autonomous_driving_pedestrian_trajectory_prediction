#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, BMW Group, all rights reserved.
#
# Redistribution and use in source and other forms, with or without modification,
# are permitted only in BMW internal usage. Any other companies or third-party
# SHOULD NOT use it.
#
# This software is provided by the copyright holders and contributors "as is" and
# any express or implied warranties, including, but not limited to, the implied
# warranties of merchantability and fitness for a particular purpose are disclaimed.
# In no event shall copyright holders or contributors be liable for any direct,
# indirect, incidental, special, exemplary, or consequential damages
# (including, but not limited to, procurement of substitute goods or services;
# loss of use, data, or profits; or business interruption) however caused
# and on any theory of liability, whether in contract, strict liability,
# or tort (including negligence or otherwise) arising in any way out of
# the use of this software, even if advised of the possibility of such damage.
#
# @Time    : 18-5-24
# @Author  : Gavin.Xu
# @Email   : Gavin.Xu@bmw.com
# @Department: EG-CN-72

import os
import numpy as np
import cv2
import math
import matplotlib.image as img
import matplotlib.pyplot as plt

import csv
import pandas as pd
import random

import datetime

from astar import AStarSearch
from config import *


class ContextGenerator(object):
    """
    A class to generate the fake hd map (context) for prediction
    """
    def __init__(self, dump_root_path, args, args_map):
        """
        Initialization of map generator

        :param dump_root_path: root path for map and trajectory file
        :param args: map context value and cost value configuration
        :param args_map: map configuration parameters
        """
        self._dump_root_path = dump_root_path
        if not os.path.exists(self._dump_root_path):
            os.makedirs(self._dump_root_path)

        self._dump_context_root_path = os.path.join(self._dump_root_path, 'context')
        self._dump_traj_root_path = os.path.join(self._dump_root_path, 'trajectory')

        if not os.path.exists(self._dump_context_root_path):
            os.makedirs(self._dump_context_root_path)
        if not os.path.exists(self._dump_traj_root_path):
            os.makedirs(self._dump_traj_root_path)

        self.context_value_cost_config = args
        self.map_config = args_map
        self._width = args_map.width
        self._height = args_map.height
        self._map_info = []              # list of map outer 8 points coordinates

    def generate_context_traj(self, context_name, traj_name, traj_num):
        """
        Generate context and trajectories and dump all the context and trajecoties into local files

        :param context_name: name of context file
        :param traj_name: name of trajectory file
        :param traj_num: number of trajectory to generate for each map
        """
        # generate random lane width and slope, then side walk and zebra crossing width
        k_x = (random.uniform(1.5, 10), random.uniform(-5, -10))[random.randint(0, 1)]
        k_y = random.uniform(-0.35, 0.55)
        width_x = random.uniform(200, 400)
        width_y = random.uniform(200, 400)
        width_side_walk = random.uniform(50, 120)
        width_zebra = random.uniform(70, 120)

        # set dump path for context, trajectory and map info
        dump_context_full_path = os.path.join(self._dump_context_root_path, context_name)
        dump_traj_full_path = os.path.join(self._dump_traj_root_path, traj_name)

        # context matrix
        context = np.ones([self._width, self._height], dtype='int64') * self.context_value_cost_config.unknown

        # point A, B, C, D defining x normal lane clock-wisely;
        # point E, F, G, D defining y normal lane clock-wisely
        point_A = (self.map_config.offset_x - width_x / 2, 0)
        point_B = ((self._height - (point_A[1] - k_x * point_A[0])) / k_x, self._height - 1)
        point_C = (point_B[0] + width_x, self._height - 1)
        point_D = (self.map_config.offset_x + width_x / 2, 0)
        point_E = (0, self.map_config.offset_y - width_y / 2)
        point_F = (0, self.map_config.offset_y + width_y / 2)
        point_G = (self._width - 1, k_y * self._width + (point_F[1] - k_y * point_F[0]))
        point_H = (self._width - 1, point_G[1] - width_y)

        print "...drawing..."

        # Draw normal lanes
        lane_x = (point_A, point_B, point_C, point_D)
        lane_y = (point_E, point_F, point_G, point_H)
        self.fill_color(context, lane_x, self.context_value_cost_config.normal_lane)
        self.fill_color(context, lane_y, self.context_value_cost_config.normal_lane)

        # Draw junction area
        point_a = self.get_inter(point_A, k_x, point_E, k_y)
        point_b = self.get_inter(point_A, k_x, point_F, k_y)
        point_c = self.get_inter(point_D, k_x, point_F, k_y)
        point_d = self.get_inter(point_D, k_x, point_E, k_y)
        junction_area = (point_a, point_b, point_c, point_d)
        self.fill_color(context, junction_area, self.context_value_cost_config.junction_area)

        # Draw side walks
        point1, point2, ___, ___, point5, point6, ___, ___ = self.get_outer_8_points(
                       lane_x, k_x, k_y, width_side_walk)
        ___, ___, point3, point4, ___, ___, point7, point8 = self.get_outer_8_points(
                       lane_y, k_x, k_y, width_side_walk)
        map_info = [point1, point2, point3, point4, point5, point6, point7, point8]
        self.fill_color(context, (point1, point2, point_B, point_A), self.context_value_cost_config.side_walk,
                        to_replace=self.context_value_cost_config.unknown)
        self.fill_color(context, (point_F, point3, point4, point_G), self.context_value_cost_config.side_walk,
                        to_replace=self.context_value_cost_config.unknown)
        self.fill_color(context, (point_D, point_C, point5, point6), self.context_value_cost_config.side_walk,
                        to_replace=self.context_value_cost_config.unknown)
        self.fill_color(context, (point8, point_E, point_H, point7), self.context_value_cost_config.side_walk,
                        to_replace=self.context_value_cost_config.unknown)

        # Draw zebra walks
        point1, point2, point3, point4, point5, point6, point7, point8 = self.get_outer_8_points(
                        junction_area, k_x, k_y, width_zebra)
        self.fill_color(context, (point1, point2, point_b, point_a), self.context_value_cost_config.zebra_crossing)
        self.fill_color(context, (point_b, point3, point4, point_c), self.context_value_cost_config.zebra_crossing)
        self.fill_color(context, (point_d, point_c, point5, point6), self.context_value_cost_config.zebra_crossing)
        self.fill_color(context, (point8, point_a, point_d, point7), self.context_value_cost_config.zebra_crossing)

        # Draw obstacles
        for obs in range(self.map_config.n_obstacle):
            size = self.map_config.obstacle_size // 2
            x = random.randint(self._width/4, self._width*3/4)
            y = random.randint(self._height/4, self._height*3/4)
            pos = ((x - size, y-size), (x - size, y + size), (x + size, y + size), (x + size, y - size))
            self.fill_color(context, pos, self.context_value_cost_config.static_obstacles)

        # save map jpg
        self.visual_context(context, dump_context_full_path)

        # get trajectory with noise
        trajectory, traj_info = self.get_traj(context, self.context_value_cost_config, num=traj_num)
        self.write_traj(trajectory, dump_traj_full_path)

    def generate_all(self, amount=1, traj_num=1, start=1):
        """
        invoke the context generator and trajectory generator to generate (context, trajectory) pair

        :param amount: the number of map to generate, 1 map as default
        :param traj_num: number of trajectory per map, 1 path as default
        :param start: # map to start, 1 as default
        """
        print "generating maps..."

        start_time = datetime.datetime.now()
        batch_start_time = start_time
        print "time stamp: " + str(start_time)

        for i in range(amount):
            print "\nmaps " + str(i + start) + "/" + str(amount + start) + " in progress..."
            name = str(i + start)

            # configure file name
            context_name = name + '_context.jpg'
            traj_name = name + '_traj.csv'

            # generate map with trajectory
            self.generate_context_traj(context_name, traj_name, traj_num)

            # get time taken
            if (i + 1) % 10 == 0:
                batch_end_time = datetime.datetime.now()
                print "\ntime taken: " + str(batch_end_time - start_time) + "\n"
                #batch_start_time = batch_end_time

        print str(i+1) + " maps with trajectory generated. \nFinished."
        print "Total time taken: " + str(datetime.datetime.now() - start_time)

    def fill_color(self, context, rectangle, color, to_replace=-1):
        """
        fill in the area of the rectangle with the color value, or only replace the specified
        pixel value with the color in the rectangle if specified

        :param context: context
        :param rectangle: a list of 4 coordinates of tuple (x, y) that defines the rectangle clock-wisely
        :param color: context color integer
        :param to_replace: context value to replace or -1
        """
        for i in range(context.shape[0]):
            for j in range(context.shape[1]):
                point = (i, j)
                if self.is_inside(point, rectangle) and \
                        (to_replace < 0 or context[i][j] == to_replace):
                        context[i][j] = color

    def is_inside(self, p, rectangle):
        """
        determine if the point is inside the rectangle
        :param p: a point, it is a tuple (x, y)
        :param rectangle: a list of 4 coordinates of tuple (x, y) that defines the rectangle clock-wisely
        :return: true if a point is in the rectangle, false otherwise
        """
        # a point is inside a parallelogram if the point is at right hand side of the four side
        return self.get_cross(rectangle[0], rectangle[1], p) <= 0 and \
               self.get_cross(rectangle[1], rectangle[2], p) <= 0 and \
               self.get_cross(rectangle[2], rectangle[3], p) <= 0 and \
               self.get_cross(rectangle[3], rectangle[0], p) <= 0

    def get_cross(self, a, b, p):
        """
        get the cross product of point |a b| x |b p|

        :param a, b, p: point, it is a tuple (x, y)
        :return: cross product |a b| x |b p|
        """
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

    def get_inter(self, p1, k1, p2, k2):
        """
        get the intersection point of two lines defined by a point on the line and the slope

        :return: intersection point, it is a tuple (x, y)
        """
        b1 = p1[1] - k1 * p1[0]
        b2 = p2[1] - k2 * p2[0]
        return self.get_inter_(k1, b1, k2, b2)

    def get_inter_(self, k1, b1, k2, b2):
        """
        get the interseciton point of two lines defined their slopes and y-intersects

        :return: intersection point, it is a tuple (x, y)
        """
        x = (b2 - b1) / (k1 - k2)
        y = k1 * x + b1
        point = (x, y)

        return point

    def get_outer_8_points(self, junction_points, k1, k2, width):
        """
        get the eight external points that define the  four zebra crossings or side walks in the context

        :param junction_points: a list of four points of the form tuple (x, y) that define the intersection points
        :param k1: slope at x direction
        :param k2: slope at y direction
        :param width: width of the zebra crossing or side walk
        :return: eight points clock-wisely. each is a tuple (x, y)
        """
        a, b, c, d = junction_points[0], junction_points[1], junction_points[2], junction_points[3]
        delta_d_x = abs(width / math.cos(math.atan(k1)))
        delta_d_y = abs(width / math.cos(math.atan(k2)))
        b1 = b[1] - k1 * b[0] + delta_d_x * k1 / abs(k1)   # y-intersect after line is shifted
        b2 = b[1] - k2 * b[0] + delta_d_y
        b3 = d[1] - k1 * d[0] - delta_d_x * k1 / abs(k1)
        b4 = d[1] - k2 * d[0] - delta_d_y

        point1 = self.get_inter_(k1, b1, k2, a[1] - k2 * a[0])
        point2 = self.get_inter_(k1, b1, k2, b[1] - k2 * b[0])
        point3 = self.get_inter_(k2, b2, k1, b[1] - k1 * b[0])
        point4 = self.get_inter_(k2, b2, k1, c[1] - k1 * c[0])
        point5 = self.get_inter_(k1, b3, k2, c[1] - k2 * c[0])
        point6 = self.get_inter_(k1, b3, k2, d[1] - k2 * d[0])
        point7 = self.get_inter_(k2, b4, k1, d[1] - k1 * d[0])
        point8 = self.get_inter_(k2, b4, k1, a[1] - k1 * a[0])
        return point1, point2, point3, point4, point5, point6, point7, point8

    def get_points_on_line(self, p1, p2, num=3, noise=32):
        """
        get points on the line of equal distance defined by p1 and p2
        :param p1: point 1. it is a tuple (x, y)
        :param p2: point 2. it is a typle (x, y)
        :param num: number of points on the line of equal distance to get
        :param noise: pixel noise to add to additional points, generated as random float in range [-noise, noise]
        :return: a list of additional points, each is a tuple (x, y)
        """

        x1, y1 = p1
        x2, y2 = p2
        points_between = []
        # theta = abs(math.atan((y2- y1) / (x2 - x1))) if x2 != x1 else math.asin(1)

        # d = math.sqrt((x2- x1)**2 + (y2 - y1)**2) / (num+1)     # length of each segment after adding in-between points
        #delta_x, delta_y = d * math.cos(theta), d * math.sin(theta)
        delta_x, delta_y = (x2 - x1) / (num + 1), (y2 - y1) / (num + 1)

        for i in range(num):
            x, y = x1 + delta_x * (i + 1), y1 + delta_y * (i + 1)   # new points coordinates
            x += random.uniform(-noise, noise)  # add noise to x
            y += random.uniform(-noise, noise)  # add noise to y
            points_between.append((x, y))

        return points_between

    def visual_context(self, context, dump_context_full_path):
        """
        dump context map into local image, and map_info to txt file

        :param context: context numpy matrix
        :param map_info: 8 external points defining side walks
        :param dump_context_full_path: path for context jpg
        """
        cv2.imwrite(dump_context_full_path, context)
        print "Map generated to " + dump_context_full_path

    def get_traj(self, context_map, args, num=1):
        """
        get random trajectory for the map

        :param context_map: map matrix
        :param args: context value and cost configuration parameters
        :param num: number of trajectory to generate
        :return: trajectory, a list of trajectories Y={y1, y2, ...}, each trajectory yi is
                 a list of pixel position tuple yi = {(xi1, yi1), (xi2, yi2), ...}
                 trajectory information, tuple (uuid, g value, h value) for each node in each trajectory
        """
        trajectory_per_map = []
        traj_info_per_map = []
        for i in range(num):
            print "       generating trajectory " + str(i) + "    ... "
            # build astar map from map
            a_star_generator = AStarSearch(context_map, args)

            # get node ids for each class
            zebra_crossing_nodes = a_star_generator.node_collection['zebra_crossing']
            side_walk_nodes = a_star_generator.node_collection['side_walk']
            junction_area_nodes = a_star_generator.node_collection['junction_area']

            # specify the start point and end point for the trajectory
            while True:
                start_point = random.choice(side_walk_nodes)
                end_point = random.choice(side_walk_nodes)
                x1, y1 = a_star_generator.map2pixel(start_point) #start_point // a_star_generator.grid_cols, start_point % a_star_generator.grid_cols
                x2, y2 = a_star_generator.map2pixel(end_point) #end_point // a_star_generator.grid_cols, end_point % a_star_generator.grid_cols
                if math.sqrt((x2 - x1)**2 + (y2 - y1)**2) >= 450:
                    break
            # print start_point
            # print (x1, y1)
            # print end_point
            # print (x2, y2)

            # get trajectory
            trajectory, traj_info = a_star_generator.search((x1, y1), (x2, y2))

            # process trajectory by adding additional points with random noises
            trajectory = self.add_noise_traj(trajectory)

            trajectory_per_map.append(trajectory)
            traj_info_per_map.append(traj_info)

        return trajectory_per_map, traj_info_per_map

    def add_noise_traj(self, traj):
        """
        Add additional points for the trajectory with random noises to mimic a human path

        :param traj: trajectory, it is a list of pixel position tuple (x, y)
        :return: trajectory with noise
        """
        index = 0
        len_traj = len(traj)
        for i in range(len_traj - 1):
            p1, p2 = traj[index], traj[index+1]
            index += 1
            intense_points = self.get_points_on_line(p1, p2, num=3, noise=1)     # get in-between points for each two points

            # the pixel distance between two points is 16 or 16*sqr2
            # we add 3 additional points in between as default
            for point in intense_points:
                traj.insert(index, point)
                index += 1

        return traj

    def write_traj(self, traj, dump_traj_full_path):
        """
        write trajectory data to csv file. Each row is of form x, y. Each trajectory is seperated
        by an empty row.

        :param traj: a list of trajectory(ies) Y=[Y1, Y2,...], each trajectory is a list of pixel point of
                     the form tuple {x, y}
        :param dump_traj_full_path: path for trajectory file
        """

        with open(dump_traj_full_path, 'wb') as csvfile:
            csvwritter = csv.writer(csvfile, dialect='excel')
            csvwritter.writerow(['x', 'y'])
            for trajectory in traj:
                for i, each_point in enumerate(trajectory):
                    csvwritter.writerow([each_point[0], each_point[1]])
                csvwritter.writerow("")

        print "       " + str(len(traj)) + " trajectories saved as " + dump_traj_full_path


if __name__ == '__main__':
    n_map = 4001             # number of map
    n_traj_per_map = 1      # number of trajectory for each map
    start = 2000             # new map start from here

    # get configurations
    args = context_value_cost_config()
    args_map = deploy_args_map()

    # Generate new maps with trajectories
    context_generator = ContextGenerator('../../dataset/', args, args_map)
    context_generator.generate_all(amount=n_map, start=start)

    for i in range(10):
        # randomly visualize trajectory on map
        num = random.randint(1, start + n_map)
        map_path = '../../dataset/context/'+str(num)+'_context.jpg'
        traj_path = '../../dataset/trajectory/'+str(num)+'_traj.csv'
        im = plt.imread(map_path)
        implot = plt.imshow(im)

        df = pd.read_csv(traj_path, dtype=float)
        for row in df.values:
            if len(row) > 0:
                plt.scatter([row[1]], [row[0]])  # invert x and y for plt plot (keep x value for vertical position)

        plt.show()

    # # Visualize trajectory on map
    # map_path = '../../dataset/context/10_context.jpg'
    # traj_path = '../../dataset/trajectory/10_traj.csv'
    # im = plt.imread(map_path)
    # implot = plt.imshow(im)
    #
    # df = pd.read_csv(traj_path, dtype=float)
    # for row in df.values:
    #     if len(row) > 0:
    #         plt.scatter([row[1]], [row[0]])         # invert x and y for plt plot (keep x value at vertical position)
    #
    # plt.show()



import numpy as np
import math


class AStarNode(object):
    """
    Node definition in A star algorithm
    """
    def __init__(self, uuid, g, h, grid_type, parent_node, raw_width, raw_height, grid_size, walkable, cost, adjacent_list):
        self._uuid = uuid
        self._g = g
        self._h = h
        self._type = grid_type
        self._parent_node = parent_node
        self._raw_width = raw_width
        self._raw_height = raw_height
        self._grid_size = grid_size
        self._walkable = walkable
        self._cost = cost
        self._adjacent_list = adjacent_list

    @property
    def uuid(self):
        return self._uuid

    @property
    def g(self):
        return self._g

    @property
    def h(self):
        return self._h

    @property
    def type(self):
        return self._type

    @property
    def parent_node(self):
        return self._parent_node


    @property
    def raw_width(self):
        return self._raw_width

    @property
    def raw_height(self):
        return self._raw_height

    @property
    def grid_size(self):
        return self._grid_size

    @property
    def walkable(self):
        return self._walkable

    @property
    def cost(self):
        return self._cost

    @property
    def adjacent_list(self):
        return self._adjacent_list

    @h.setter
    def h(self, other):
        self._h = other

    @g.setter
    def g(self, other):
        self._g = other

    @parent_node.setter
    def parent_node(self, other):
        self._parent_node = other


class AStarSearch(object):
    """
    A star search algorithm

    Reference: https://blog.csdn.net/hitwhylz/article/details/23089415
    """
    def __init__(self, context, args):
        """
        initialization of the AStarSearch class

        :param context: matrix (grid) of the context, 1280 x 1280
        :param args: context and cost value configuration paramters
        """
        self._context = context
        self._start_point = None
        self._end_point = None

        self.k = args.grid_size
        self.alpha = args.alpha       # alpha constant, controlling distance in g
        self.beta = args.beta          # beta constant, controlling h
        self.grid_rows, self.grid_cols = context.shape[0] // self.k, context.shape[1] // self.k

        self._context_value_config = {
            'unknown': args.unknown,
            'zebra_crossing': args.zebra_crossing,
            'normal_lane': args.normal_lane,
            'side_walk': args.side_walk,
            'static_obstacles': args.static_obstacles,
            'junction_area': args.junction_area
        }
        self._cost_config = {
            'unknown': args.unknown_cost,
            'zebra_crossing': args.zebra_crossing_cost,
            'normal_lane': args.normal_lane_cost,
            'side_walk': args.side_walk_cost,
            'static_obstacles': args.static_obstacles_cost,
            'junction_area': args.junction_area_cost
        }

        # collect node ids to each type
        self.node_collection = {
            'unknown': [],
            'zebra_crossing': [],
            'normal_lane': [],
            'side_walk': [],
            'static_obstacles': [],
            'junction_area': []
        }
        self._grid_matrix = self.build_grid()   # build astar node grid

    def build_grid(self):
        """
        build grid based on context matrix: for calculation speed, here we rebuild the context into a
        (1280 / K) x (1280 / K) size matrix, each cell is a K X K pixels patch in original context.

        :return: grid matrix of astar nodes
        """
        # build astar grid matrix
        grid_matrix = np.empty((self.grid_rows, self.grid_cols), dtype=object)

        uuid = 0

        for row_idx in range(self.grid_rows):
            for col_idx in range(self.grid_cols):

                # slice a k x k size patch to build each astar node
                grid = self._context[self.k * row_idx: self.k * row_idx + self.k,
                                     self.k * col_idx: self.k * col_idx + self.k]

                # get grid type and cost of the patch
                grid_type, _cost = self.get_cost(grid)

                # determine if the patch is walkable
                _walkable = True if _cost < self._cost_config['unknown'] else False

                # get uuid(s) of all adjacent patch(es) of the patch
                _adjacent_list = self.get_adjacent_list(row_idx, col_idx)

                # build the astar node from the patch
                # record the uuid of the node to the list of its corresponding type
                node = AStarNode(uuid, -1, -1, grid_type, None, self.grid_cols, self.grid_rows, self.k, _walkable, _cost, _adjacent_list)
                grid_matrix[row_idx][col_idx] = node
                self.node_collection[grid_type].append(uuid)

                uuid += 1

        return grid_matrix

    def search(self, start_point, end_point):
        """
        Search the trajectory from start point to end point using the A* algorithm

        :param start_point: pixel position of the starting point, it's a tuple (x, y)
        :param end_point: pixel position of the ending point, it's a tuple (x, y)
        :return: trajectory from start point to end, it is a list of pixel position of tuple (x, y);
                 or None if no path found
        """
        self._start_point = start_point
        self._end_point = end_point

        # get the astar node that corresponds to the start point,
        # initialize its g, h value to be zero
        curr_node = self.map2node(self._start_point)
        curr_node.g, curr_node.h = 0, 0

        open_list = [curr_node]     # shopping list
        closed_list = []            # closed list for node that have been checked

        # start searching
        while len(open_list) != 0:
            curr_node = self.get_minimum(open_list)   # find the smallest node
            closed_list.append(curr_node)

            for adj_node_id in curr_node.adjacent_list:

                # get the adjacent node from its uuid
                adj_node = self._grid_matrix[adj_node_id // self.grid_cols][adj_node_id % self.grid_cols]

                # if walkable and not visited
                if adj_node.walkable and not closed_list.__contains__(adj_node):
                    # if not yet added in the open list
                    # add node to the open list and set its g, h and parent node
                    if not open_list.__contains__(adj_node):
                        open_list.append(adj_node)
                        adj_node.g, adj_node.h = self.get_g_h(curr_node, adj_node)
                        adj_node.parent_node = curr_node

                        if adj_node_id == self.map2node(self._end_point).uuid:  # reached end point, then path found and search finished
                            return self.build_path(self.map2node(self._end_point))

                    # if already in the open list,
                    # update its g, h and parent node if find a better path
                    else:
                        g, h = self.get_g_h(curr_node, adj_node)
                        if g + h <= adj_node.g + adj_node.h:            # update with better path if found
                            adj_node.g, adj_node.h = g, h
                            adj_node.parent_node = curr_node

            open_list.remove(curr_node)

        return None

    def build_path(self, end_node):
        """
        Trace back the parent node of the end node to build the trajectory

        :param end_node: astar node
        :return node_path: trajectory, a list of the points of the form tuple (x, y)
        :return node_path_id: trajectory information, a list of tuple (uuid, g value, h value)
        """
        node_path = []
        node_path_id = []

        while end_node is not None:
            x, y = self.map2pixel(end_node.uuid)
            node_path.insert(0, (x, y))
            node_path_id.insert(0, (end_node.uuid, end_node.g, end_node.h))
            end_node = end_node.parent_node

        return node_path, node_path_id

    def get_cost(self, grid):
        """
        get the cost of the k x k grid (default k=16).
        The type of the grid (zebra crossing, normal lane, side walk, etc) is determined
        by the most commonly appear context value in the grid

        :param grid: k x k size matrix
        :return: grid type and the cost value
        """
        grid_cell = grid.reshape((self.k * self.k))

        # get the most common context value in the grid
        most_common_value = np.bincount(grid_cell).argmax()

        # get the grid type from the context value
        grid_type = self._cost_config.keys()[self._context_value_config.values().index(most_common_value)]

        return grid_type, self._cost_config[grid_type]

    def get_adjacent_list(self, x, y):
        """
        Get adjacent node ids of the node in the context

        :param x: row
        :param y: column
        :return: a list of node ids
        """
        this_node = x*self.grid_rows + y
        adj_list = [this_node]
        col_prev = y - 1 if y > 0 else y
        col_next = y + 1 if y < self.grid_cols-1 else y
        row_prev = x - 1 if x > 0 else x
        row_next = x + 1 if x < self.grid_cols-1 else x

        adj_list.append(row_prev * self.grid_rows + col_prev)
        adj_list.append(row_prev * self.grid_rows + y)
        adj_list.append(row_prev * self.grid_rows + col_next)
        adj_list.append(x * self.grid_rows + col_prev)
        adj_list.append(x * self.grid_rows + col_next)
        adj_list.append(row_next * self.grid_rows + col_prev)
        adj_list.append(row_next * self.grid_rows + y)
        adj_list.append(row_next * self.grid_rows + col_next)

        while adj_list.__contains__(this_node):   # remove the center node to leave only its adjacencies
            adj_list.remove(this_node)

        return list(set(adj_list))

    def get_g_h(self, curr_node, adj_node):
        """
        get the g and h value of the adj_node walking from the curr_node

        :param curr_node: astar node
        :param adj_node: astar node
        :return: g and h value
        """
        id_delta = abs(adj_node.uuid - curr_node.uuid)

        # distance from current node to adj node
        distance = self.k if id_delta == 1 or id_delta == self.grid_cols else self.k * math.sqrt(2)
        distance += curr_node.g
        g = adj_node.cost + distance * self.alpha           # g value

        # pixel position of the adj node and end point
        x, y = self.map2pixel(adj_node.uuid)
        x_, y_ = self.map2pixel(self.map2node(self._end_point).uuid)
        man_h = self.beta * (abs(x_ - x) + abs(y_ - y))      # manhatann distance

        return g, man_h

    def get_minimum(self, shopping_list):
        """
        Find the astar node of the smallest F value in the list

        :param shopping_list: a list of astar nodes
        :param shopping_list: a list of astar nodes
        :return: the astar node of the smallest F
        """
        smallest_node = shopping_list[0]

        for x in shopping_list:
            if x.g + x.h <= smallest_node.g + smallest_node.h:
                smallest_node = x

        return smallest_node

    def map2pixel(self, uuid):
        """
        map the astar_node uuid (grid cell) into the original context pixel position

        :param astar_node: astar node uuid
        :return: center position of the grid cell in raw context matrix, it is a (x,y) tuple
        """
        x = uuid // self.grid_cols
        y = uuid % self.grid_cols
        pixel_pos = (self.k / 2 + x * self.k, self.k / 2 + y * self.k)

        return pixel_pos

    def map2node(self, pixel_pos):
        """
        map the pixel position to astar node

        :param pixel: grid cell position tuple (x, y), x is vertical direction
        :return: astar_node
        """

        x = pixel_pos[0] // self.k
        y = pixel_pos[1] // self.k
        return self._grid_matrix[x][y]










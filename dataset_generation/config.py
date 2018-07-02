import argparse


def deploy_args_map():
    """
    configure the deploy arguments
    :return: args
    """
    parser = argparse.ArgumentParser()

    # width of the context map
    parser.add_argument('--width', type=int, default=1280,
                        help='width of the context map')

    # height of the context map
    parser.add_argument('--height', type=int, default=1280,
                        help='height of the context map')

    # x offset on the map, defining the position of the  vertical normal lane
    parser.add_argument('--offset_x', type=int, default=600,
                        help='vertical normal lane offset in the map')

    # y offset on the map, defining the position of the horizon normal lane
    parser.add_argument('--offset_y', type=int, default=700,
                        help='horizontal normal lane offset in the map')

    # the slope of the vertical normal lane
    #k_x = (random.uniform(1.5, 10), random.uniform(-5, -10))[random.randint(0, 1)]
    #parser.add_argument('--k_x', type=float,
    #                    default=k_x,
    #                    help='slope of the vertical normal lane, in the range of (1.5, 10) && (-5, -10)')

    # the slope of the horizon normal lane
    #k_y = random.uniform(-0.35, 0.55)
   # parser.add_argument('--k_y', type=float,
    #                    default=k_y,
    #                    help='slope of the horizon normal lane, '
    #                         'randomly generated in the range of (-0.35, 0.55)')

    # width of the vertical normal lane
    #width_x = random.uniform(200, 400)
    #parser.add_argument('--width_x', type=float, default=width_x,
     #                   help='width of the vertical normal lane, '
     #                        'randomly generated in the range of (200, 400) pixel value')

    # width of the horizon normal lane
    #width_y = random.uniform(200, 400)
    #parser.add_argument('--width_y', type=float, default=width_y,
    #                    help='width of the horizon normal lane, '
     #                        'randomly generated in the range of (200, 400) pixel value')

    # width of the side walk
    #width_side = random.uniform(50, 120)
   # parser.add_argument('--width_side_walk', type=float, default=width_side,
    #                    help='width of the side walk, randomly generated in the range of (50, 120) pixel value')

    # width of the zebra crossing
   # width_zebra = random.uniform(70, 120)
    #parser.add_argument('--width_zebra', type=float, default=width_zebra,
    #                    help='width of the zebra crossing, randomly generated in the range of (70, 120) pixel value')

    # number of static obstacles on the map
    parser.add_argument('--n_obstacle', type=int, default=2,
                        help='number of static obstacle(s) on the map')

    # size of the square static obstacle
    parser.add_argument('--obstacle_size', type=int, default=60,
                        help='pixel size of the static obstacle')

    args = parser.parse_args()

    return args


def context_value_cost_config():
    """
        configure the context value and cost value for different context type in the map
        :return: args
    """
    parser = argparse.ArgumentParser()

    # size of the grid cell as an astar node
    parser.add_argument('--grid_size', type=int, default=16,
                        help='size of the grid cell as an astar node')

    # parameter alpha annealed to distance in g = cost + distance for astar node
    parser.add_argument('--alpha', type=float, default=1,
                        help='alpha parameter annealed to distance in g = cost + distance')

    # paramter beta annealed to h for astar node
    parser.add_argument('--beta', type=float, default=1,
                        help='beta parameter annealed to manhattan distance h')

    # pixel value for unknown area
    parser.add_argument('--unknown', type=int, default=0,
                        help='pixel value for unknown context area')

    # pixel value for zebra crossing area
    parser.add_argument('--zebra_crossing', type=int, default=50,
                        help='pixel value for zebra crossing context area')

    # pixel value for normal lane area
    parser.add_argument('--normal_lane', type=int, default=100,
                        help='pixel value for normal lane context area')

    # pixel value for side walk area
    parser.add_argument('--side_walk', type=int, default=150,
                        help='pixel value for side walk context area')

    # pixel value for static obstacle area
    parser.add_argument('--static_obstacles', type=int, default=200,
                        help='pixel value for obstacle context area')

    # pixel value for junction area
    parser.add_argument('--junction_area', type=int, default=250,
                        help='pixel value for junction context area')

    # cost value for unknown area
    parser.add_argument('--unknown_cost', type=int, default=200,
                        help='cost value for unknown context area')

    # cost value for zebra crossing area
    parser.add_argument('--zebra_crossing_cost', type=int, default=12,
                        help='cost value for zebra crossing area')

    # cost value for normal lane area
    parser.add_argument('--normal_lane_cost', type=int, default=100,
                        help='cost value for normal lane area')

    # cost value for side walk area
    parser.add_argument('--side_walk_cost', type=int, default=10,
                        help='cost value for side walk area')

    # cost value for static obstacle area
    parser.add_argument('--static_obstacles_cost', type=int, default=200,
                        help='cost value for static obstacle area')

    # cost value for junction area
    parser.add_argument('--junction_area_cost', type=int, default=100,
                        help='cost value for junction area')

    args = parser.parse_args()

    return args

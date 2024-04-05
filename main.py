import numpy as np
import matplotlib.pyplot as plt


def plot_points_and_lines(hull, points):

    x = [point[0] for point in hull]
    y = [point[1] for point in hull]
    plt.plot(x, y, 'bo')

    x_all = [point[0] for point in points]
    y_all = [point[1] for point in points]
    plt.plot(x_all, y_all, 'bo')

    for i in range(len(hull)):
        plt.plot([hull[i][0], hull[(i+1) % len(hull)][0]], [hull[i][1], hull[(i+1) % len(hull)][1]], 'r-')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)

    plt.show()


def plot_triangles(triangles):

    plt.figure()

    for triangle in triangles:
        x = [point[0] for point in triangle + [triangle[0]]]
        y = [point[1] for point in triangle + [triangle[0]]]
        plt.plot(x, y, 'b-')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)

    plt.show()


def generate_random_points(num_points, max_coord, min_coord):
    return np.random.rand(num_points, 2) * (max_coord - min_coord) + min_coord


def find_leftmost_lowest_point(points):

    min_x_index = np.argmin(points[:, 0])

    points_with_min_x = points[points[:, 0] == points[min_x_index, 0]]

    leftmost_lowest_index = np.argmin(points_with_min_x[:, 1])

    return np.where((points == points_with_min_x[leftmost_lowest_index]).all(axis=1))[0][0]


def rotate(a, b, c):
    return (b[0] - a[0])*(c[1] - b[1]) - (b[1] - a[1])*(c[0] - b[0])


def jarvis(points):
    hull = []
    min_point_idx = find_leftmost_lowest_point(points)
    indexes = list(range(points.shape[0]))
    indexes.remove(min_point_idx)
    indexes.append(min_point_idx)
    hull.append(min_point_idx)

    while True:
        right = indexes[0]
        for i in indexes:
            if rotate(points[hull[-1]], points[right], points[i]) < 0:
                right = i
        if right == min_point_idx:
            break
        else:
            hull.append(right)
            indexes.remove(right)

    return points[hull]


def generate_random_point_in_triangle(triangle):
    a, b, c = triangle
    point = np.random.rand(2)
    if point[0] + point[1] > 1:
        point[0], point[1] = 1 - point[0], 1 - point[1]

    v_1, v_2 = b - a, c - a
    A = np.vstack((v_1, v_2))
    point = np.dot(point, A) + a
    return point


def area_of_triangle(triangle):
    a, b, c = triangle
    return abs(((b[0] - a[0])*(c[1] - b[1]) - (b[1] - a[1])*(c[0] - b[0])) / 2)


def triangulate(hull):
    triangles = []
    centroid_point = np.mean(hull, axis=0)
    for i in range(len(hull)):
        triangles.append([centroid_point, hull[i], hull[(i + 1) % len(hull)]])
    return np.array(triangles)


def calculate_probs(triangles):
    areas = np.array([area_of_triangle(triangle) for triangle in triangles])
    probs = areas / np.sum(areas)
    return probs


def get_random_triangle_idx(probs):
    return np.random.choice(np.array(range(probs.size)), p=probs)


#np.random.seed(454223)
points = generate_random_points(1000, 30, -50)
hull = jarvis(points)
triangles = triangulate(hull)
probs = calculate_probs(triangles)

uniform_points = []
for i in range(10000):
    idx = get_random_triangle_idx(probs)
    point = generate_random_point_in_triangle(triangles[idx])
    uniform_points.append(point)

plot_points_and_lines(hull, uniform_points)
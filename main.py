#!/usr/bin/env python3

import random
from typing import Sized
import subprocess

import matplotlib as mpl
import numpy as np
import svgwrite
from PIL import Image
from scipy.spatial import Delaunay, KDTree

mpl.use('TkAgg')


TL_CODE = '#2f6dbd'
TR_CODE = '#89bafa'
BL_CODE = '#fc923a'
BR_CODE = '#ffc99c'


def colorcode2rgb(code: str) -> np.ndarray:
    code = code.lstrip('#')
    assert len(code) == 6, 'Invalid color code'
    arr = np.zeros((3, 1))
    for i in range(0, len(code), 2):
        arr[i//2, 0] = int(code[i:i+2], 16)
    return arr


class ColorMap:
    def __init__(self, tl_code: str, tr_code: str, bl_code: str, br_code: str):
        self.tl = colorcode2rgb(tl_code)
        self.tr = colorcode2rgb(tr_code)
        self.bl = colorcode2rgb(bl_code)
        self.br = colorcode2rgb(br_code)

    def get_code(self, x: float, y: float) -> np.ndarray:
        return (
            x * y * self.tl + (1 - x) * y * self.tr +
            x * (1 - y) * self.bl + (1 - x) * (1 - y) * self.br
        ).astype(np.uint8)

# looks nicenp.ndarray
def draw_image() -> None:
    # h, w
    img_size = (100, 100)

    cmap = ColorMap(TL_CODE, TR_CODE, BL_CODE, BR_CODE)
    # img = Image.new('RGB', img_size, (255, 255, 255))
    img_arr = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    for i in range(img_size[0]):
        fi = i / img_size[0]
        for j in range(img_size[1]):
            fj = j / img_size[1]
            pixel = cmap.get_code(fi, fj).flatten().tolist()
            img_arr[i,j] = pixel

    img = Image.fromarray(img_arr)
    img.save('output/colored_square.png')


def gen_points(
        w: float, h: float, density: float,
        circ_x: float, circ_y: float, circ_r: float
) -> list[tuple[float, float]]:
    """
    creates a list of points in the range from 0 to w and 0 to h, sparing a defined circle
    :param w: width of the image
    :param h: height of the image
    :param density: avg. amount of points per area
    :param circ_x: x coord of center of the circle to spare
    :param circ_y: y coord of center of the circle to spare
    :param circ_r: radius of the circle to spare
    :return: list of the created points (x, y)
    """
    n_points = int(w * h * density)
    point_list = list()

    for i in range(n_points):
        p = (
            random.random() * w,
            random.random() * h
        )

        if (p[0] - circ_x)**2 + (p[1] - circ_y)**2 < circ_r**2:
            continue

        point_list.append(p)

    return point_list


def gen_circle_points(circ_x: float, circ_y: float, circ_r: float, n_points: int) -> list[tuple[float, float]]:
    point_list = list()

    for i in range(n_points):
        ang = 2 * np.pi * i / n_points
        x = circ_x + circ_r * np.cos(ang)
        y = circ_y + circ_r * np.sin(ang)
        point_list.append((x, y))

    return point_list

def gen_square_points(w: float, h: float, density: float, min_dist: float) -> list[tuple[float, float]]:
    """
    Place points randomly on the edge of a rectangle
    :param w: width of the rectangle
    :param h: height of the rectangle
    :param density: points per len unit
    :param min_dist: minimum distance between two points
    :return: list of the created points (x, y)
    """

    point_list = list()
    point_list += [(0, random.random() * h) for _ in range(int(density * h))]
    point_list += [(w, random.random() * h) for _ in range(int(density * h))]
    point_list += [(random.random() * w, 0) for _ in range(int(density * w))]
    point_list += [(random.random() * w, h) for _ in range(int(density * w))]

    corner_points = [(0, 0), (w, 0), (0, h), (w, h)]

    for i in range(len(point_list)-1, 0, -1):
        if (point_list[i][0] - point_list[i-1][0])**2 + (point_list[i][1] - point_list[i-1][1])**2 < min_dist**2:
            point_list.pop(i)
            break

        for point in corner_points:
            if (point[0] - point_list[i][0])**2 + (point[1] - point_list[i][1])**2 < min_dist**2:
                point_list.pop(i)
                break

    point_list += corner_points

    return point_list


def create_svg(
        simplices: Sized, pt_arr: np.ndarray, colmap: ColorMap,
        w: float, h: float, output_file="output/triangles.svg"
) -> None:
    """
    Generate an SVG file with triangles based on the triangulation.
    :param simplices: list of simplices (triangle corners) to visualize
    :param pt_arr: Array of points used for triangulation
    :param colmap: ColorMap object for computing triangle colors
    :param w: Width of the canvas
    :param h: Height of the canvas
    :param output_file: Path to save the output SVG file
    """

    scale = 100
    h_color_range = 0.3

    dwg = svgwrite.Drawing(output_file, profile='tiny', size=(f"{w*scale}", f"{h*scale}"))
    for simplex in simplices:
        pts = [pt_arr[simplex[i]] for i in range(3)]

        avg_x = np.mean([p[0] / w for p in pts])
        avg_y = np.mean([p[1] / h for p in pts])
        y_min = max(avg_y - h_color_range/2, 0.)
        y_max = min(avg_y + h_color_range/2, 1.)
        y_val = random.uniform(y_min, y_max)
        x_val = random.random()

        color = colmap.get_code(x_val, y_val).flatten()
        hex_color = f"#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}"
        dwg.add(dwg.polygon([(p[0]*scale, p[1]*scale) for p in pts], fill=hex_color, stroke="white", stroke_width=2))
    dwg.save()


def main():
    w = 5.
    h = 30.
    cx = w / 2
    cy = 6.
    cr = 1.75
    min_dist = 0.5
    thick = 0.1

    # get randomized points (without circle
    pts = gen_points(w, h, 6, cx, cy, cr)

    # draw circle and square around everything
    static_points = list()
    static_points += gen_circle_points(cx, cy, cr, 12)
    static_points += gen_square_points(w, h, 2, min_dist)

    # find all points w/ neighbours closer than min_dist
    tree = KDTree(np.array(static_points + pts))
    pairs = tree.query_pairs(min_dist)
    rm_index_set = set()
    for (i, j) in pairs:
        if i in rm_index_set or j in rm_index_set:
            continue
        if i >= len(static_points):
            rm_index_set.add(i - len(static_points))
            continue
        if j >= len(static_points):
            rm_index_set.add(j - len(static_points))
            continue

    # remove all points having too close neighbours
    rm_indices = list(rm_index_set)
    rm_indices.sort(reverse=True)
    for i in rm_indices:
        pts.pop(i)

    pts += static_points
    pt_arr = np.array(pts)

    # triangulate points
    tri = Delaunay(pt_arr)

    # filter the skipped circle    
    simplices = list()
    mid_p = np.array([cx, cy])
    for simplex in tri.simplices:
        skip = True
        for i in range(3):
            distance = np.linalg.norm(mid_p - pt_arr[simplex[i]])
            if distance > cr + 1e-5:
                skip = False
                break

        if skip:
            continue

        simplices.append(simplex)

    # generate SVG file
    svg_path = "output/triangles.svg"
    colmap = ColorMap(TL_CODE, TR_CODE, BL_CODE, BR_CODE)
    create_svg(simplices, pt_arr, colmap, w, h, svg_path)

    # call inkscape to convert the SVG into a PDF to use it in latex
    pdf_path = "output/triangles.pdf"
    try:
        subprocess.run(["inkscape", svg_path, "--export-type=pdf", "--export-filename", pdf_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting SVG to PDF: {e}")
    


# def main() -> None:
#     draw_image()


if __name__ == '__main__':
    main()

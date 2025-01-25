#!/usr/bin/env python3
from __future__ import annotations

import random
import subprocess
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import svgwrite
from PIL import Image
from scipy.spatial import Delaunay, KDTree

from areas import Area, Circle, gen_octaeda, ConvexPolygon

# mpl.use('TkAgg')


TL_CODE = '#ffc99c'
# TR_CODE = '#fc923a'
TR_CODE = '#ff831c'
BL_CODE = '#89bafa'
# BL_CODE = '#94e2ff'
BR_CODE = '#2f6dbd'
A4_W_CM = 21.0
A4_H_CM = 29.7


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
            (1 - x) * (1 - y) * self.tl + x * (1 - y) * self.tr +
            (1 - x) * y * self.bl + x * y * self.br
        ).astype(np.uint8)

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
            # i for images is line, j is column
            pixel = cmap.get_code(fj, fi).flatten().tolist()
            img_arr[i,j] = pixel

    img = Image.fromarray(img_arr)
    import os
    os.makedirs('output', exist_ok=True)
    img.save('output/colored_square.png')


def scatter_points(
        w: float, h: float, density: float
) -> list[tuple[float, float]]:
    """
    creates a list of points in the range from 0 to w and 0 to h, sparing a defined circle
    :param w: width of the image
    :param h: height of the image
    :param density: avg. amount of points per unit area
    :return: list of the created points (x, y)
    """
    n_points = int(w * h * density)
    point_list = list()

    for i in range(n_points):
        p = (
            random.random() * w,
            random.random() * h
        )

        point_list.append(p)

    return point_list


def create_svg(
        simplices: list, pt_arr: np.ndarray, colmap: ColorMap,
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

    os.makedirs('output', exist_ok=True)
    dwg.save()


def main():
    w = (A4_W_CM - 2 * 0.3) * 0.3
    h = A4_H_CM - 2 * 0.3
    cen_x = w / 2
    circ_y = 4.5
    circ_r = 2.5
    rect_margin = 0.25
    rect1_h = 0.5
    rect1_y = circ_y + circ_r + rect1_h + 1
    rect1_w = 2 * (circ_r - rect_margin)
    rect2_y = A4_H_CM - 2.
    rect2_w = 2 * (circ_r - rect_margin)
    rect2_h = 1.5
    min_dist = 1.5

    print('generating points')

    # get randomized points
    pts = scatter_points(w, h, 6)

    circ = Circle(cen_x, circ_y, circ_r, 0.15, 'circle')
    rect1 = gen_octaeda(cen_x, rect1_y, rect1_w, rect1_h, rect_margin, 0.6, 'rect1')
    rect2 = gen_octaeda(cen_x, rect2_y, rect2_w, rect2_h, rect_margin, 0.6, 'rect2')
    areas = [circ, rect1, rect2]

    n_filtered = 0
    for area in areas:
        print(f'filtering points for area {area.name}')

        for point_index in range(len(pts), 0, -1) :
            point = pts[point_index-1]
            if area.contains_point(point):
                n_filtered += 1
                pts.pop(point_index-1)
                continue

    print(f'filtered {n_filtered} points')

    # draw circle and square around everything
    static_points = list()
    static_points += ConvexPolygon([(0, 0), (w, 0), (w, h), (0, h)], 'frame')\
                        .gen_points(0.75, min_dist * 0.5)
    static_points += circ.gen_points(12)
    static_points += rect1.gen_points(0.75, min_dist * 0.2)
    static_points += rect2.gen_points(0.75, min_dist * 0.2)

    # find all points w/ neighbours closer than min_dist
    tree = KDTree(np.array(static_points + pts))
    pairs = tree.query_pairs(min_dist)
    rm_index_set = set()
    for (i, j) in pairs:
        if i in rm_index_set or j in rm_index_set:
            continue
        if i >= len(static_points):
            rm_index_set.add(i - len(static_points))
            # continue
        if j >= len(static_points):
            rm_index_set.add(j - len(static_points))
            # continue

    # remove all points having too close neighbours
    rm_indices = list(rm_index_set)
    rm_indices.sort(reverse=True)
    for i in rm_indices:
        pts.pop(i)

    pts = static_points + pts
    pt_arr = np.array(pts)

    # triangulate points
    tri = Delaunay(pt_arr)

    # filter the skipped circle
    simplices = list(tri.simplices)
    n_points_touching_polygons = 0
    n_edges_crossing_polygons = 0
    for area in areas:
        print(f'filtering simplices for area {area.name}')
        for i_simplex in range(len(simplices), 0, -1):
            simplex = simplices[i_simplex-1]
            keep = False

            n_outside_edges = 0
            for i in range(3):
                point = pt_arr[simplex[i]]

                if not area.contains_point_with_margin(point, 1e-7):
                    n_points_touching_polygons += 1
                    keep = True

            if keep:
                continue

            simplices.pop(i_simplex-1)

    print(f'n_points_touching_polygons: {n_points_touching_polygons}')
    print(f'n_edges_crossing_polygons: {n_edges_crossing_polygons}')

    # plt.plot([t[0] for t in static_points], [t[1] for t in static_points], 'x')
    # plt.plot([t[0] for t in rect1.points], [t[1] for t in rect1.points])
    # plt.show()

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
    


if __name__ == '__main__':
    draw_image()
    main()


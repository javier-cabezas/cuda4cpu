# CUDA for CPU allows you to compile and execute CUDA kernels on CPUs.
#
# Copyright (C) 2014 Javier Cabezas
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Square cuda4cpu logo: the wordmark stacked as "cuda" over "4cpu", in
Ubuntu Bold as outlines, with the social preview's three-color bar, on a
transparent background. The SVG switches colors with the browser's scheme;
the PNG, which can't, uses colors readable on light and dark pages.
Needs fontTools, Pillow and the Ubuntu font (fonts-ubuntu). Usage:

    python3 docs/images/make-logo.py docs/images/logo.svg docs/images/logo.png 512 [PREVIEW_PREFIX]"""
import sys
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
from fontTools.pens.recordingPen import DecomposingRecordingPen
from PIL import Image, ImageDraw, ImageChops

FONT = '/usr/share/fonts/truetype/ubuntu/Ubuntu[wdth,wght].ttf'
GREEN, DARK, BLUE = 'green', 'dark', 'blue'          # the colors' roles
LIGHT = {GREEN: '#609700', DARK: '#24292f', BLUE: '#005aa0'}       # the README logo's, the social preview's
DARK_SCHEME = {GREEN: '#8cc83c', DARK: '#e6edf3', BLUE: '#4aa3ff'}
# Readable on both, at least 3:1 against white and #0d1117 (WCAG, large text)
BOTH = {GREEN: '#609700', DARK: '#7d8590', BLUE: '#2a7fd4'}
SIZE = 512                      # the square, in SVG units
BAR = 20                        # the bar's height
WIDTH = 0.76                    # the widest line's share of the square
LEADING = 0.86                  # baseline to baseline, in ems

out_svg, out_png, png_size = sys.argv[1], sys.argv[2], int(sys.argv[3])
preview = sys.argv[4] if len(sys.argv) > 4 else None

font = instancer.instantiateVariableFont(TTFont(FONT), {'wght': 700, 'wdth': 100})
cmap, glyphs, hmtx = font.getBestCmap(), font.getGlyphSet(), font['hmtx']
upem = font['head'].unitsPerEm


def contours(char):
    """The glyph's contours, as lists of points (quadratic curves flattened),
    in font units, and its advance"""
    name = cmap[ord(char)]
    pen = DecomposingRecordingPen(glyphs)
    glyphs[name].draw(pen)
    result, current = [], []
    for op, args in pen.value:
        if op == 'moveTo':
            current = [args[0]]
        elif op == 'lineTo':
            current.append(args[0])
        elif op == 'qCurveTo':
            # TrueType: implied on-curve points between consecutive off-curve ones
            start = current[-1]
            offs, end = list(args[:-1]), args[-1]
            points = [start]
            for i, off in enumerate(offs):
                nxt = end if i == len(offs) - 1 else ((off[0] + offs[i + 1][0]) / 2, (off[1] + offs[i + 1][1]) / 2)
                p0 = points[-1]
                for s in range(1, 9):
                    t = s / 8
                    points.append(((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * off[0] + t * t * nxt[0],
                                   (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * off[1] + t * t * nxt[1]))
            current.extend(points[1:])
        elif op == 'curveTo':
            p0, (c1, c2, p3) = current[-1], args
            for s in range(1, 13):
                t = s / 12
                current.append(tuple((1 - t) ** 3 * p0[k] + 3 * (1 - t) ** 2 * t * c1[k] + 3 * (1 - t) * t * t * c2[k]
                                     + t ** 3 * p3[k] for k in range(2)))
        elif op in ('closePath', 'endPath'):
            if current:
                result.append(current)
            current = []
    return result, hmtx[name][0]


def line(text):
    """The line's glyphs as (color, contours offset by their x), its width and
    vertical extent, in font units"""
    x, parts, ymin, ymax = 0, [], 0, 0
    for char, color in text:
        cs, advance = contours(char)
        parts.append((color, [[(px + x, py) for px, py in c] for c in cs]))
        for c in cs:
            ymin, ymax = min(ymin, *(p[1] for p in c)), max(ymax, *(p[1] for p in c))
        x += advance
    return parts, x, ymin, ymax


lines = [line([(c, GREEN) for c in 'cuda']),
         line([('4', DARK)] + [(c, BLUE) for c in 'cpu'])]
scale = WIDTH * SIZE / max(width for _, width, _, _ in lines)
# The block's ink: from the first line's top to the second's bottom (its p)
top = lines[0][3]
bottom = lines[1][2] - LEADING * upem
height = (top - bottom) * scale
area_top = BAR
y0 = area_top + (SIZE - area_top - height) / 2 + top * scale   # first baseline

shapes = []                     # (color, contours in SVG units)
for i, (parts, width, _, _) in enumerate(lines):
    x0 = (SIZE - width * scale) / 2
    baseline = y0 + i * LEADING * upem * scale
    for color, cs in parts:
        shapes.append((color, [[(x0 + px * scale, baseline - py * scale) for px, py in c] for c in cs]))

bar = [(GREEN, 0, 0.55), (DARK, 0.55, 0.62), (BLUE, 0.62, 1.0)]   # the social preview's proportions

# SVG: one path per glyph, even-odd fill, colored by class
def css(palette):
    return ' '.join(f'.{role} {{ fill: {color}; }}' for role, color in palette.items())


paths = []
for role, cs in shapes:
    d = ' '.join('M' + ' L'.join(f'{x:.2f},{y:.2f}' for x, y in c) + ' Z' for c in cs)
    paths.append(f'  <path class="{role}" fill-rule="evenodd" d="{d}"/>')
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}" width="{SIZE}" height="{SIZE}">',
       '  <title>cuda4cpu</title>',
       '  <style>',
       f'    {css(LIGHT)}',
       f'    @media (prefers-color-scheme: dark) {{ {css(DARK_SCHEME)} }}',
       '  </style>']
svg += [f'  <rect class="{role}" x="{a * SIZE:.2f}" y="0" width="{(b - a) * SIZE:.2f}" height="{BAR}"/>'
        for role, a, b in bar]
svg += paths + ['</svg>']
open(out_svg, 'w').write('\n'.join(svg) + '\n')


def render(palette, size, background=(0, 0, 0, 0)):
    """The shapes drawn 4 times larger and reduced, for antialiasing"""
    k = size * 4 / SIZE
    image = Image.new('RGBA', (size * 4, size * 4), background)
    draw = ImageDraw.Draw(image)
    for role, a, b in bar:
        draw.rectangle([a * SIZE * k, 0, b * SIZE * k - 1, BAR * k - 1], fill=palette[role])
    for role, cs in shapes:
        mask = Image.new('1', image.size, 0)
        for c in cs:            # even-odd: each contour flips the pixels inside it
            layer = Image.new('1', image.size, 0)
            ImageDraw.Draw(layer).polygon([(x * k, y * k) for x, y in c], fill=1)
            mask = ImageChops.logical_xor(mask, layer)
        image.paste(palette[role], mask=mask)
    return image.resize((size, size), Image.LANCZOS)


render(BOTH, png_size).save(out_png, optimize=True)
if preview:
    # How each looks: the PNG on white and on GitHub's dark background, and
    # the SVG's two schemes on theirs
    for name, palette, background in (('png-light', BOTH, '#ffffff'), ('png-dark', BOTH, '#0d1117'),
                                      ('svg-light', LIGHT, '#ffffff'), ('svg-dark', DARK_SCHEME, '#0d1117')):
        render(palette, 256, background).convert('RGB').save(f'{preview}-{name}.png')
print('scale', round(scale, 4), 'ink height', round(height, 1))

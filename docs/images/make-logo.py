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
transparent background, in colors readable on light and dark pages alike:
an SVG shown as an image can't know the page's theme (prefers-color-scheme
there is the operating system's), so neither file switches colors.
Needs fontTools, Pillow and the Ubuntu font (fonts-ubuntu). Usage:

    python3 docs/images/make-logo.py docs/images/logo.svg docs/images/logo.png 512 [PREVIEW_PREFIX]"""
import sys
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
from fontTools.pens.recordingPen import DecomposingRecordingPen
from PIL import Image, ImageDraw, ImageChops

FONT = '/usr/share/fonts/truetype/ubuntu/Ubuntu[wdth,wght].ttf'
GREEN, DARK, BLUE = 'green', 'dark', 'blue'          # the colors' roles
# The README logo's green, and a slate and a blue in its spirit, each at
# least 3:1 against white and GitHub's dark #0d1117 (WCAG, large text). The
# social preview's #24292f and #005aa0 vanish on dark pages.
PALETTE = {GREEN: '#609700', DARK: '#7d8590', BLUE: '#2a7fd4'}
SIZE = 512                      # the square, in SVG units
BAR = 20                        # the bar's height
WIDTH = 0.76                    # the widest line's share of the square
LEADING = 0.86                  # baseline to baseline, in ems

if len(sys.argv) not in (4, 5):
    sys.exit('usage: make-logo.py logo.svg logo.png SIZE [PREVIEW_PREFIX]')
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
            # TrueType: implied on-curve points between consecutive off-curve
            # ones. A contour of off-curve points only ends with None, and
            # starts at the midpoint of its last and first points.
            offs, end = list(args[:-1]), args[-1]
            if end is None:
                start = ((offs[-1][0] + offs[0][0]) / 2, (offs[-1][1] + offs[0][1]) / 2)
                current, end = [start], start
            if not offs:                        # one point: a line
                current.append(end)
                continue
            points = [current[-1]]
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

# The social preview's proportions. Each segment starts at 0 and covers the
# next's start, so there's no seam between two antialiased edges.
bar = [(BLUE, 0, 1.0), (DARK, 0, 0.62), (GREEN, 0, 0.55)]

# SVG: one path per glyph, even-odd fill, colors as attributes (some
# viewers drop <style>)
paths = []
for role, cs in shapes:
    d = ' '.join('M' + ' L'.join(f'{x:.2f},{y:.2f}' for x, y in c) + ' Z' for c in cs)
    paths.append(f'  <path fill="{PALETTE[role]}" fill-rule="evenodd" d="{d}"/>')
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}" width="{SIZE}" height="{SIZE}">',
       '  <!-- SPDX-License-Identifier: Apache-2.0. Generated by make-logo.py -->',
       '  <title>cuda4cpu</title>']
svg += [f'  <rect fill="{PALETTE[role]}" x="{a * SIZE:.2f}" y="0" width="{(b - a) * SIZE:.2f}" height="{BAR}"/>'
        for role, a, b in bar]
svg += paths + ['</svg>']
with open(out_svg, 'w', encoding='utf-8') as f:
    f.write('\n'.join(svg) + '\n')


def render(palette, size, background=(0, 0, 0, 0)):
    """The shapes drawn 4 times larger and reduced, for antialiasing"""
    k = size * 4 / SIZE
    image = Image.new('RGBA', (size * 4, size * 4), background)
    draw = ImageDraw.Draw(image)
    for role, a, b in bar:
        draw.rectangle([round(a * SIZE * k), 0, round(b * SIZE * k) - 1, round(BAR * k) - 1], fill=palette[role])
    for role, cs in shapes:
        mask = Image.new('1', image.size, 0)
        for c in cs:            # even-odd: each contour flips the pixels inside it
            layer = Image.new('1', image.size, 0)
            ImageDraw.Draw(layer).polygon([(x * k, y * k) for x, y in c], fill=1)
            mask = ImageChops.logical_xor(mask, layer)
        image.paste(palette[role], mask=mask)
    return image.resize((size, size), Image.BOX)       # each pixel, the mean of its 4x4


render(PALETTE, png_size).save(out_png, optimize=True)
if preview:
    # How it looks on white and on GitHub's dark background
    for name, background in (('light', '#ffffff'), ('dark', '#0d1117')):
        render(PALETTE, 256, background).convert('RGB').save(f'{preview}-{name}.png')
print('scale', round(scale, 4), 'ink height', round(height, 1))

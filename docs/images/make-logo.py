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

"""cuda4cpu's logos, in Ubuntu Bold as outlines, on transparent backgrounds:

- logo.svg and logo.png (512x512): the wordmark stacked as "cuda" over
  "4cpu", with the social preview's three-color bar, in colors readable on
  light and dark pages alike. An SVG shown as an image can't know the page's
  theme (prefers-color-scheme there is the operating system's), so it
  doesn't switch colors.
- wordmark.svg: "cuda4cpu" on one line, for the README, in the same
  colors. Not a <picture> with a variant per theme: GitHub's new repository
  view picks its source by the operating system's scheme when the script
  that follows GitHub's theme doesn't load, which leaves a dark 4 on a dark
  page; and the #gh-dark-mode-only links that work around it are deprecated.

Needs fontTools, Pillow and the Ubuntu font (fonts-ubuntu). Usage:

    python3 docs/images/make-logo.py docs/images [PREVIEW_DIR]"""
import os
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
PNG_SIZE = 512
WORDMARK_HEIGHT = 75            # the README logo's, as it's shown
MARGIN = 0.06                   # around the wordmark's ink, on every side, in ems

if len(sys.argv) not in (2, 3):
    sys.exit('usage: make-logo.py OUTPUT_DIR [PREVIEW_DIR]')
out_dir = sys.argv[1]
preview = sys.argv[2] if len(sys.argv) > 2 else None

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


def svg(shapes, palette, width, height, bar=()):
    """One path per glyph, even-odd fill, colors as attributes (some viewers
    drop <style>)"""
    text = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:g} {height:g}" '
            f'width="{width:g}" height="{height:g}">',
            '  <!-- SPDX-License-Identifier: Apache-2.0. Generated by make-logo.py -->',
            '  <title>cuda4cpu</title>']
    text += [f'  <rect fill="{palette[role]}" x="{a * width:.2f}" y="0" width="{(b - a) * width:.2f}" height="{BAR}"/>'
             for role, a, b in bar]
    for role, cs in shapes:
        d = ' '.join('M' + ' L'.join(f'{x:.2f},{y:.2f}' for x, y in c) + ' Z' for c in cs)
        text.append(f'  <path fill="{palette[role]}" fill-rule="evenodd" d="{d}"/>')
    return '\n'.join(text + ['</svg>']) + '\n'


def render(shapes, palette, width, height, scale, bar=(), background=(0, 0, 0, 0)):
    """The shapes, about scale pixels per unit (the image's size rounded),
    drawn 4 times larger and reduced, for antialiasing"""
    size = (round(width * scale), round(height * scale))
    kx, ky = size[0] * 4 / width, size[1] * 4 / height
    image = Image.new('RGBA', (size[0] * 4, size[1] * 4), background)
    draw = ImageDraw.Draw(image)
    for role, a, b in bar:
        draw.rectangle([round(a * width * kx), 0, round(b * width * kx) - 1, round(BAR * ky) - 1], fill=palette[role])
    for role, cs in shapes:
        mask = Image.new('1', image.size, 0)
        for c in cs:            # even-odd: each contour flips the pixels inside it
            layer = Image.new('1', image.size, 0)
            ImageDraw.Draw(layer).polygon([(x * kx, y * ky) for x, y in c], fill=1)
            mask = ImageChops.logical_xor(mask, layer)
        image.paste(palette[role], mask=mask)
    return image.resize(size, Image.BOX)       # each pixel, the mean of its 4x4


def write(name, text):
    with open(os.path.join(out_dir, name), 'w', encoding='utf-8') as f:
        f.write(text)


def previews(name, shapes, palettes, width, height, scale, bar=()):
    """How it looks on white and on GitHub's dark background"""
    if preview:
        for scheme, background in (('light', '#ffffff'), ('dark', '#0d1117')):
            render(shapes, palettes[scheme], width, height, scale, bar, background).convert('RGB').save(
                os.path.join(preview, f'{name}-{scheme}.png'))


def positioned(parts, x0, baseline, scale):
    return [(color, [[(x0 + px * scale, baseline - py * scale) for px, py in c] for c in cs]) for color, cs in parts]


#
# The square logo
#

lines = [line([(c, GREEN) for c in 'cuda']),
         line([('4', DARK)] + [(c, BLUE) for c in 'cpu'])]
scale = WIDTH * SIZE / max(width for _, width, _, _ in lines)
# The block's ink: from the first line's top to the second's bottom (its p)
top = lines[0][3]
bottom = lines[1][2] - LEADING * upem
height = (top - bottom) * scale
y0 = BAR + (SIZE - BAR - height) / 2 + top * scale        # first baseline
shapes = []                     # (color, contours in SVG units)
for i, (parts, width, _, _) in enumerate(lines):
    shapes += positioned(parts, (SIZE - width * scale) / 2, y0 + i * LEADING * upem * scale, scale)
# The social preview's proportions. Each segment starts at 0 and covers the
# next's start, so there's no seam between two antialiased edges.
bar = [(BLUE, 0, 1.0), (DARK, 0, 0.62), (GREEN, 0, 0.55)]
write('logo.svg', svg(shapes, PALETTE, SIZE, SIZE, bar))
render(shapes, PALETTE, SIZE, SIZE, PNG_SIZE / SIZE, bar).save(os.path.join(out_dir, 'logo.png'), optimize=True)
previews('logo', shapes, {'light': PALETTE, 'dark': PALETTE}, SIZE, SIZE, 0.5, bar)

#
# The wordmark: its ink, with the same margin on every side, scaled to the
# README logo's height
#

parts, _, ymin, ymax = line([(c, GREEN) for c in 'cuda'] + [('4', DARK)] + [(c, BLUE) for c in 'cpu'])
xs = [x for _, cs in parts for c in cs for x, _ in c]
margin = MARGIN * upem
scale = WORDMARK_HEIGHT / (ymax - ymin + 2 * margin)
word_width = round((max(xs) - min(xs) + 2 * margin) * scale, 2)
shapes = positioned(parts, (margin - min(xs)) * scale, (ymax + margin) * scale, scale)
write('wordmark.svg', svg(shapes, PALETTE, word_width, WORDMARK_HEIGHT))
previews('wordmark', shapes, {'light': PALETTE, 'dark': PALETTE}, word_width, WORDMARK_HEIGHT, 4)
print('wordmark', word_width, 'x', WORDMARK_HEIGHT)

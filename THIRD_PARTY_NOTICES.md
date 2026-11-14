# Third-party notices

This project is released under the MIT License (see `LICENSE`). Parts of the simulator in `osu_sim/`
are ported from the two open-source projects below, both under the MIT License. Each file's module
docstring names the source files it follows.

## osu! (lazer)

https://github.com/ppy/osu, used for hit windows, the rule for which object a click counts for,
slider tracking, spinner counting, modifiers, object timing and the Hidden fade
(`osu_sim/judge.py`, `mods.py`, `playable.py`, `renderer.py`, `slider.py`).

Copyright (c) 2025 ppy Pty Ltd <contact@ppy.sh>.

## osu-framework

https://github.com/ppy/osu-framework, used for slider path approximation (`osu_sim/slider.py`).

Copyright (c) 2024 ppy Pty Ltd <contact@ppy.sh>.

## Licence text of both projects

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Game data

osu! is a trademark of ppy Pty Ltd. Beatmaps and replays belong to their mappers and players and
are not part of this repository.

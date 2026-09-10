# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pre-compile the C++ profiling observer in this environment.

    python -m talos.compile_observer

Exits non-zero if the build fails. Run once right after `pip install`
(e.g. as a Dockerfile step) so a broken toolchain or header mismatch
surfaces at install time instead of silently disabling profiling at
training time. The build is cached, so training-time loads then hit the
cache instead of paying the ~1 min compile.

`python -m talos.install_skills` already runs this as its tail step; this
module is the standalone entry for environments where the skills are
installed separately from the training image.
"""

from __future__ import annotations

import sys

from talos.torch_perf_analysis.python.profiler import compile_observer_main

if __name__ == "__main__":
    sys.exit(compile_observer_main())

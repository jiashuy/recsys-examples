# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Turn a case's option dict into the command line of an argparse/click worker."""

from typing import Any, List, Mapping


def as_flags(options: Mapping[str, Any]) -> List[str]:
    """``{"batch_size": 32, "caching": True}`` -> ``["--batch-size", "32", "--caching"]``.

    A ``True`` value is a bare flag, ``False`` / ``None`` drops the option, and
    anything else is ``--name value``. Underscores in a name become dashes, which
    is how the workers spell their options.
    """
    flags: List[str] = []
    for name, value in options.items():
        flag = "--" + name.replace("_", "-")
        if value is True:
            flags.append(flag)
        elif value is False or value is None:
            continue
        else:
            flags += [flag, str(value)]
    return flags

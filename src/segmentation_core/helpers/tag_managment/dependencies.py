# (c) McKinsey & Company 2016 – Present
# All rights reserved
#
#
# This material is intended solely for your internal use and may not be reproduced,
# disclosed or distributed without McKinsey & Company's express prior written consent.
# Except as otherwise stated, the Deliverables are provided ‘as is’, without any express
# or implied warranty, and McKinsey shall not be obligated to maintain, support, host,
# update, or correct the Deliverables. Client guarantees that McKinsey’s use of
# information provided by Client as authorised herein will not violate any law
# or contractual right of a third party. Client is responsible for the operation
# and security of its operating environment. Client is responsible for performing final
# testing (including security testing and assessment) of the code, model validation,
# and final implementation of any model in a production environment. McKinsey is not
# liable for modifications made to Deliverables by anyone other than McKinsey
# personnel, (ii) for use of any Deliverables in a live production environment or
# (iii) for use of the Deliverables by third parties; or
# (iv) the use of the Deliverables for a purpose other than the intended use
# case covered by the agreement with the Client.
# Client warrants that it will not use the Deliverables in a "closed-loop" system,
# including where no Client employee or agent is materially involved in implementing
# the Deliverables and/or insights derived from the Deliverables.

"""Helper for tag dependecy management."""
from typing import Set


def _bfs(key: str, edges: dict) -> Set[str]:
    """Breadth first search through a dict of edges."""
    if key not in edges:
        return set()

    collected = set()
    queue = [key]

    while queue:
        k = queue.pop(0)
        collected.add(k)
        queue.extend(edges.get(k, set()) - collected)

    collected.remove(key)
    return collected


class DependencyGraph:
    """Helper class to hold and manage tag dependencies."""

    def __init__(self):
        """New DependencyGraph."""
        self.dependencies = {}
        self.dependents = {}

    def add_dependency(self, tag: str, depends_on: str):
        """Adds new dependency.

        Internally, this is stored both as "A has dependency B" and "B has dependent A".

        Args:
            tag: dependent
            depends_on: dependency
        """
        self.dependencies.setdefault(tag, set()).add(depends_on)
        self.dependents.setdefault(depends_on, set()).add(tag)

    def remove_dependency(self, tag: str, depends_on: str):
        """Removes a previously added dependency.

        Args:
            tag: dependent
            depends_on: dependency
        """
        self.dependencies[tag].remove(depends_on)
        if not self.dependencies[tag]:
            self.dependencies.pop(tag)
        self.dependents[depends_on].remove(tag)
        if not self.dependents[depends_on]:
            self.dependents.pop(depends_on)

    def get_dependents(self, tag: str) -> Set[str]:
        """Get all dependents (and dependents of dependents) of `tag`.

        Args:
            tag: starting tag
        """
        return _bfs(tag, self.dependents)

    def get_dependencies(self, tag: str) -> Set[str]:
        """Get all dependencies (and dependencies of dependencies) of `tag`.

        Args:
            tag: starting tag
        """
        return _bfs(tag, self.dependencies)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent bounded section-graph contract oracle; no USD/client imports.

This profile covers all eleven spec types with bounded ASCII names and absolute
prim/property target paths, numeric fields, bool/token/specifier/variability
scalars, typed token/path vectors, path list operations and the presence value
families of scene_graph_wire. Other path syntax and value families remain
explicitly unsupported. It models wire application, not authorization to issue
captured one-spec edits.

Commands that meet a changed graph follow the Live Edit service's rules: a
Create, Delete or Move of a missing target is skipped; field and sample data
for a missing section is kept as orphan data that is never projected; a Create
at an occupied (parent, name) slot deletes the occupant's subtree first; a
Create of a live identity is an invalid argument; a Move to a missing parent,
into its own subtree or onto an ancestor's name deletes the moved subtree, and
a Move onto any other occupied name deletes the occupant first. Equal old and
new parents in a Move select a rename under the stored parent, without a
sibling-name check, so two siblings can share one name and path; the (parent,
name) slot then resolves to the lower identity.

A Reorder replaces the parent's entire stored children list with its list,
verbatim and across every child list (the service ignores childrenListId). A
listed ID that is not a live section, or whose parent is another section, is an
invalid argument with the service's status text. A child the list omits stays a
section but is unlisted: it and its subtree are unreachable, so the projected
state, like the service's catch-up walk over stored children, omits them. A
later Reorder that lists it, or a Move, makes it reachable again. With
`per_list`, the model follows a service that orders the list per child list
instead: it applies the list exactly as USD's SdfApplyListOrdering does to each
child list, as a reader applies a delivered Reorder, ignoring listed IDs that
are not current children and every repeat after an ID's first occurrence, so no
child becomes unlisted. A Delete
walks listed children only, like the service's cascade over stored children
lists: an unlisted descendant of a deleted section stays as a stranded row whose
parent is missing, unreachable with its subtree until a Move gives it a parent.
"""
import copy
import re

from scene_graph_wire import decode, Delta, GraphWireError, Paths, TimeSamples, checked_path

ROOT, PRIM, ATTRIBUTE, RELATIONSHIP, VARIANT, VARIANT_SET = 7, 6, 1, 8, 10, 11
CONNECTION, EXPRESSION, MAPPER, MAPPER_ARG, TARGET = 2, 3, 4, 5, 9
CHILD_LISTS = {ROOT: ('primChildren',), PRIM: ('primChildren', 'properties', 'variantSetChildren'),
               VARIANT: ('primChildren', 'properties', 'variantSetChildren'), VARIANT_SET: ('variantChildren',),
               ATTRIBUTE: ('connectionChildren', 'expressionChildren', 'mapperChildren'),
               RELATIONSHIP: ('targetChildren', 'expressionChildren'), TARGET: ('properties',), MAPPER: ('mapperArgChildren',)}
CHILD_TYPES = {'primChildren': (PRIM,), 'properties': (ATTRIBUTE, RELATIONSHIP),
               'variantSetChildren': (VARIANT_SET,), 'variantChildren': (VARIANT,),
               'connectionChildren': (CONNECTION,), 'expressionChildren': (EXPRESSION,),
               'mapperChildren': (MAPPER,), 'targetChildren': (TARGET,), 'mapperArgChildren': (MAPPER_ARG,)}
CHILD_FIELDS = set(CHILD_TYPES)
PATH_FIELDS = {'connectionChildren', 'targetChildren', 'mapperChildren'}
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
PROPERTY = re.compile(r"[A-Za-z_][A-Za-z_0-9]*(?::[A-Za-z_][A-Za-z_0-9]*)*\Z")
VARIANT_NAME = re.compile(r"[A-Za-z_0-9-]+\Z")
MAX_SECTIONS, MAX_FIELDS, MAX_DEPTH, MAX_PATH_BYTES = 4096, 16384, 64, 1024 * 1024
MAX_WORK, MAX_HISTORY_BYTES, MAX_COMMITS = 4 * 1024 * 1024, 16 * 1024 * 1024, 4096


class GraphError(ValueError):
    """`status_message` is set only where the reference reproduces the service's status text."""
    def __init__(self, message, kind="unsupported", status_message=None):
        self.kind = kind
        self.status_message = status_message
        super().__init__(message)


def reject(message, kind="unsupported"):
    raise GraphError(message, kind)


def duplicate_section(identity):
    message = f"delta error: duplicate section_id {identity} in CreateSection"
    raise GraphError(message, "malformed", message)


def reorder_rejected(message):
    message = "delta error: " + message
    raise GraphError(message, "malformed", message)


def root():
    return {"id": 1, "parent": 0, "type": ROOT, "name": "", "order": 0, "path": "/", "fields": {}}


def group(nodes, identity):
    child = nodes[identity]
    if identity == 1:
        reject("root has no parent group", "malformed")
    parent = nodes.get(child["parent"])
    if parent is None:
        reject("missing graph parent")
    parent_type, child_type = parent["type"], child["type"]
    for field in CHILD_LISTS.get(parent_type, ()):
        if child_type in CHILD_TYPES[field]:
            if parent_type == TARGET and child_type != ATTRIBUTE:
                break
            return parent['id'], field
    reject("incompatible parent and child types", "malformed")


def groups(nodes, unlisted=frozenset()):
    """Listed members of each (parent, child list), in stored order."""
    result = {}
    for identity in nodes:
        if identity != 1 and identity not in unlisted:
            result.setdefault(group(nodes, identity), []).append(identity)
    for identities in result.values():
        identities.sort(key=lambda identity: nodes[identity]["order"])
    return result


def validate(nodes, spend, unlisted=frozenset()):
    spend(len(nodes))
    if len(nodes) > MAX_SECTIONS:
        reject("section bound", "limit")
    if 1 not in nodes or any(nodes[1][key] != value for key, value in root().items() if key not in ("fields", "path", "order")):
        reject("invalid pseudoroot", "malformed")
    paths, path_bytes = {}, 0
    for identity, node in nodes.items():
        if not 0 < identity < 2**64 or node["id"] != identity:
            reject("invalid section identity", "malformed")
        if node["type"] not in range(1, 12):
            reject("spec type outside graph oracle profile")
        stranded = identity != 1 and node['parent'] not in nodes
        if stranded and identity not in unlisted:
            reject("cyclic or disconnected graph")
        if identity != 1:
            if not node['name']:
                reject('empty nonroot name', 'malformed')
            if node['type'] in (CONNECTION, TARGET, MAPPER):
                try:
                    checked_path(node['name'])
                except GraphWireError as error:
                    reject(str(error), error.kind)
            elif node['type'] == EXPRESSION:
                if node['name'] != 'expression':
                    reject('noncanonical expression name', 'malformed')
            else:
                pattern = VARIANT_NAME if node['type'] == VARIANT else IDENTIFIER if node['type'] in (PRIM, VARIANT_SET, MAPPER_ARG) else PROPERTY
                if not pattern.fullmatch(node["name"]):
                    reject("name outside ASCII graph oracle profile")
            if not stranded:
                group(nodes, identity)
        # A stranded row keeps its last derived path and roots its subtree's paths.
        chain, seen, current, path, parent_type = [], set(), identity, "/", ROOT
        while current != 1:
            spend(1)
            if current in seen or current not in nodes:
                reject("cyclic or disconnected graph")
            if nodes[current]["parent"] not in nodes and current in unlisted:
                path, parent_type, stranded = nodes[current]["path"], nodes[current]["type"], True
                break
            seen.add(current)
            chain.append(nodes[current])
            if len(chain) > MAX_DEPTH:
                reject("graph depth bound", "limit")
            current = nodes[current]["parent"]
        for child in reversed(chain):
            if child["type"] == PRIM:
                path += ("" if parent_type in (ROOT, VARIANT) else "/") + child["name"]
            elif child['type'] == VARIANT_SET:
                path += '{' + child['name'] + '=}'
            elif child['type'] == VARIANT:
                # The graph parent is the set's /P{set=} spec, although Sdf
                # GetParentPath strips the entire selection. Fill that final
                # empty selection without stripping enclosing nested variants.
                path = path[:-1] + child['name'] + '}'
            elif child['type'] in (CONNECTION, TARGET, MAPPER):
                path += ('.mapper' if child['type'] == MAPPER else '') + '[' + child['name'] + ']'
            else:
                path += "." + child["name"]
            parent_type = child['type']
        other = None if stranded else paths.get(path)
        if other is not None and (nodes[other]["parent"], nodes[other]["name"]) != (node["parent"], node["name"]):
            reject("destination path collision")
        # Same-parent, same-name siblings share a path: the service's rename has no sibling-name check.
        if not stranded:
            paths.setdefault(path, identity)
        path_bytes += len(path.encode("utf-8"))
        spend(len(path))
        if path_bytes > MAX_PATH_BYTES:
            reject("derived path storage bound", "limit")
        node["path"] = path
    orders = {}
    for identity, node in nodes.items():
        if identity != 1 and identity not in unlisted:
            slots = orders.setdefault(node["parent"], set())
            if node["order"] in slots:
                reject("ambiguous child order")
            slots.add(node["order"])
    if sum(len(node["fields"]) for node in nodes.values()) > MAX_FIELDS:
        reject("field bound", "limit")


def descends(nodes, identity, ancestor):
    """Whether `identity` is `ancestor` or lies in its subtree."""
    while identity in nodes:
        if identity == ancestor:
            return True
        identity = nodes[identity]["parent"]
    return False


def occupant(nodes, parent, name, mover):
    """The service resolves a (parent, name) slot to its lowest live identity."""
    return min((identity for identity, node in nodes.items()
                if identity != mover and node["parent"] == parent and node["name"] == name), default=None)


def hidden(nodes, unlisted):
    """Unlisted sections and their subtrees: unreachable from the root through stored children."""
    result = set()
    for identity in nodes:
        chain, current = [], identity
        while current in nodes and current != 1 and current not in result and current not in unlisted:
            chain.append(current)
            current = nodes[current]["parent"]
        if current in result or current in unlisted:
            result.update(chain)
            result.add(current)
    return result


def next_order(nodes, parent):
    order = max((node["order"] for node in nodes.values() if node["parent"] == parent), default=-1) + 1
    if order > 2**32 - 1:
        reject("sibling order exhausted")
    return order


def apply_list_ordering(items, order):
    """`items` ordered by the partial `order` as OpenUSD's SdfApplyListOrdering does (sdf/listOp.cpp).

    Each item `order` names, in order, moves to the end with the unnamed items that follow it; unnamed items
    before the first named one stay in front. Names not in `items` and repeats are ignored.
    """
    named = []
    for item in order:
        if item in items and item not in named:
            named.append(item)
    if not named:
        return list(items)
    scratch, result = list(items), []
    for item in named:
        start = scratch.index(item)
        end = start + 1
        while end < len(scratch) and scratch[end] not in named:
            end += 1
        result += scratch[start:end]
        del scratch[start:end]
    return scratch + result


def ranked(state):
    """`state` with each nonroot order replaced by the node's index among its parent's children, the service's
    sibling index; the root is in no children list and keeps its stored order. Order values are not
    observable; only the sibling sequence they give is."""
    siblings = {}
    for node in state:
        if node["id"] != 1:
            siblings.setdefault(node["parent"], []).append((node["order"], node["id"]))
    ranks = {identity: rank for children in siblings.values() for rank, (_, identity) in enumerate(sorted(children))}
    return [dict(node, order=ranks.get(node["id"], node["order"])) for node in state]


def place(nodes, unlisted, changes, spend):
    """Order a diff's restated sections among their parents' other children.

    A diff's section order is the section's index in its parent's current children list, and the service
    restates only changed sections, so an unchanged sibling keeps an index from an older list. Each restated
    section is inserted at its index among the parent's other listed children, in index order, and those
    children are renumbered from zero.
    """
    placed = {}
    for command in changes:
        if command.section != 1:
            placed.setdefault(command.parent, []).append((command.order, command.section))
    for parent, entries in placed.items():
        restated = {identity for _, identity in entries}
        kept = sorted((node["order"], identity) for identity, node in nodes.items()
                      if identity != 1 and node["parent"] == parent and identity not in restated
                      and identity not in unlisted)
        spend(len(kept) + len(entries))
        children = [identity for _, identity in kept]
        for order, identity in sorted(entries, key=lambda entry: entry[0]):
            children.insert(min(order, len(children)), identity)
        for order, identity in enumerate(children):
            nodes[identity]["order"] = order


class Model:
    """An entire batch validates before scene, sequence or history can change.

    State is returned by value. Failed parsing/conflicts never mutate scenes returned
    earlier or the retained commit history. Commands use section IDs;
    paths are derived, never a replacement for identity.
    """
    def __init__(self):
        self._nodes = {1: root()}
        self._unlisted = frozenset()
        self._stale = frozenset()
        self._orphans = {}
        self._history = ()
        self._history_sizes = ()
        self._sequence = None
        self._history_bytes = 0
        self._source_format = ""

    @property
    def state(self):
        """Reachable sections: the service's catch-up walk from the root over stored children lists."""
        invisible = hidden(self._nodes, self._unlisted)
        return [copy.deepcopy(self._nodes[identity]) for identity in sorted(self._nodes) if identity not in invisible]

    @property
    def detached(self):
        """Sections an omitting Reorder left unreachable, with their subtrees, keyed by section ID."""
        return {identity: copy.deepcopy(self._nodes[identity])
                for identity in sorted(hidden(self._nodes, self._unlisted))}

    @property
    def orphans(self):
        """Field/sample data committed for missing sections, keyed by section ID.

        Reference policy, service behavior unknown: orphan data for a retired
        identity stays apart when a Create reuses that identity; the new
        section starts with no fields and never projects the orphans.
        """
        return copy.deepcopy(self._orphans)

    @property
    def history(self):
        return self._history

    def trim_history(self, count):
        """Keep only the newest `count` history entries, as the service trims its update stream."""
        if type(count) is not int or count < 0:
            raise ValueError("history retention count must be a nonnegative integer")
        drop = max(len(self._history) - count, 0)
        if drop:
            self._history_bytes -= sum(self._history_sizes[:drop])
            self._history, self._history_sizes = self._history[drop:], self._history_sizes[drop:]

    @property
    def sequence(self):
        return self._sequence

    @property
    def source_format(self):
        return self._source_format

    def apply(self, data, sequence=None, *, delivered=False, per_list=False):
        """Apply one update. A `delivered` update is one a reader receives: the service never commits a Create of
        a live identity, so a delivered one repeats a commit the reader holds (a catch-up without a proven stable
        cut delivers again the commits its snapshot folded in) and changes nothing. A delivered Reorder orders each
        child list it names exactly as USD's SdfApplyListOrdering does, ignoring non-children and repeats after an
        ID's first occurrence. `per_list` applies a Reorder as a service that orders the list per child list does:
        as a delivered one."""
        try:
            # An empty upload is an empty delta that still consumes a sequence (service SDD §8.4).
            delta = decode(data) if data else Delta()
        except GraphWireError as error:
            raise GraphError(str(error), getattr(error, "kind", "malformed")) from error
        if sequence is not None and (type(sequence) is not int or not 0 < sequence < 2**64 or
                                     self._sequence is not None and sequence <= self._sequence):
            reject("service sequence must increase", "malformed")
        history_bytes = self._history_bytes + len(data)
        if len(self._history) >= MAX_COMMITS or history_bytes > MAX_HISTORY_BYTES:
            reject("history retention bound", "limit")
        work = MAX_WORK

        def spend(amount):
            nonlocal work
            if amount > work:
                reject("graph work bound", "limit")
            work -= amount

        spend(len(data) + len(self._nodes) + len(self._source_format.encode("utf-8")) +
              sum(len(node["fields"]) + sum(len(v.value) for v in node['fields'].values()
                  if type(v) is TimeSamples) for node in self._nodes.values()) +
              sum(len(fields) for fields in self._orphans.values()))
        reset = delta.is_diff and not delta.base_version
        nodes = {1: root()} if reset else copy.deepcopy(self._nodes)
        orphans = {} if reset else copy.deepcopy(self._orphans)
        unlisted = set() if reset else set(self._unlisted)
        # (section, parent) pairs: a parent other than the stored one whose
        # service children list still names the section.
        stale = set() if reset else set(self._stale)
        source_format = delta.source_format or ("" if reset else self._source_format)
        affected = set()

        def mark(identity):
            if identity != 1 and nodes[identity]["parent"] in nodes:  # A stranded row is in no child field.
                affected.add(group(nodes, identity))

        def cascade(top):
            doomed = {top}
            while True:
                spend(len(nodes) + len(stale))
                expanded = doomed | {identity for identity, node in nodes.items()
                                     if node["parent"] in doomed and identity not in unlisted}
                expanded |= {identity for identity, parent in stale if parent in doomed}
                if expanded == doomed:
                    break
                doomed = expanded
            for identity in doomed:
                mark(identity)  # A stale-listed section leaves its surviving parent's child field.
            for identity in doomed:
                del nodes[identity]
                unlisted.discard(identity)
            stale.difference_update({pair for pair in stale if pair[0] in doomed or pair[1] in doomed})

        def target(identity):
            """Live fields, or the orphan store for a missing section."""
            return nodes[identity]["fields"] if identity in nodes else orphans.setdefault(identity, {})

        if delta.is_diff:
            changed = set()
            for command in delta.commands:
                spend(1)
                if command.tag not in (5, 6):
                    reject("ordinary command in diff", "malformed")
                if command.section in changed:
                    reject("repeated diff identity")
                changed.add(command.section)
                if command.section in nodes:
                    mark(command.section)
            for command in delta.commands:
                spend(1)
                unlisted.discard(command.section)  # A diff restates final visible metadata.
                if command.tag == 6:
                    if command.section == 1:
                        reject("cannot delete root", "malformed")
                    nodes.pop(command.section, None)
                else:
                    fields = nodes.get(command.section, {}).get("fields", {})
                    nodes[command.section] = dict(id=command.section, parent=command.parent, type=command.spec_type,
                                                  name=command.name, order=command.order, path="", fields=fields)
            # A retracted section's list is gone; a restated parent lists the section.
            stale.difference_update({pair for pair in stale if pair[0] not in nodes or pair[1] not in nodes
                                     or nodes[pair[0]]["parent"] == pair[1]})
            place(nodes, unlisted, [command for command in delta.commands if command.tag == 5], spend)
            validate(nodes, spend, unlisted)
            for identity in changed:
                if identity in nodes:
                    mark(identity)
        else:
            for command in delta.commands:
                spend(1)
                if command.tag in (1, 2, 3) and command.section in (0, 1):
                    reject("ordinary structural operation needs nonroot identity", "malformed")
                if command.tag in (1, 3) and command.parent == command.section:
                    reject("self-parenting structural command", "malformed")
                if command.tag == 1:
                    if command.section in nodes:
                        if delivered:
                            continue
                        duplicate_section(command.section)
                    if command.parent not in nodes:
                        continue
                    replaced = occupant(nodes, command.parent, command.name, command.section)
                    if replaced is not None:
                        cascade(replaced)
                    nodes[command.section] = dict(id=command.section, parent=command.parent, type=command.spec_type,
                                                  name=command.name, order=next_order(nodes, command.parent), path="", fields={})
                    mark(command.section)
                elif command.tag == 2:
                    if command.section in nodes:
                        cascade(command.section)
                elif command.tag == 3:
                    node = nodes.get(command.section)
                    if node is None:
                        continue
                    if command.old_parent == command.parent:
                        # Equal command parents select a rename under the stored parent.
                        mark(command.section)
                        node["name"] = command.name
                        mark(command.section)
                    elif command.parent not in nodes or descends(nodes, command.parent, command.section):
                        cascade(command.section)
                    else:
                        replaced = occupant(nodes, command.parent, command.name, command.section)
                        if replaced is not None and descends(nodes, command.section, replaced):
                            cascade(command.section)
                        else:
                            if replaced is not None:
                                cascade(replaced)
                            mark(command.section)
                            # Only the carried old parent's list drops the section; the
                            # new parent's list gains it unless it already names it.
                            identity, stored = command.section, node["parent"]
                            stale.discard((identity, command.old_parent))
                            if stored != command.parent:
                                if identity not in unlisted and command.old_parent != stored:
                                    stale.add((identity, stored))
                                stale.discard((identity, command.parent))
                                node["order"] = next_order(nodes, command.parent)
                            elif identity in unlisted:
                                node["order"] = next_order(nodes, command.parent)
                            unlisted.discard(identity)  # Listed under its new parent.
                            node.update(parent=command.parent, name=command.name)
                            mark(command.section)
                elif command.tag == 4:
                    parent, listed = command.section, command.children
                    seen, positions = set(), {}
                    for position, identity in enumerate(listed):
                        spend(1)
                        if identity in seen:
                            if delivered or per_list:
                                continue  # USD's list ordering keeps an ID's first occurrence.
                            reorder_rejected(f"ReorderChildren({parent}) lists section_id {identity} "
                                             "more than once in children_list")
                        seen.add(identity)
                        if (delivered or per_list) and (identity not in nodes or nodes[identity]["parent"] != parent):
                            continue
                        positions[identity] = position
                        if identity not in nodes:
                            reorder_rejected(f"ReorderChildren contained unknown section_id {identity} "
                                             f"for parent {parent}")
                        if nodes[identity]["parent"] != parent:
                            reorder_rejected(f"ReorderChildren({parent}) listed section_id {identity} "
                                             f"whose actual parent_id is {nodes[identity]['parent']}")
                    if parent not in nodes:
                        continue  # The list names only stranded rows, which stay unreachable.
                    before = groups(nodes, unlisted)
                    siblings = sorted((identity for identity, node in nodes.items() if node["parent"] == parent),
                                      key=lambda identity: nodes[identity]["order"])
                    spend(len(siblings))
                    if delivered or per_list:
                        # A reader orders each child list the Reorder names on its own, as USD applies a
                        # partial order, within the positions that list holds among the parent's children.
                        named = sorted(positions, key=positions.get)
                        slots, orders = {}, {}
                        for position, identity in enumerate(siblings):
                            slots.setdefault(group(nodes, identity), []).append(position)
                        for identity in named:
                            orders.setdefault(group(nodes, identity), []).append(identity)
                        placed = list(siblings)
                        for key, order in orders.items():
                            members = [siblings[position] for position in slots[key]]
                            for position, identity in zip(slots[key], apply_list_ordering(members, order)):
                                placed[position] = identity
                        for order, identity in enumerate(placed):
                            nodes[identity]["order"] = order
                        unlisted.difference_update(siblings)
                    else:
                        # The service stores the list verbatim as the entire children list;
                        # omitted children keep their rows but become unlisted.
                        omitted = [identity for identity in siblings if identity not in positions]
                        for identity, position in positions.items():
                            nodes[identity]["order"] = position
                        for order, identity in enumerate(omitted, len(listed)):
                            nodes[identity]["order"] = order
                        unlisted.difference_update(positions)
                        unlisted.update(omitted)
                    # The list replaces the parent's whole children list, stale entries too.
                    stale.difference_update({pair for pair in stale if pair[1] == parent})
                    after = groups(nodes, unlisted)
                    affected.update(key for key in before.keys() | after.keys()
                                    if key[0] == parent and before.get(key) != after.get(key))
                else:
                    reject("diff command in ordinary update", "malformed")
                validate(nodes, spend, unlisted)

        members = groups(nodes, unlisted)
        for parent, key in affected:
            if parent not in nodes:
                continue
            names = tuple(nodes[identity]["name"] for identity in members.get((parent, key), ()))
            if names:
                nodes[parent]["fields"][key] = Paths(names) if key in PATH_FIELDS else names
            else:
                nodes[parent]["fields"].pop(key, None)
        explicit = set()
        for field in delta.fields:
            spend(1)
            if field.key == "timeSamples" and field.value is not None and type(field.value) is not TimeSamples:
                reject("timeSamples field requires a sample map", "malformed")
            fields = target(field.section)
            if field.section in nodes and field.key in CHILD_FIELDS:
                if field.key not in CHILD_LISTS.get(nodes[field.section]['type'], ()):
                    reject("child field incompatible with type", "malformed")
                if field.value is not None and type(field.value) is not (Paths if field.key in PATH_FIELDS else tuple):
                    reject("child field requires matching path/token vector", "malformed")
                explicit.add((field.section, field.key))
            if field.value is None:
                fields.pop(field.key, None)
            else:
                fields[field.key] = field.value
        # All map replacements precede per-key edits, independent of setOrder.
        # Python dictionaries, like std::map, retain the existing equivalent
        # key's signed-zero representation when replacing its value.
        for sample in delta.samples:
            fields = target(sample.section)
            old = fields.get('timeSamples', TimeSamples())
            spend(1 + len(old.value))
            values = dict(old.value)
            if sample.value is None:
                if sample.time not in values:
                    continue  # Missing-key erase preserves authored empty maps.
                del values[sample.time]
            else:
                values[sample.time] = sample.value
            if values:
                fields['timeSamples'] = TimeSamples(tuple(sorted(values.items())))
            else:
                fields.pop('timeSamples', None)
        for parent, key in explicit:
            identities = members.get((parent, key), ())
            names = nodes[parent]["fields"].get(key, ())
            if type(names) is Paths:
                names = names.value
            if sorted(names) != sorted(nodes[identity]["name"] for identity in identities):
                reject("final child field contradicts graph")
            # Duplicate sibling names keep their relative order.
            queues = {}
            for identity in identities:
                queues.setdefault(nodes[identity]["name"], []).append(identity)
            slots = sorted(nodes[identity]["order"] for identity in identities)
            for name, slot in zip(names, slots):
                nodes[queues[name].pop(0)]["order"] = slot
        for identity in [identity for identity, fields in orphans.items() if not fields]:
            del orphans[identity]
        validate(nodes, spend, unlisted)
        if sum(len(fields) for fields in orphans.values()) > MAX_FIELDS:
            reject("orphan field bound", "limit")
        for node in nodes.values():
            for key in node["fields"]:
                spend(1)
                if key in CHILD_FIELDS and key not in CHILD_LISTS.get(node['type'], ()):
                    reject("retained child field incompatible with type", "malformed")
        history = self._history + ((sequence, delta),)
        sizes = self._history_sizes + (len(data),)
        # Materialize caller copy before committing even on an allocation failure.
        invisible = hidden(nodes, unlisted)
        result = [copy.deepcopy(nodes[identity]) for identity in sorted(nodes) if identity not in invisible]
        self._nodes, self._history, self._orphans = nodes, history, orphans
        self._unlisted = frozenset(unlisted)
        self._stale = frozenset(stale)
        self._history_sizes = sizes
        self._source_format = source_format
        self._history_bytes = history_bytes
        if sequence is not None:
            self._sequence = sequence
        return result

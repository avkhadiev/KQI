#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a module for knowledge quantification index (KQI) '

__author__ = 'Huquan Kang' # And me, a dumbass


import math
import datetime
import igraph

from tqdm import tqdm


class DiGraph():
    def __init__(self) -> None:
        '''
        self.__pred = {node: [pred, ...]}
        self.__succ = {node: [succ, ...]}
        decay: Percentage of decay per decade, from 0 to 1.
        '''
        self.__pred = {}
        self.__succ = {}
        self.__date = {}
        self.__DECAY = 1
        self.__TODAY = datetime.date.today()
        self.__UPDATE_FLAG = False
        self.__volume = {}
        self.__graph_volume = 0

    def add_node(self, v, pred: list, date: datetime.date = datetime.date.today()):
        '''
        Add node like BA model. This function will override the old node.
        '''
        if self.__pred.__contains__(v):
            raise Exception('Add repeated node!')

        self.__pred[v] = [u for u in set(pred)]
        for u in self.__pred[v]:
            if not self.__succ.__contains__(u):
                self.__succ[u] = [v]
            else:
                self.__succ[u].append(v)
        self.__date[v] = date
        self.__UPDATE_FLAG = True

    def remove_node(self, v):
        if not self.__pred.__contains__(v):
            raise Exception('Remove non-existed node!')
        try:
            preds = self.__pred.pop(v)
            for u in preds:
                self.__succ[u].remove(v)

            self.__date.pop(v)
        except:
            pass

        self.__UPDATE_FLAG = True

    def set_today(self, date: datetime.date):
        self.__TODAY = date

    def set_decay(self, decay):
        self.__DECAY = decay

    def weight(self, v):
        return self.__DECAY**((self.__TODAY-self.__date[v]).days/365/10)

    def date(self, v):
        return self.__date[v]

    def nodes(self):
        for v in self.__pred.keys():
            yield v

    def number_of_nodes(self):
        return self.__pred.__len__()

    def number_of_edges(self):
        return sum(map(lambda k: len(k), self.__pred.values()))

    def exist_node(self, v):
        return v in self.__pred

    def successors(self, u, when: datetime.date = None):
        if not self.__succ.__contains__(u):
            return []
        if when:
            for v in self.__succ[u]:
                if self.__date[v] < when:
                    yield v
        else:
            for v in self.__succ[u]:
                yield v

    def predecessors(self, v):
        if not self.__pred.__contains__(v):
            return []
        for u in self.__pred[v]:
            yield u

    def in_degree(self, v):
        return len(self.__pred[v])

    def out_degree(self, u, when: datetime.date = None):
        if u not in self.__succ:
            return 0
        if when:
            return len(self.successors(u, when))
        else:
            return len(self.__succ[u])

    def out_degree_weighted(self, u):
        return sum(map(lambda k: self.weight(k), self.successors(u)))

    def ancestors(self, v):
        access_set = {v}
        open_list = [v]
        while len(open_list):
            node = open_list.pop()
            for v in self.predecessors(node):
                if v not in access_set:
                    access_set.add(v)
                    open_list.append(v)
        return access_set - {v}

    def descendants(self, v):
        access_set = {v}
        open_list = [v]
        while len(open_list):
            node = open_list.pop()
            for v in self.successors(node):
                if v not in access_set:
                    access_set.add(v)
                    open_list.append(v)
        return access_set - {v}

    def __clear_cache(self):
        self.__volume = {}
        self.__graph_volume = 0
        self.__partition_tree()

    def __strongly_connected_components(self):
        preorder = {}
        lowlink = {}
        scc_found = set()
        scc_queue = []
        i = 0  # Preorder counter
        for source in self.nodes():
            if source not in scc_found:
                queue = [source]
                while queue:
                    v = queue[-1]
                    if v not in preorder:
                        i = i + 1
                        preorder[v] = i
                    done = True
                    for w in self.successors(v):
                        if w not in preorder:
                            queue.append(w)
                            done = False
                            break
                    if done:
                        lowlink[v] = preorder[v]
                        for w in self.successors(v):
                            if w not in scc_found:
                                if preorder[w] > preorder[v]:
                                    lowlink[v] = min([lowlink[v], lowlink[w]])
                                else:
                                    lowlink[v] = min([lowlink[v], preorder[w]])
                        queue.pop()
                        if lowlink[v] == preorder[v]:
                            scc = {v}
                            while scc_queue and preorder[scc_queue[-1]] > preorder[v]:
                                k = scc_queue.pop()
                                scc.add(k)
                            scc_found.update(scc)
                            yield scc
                        else:
                            scc_queue.append(v)

    # TODO: THIS IS SLOP; REVIEW TO MAKE SURE SOUND REMOVAL OF ALL CYCLES
    def remove_cycles(self):
        """
        Enforces a strict Directed Acyclic Graph (DAG) by pruning
        any citation that violates temporal causality.
        """
        count = 0
        # 1. First pass: Handle temporal violations and identity self-loops
        for v in list(self.nodes()):
            # Predecessors (u) must be OLDER than v.
            # If u is newer or the same age, we check ID as a tie-breaker.
            invalid_preds = [
                u for u in self.predecessors(v)
                if u not in self.__pred or
                self.__date[u] > self.__date[v] or
                (self.__date[u] == self.__date[v] and u >= v) # Tie-breaker
            ]

            for u in invalid_preds:
                self.__pred[v].remove(u)
                if v in self.__succ.get(u, []):
                    self.__succ[u].remove(v)
                count += 1

        # 2. Second pass: Systematic Cycle Removal (Feedback Arc Set)
        # This catches any remaining complex cycles (A->B->C->A)
        # that survived the temporal filter.
        import networkx as nx

        # Build a temporary graph to find cycles
        temp_G = nx.DiGraph()
        for v in self.nodes():
            for u in self.predecessors(v):
                temp_G.add_edge(u, v)

        while not nx.is_directed_acyclic_graph(temp_G):
            # Find a single cycle and break one edge
            cycle = nx.find_cycle(temp_G, orientation='original')
            u, v, _ = cycle[0]
            # Remove it from your internal data
            self.__pred[v].remove(u)
            self.__succ[u].remove(v)
            # Remove from temp graph to check next cycle
            temp_G.remove_edge(u, v)
            count += 1

        print(f'Total edges pruned to enforce DAG: {count}')
        self.__UPDATE_FLAG = True

    def topological_sort(self):
        try:
            if not self.__UPDATE_FLAG:
                return self.sorted_list
        except:
            pass
        print('start topological sort')
        indegree_map = {v: self.in_degree(
            v) for v in self.nodes() if self.in_degree(v) > 0}
        # These nodes have zero indegree and ready to be returned.
        zero_indegree = [v for v in self.nodes() if self.in_degree(v) == 0]

        self.sorted_list = []
        with tqdm(total=self.number_of_nodes()) as bar:
            while zero_indegree:
                node = zero_indegree.pop()
                for child in self.successors(node):
                    indegree_map[child] -= 1
                    if indegree_map[child] == 0:
                        zero_indegree.append(child)
                        del indegree_map[child]

                self.sorted_list.append(node)
                bar.update()

        for v in indegree_map.keys():
            if self.__date[v] < self.__TODAY:
                raise Exception(
                    "Graph contains a cycle or graph changed during iteration")
        print('end topological sort')
        return self.sorted_list

    def __partition_tree(self):
        """
        Calculate volume.
        """
        for node in tqdm(reversed(self.topological_sort())):
            self.__volume[node] = self.out_degree_weighted(
                node) + sum(map(lambda k: self.__volume[k] / self.in_degree(k), self.successors(node)))

        self.__graph_volume = sum([self.weight(v) + self.__volume[v]
                                  for v in self.nodes() if self.in_degree(v) == 0])

    def kqi(self, v: int) -> float:
        """
        Calculate KQI.
        """
        if self.__UPDATE_FLAG:
            self.__clear_cache()
            self.__UPDATE_FLAG = False

        if v == 0:
            raise Exception('Can not access super root!')

        if self.__volume[v] == 0:
            return 0

        in_deg = self.in_degree(v)
        if in_deg == 0:
            return -self.__volume[v] / self.__graph_volume * math.log2(self.__volume[v] / self.__graph_volume)

        return sum(map(lambda k: -self.__volume[v]/in_deg/self.__graph_volume
                       * math.log2(self.__volume[v]/in_deg/self.__volume[k]), self.predecessors(v)))

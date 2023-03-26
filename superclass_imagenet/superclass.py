# CODE ADAPTED FROM https://github.com/MadryLab/robustness/blob/master/robustness/tools/imagenet_helpers.py


import os
import json
from itertools import product
import numpy as np
import wn

en = wn.Wordnet('omw-en')


class Node():
    '''
    Class for representing a node in the ImageNet/WordNet hierarchy. 
    '''
    def __init__(self, wnid, parent_wnid=None, name=""):
        """
        Args:
            wnid (str) : WordNet ID for synset represented by node
            parent_wnid (str) : WordNet ID for synset of node's parent
            name (str) : word/human-interpretable description of synset 
        """

        self.wnid = wnid
        self.name = name
        self.class_num = -1
        self.parent_wnid = parent_wnid
        self.descendant_count_in = 0
        self.descendants_all = set()
    
    def add_child(self, child):
        """
        Add child to given node.

        Args:
            child (Node) : Node object for child
        """
        child.parent_wnid = self.wnid
    
    def __str__(self):
        return f'Name: ({self.name}), ImageNet Class: ({self.class_num}), Descendants: ({self.descendant_count_in})'
    
    def __repr__(self):
        return f'Name: ({self.name}), ImageNet Class: ({self.class_num}), Descendants: ({self.descendant_count_in})'


class ImageNetHierarchy():
    '''
    Class for representing ImageNet/WordNet hierarchy. 
    '''
    def __init__(self, wnid_to_label):
        assert all(x not in wnid_to_label for x in ['n09450163', 'n10994097', 'n11196627', 'n11318824']), \
            'WNIDs n09450163, n10994097, n11196627, n11318824 are not supported.'
        
        self.in_wnids, self.wnid_to_num = wnid_to_label.keys(), wnid_to_label
        self.tree = {}
            
        with open('wordnet.is_a.txt', 'r') as f:
            for line in f.readlines():
                parent_wnid, child_wnid = line.strip('\n').split(' ')
                parentNode = self.get_node(parent_wnid)
                childNode = self.get_node(child_wnid)
                parentNode.add_child(childNode)
                
        for wnid in self.in_wnids:
            self.tree[wnid].descendant_count_in = 0
            self.tree[wnid].class_num = self.wnid_to_num[wnid]
            
        for wnid in self.in_wnids:
            node = self.tree[wnid]
            while node.parent_wnid is not None:
                self.tree[node.parent_wnid].descendant_count_in += 1
                self.tree[node.parent_wnid].descendants_all.update(node.descendants_all)
                self.tree[node.parent_wnid].descendants_all.add(node.wnid)
                node = self.tree[node.parent_wnid]
        
        del_nodes = [wnid for wnid in self.tree \
                     if (self.tree[wnid].descendant_count_in == 0 and self.tree[wnid].class_num == -1)]
        for d in del_nodes:
            self.tree.pop(d, None)
                        
        assert all([k.descendant_count_in > 0 or k.class_num != -1 for k in self.tree.values()])

        self.wnid_sorted = sorted(sorted([(k, v.descendant_count_in, len(v.descendants_all)) \
                                        for k, v in self.tree.items()
                                        ],
                                        key=lambda x: x[2], 
                                        reverse=True
                                        ),
                                key=lambda x: x[1], 
                                reverse=True
                                )

    def get_node(self, wnid):
        """
        Add node to tree.

        Args:
            wnid (str) : WordNet ID for synset represented by node

        Returns:
            A node object representing the specified wnid.
        """
        if wnid not in self.tree:
            self.tree[wnid] = Node(wnid, name=', '.join(en.synset(f'omw-en-{wnid[1:]}-n').lemmas()))
        return self.tree[wnid]

    def is_ancestor(self, ancestor_wnid, child_wnid):
        """
        Check if a node is an ancestor of another.

        Args:
            ancestor_wnid (str) : WordNet ID for synset represented by ancestor node
            child_wnid (str) : WordNet ID for synset represented by child node

        Returns:
            A boolean variable indicating whether or not the node is an ancestor
        """
        return (child_wnid in self.tree[ancestor_wnid].descendants_all)

    def get_descendants(self, node_wnid, in_imagenet=False):
        """
        Get all descendants of a given node.

        Args:
            node_wnid (str) : WordNet ID for synset for node
            in_imagenet (bool) : If True, only considers descendants among 
                                ImageNet synsets, else considers all possible
                                descendants in the WordNet hierarchy

        Returns:
            A set of wnids corresponding to all the descendants
        """        
        if in_imagenet:
            return set([self.wnid_to_num[ww] for ww in self.tree[node_wnid].descendants_all
                        if ww in set(self.in_wnids)])
        else:
            return self.tree[node_wnid].descendants_all
    
    def get_superclasses(self, n_superclasses, 
                         ancestor_wnid=None, superclass_lowest=None, 
                         balanced=True):
        """
        Get superclasses by grouping together classes from the ImageNet dataset.

        Args:
            n_superclasses (int) : Number of superclasses desired
            ancestor_wnid (str) : (optional) WordNet ID that can be used to specify
                                common ancestor for the selected superclasses
            superclass_lowest (set of str) : (optional) Set of WordNet IDs of nodes
                                that shouldn't be further sub-classes
            balanced (bool) : If True, all the superclasses will have the same number
                            of ImageNet subclasses

        Returns:
            superclass_wnid (list): List of WordNet IDs of superclasses
            class_ranges (list of sets): List of ImageNet subclasses per superclass
            label_map (dict): Mapping from class number to human-interpretable description
                            for each superclass
        """             
        
        assert superclass_lowest is None or \
               not any([self.is_ancestor(s1, s2) for s1, s2 in product(superclass_lowest, superclass_lowest)])
         
        superclass_info = []
        for (wnid, ndesc_in, ndesc_all) in self.wnid_sorted:
            
            if len(superclass_info) == n_superclasses:
                break
                
            if ancestor_wnid is None or self.is_ancestor(ancestor_wnid, wnid):
                keep_wnid = [True] * (len(superclass_info) + 1)
                superclass_info.append((wnid, ndesc_in))
                
                for ii, (w, d) in enumerate(superclass_info):
                    if self.is_ancestor(w, wnid):
                        if superclass_lowest and w in superclass_lowest:
                            keep_wnid[-1] = False
                        else:
                            keep_wnid[ii] = False
                
                for ii in range(len(superclass_info) - 1, -1, -1):
                    if not keep_wnid[ii]:
                        superclass_info.pop(ii)
            
        superclass_wnid = [w for w, _ in superclass_info]
        class_ranges, label_map = self.get_subclasses(superclass_wnid, 
                                    balanced=balanced)
                
        return superclass_wnid, class_ranges, label_map

    def get_subclasses(self, superclass_wnid, balanced=True):
        """
        Get ImageNet subclasses for a given set of superclasses from the WordNet 
        hierarchy. 

        Args:
            superclass_wnid (list): List of WordNet IDs of superclasses
            balanced (bool) : If True, all the superclasses will have the same number
                            of ImageNet subclasses

        Returns:
            class_ranges (list of sets): List of ImageNet subclasses per superclass
            label_map (dict): Mapping from class number to human-interpretable description
                            for each superclass
        """      
        ndesc_min = min([self.tree[w].descendant_count_in for w in superclass_wnid]) 
        class_ranges, label_map = [], {}
        for ii, w in enumerate(superclass_wnid):
            descendants = self.get_descendants(w, in_imagenet=True)
            if balanced and len(descendants) > ndesc_min:
                descendants = set([dd for ii, dd in enumerate(sorted(list(descendants))) if ii < ndesc_min])
            class_ranges.append(descendants)
            label_map[ii] = self.tree[w].name
            
        for i in range(len(class_ranges)):
            for j in range(i + 1, len(class_ranges)):
                assert(len(class_ranges[i].intersection(class_ranges[j])) == 0)
                
        return class_ranges, label_map


if __name__ == '__main__':
    NUM_SUPERCLASSES = 10
    
    # this is just an example construction of wnid_to_label
    # we should replace this with our GeoYFCC-CLIP 500-class wnids and corresponding labels (0-499)
    with open('imagenet21k_wordnet_ids.txt', 'r') as f:
        wnids = [line.strip('\n') for line in f.readlines()]
    np.random.seed(2389458523)
    sample_wnids = np.random.choice(wnids, size=5000, replace=False)
    wnid_to_label = {wnid: i for i, wnid in enumerate(sample_wnids)}
    
    h = ImageNetHierarchy(wnid_to_label)
    superclass_wnid, class_ranges, label_map = h.get_superclasses(NUM_SUPERCLASSES)
    
    print('Superclasses:', label_map)
    print('Class collapsing recipe:')
    for i, class_range in enumerate(class_ranges):
        print(f'Superclass {i}:', sorted(list(class_range)))

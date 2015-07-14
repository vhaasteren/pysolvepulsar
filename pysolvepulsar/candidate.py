#!/usr/bin/python
# -*- coding: utf-8 -*-
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab
"""
candidate:  Representation of candidate solutions

"""

from __future__ import print_function
from __future__ import division
import numpy as np
import copy

class CandidateSolution(object):
    """
    Class that represents a single candidate solution for a pulsar. Often this
    solution will need to be refined further.
    """

    def __init__(self, copyobject=None):
        """Initialize a new candidate. Either empty, or make a copy

        Initialize a new candidate. Either empty, or make a copy

        :param copyobject:
            If not none, the parameters from this object will be copied
        """
        self.init_empty()

        if copyobject is not None:
            self.copy_from_object(copyobject)

    def init_empty(self):
        """Initialize this object with empty attributes
        """
        self._pars = []
        self._patches = []
        self._rpn = []

        # Tree references
        self._parent = None                 # Parent solution
        self._delete_children = []          # Children through deletion
        self._delete_history = []           # Indices deleted observations
        self._merge_children = []           # Children through patch-mergers

        # P-values
        self._delete_pvals_hist = []        # History of deletion p-values
        self._parent_chi2pval = 1.0         # Chi2 p-value of parent
        self._parent_efacpval = 1.0         # Efac p-value of parent
        self._proposed_pval = (1.0, None)   # Prior-proposal probability
        self._origin = ("root",)            # Proposal origin
        self._chi2 = 1.0                    # Chi2 value of current solution
        self._efac_pval = 1.0               # Efac p-value of current solution
        self._dof = 0                       # Chi2 degrees of freedom
        self._child_merge_pvals = []        # All children merge prior pvals
        self._hist_pval = None              # Priority p-value due to history

    def set_start_solution(self, pars, nobs):
        """Set the start solution

        Set the starting solution

        :param pars:
            Start parameters

        :param nobs:
            The number of observations
        """
        self._pars = pars.copy()
        self.init_separate_patches()

    def set_solution(self, pars, patches, rpn):
        """Set the candidate solution

        :param pars:
            Start parameters

        :param paches:
            Coherent patches

        :param rpn:
            Relative phase numbers within the patches
        """
        self._pars = pars.copy()
        self.set_patches(patches, rpn)

    def copy_from_object(self, copyobject):
        """Copy all the information from an object into this one.

        Copy all the information from an object into this one. This object will
        be an orphan: the parent/child information is not copied.

        :param copyobject:
            The CandidateSolution object to copy stuff from
        """
        self._pars = copy.deepcopy(copyobject._pars)
        self._rpn = copy.deepcopy(copyobject._rpn)
        self._patches = copy.deepcopy(copyobject._patches)

    def __eq__(self, other):
        """Compare p-values == """
        if isinstance(other, CandidateSolution):
            rv = (self.pval == other.pval)
            if rv and self.pval_log is not None and other.pval_log is not None:
                rv = (self.pval_log == other.pval_log)
            return rv
        return NotImplemented

    def __ne__(self, other):
        """Compare p-values != """
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __gt__(self, other):
        """Compare p-values >"""
        if isinstance(other, CandidateSolution):
            rv = (self.pval > other.pval)
            test = (self.pval == other.pval)
            if test and \
                    self.pval_log is not None and other.pval_log is not None:
                rv = (self.pval_log < other.pval_log)
            return rv
        return NotImplemented

    def __ge__(self, other):
        """Compare p-values >="""
        result = self.__lt__(other)
        if result is NotImplemented:
            return result
        return not result

    def __lt__(self, other):
        """Compare p-values <"""
        if isinstance(other, CandidateSolution):
            test = (self.pval == other.pval)
            rv = (self.pval < other.pval)
            if test and \
                    self.pval_log is not None and other.pval_log is not None:
                rv = (self.pval_log > other.pval_log)
            return rv
        return NotImplemented

    def __le__(self, other):
        """Compare p-values <="""
        result = self.__gt__(other)
        if result is NotImplemented:
            return result
        return not result

    def have_child_phase_shift_pval(self, rpnshift, pi1, pi2):
        """Return True if we have the p-value for this phase shift in memory

        Return True if we have the p-value for this phase shift in memory

        :param rpnshift:
            The integer shift in relative phase between patch pi1 and pi2

        :param pi1:
            Index of coherence patch 1

        :param pi2:
            Index of coherence patch 2. Must be pi+1 or pi-1, since we are only
            merging adjacent coherence patches...
        """
        if not pi2 in [pi1+1, pi1-1]:
            raise ValueError("Can only merge adjacent coherent patches")
        
        # Sort the indices and determine the order of the tree
        pi1, pi2 = (pi1, pi2) if pi1 < pi2 else (pi2, pi1)
        tree_index = pi1
        order = np.abs(rpnshift)
        nelements = 2*order+1

        rv = True
        if len(self._child_merge_pvals[pi1]) < nelements:
            rv = False

        return rv

    def get_child_index_from_phase_shift(self, rpnshift, pi1, pi2):
        """Get the child-index for a particular proposed phase shift

        Given a shift in relative phase `rpnshift` between patch pi1 and patch
        pi2, return the index of the child candidate

        :param rpnshift:
            The integer shift in relative phase between patch pi1 and pi2

        :param pi1:
            Index of coherence patch 1

        :param pi2:
            Index of coherence patch 2. Must be pi+1 or pi-1, since we are only
            merging adjacent coherence patches...
        """
        if not pi2 in [pi1+1, pi1-1]:
            raise ValueError("Can only merge adjacent coherent patches")
        
        # Sort the indices and determine the order of the tree
        pi1, pi2 = (pi1, pi2) if pi1 < pi2 else (pi2, pi1)
        tree_index = pi1
        order = np.abs(rpnshift)
        nelements = 2*order+1

        # Properly grow the tree
        self.grow_merge_child_list(tree_index, order)

        mclen = len(self._merge_children[tree_index])
        neworder = (mclen-1)/2
        return order + rpnshift

    def grow_merge_child_list(self, tree_index, order):
        """Given the tree index of the child mergers, grow the child list

        Grow the child list for a specific tree index properly (per two or same
        order).

        :param tree_index:
            Index of the tree (which patches to merge)

        :param order:
            Grow the tree until this order
        """
        if order<0:
            raise ValueError("Order of the tree must be >= 0")
        if len(self._merge_children) == 0:
            self._merge_children = [[] for ii in range(len(self._patches)-1)]
        mclen = len(self._merge_children[tree_index])
        nelements = 2*order+1
        if mclen == 0:
            self._merge_children[tree_index] = [None] * nelements
        elif (mclen-1) % 2 == 0:
            # Grow properly
            while mclen < nelements:
                self._merge_children[tree_index].append(None)
                self._merge_children[tree_index].insert(0, None)
                mclen += 2
        else:
            raise ValueError("_merge_children not grown properly")

    def get_ssp_len(self):
        """Return the number of single-point patches

        Return the number of single-point patches
        """
        nop = np.array([len(patch) for patch in self._patches])
        return np.sum(nop == 1)

    def get_patch_from_sspind(self, sspind):
        """Return the patch index from the single-point patch index

        Return the patch index from the single-point patch index

        :param sspind:
            Single point patch index
        """
        nop = np.array([len(patch) for patch in self._patches])
        return np.where(nop == 1)[0][sspind]

    def get_patch_from_obsind(self, oind):
        """Return the patch index form the observation index

        Return the patch index from the observation index

        :param obsind:
            Observation index
        """
        for ii, patch in enumerate(self._patches):
            if oind in patch:
                return ii

        return None

    def get_obsind2patch_map(self):
        """Return the mapping from observation index to patch index
        
        :return:
            Dictionary of the mapping
        """
        mapping = {}

        for ii, patch in enumerate(self._patches):
            for obs in patch:
                mapping[obs] = ii

        return mapping

    def get_sspind_from_patchind(self, pind):
        """Return the sspind, given the patch index

        Return the single-point patch index, given the patch index

        :param pind:
            The index of the patch
        """
        nop = np.array([len(patch) for patch in self._patches])
        fullinds = np.zeros(len(nop), dtype=np.int)
        pmsk = (nop == 1)
        fullinds[pmsk] = np.arange(np.sum(pmsk))
        return fullinds[pind]

    def delete_observation(self, oind):
        """Delete an observation

        Delete an observation. If the patch is a single-point-patch, delete the
        patch
        """
        pind = self.get_patch_from_obsind(oind)
        if pind is None:
            raise ValueError("Observation not included in a patch")

        ind = self._patches[pind].index(oind)
        temp = self._patches[pind].pop(ind)
        temp = self._rpn[pind].pop(ind)
        if len(self._patches[pind]) == 0:
            temp = self._patches.pop(pind)
            temp = self._rpn.pop(pind)

    def integrate_in_tree(self, origin=('root',), parent=None,
            parent_efacpval=1.0, parent_chi2pval=1.0, proposed_pval=(1.0,None)):
        """Integrate this solution in the candidate solution tree

        Integrate this solution in the candidate solution tree, and have the
        parent do the same.

        NOTE: the patches should already be joined here

        :param origin:
            What the origin of this candidate solution is. Available options are
            ('root',), ('delete', patch_ind), ('rps', rpn_shift, patch_ind1, patch_ind2)

        :param parent:
            Reference to the parent object

        :param parent_efacpval:
            The efac p-value of the parent solution

        :param parent_chi2pval:
            The p-value of the parent chi2 solution

        :param proposed_pval:
            The p-value tuple of this candidate, prior to
            evaluation/optimization
        """
        if origin == 'root':
            self._origin = ('root',)
            self._parent = None
            self._parent_chi2pval = 1.0
            self._parent_efacpval = 1.0
        else:
            self._origin = origin
            self._parent = parent
            self._parent_chi2pval = parent_chi2pval
            self._parent_efacpval = parent_efacpval
            self._proposed_pval = proposed_pval

            if origin[0] == 'delete':
                # Deleted point is always initiated from the parent
                self._delete_pvals_hist.append(proposed_pval)
            else:
                # It's a relative pulse-number shift
                # TODO: FIX THISSS!!!
                #merge_ind = k blah
                #self._parent._merge_children blah
                pass

        self._delete_children = [None] * self.get_ssp_len()
        self._merge_children = [[] for ii in range(len(self._patches)-1)]

    def initialize_children_rpn0(self, pulse_numbers):
        """This candidate has just been created. Create rpn=0 children

        This candidate has just been created. Create candidate solutions for
        patch joining, all with relative phase shift = 0

        :param pulse_numbers:
            The pulse numbers, relative to the PEPOCH
        """
        add = []
        # There are only npatch-1 joins possible
        for pind1, patch in enumerate(self._patches[:-1]):
            pind2 = pind1+1
            child_ind = self.get_child_index_from_phase_shift(0, pind1, pind2)
            if self.have_child_phase_shift_pval(0, pind1, pind2):
                add.append(self.add_proposal_candidate_rpn(0, pind1, pind2,
                        pulse_numbers))
            else:
                # This phase shift/merger is so unlikely, we are not bothering
                # to try even
                # TODO: If all else fails, we should be able to do this type of
                #       merger anyway
                pass

        return add

    def notify_patches_merged(self, rpn, pind1, pind2, delete_pvals,
            pulse_numbers):
        """Patches pind1, pind2 have merged. Deal with it.

        Patches pind1, pind2 have merged. Deal with it by adding the option to
        delete these patches if they are single-point patches.

        :param rpn:
            Relative phase number that was evaluated

        :param pind1:
            Index of patch 1

        :param pind2:
            Index of patch 2

        :param delete_pvals:
            Deletion prior probabilities of all the observations

        :param pulse_numbers:
            The pulse numbers, relative to the PEPOCH
        """
        pind1, pind2 = (pind1, pind2) if pind1 < pind2 else (pind2, pind1)
        add = []
        # Deletion of patch 1
        if rpn==0 and len(self._patches[pind1]) == 1:
            add.append(self.add_proposal_candidate_delete(pind1, delete_pvals))

        # Deletion of patch 2
        if rpn==0 and len(self._patches[pind2]) == 1:
            add.append(self.add_proposal_candidate_delete(pind2, delete_pvals))

        # Relative phase shift expansion
        neworder = np.abs(rpn)+1
        for rps in [-neworder, neworder]:
            child_ind = self.get_child_index_from_phase_shift(rps, pind1, pind2)
            if self._merge_children[pind1][child_ind] is None and \
                    self.have_child_phase_shift_pval(rps, pind1, pind2):
                add.append(self.add_proposal_candidate_rpn(rps, pind1, pind2,
                        pulse_numbers))
            elif not self.have_child_phase_shift_pval(rps, pind1, pind2):
                # This phase shift/merger is so unlikely, we are not bothering
                # to try even
                # TODO: If all else fails, we should be able to do this type of
                #       merger anyway
                pass

        return add

    def add_proposal_candidate_delete(self, pind, delete_pvals):
        """Add the proposal to delete patch number pind

        Add the proposal to delete patch number pind to the tree

        :param pind:
            Index number of the patch

        :param delete_pvals:
            Deletion prior probabilities of all the observations
        """
        sspind = self.get_sspind_from_patchind(pind)

        newcand = CandidateSolution(self)
        temp = newcand._patches.pop(pind)
        origin = ('delete', pind)
        efac_pval, chi2, dof = self._efac_pval, self._chi2, self._dof
        parent_chi2pval = sst.chi2.sf(chi2, dof) if dof > 0 else 1.0
        proposed_pval = (delete_pvals[self._patches[pind][0]], None)
        newcand.integrate_in_tree(origin, self, efac_pval,
                parent_chi2pval, proposed_pval)

        self._delete_children[sspind] = newcand

        # Add the deleted observation to the history present in newcand
        obsind = self._patches[sspind][0]
        ind = bisect.bisect_left(newcand._delete_history, obsind)
        newcand._delete_history.insert(ind, obsind)

        return newcand

    def add_proposal_candidate_rpn(self, rps, pind1, pind2, pulse_numbers):
        """Add the proposal to perform relative phase shifts between patch 1&2

        Add the proposal to shift the phase between patch pind1 and patch pind2

        :param rps:
            Relative phase shift

        :param pind1:
            Index number of patch 1

        :param pind2:
            Index number of patch 2

        :param pulse_numbers:
            The pulse numbers, relative to the PEPOCH
        """
        pind1, pind2 = (pind1, pind2) if pind1 < pind2 else (pind2, pind1)
        newcand = CandidateSolution(self)

        newcand.join_patches(pind1, pind2, pulse_numbers, rpnjump=rps)
        origin = ('rps', rps, pind1, pind2)

        ch_index = self.get_child_index_from_phase_shift(rps, pind1, pind2)

        efac_pval, chi2, dof = self._efac_pval, self._chi2, self._dof
        parent_chi2pval = sst.chi2.sf(chi2, dof) if dof > 0 else 1.0
        proposed_pval = self._child_merge_pvals[pind1][ch_index]
        newcand.integrate_in_tree(origin, self, efac_pval,
                parent_chi2pval, proposed_pval)

        self._merge_children[pind1][ch_index] = newcand

        return newcand

    def init_separate_patches(self, nobs):
        """Initialize the individual coherence patches
        """
        self._patches = [[ind] for ind in range(nobs)]
        self._rpn = [[0] for ind in range(nobs)]

    def set_patches(self, patches, rpn):
        """Set the patches to something specific
        """
        self._patches = copy.deepcopy(patches)
        self._rpn = copy.deepcopy(rpn)

    def join_patches(self, ind1, ind2, apn, rpnjump=0):
        """Join patches ind1, and ind2

        Given patch indices ind1 and ind2, join those two patches. Even though
        we usually only merge adjacent patches, ind1 and ind2 are allowed to be
        arbitrary.

        :param ind1:
            Index of patch one

        :param ind2:
            Index of patch two

        :param apn:
            Absolute pulse numbers (relative to PEPOCH) for all TOAs (np array)

        :param rpnjump:
            Jump in relative pulse number that we'll add to patch 2 (default=0)
        """
        ind1, ind2 = (ind1, ind2) if (ind1 < ind2) else (ind2, ind1)  # Sort
        patches1, rpns1 = self._patches[:ind1], self._rpn[:ind1]
        patches2, rpns2 = self._patches[ind1+1:ind2], self._rpn[ind1+1:ind2]
        patches3, rpns3 = self._patches[ind2+1:], self._rpn[ind2+1:]
        patch1, rpn1 = self._patches[ind1], self._rpn[ind1]
        patch2, rpn2 = self._patches[ind2], self._rpn[ind2]

        # Relative pulse number between patches
        iprpn = apn[patch2[0]]-apn[patch1[0]] + rpnjump
        patch2rpn = [rpn2[ii] + iprpn for ii in range(len(rpn2))]

        # Make new patch
        newpatch = patch1 + patch2
        newrpn = rpn1 + patch2rpn

        # Save all patches
        self._patches = patches1 + [newpatch] + patches2 + patches3
        self._rpn = rpns1 + [newrpn] + rpns2 + rpns3

    def split_patch(self, pind, msk):
        """Split a patch in two

        Given patch index pind, and observation index oind, split the patch
        after oind.

        :param pind:
            Index of the patch

        :param msk:
            Mask, relative to this patch's content, of what to separate out
        """
        patches1, rpns1 = self._patches[:pind], self._rpn[:pind]
        patches2, rpns2 = self._patches[pind+1:], self._rpn[pind+1:]

        patch1 = list(np.array(self._patches[pind])[np.logical_not(msk)])
        rpn1 = list(np.array(self._rpn[pind])[np.logical_not(msk)])
        patch2 = list(np.array(self._patches[pind])[msk])
        rpn2 = list(np.array(self._rpn[pind])[msk])

        # Re-shift the rpns
        rpn1 = [rp-np.min(rpn1) for rp in rpn1]
        rpn2 = [rp-np.min(rpn2) for rp in rpn2]

        self._patches = patches1 + [patch1, patch2] + patches2
        self._rpn = rpns1 + [rpn1, rpn2] + rpns2

    def add_rpn_jump(self, pind, msk, rpnjump=0):
        """Add a shift in relative phase number for the masked observations

        Add a shift in relative phase number for the masked observations.

        :param pind:
            Index of the patch

        :param msk:
            Mask, relative to this patch's content, which rpn's to shift

        :param rpnjump:
            By how many pulse numbers to shift the observations
        """
        newrpn = np.array(self._rpn[pind])
        newrpn[msk] += rpnjump
        self._rpn[pind] = list(newrpn)

    def get_patches(self, fitpatch=None):
        """Get the patches, minus element `fitpatch` if not None
        """
        if fitpatch is None:
            patches = self._patches
        else:
            patches = self._patches[:fitpatch] + self._patches[(fitpatch+1):]
        return patches

    def get_origin_patches(self):
        """Return the rpn and patches ids merged in order to get this solution

        Return the relative phase number and the patch numbers that were merged
        in the parent in order to get this solution
        """
        if self._origin[0] == 'rps':
            return (self._origin[1], self._origin[2], self._origin[3])
        else:
            return NotImplemented

    def get_number_of_toas(self, exclude_nonconnected=False):
        """Get the number of toas, possibly excluding non-connected epochs

        :param exclude_nonconnected:
            If True, exclude coherent patches of length 1
        """
        ntoas = 0
        for pp, patch in enumerate(self._patches):
            ntoas += len(patch) if len(patch) > 1 or \
                    not exclude_nonconnected else 0

        return ntoas

    def get_rpns(self, fitpatch=None):
        """Get the relative pulse numbers, minus element `fitpatch` if not None
        """
        if fitpatch is None:
            rpn = self._rpn
        else:
            rpn = self._rpn[:fitpatch] + self._rpn[(fitpatch+1):]
        return rpn

    def register_optimized_results(self, efac_pval, chi2, dof, child_merge_pvals):
        """After running an optimizer on this candidate, register the results

        After running an optimizer on this candidate, register all the
        probability results, so we can initialize children. If this is
        merged-patch solution with zero relative phase, also create the option
        to delete the point in the parent.

        :param efac_pval
            Efac p-value of the solution

        :param chi2:
            Chi^2 value of the solution

        :param dof:
            Number of effective degrees of freedom of the solution

        :param child_merge_pvals:
            A tree of all possible children p-values, as predicted by the
            Gaussian process approximation. The tree is 'only' populated up to
            some lowest p-value
        """
        self._efac_pval = efac_pval
        self._chi2 = chi2
        self._dof = dof
        self._child_merge_pvals = child_merge_pvals

    def is_parent_delete_trigger(self):
        """Returns true when the parent needs a delete addition when evaluated

        This function returns True when we need to notify the parent when this
        candidate is evaluated. This happens when we are a '0' relative phase
        candidate, are evaluated, and we have a parent.
        """
        rv = False
        if self._origin[0] == 'rps' and self._origin[1] == 0:
            rv = True
        return rv

    def is_parent_rpn_trigger(self):
        """Returns true when the parent might need rpn additions when evaluated

        This function returns True when the parent needs to be notified when we
        evaluate this candidate. This happens when we are non-root, and we are a
        relative phase shift.
        """
        rv = False
        if self._origin[0] == 'rps':
            rv = True
        return rv

    def get_root_map(self, inds):
        """Get the mapping from current patch id, to root patch id

        Due to the deletion of observation, and the merging of patches, the
        current patch id's do not correspond to observations anymore. This
        function returns the mapping of the current patches to the original
        observations in root
        """
        inds = np.atleast_1d(inds)
        if self._origin[0] == 'root':
            mapping = inds
        else:
            parent_inds = self._parent.map_from_child(inds, self._origin)
            mapping = self._parent.get_root_map(parent_inds)

        return mapping

    def map_from_child(self, inds, origin):
        """Given indices from a child, map to this candidate

        Given the patch indices from a child, map those indices to this
        candidate. Handy for recursive use

        :param inds:
            Indices of the child (numpy array)

        :param origin:
            Description of what happened from here to the child
        """
        if origin[0] == 'delete':
            dp = origin[1]
            newinds = inds.copy()
            newinds[dp:] += 1
        elif origin[0] == 'rps':
            # Phase shifting is just like deleting a patch
            dp = origin[3]
            newinds = inds.copy()
            newinds[dp:] += 1
        elif origin[0] == 'root':
            # This should never happen!
            newinds = inds
        return newinds

    def get_history(self):
        """Obtain the history of this candidate

        Obtain the history of this candidate as a list of origins.

        TODO: This is still an O(n^2) procedure
        TODO: DEBUG THIS!!!!
        """
        hist = []
        # First get the full history trace of the parent
        if self._origin[0] != 'root':
            hist += self._parent.get_history()
            mapping = self._parent.get_root_map(np.arange(len(self._parent._patches)))

        if self._origin[0] == 'root':
            hist += ['r']
        elif self._origin[0] == 'delete':
            obsind = mapping[self._origin[1]]
            hist += ['d-'+str(obsind)]
        elif self._origin[0] == 'rps':
            obsind = mapping[self._origin[3]]
            hist += ['s-'+str(self._origin[1])+'-'+str(obsind)]
        else:
            raise NotImplementedError("{0} not an origin ID".self._origin[0])

        return hist

    def get_history_str(self):
        """Return a string of the history of this candidate"""
        return ', '.join(self.get_history())

    def get_history_hashable(self):
        first = str([patch for patch in self._patches if len(patch) > 1])
        second = str([rpn for rpn in self._rpn if len(rpn) > 1])
        third = str(self._delete_history)
        return first+second+third

    def get_history_len(self, count_del=False):
        """Return the length of the history

        Return how many times we have iterated this candidate

        :param cound_del:
            Whether or not we count deletion as an iteration
        """
        parent = self
        length = 1
        while parent._origin[0] != 'root':
            if count_del or parent._origin[0] != 'delete':
                length += 1

            parent = parent._parent

        return length

    def is_coherent(self):
        """Is this candidate solution coherent?"""
        #TODO: We should do proper checking of chi2 values and shit
        return len(self._patches) == 1

    @property
    def pars(self):
        return self._pars

    @pars.setter
    def pars(self, value):
        self._pars = value

    @property
    def npars(self):
        return len(self.pars)

    @property
    def pval(self):
        return self._proposed_pval[0] * self._parent_efacpval * self.hist_pval

    @property
    def pval_log(self):
        #return False if self._proposed_pval[1] is None else True
        return self._proposed_pval[1]

    @property
    def hist_pval_old(self):
        if self._hist_pval is None:
            self._hist_pval = np.log2(self.get_history_len()+1)
        return self._hist_pval

    @property
    def hist_pval(self):
        pls = np.array([len(patch) for patch in self._patches])
        nps = np.sum(pls > 1)    # Number of patches with nobs > 1
        nobs = np.max(pls)
        return np.log2(nobs+1) #* np.log(nps+5)/np.log(5)

    @property
    def nobs(self):
        return self.get_number_of_toas(exclude_nonconnected=False)

    @property
    def obs_inds(self):
        """The candidate has deleted points. So we need to map from shorter list
        to longer list
        """
        return np.array(sum(self._patches, []), dtype=np.int)

    @property
    def obs_inds_inv(self):
        """This is the inverse mapping of 'obs_inds'
        """
        obs_inds = self.obs_inds
        rv = np.zeros(np.max(obs_inds)+1)
        for ii, p in enumerate(obs_inds):
            rv[p] = ii
        return rv

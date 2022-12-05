import numpy as np
import itertools
from .materials import Wafer


def front_end_transmission(E, diamond_thickness=300e-6):
    cvd = Wafer(material="C", thickness=diamond_thickness, density=3.52)
    t = cvd.calc_transmission(E)
    return t


class Filter:
    def __init__(self, *wafers, **kwargs):
        """ wafers is a list each element is
            (Material, thickness) or (Material, thickness, density)
            If there is only one wafer there is no need to have a list of list
            an Instance can be defined as
            Filter("Si",10e-6)
            Filter(material="Si",thickness=10e-6)
            Filter( ("Si",10e-6) )
            Filter( ( ("Si",10e-6), ("Fe",5e-6) ) )
        """

        if len(wafers) > 0 and len(kwargs) > 0:
            raise ValueError(
                "Cannot define filter with positional and keyword arguments"
            )

        if len(kwargs) > 0:
            wafers = [Wafer(**kwargs)]
        else:

            # check if called as Wafer(mater,thick)
            if len(wafers) > 1:
                wafers = (wafers,)

            # check if called as Wafer( ((m1,t1),(m2,t2)) )
            if isinstance(wafers[0][0], (tuple, list)):
                wafers = wafers[0]

            wafers = [Wafer(*w) for w in wafers]

        self.wafers = wafers

    def calc_transmission(self, E):
        T = 1
        for wafer in self.wafers:
            T *= wafer.calc_transmission(E,cross_section_kind="total")
        return T

    def __str__(self):
        return "+".join([str(w) for w in self.wafers])

    def __repr__(self):
        return self.__str__()


class EmptySlot:
    def __init__(self):
        pass

    def calc_transmission(self, E):
        return 1

    def __repr__(self):
        return f"empty"

    def __str__(self):
        return f"Em"

def _complete_with_zeros(mylist,naxis=6):
    mylist = tuple(mylist) + (0,)*(naxis-len(mylist))
    return mylist

class Filters:
    def __init__(self, att_list):
        """ att_list has to be a list of list
            [ [ axis1_filter1, axis1_filter2, axis1_filter3, axis1_filter4],
              [ axis2_filter1, axis2_filter2, axis2_filter3] ] """
        self._att_list = att_list
        self.naxis = len(att_list)
        self._nfilters_per_axis = [len(axis) for axis in att_list]
        self._att = list(
            itertools.product(*[range(n) for n in self._nfilters_per_axis])
        )

        # select combinations that protect downbeam filters with upbeam ones
        # i.e. axis3 will only be used if axis1 and axis2 are not empty
        temp = []
        iloop = list(reversed(range(self.naxis)))
        for att in self._att:
            attenuators = [self._att_list[axis][filtnum] for axis,filtnum in enumerate(att)]
            # find most downbeam filter that is not empty
            axis = 0
            for i in iloop:
                if not isinstance(attenuators[i],EmptySlot):
                    last_non_empty_axis = i
                    break
            # verify that all previous filters are not empty
            if axis != 0 and all( [not isinstance(attenuators[i],EmptySlot) for i in range(last_non_empty_axis)] ):
                temp.append(att)

        self._att_protect_downbeam = temp
        self.CACHE = dict()
        self.CACHE_protect_downbeam = dict()




    def calc_transmission(self, E, filter_selection):
        """ filter_selection is list (0,2,3), each number is filter number in axis0,axis1,axis2 """
        T = 1
        for axis, filter_number in enumerate(filter_selection):
            T *= self._att_list[axis][filter_number].calc_transmission(E)
        return T

    def _calc_all_transmissions(self, E):
        if E not in self.CACHE:
            data = np.asarray([self.calc_transmission(E, filters) for filters in self._att])
            self.CACHE[E]=data
        return self.CACHE[E]

    def _calc_all_transmissions_protect_downbeam(self, E):
        if E not in self.CACHE_protect_downbeam:
            data = np.asarray([self.calc_transmission(E, filters) for filters in self._att_protect_downbeam])
            self.CACHE_protect_downbeam[E]=data
        return self.CACHE_protect_downbeam[E]

    def _show_axis(self, axis=0):
        filters = self._att_list[axis]
        s = f"axis{axis} |"
        for f in filters:
            s += str(f) + "|"
        return s

    def _show_combination(self, filter_selection):
        filters = [
            self._att_list[axis][num] for axis, num in enumerate(filter_selection)
        ]
        return "|".join([str(f) for f in filters])

    def __repr__(self):
        return "\n".join([self._show_axis(axis) for axis in range(self.naxis)])

    def calc_best_transmission(self, E, requested_transmission, verbose=False,use_protect_downbeam=False):
        """ E must be a float, can't be a vector """
        E = float(E)
        if use_protect_downbeam:
            t = self._calc_all_transmissions_protect_downbeam(E)
            best = np.argmin(np.abs(t - requested_transmission))
            best_combination = self._att_protect_downbeam[best]
        else:
            t = self._calc_all_transmissions(E)
            best = np.argmin(np.abs(t - requested_transmission))
            best_combination = self._att[best]
        t_1E = t[best]
        t_2E = self.calc_transmission(2 * E, best_combination)
        t_3E = self.calc_transmission(3 * E, best_combination)
        if verbose:
            print(f"Finding set for T={requested_transmission:.3g} @ {E:.3f} keV")
            print(f"best set is {best_combination}:")
            print(f"  {self._show_combination(best_combination)}")
            print(
                f"transmission @  E is {float(t[best]):.3g} (asked {requested_transmission:.3g})"
            )
            print(f"transmission @ 2E is {t_2E:.3g}")
            print(f"transmission @ 3E is {t_3E:.3g}")
        return DataStorage(
            bestset_key=best_combination,
            bestset=[self._att_list[i][f] for i,f in enumerate(best_combination)],
            transmission=t_1E,
            energy=E,
            transmission_requested=requested_transmission,
            t1E=t_1E,
            t2E=t_2E,
            t3E=t_3E,
        )


def test_filter_definition():
    f = Filter("C", 1e-6)
    print("This should be 1um of C:", str(f))
    f = Filter(("C", 2e-6))
    print("This should be 2um of C:", str(f))
    # f = Filter((("C",3e-6),))
    # print("This should be 3um of C:",str(f))
    f = Filter((("C", 3e-6), ("Si", 2e-6)))
    print("This should be 3um of C+2um of Si:", str(f))
    f = Filter(material="C", thickness=1e-6)
    print("This should be 1um of C (using kwargs):", str(f))


def test_filters():
    empty = EmptySlot()
    axis0 = [empty, Filter("C", 10e-6), Filter(material="C", thickness=20e-6)]
    axis1 = [
        empty,
        Filter(("Si", 10e-6)),
        Filter(("Si", 20e-6)),
        Filter(("Si", 20e-6)),
        Filter(material="Si", thickness=40e-6),
    ]
    axis2 = [
        empty,
        Filter(material="Fe", thickness=10e-6),
        Filter(material="Si", thickness=20e-6),
    ]

    axis3 = [
        empty,
        Filter(("Al", 50e-6), ("Si", 20e-6)),
        Filter(("Al", 150e-6), ("C", 200e-6)),
    ]

    filters = Filters([axis0, axis1, axis2, axis3])

    # test few cases

    #E = 8
    #for t in [0.01, 0.03, 0.1, 0.3, 0.85, 1]:
    #    filters.calc_best_transmission(E, t)
    return filters


axis = [[EmptySlot(), Filter("Si", 10e-6 * (2 ** i))] for i in range(0, 10)]
axis = [[EmptySlot(), Filter("Si", 10e-6 * (2 ** i))] for i in range(0, 3)]
power_filter = Filters(axis)


if __name__ == "__main__":
    test_filter_definition()
    test_filters()

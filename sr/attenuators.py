import numpy as np
import itertools
import time
from .materials import Wafer

try:
    from datastorage import DataStorage
except ImportError:
    DataStorage = dict

def front_end_transmission(E, diamond_thickness=300e-6):
    cvd = Wafer(name="diamond", thickness=diamond_thickness)
    t = cvd.calc_transmission(E)
    return t


class Filter:
    def __init__(self, *wafers, **kwargs):
        """ wafers is a list each element is
            (Material, thickness) or (Material, thickness, density)
            If there is only one wafer there is no need to have a list of list
            an Instance can be defined as
            Filter("Si",10e-6)
            Filter(name="Si",thickness=10e-6)
            Filter( ("Si",10e-6) )
            Filter( ( ("Si",10e-6), ("Fe",5e-6) ) )
            
            the Filter class also accepts optional keyword-only arguments
            f_move_in=move_in()
            f_move_out=move_out()
            f_is_in=is_in()
            get_E to get X-ray energy to use
        """
        self.move_in = kwargs.pop("f_move_in",None)
        self.move_out = kwargs.pop("f_move_out",None)
        self.is_in = kwargs.pop("f_is_in",None)
        self.get_E = kwargs.pop("get_E",None)

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
            if len(wafers) > 0 and isinstance(wafers[0][0], (tuple, list)):
                wafers = wafers[0]

            wafers = [Wafer(*w) for w in wafers]

        self.wafers = wafers

    def calc_transmission(self, E=None):
        if E is None: E=self.get_E()
        T = 1
        for wafer in self.wafers:
            T *= wafer.calc_transmission(E,cross_section_kind="total")
        return T

    def __str__(self):
        return "+".join([str(w) for w in self.wafers])

    def __repr__(self):
        return self.__str__()


class EmptySlot(Filter):
    def __init__(self,*args,**kwargs):
        super().__init__(**kwargs)

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
    def __init__(self, att_list,get_E=None):
        """ att_list has to be a list of list
            [ [ axis1_filter1, axis1_filter2, axis1_filter3, axis1_filter4],
              [ axis2_filter1, axis2_filter2, axis2_filter3] ] 
            get_E maybe passed to auto-get E (for integration in bliss)
        """
        self.get_E=get_E
        self._att_list = att_list
        if get_E is not None:
            for att in att_list:
                for pos in att: pos.get_E=get_E
        self.naxis = len(att_list)
        self._nfilters_per_axis = [len(axis) for axis in att_list]
        self._has_status = all( [all([f.is_in is not None for f in a]) for a in att_list] )
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
        self._CACHE = dict()
        self._CACHE_protect_downbeam = dict()

    def calc_transmission(self, E=None, filter_selection=None):
        """ 
            filter_selection is list (0,2,3), each number is filter number in axis0,axis1,axis2
            if E is None, it will try to get it from get_E function
            if filter_selection is None, it will try to get it from hardware
        """
        if E is None: E = self.get_E()
        if E is None: raise ValueError("Energy not provided or not able to get it")
        
        if filter_selection is None: filter_selection = self._status()
        if filter_selection is None: raise ValueError("Filter selection not provided or not able to get it")
       
        T = 1
        for axis, filter_number in enumerate(filter_selection):
            T *= self._att_list[axis][filter_number].calc_transmission(E)
        return T

    def _calc_all_transmissions(self, E):
        if E not in self._CACHE:
            data = np.asarray([self.calc_transmission(E, filters) for filters in self._att])
            self._CACHE[E]=data
        return self._CACHE[E]

    def _calc_all_transmissions_protect_downbeam(self, E):
        if E not in self._CACHE_protect_downbeam:
            data = np.asarray([self.calc_transmission(E, filters) for filters in self._att_protect_downbeam])
            self._CACHE_protect_downbeam[E]=data
        return self._CACHE_protect_downbeam[E]

    def _show_axis(self, axis=0):
        filters = self._att_list[axis]
        s = f"axis{axis} |"
        for f in filters:
            s += str(f) + "|"
        return s

    def _calc_transmission_info(self, E, filter_selection):
        """ E must be a float, can't be a vector """
        if E is None: E=self.get_E()
        E = float(E)
        t_1E = self.calc_transmission(    E, filter_selection)
        t_2E = self.calc_transmission(2 * E, filter_selection)
        t_3E = self.calc_transmission(3 * E, filter_selection)
        filters=[self._att_list[i][f] for i,f in enumerate(filter_selection)]
        info_str = self._show_combination(filter_selection)
        sum_mat = dict()
        filters_in = [filters[i] for i in range(self.naxis) if filter_selection[i]]
        for f in filters_in:
            # each filter might be made of more than one kind of material
            for w in f.wafers:
                if w.formula not in sum_mat: sum_mat[w.formula] = 0
                sum_mat[w.formula] += w.thickness
        txt = []
        for k,l in sum_mat.items():
            txt.append(f"{k}, {l*1e3} mm")
        if len(txt) == 0:
            txt = ""
        else:
            txt = "; ".join(txt)


        return DataStorage(
            filter_selection=filter_selection,
            filters=filters,
            filters_string=info_str,
            total_lengths = txt,
            transmission=t_1E,
            energy=E,
            t1E=t_1E,
            t2E=t_2E,
            t3E=t_3E,
        )


    def _show_combination(self, filter_selection):
        filters = [
            self._att_list[axis][num] for axis, num in enumerate(filter_selection)
        ]
        return "|".join([str(f) for f in filters])

    def __repr__(self):
        return "\n".join([self._show_axis(axis) for axis in range(self.naxis)])

    def calc_best_transmission(self, requested_transmission, E=None,verbose=False,use_protect_downbeam=False,move=False,wait_move=0.1):
        """ E must be a float, can't be a vector """
        if E is None: E=self.get_E()
        E = float(E)
        if use_protect_downbeam:
            t = self._calc_all_transmissions_protect_downbeam(E)
            best = np.argmin(np.abs(t - requested_transmission))
            best_combination = self._att_protect_downbeam[best]
        else:
            t = self._calc_all_transmissions(E)
            best = np.argmin(np.abs(t - requested_transmission))
            best_combination = self._att[best]
        ret = self._calc_transmission_info(E,best_combination)
        ret["requested_transmission"] = requested_transmission
        t_1E = ret["t1E"]
        t_2E = ret["t2E"]
        t_3E = ret["t3E"]
        if verbose:
            print(f"Finding set for T={requested_transmission:.3g} @ {E:.3f} keV")
            print(f"best set is {best_combination}:")
            print(f"  {self._show_combination(best_combination)}")
            print(
                f"transmission @  E is {float(t_1E):.3g} (asked {requested_transmission:.3g})"
            )
            print(f"transmission @ 2E is {t_2E:.3g}")
            print(f"transmission @ 3E is {t_3E:.3g}")
        if move:
            for iaxis,chosen_filter in enumerate(best_combination):
                f = self._att_list[iaxis][chosen_filter]
                if f.move_in is None: raise ValueError("No moving function defined!")
                f.move_in()
                if isinstance(wait_move,(int,float)) and wait_move>0:
                    time.sleep(wait_move)
        return ret

    def _status(self):
        if not self._has_status: return None
        status = []
        for a in self._att_list:
            for f in a:
                if isinstance(f,EmptySlot): continue
                if f.is_in():
                    status.append(1)
                else:
                    status.append(0)
        return status

    def status(self,E=None):
        if E is None: E = self.get_E()
        status = self._status()
        return self._calc_transmission_info(E,status)
                
        


def test_filter_definition():
    f = Filter("C", 1e-6)
    print("This should be 1um of C:", str(f))
    f = Filter(("C", 2e-6))
    print("This should be 2um of C:", str(f))
    # f = Filter((("C",3e-6),))
    # print("This should be 3um of C:",str(f))
    f = Filter((("C", 3e-6), ("Si", 2e-6)))
    print("This should be 3um of C+2um of Si:", str(f))
    f = Filter(name="C", thickness=1e-6)
    print("This should be 1um of C (using kwargs):", str(f))


def test_filters():
    empty = EmptySlot()
    axis0 = [empty, Filter("C", 10e-6), Filter(name="C", thickness=20e-6)]
    axis1 = [
        empty,
        Filter(("Si", 10e-6)),
        Filter(("Si", 20e-6)),
        Filter(("Si", 20e-6)),
        Filter(name="Si", thickness=40e-6),
    ]
    axis2 = [
        empty,
        Filter(name="Fe", thickness=10e-6),
        Filter(name="Si", thickness=20e-6),
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
    #    filters.calc_best_transmission(t,E=E)
    return filters


#axis = [[EmptySlot(), Filter("Si", 10e-6 * (2 ** i))] for i in range(0, 10)]
#axis = [[EmptySlot(), Filter("Si", 10e-6 * (2 ** i))] for i in range(0, 3)]
#power_filter = Filters(axis)


if __name__ == "__main__":
    test_filter_definition()
    test_filters()

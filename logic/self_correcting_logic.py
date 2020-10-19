from typing import List,Tuple,Dict,Any
import numpy as np

class Self_Correcting_logic:
    """ Define Self_Correcting logic.
    Args: None
    Attributes:
        num_predicate: constant int, 
        num_formula: constant int, 
        BEFORE: constant int,
            denote 'before' relation in time template
        EQUAL: constant int, 
            denote 'after' relation in time template
        R_matrix: 0-1 np.array with shape(num_predicate, num_formula). 
            R_matrix[predicate][formula]=1 means this formula involves this predicate.
            R_matrix is defined by predicate_indices.
        logic_rule_list: list(type=int) list, 
            experts defined logic formulas(rules).
                
        time_template_list: np.array(type=int) list, 
            experts defined time templates (temporal relations).
        formula_ind_to_rule_ind: int list, 
            experts defined mapping from formula, to logic formula(rule) and time templates 
            formula_ind_to_rule_ind[formula_ind] = index of logic formula(rule) or time templates 
        
    Methods:
        Public methods: None
        Private methods:
            __get_R_matrix:
            __get_logic_formulas: Set expert logic formulas via hard coding.
    """
    def __init__(self):
        self.num_predicate = 2 # num_predicate is same as num_node
        self.num_formula = 1

        # use 100 to indicate BEFORE, use 1000 to indicate EQUAL
        self.BEFORE = 100
        self.EQUAL = 1000
        
        self.R_matrix = self.__get_R_matrix()
        self.logic_rule_list, self.time_template_list, self.formula_ind_to_rule_ind = self.__get_logic_formulas()

        # For Hawkes (and other AutoRegression-like PPs), the triggering predicate and triggered predicate are same.
        # To fit our logic formulation, we thus copy(mapping) the target to form a virtual predicate.
        # predicate_mapping[i] = j, means preidcate i is real, predicate j is virtual.
        # predicate[1] (i.e. B) is target(real) predicate
        # predicate[0] (i.e. A) is copied(virtual) predicate
        self.predicate_mapping = {1:[0]}


    def __get_R_matrix(self) -> np.ndarray:
        """define R matrix.
        Args: None
        Temporal variables:
            predicate_indices: list of int lists, 
                predicate_indices[f] = pred_inds
                means formula f involves predicates specified by pred_inds.
        Returns:
            R_matrix: 0-1 np.ndarray with shape(num_predicate, num_formula). 
                R_matrix[predicate][formula]=1 means this formula involves this predicate.
        """
        #relation matrix
        R_matrix = np.zeros((self.num_predicate, self.num_formula), dtype=int)

        predicate_indices = [[0,1],]

        for idx, predicates in enumerate(predicate_indices):
            R_matrix[predicates,idx] = 1
        
        return R_matrix

    def __get_logic_formulas(self) -> Tuple[List]:
        """Set expert logic formulas via hard coding.
        Args: None
        Returns: 
            (detailed info refers to class doc)
            logic_rule_list: np.array list, 
                [logic_f0,logic_f1,..]
            time_template_list: np.array list, 
                [time_template_f0(),..,]
            formula_ind_to_rule_ind: int list, 
                formula_ind_to_rule_ind[formula_ind] = index of logic formula(rule) or time templates 
        """
        
        #### Begin logic_formula  ####
        # Self-Correcting, in Eq(17)
        # f0(A,B): If A, then (not B)
        # f0 = (not A) or (not B)
        logic_f0 = np.array([0,0])
        #### End logic_formula  ####
        

        #### Begin time_template  ####
        # Self-Correcting, in Eq(17)
        # f0(A,B): A -> (not B) and A before B
        # only allow time relations between A and others, thus relation is vector instead of matrix.
        time_template_f0 = np.zeros((2,),dtype=int)
        time_template_f0[1] = self.BEFORE
        #### End time_template  ####

        formula_ind_to_rule_ind = [0] * self.num_formula

        for i in range(len(formula_ind_to_rule_ind)):
            formula_ind_to_rule_ind[i] = i
        
        logic_rule_list = [logic_f0,]
        time_template_list = [time_template_f0,]

        return logic_rule_list, time_template_list, formula_ind_to_rule_ind

if __name__ == "__main__":
    logic = Self_Correcting_logic()
    print(logic.formula_ind_to_rule_ind)
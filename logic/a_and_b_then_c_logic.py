from typing import List,Tuple,Dict,Any
import numpy as np

class A_And_B_Then_C_logic:
    """
    1 Rule: A and B -> C, where C is target, A,B are indep preds.
    Annotations are same to ./logic/hawkes_logic.py, thus here we omit them.
    """
    def __init__(self):
        self.num_predicate = 3 # num_predicate is same as num_node
        self.num_formula = 1
        self.BEFORE = 100
        self.EQUAL = 1000
        
        self.R_matrix = self.__get_R_matrix()
        self.logic_rule_list, self.time_template_list, self.formula_ind_to_rule_ind, self.predicate_mapping, self.independent_predicate, self.is_duration_pred = self.__get_logic_formulas()

    def __get_R_matrix(self) -> np.ndarray:
        #relation matrix
        R_matrix = np.zeros((self.num_predicate, self.num_formula), dtype=int)
        predicate_indices = [[0,1,2]]
        # predicate_indices[i] corresponds to formula[i]
        for idx, predicates in enumerate(predicate_indices):
            R_matrix[predicates,idx] = 1
        return R_matrix

    def __get_logic_formulas(self) -> Tuple[List]:
        logic_rule_list = list()
        time_template_list = list()
        #### Begin logic_formula  ####
        # f0(A,B,C): If A and B, then C : Not A OR Not B OR C
        logic_f0 = np.array([0,0,1])
        logic_rule_list.append(logic_f0)
      
        #### Begin time_template  ####
        # f0(A,B,C):  A and B -> C  and A before B and A before C. 
        time_template_f0 = np.zeros((3,),dtype=int)
        time_template_f0[1] = self.BEFORE
        time_template_f0[2] = self.BEFORE
        time_template_list.append(time_template_f0)

        #### Begin predicate mapping
        predicate_mapping = {}

        #### Begin independent predicate
        independent_predicate = [0,1]

        #### Begin is_duration_pred
        # is_duration_pred[i] = True means pred i may keeps state in some duration.
        # otherwise, pred i keep a certain state in all time, and only switch to the other state in infinite-small time.
        is_duration_pred = [0,0,0]

        formula_ind_to_rule_ind = [0] * self.num_formula

        for i in range(len(formula_ind_to_rule_ind)):
            formula_ind_to_rule_ind[i] = i
        
        return logic_rule_list, time_template_list, formula_ind_to_rule_ind, predicate_mapping, independent_predicate, is_duration_pred

if __name__ == "__main__":
    logic = A_Then_B_logic()
    print(logic.formula_ind_to_rule_ind)
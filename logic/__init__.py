import sys
sys.path.extend(['./','../'])
import itertools
from typing import List,Tuple,Dict,Any

import numpy as np

#from logic.mimic_logic import Mimic_logic
from logic.hawkes_logic import Hawkes_logic
from logic.self_correcting_logic import Self_Correcting_logic
from logic.a_then_b_logic import A_Then_B_logic

class Logic:
    """Provide expert defiend logic rule calculations.
    Args:
        args: argparse object,
            use args.dataset_name to determine which set of expert rules to use.
    Attributes:
        logic: object,
            containing hard coding logic rules. 
    Methods:
        public: 
            get_template
        private:
            get_predicate_ind
            get_formula_ind
            get_time_template
            get_logic_rule
    """
    def __init__(self,args):
        self.args = args
        if args.dataset_name == "synthetic":
            if args.synthetic_logic_name == "hawkes":
                self.logic = Hawkes_logic()
            elif args.synthetic_logic_name == "self_correcting":
                self.logic = Self_Correcting_logic()
            elif args.synthetic_logic_name == "a_then_b":
                self.logic = A_Then_B_logic()
            else:
                raise ValueError
        else:
            raise ValueError

    
    def get_template(self, target_predicate_ind:int) -> Dict[int, Dict[str,Any]]:
        """calculate formula effect and store to template.
        Template is used to collect evidence from history of neighbors of the target_predicates.
        Args:
            target_predicate_ind: int
                index of one predicate that needs template
        Returns:
            template: dict of dicts
                template[formula_ind] = template infos of this formula
                template[formula_ind]['neighbor_combination'] = the neighbor value combination that makes formula effect non-zero.                
        """
        template = dict()

        formula_ind_list = self.get_formula_ind(predicate_ind=target_predicate_ind)
        for formula_ind in formula_ind_list:
            # neighbor = predicates in same formula
            time_template = self.get_time_template(formula_ind=formula_ind)
            predicate_ind = self.get_predicate_ind(formula_ind=formula_ind)
            predicate_ind.sort()

            neighbor_ind = predicate_ind.copy()
            neighbor_ind.remove(target_predicate_ind)
            
            target_ind_in_predicate = predicate_ind.index(target_predicate_ind)
            neighbor_ind_in_predicate = list(range(len(predicate_ind)))
            neighbor_ind_in_predicate.remove(target_ind_in_predicate)  # no effect from own

            logic_rule = self.get_logic_rule(formula_ind = formula_ind)
            neighbor_combination = logic_rule.copy()
            neighbor_combination = np.delete(neighbor_combination, target_ind_in_predicate)
            neighbor_combination = 1 - neighbor_combination  
            #neighbor_combination = 1 - logic_rule, since only when all neighbors are unsatified will the formula effect be non-zero.

            # begin calculation of formula effect, i.e. Eq(8) in paper.
            # formula_effect in this code denotes "formula effect with target predicte = 0"
            target_sign = logic_rule[target_ind_in_predicate]
            if target_sign == 0:
                # target_sign matches target predicte 
                # --> it tends to keep state 
                # --> intensity should be small 
                # --> formula effect = -1
                formula_effect = -1
            else: #target_sign == 1
                formula_effect = 1

            template[formula_ind] = dict()
            template[formula_ind]['neighbor_ind'] = neighbor_ind
            template[formula_ind]['neighbor_combination'] = neighbor_combination
            template[formula_ind]['formula_effect'] = [formula_effect, -formula_effect]
            #['formula_effect'][0] means formula_effect when cur state == 0
            #['formula_effect'][1] means formula_effect when cur state == 1
            template[formula_ind]['time_template'] = time_template
            template[formula_ind]['target_ind_in_predicate'] = target_ind_in_predicate
        return template
    
    def get_predicate_ind(self, formula_ind:int)->List[int]:
        """returns predicate indices related to formula_ind.
        """
        predicate_ind = np.nonzero(self.logic.R_matrix[:, formula_ind])
        predicate_ind = list(predicate_ind[0])
        return predicate_ind
    
    def get_formula_ind(self, predicate_ind:int)->List[int]:
        """returns formula indices related to predicate_ind.
        """
        formula_ind = np.nonzero(self.logic.R_matrix[predicate_ind, :])
        formula_ind = list(formula_ind[0])
        return formula_ind
    
    def get_time_template(self, formula_ind:int) ->np.ndarray:
        """returns time_template of given formula.
        """
        rule_ind = self.logic.formula_ind_to_rule_ind[formula_ind]
        time_template = self.logic.time_template_list[rule_ind] 
        return time_template
    
    def get_logic_rule(self, formula_ind:int) -> np.array:
        """returns logic rule coressponding to formula_ind.
        """
        rule_ind = self.logic.formula_ind_to_rule_ind[formula_ind]
        logic_rule = self.logic.logic_rule_list[rule_ind]
        return logic_rule
    

if __name__ == "__main__":
    from utils.args import get_args
    
    args = get_args()
    args.dataset_name = "synthetic"
    args.synthetic_logic_name = "hawkes"
    l = Logic(args)
    print(l.get_template(0))
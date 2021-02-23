import numpy as np
import itertools

##################################################################
#np.random.seed(1)
class Logic_Model_Generator:
    def __init__(self):

        ### the following parameters are used to manually define the logic rules
        self.num_predicate = 0  # num_predicate is same as num_node
        self.num_formula = 0
        self.BEFORE = 'BEFORE'
        self.EQUAL = 'EQUAL'
        self.AFTER = 'AFTER'
        self.Time_tolerance = 0.1
        self.body_predicate_set = list() # the index set of all body predicates
        self.head_predicate_set = list() # the index set of all head predicates
        self.decay_rate = 1 # decay kernel
        self.predicate_notation = list()

        self.body_intensity= dict()
        self.logic_template = dict()
        
    def print_rule(self):
        for head_predicate_idx, rules in self.logic_template.items():
            print("Head = {}, base = {:.4f}".format(self.predicate_notation[head_predicate_idx], self.model_parameter[head_predicate_idx]['base']))
            for rule_id, rule in rules.items():
                rule_str = "Rule{}: ".format(rule_id)
                rule_str += self.get_rule_str(rule, head_predicate_idx)
                weight = self.model_parameter[head_predicate_idx][rule_id]['weight']
                rule_str += ", weight={:.4f}".format(weight)
                print(rule_str)
    
    def get_rule_str(self, rule, head_predicate_idx):
        body_predicate_idx = rule['body_predicate_idx']
        body_predicate_sign = rule['body_predicate_sign']
        head_predicate_sign = rule['head_predicate_sign'][0]
        temporal_relation_idx = rule['temporal_relation_idx']
        temporal_relation_type = rule['temporal_relation_type']
        rule_str = ""
        negative_predicate_list = list()
        for i in range(len(body_predicate_idx)):
            if body_predicate_sign[i] == 0:
                rule_str += "Not "
                negative_predicate_list.append(body_predicate_idx[i])
            rule_str += self.predicate_notation[body_predicate_idx[i]]
            if i <= len(body_predicate_idx) - 2:
                rule_str += " ^ "
        rule_str += " --> "
        if head_predicate_sign == 0:
            rule_str += "Not "
            negative_predicate_list.append(head_predicate_idx)
        rule_str += self.predicate_notation[head_predicate_idx]
        rule_str += " , "

        for i in range(len(temporal_relation_idx)):
            if temporal_relation_idx[i][0] in negative_predicate_list:
                rule_str += "Not "
            rule_str += self.predicate_notation[temporal_relation_idx[i][0]]
            rule_str += " {} ".format(temporal_relation_type[i])
            if temporal_relation_idx[i][1] in negative_predicate_list:
                rule_str += "Not "
            rule_str += self.predicate_notation[temporal_relation_idx[i][1]]
            if i <= len(temporal_relation_idx) - 2:
                rule_str += " ^ "   
        return rule_str    

    def intensity(self, cur_time, head_predicate_idx, history):
        feature_formula = []
        weight_formula = []
        effect_formula = []
        for formula_idx in list(self.logic_template[head_predicate_idx].keys()):
            weight_formula.append(self.model_parameter[head_predicate_idx][formula_idx]['weight'])
            feature_formula.append(self.get_feature(cur_time=cur_time, head_predicate_idx=head_predicate_idx,
                                                    history=history, template=self.logic_template[head_predicate_idx][formula_idx]))
            effect_formula.append(self.get_formula_effect(cur_time=cur_time, head_predicate_idx=head_predicate_idx,
                                                       history=history, template=self.logic_template[head_predicate_idx][formula_idx]))
        intensity =  np.array(weight_formula) * np.array(feature_formula) * np.array(effect_formula)

        intensity = self.model_parameter[head_predicate_idx]['base'] + np.sum(intensity)
        intensity = np.exp(intensity)
        return intensity

    def get_feature(self, cur_time, head_predicate_idx, history, template):
        transition_time_dic = {}
        feature = 0
        for idx, body_predicate_idx in enumerate(template['body_predicate_idx']):
            transition_time = np.array(history[body_predicate_idx]['time'])
            transition_state = np.array(history[body_predicate_idx]['state'])
            mask = (transition_time <= cur_time) * (transition_state == template['body_predicate_sign'][idx])
            transition_time_dic[body_predicate_idx] = transition_time[mask]
        transition_time_dic[head_predicate_idx] = [cur_time]
        ### get weights
        # compute features whenever any item of the transition_item_dic is nonempty
        history_transition_len = [len(i) for i in transition_time_dic.values()]
        if min(history_transition_len) > 0:
            # need to compute feature using logic rules
            time_combination = np.array(list(itertools.product(*transition_time_dic.values())))
            time_combination_dic = {}
            for i, idx in enumerate(list(transition_time_dic.keys())):
                time_combination_dic[idx] = time_combination[:, i]
            temporal_kernel = np.ones(len(time_combination))
            for idx, temporal_relation_idx in enumerate(template['temporal_relation_idx']):
                time_difference = time_combination_dic[temporal_relation_idx[0]] - time_combination_dic[temporal_relation_idx[1]]
                if template['temporal_relation_type'][idx] == 'BEFORE':
                    temporal_kernel *= (time_difference < - self.Time_tolerance) * np.exp(-self.decay_rate * (cur_time - time_combination_dic[temporal_relation_idx[0]]))
                if template['temporal_relation_type'][idx] == 'EQUAL':
                    temporal_kernel *= (abs(time_difference) <= self.Time_tolerance) * np.exp(-self.decay_rate *(cur_time - time_combination_dic[temporal_relation_idx[0]]))
                if template['temporal_relation_type'][idx] == 'AFTER':
                    temporal_kernel *= (time_difference > self.Time_tolerance) * np.exp(-self.decay_rate * (cur_time - time_combination_dic[temporal_relation_idx[1]]))
            feature = np.sum(temporal_kernel)
        return feature

    def get_formula_effect(self, cur_time, head_predicate_idx, history, template):
        ## Note this part is very important!! For generator, this should be np.sum(cur_time > head_transition_time) - 1
        ## Since at the transition times, choose the intensity function right before the transition time
        head_transition_time = np.array(history[head_predicate_idx]['time'])
        head_transition_state = np.array(history[head_predicate_idx]['state'])
        if len(head_transition_time) == 0:
            cur_state = 0
            counter_state = 1 - cur_state
        else:
            idx = np.sum(cur_time > head_transition_time) - 1
            cur_state = head_transition_state[idx]
            counter_state = 1 - cur_state

        if counter_state == template['head_predicate_sign']:
            formula_effect = 1
        else:
            formula_effect = -1
        return formula_effect

    def generate_data(self, num_sample, time_horizon):
        print("Generate {} samples".format(num_sample))
        print("with following rules:")
        self.print_rule()
        for body_idx in self.body_predicate_set:
            print("Intensity {} is {}".format(self.predicate_notation[body_idx], self.body_intensity[body_idx]))
        
        data={}

        for sample_ID in np.arange(0, num_sample, 1):
            data[sample_ID] = {}
            # initialize data
            for predicate_idx in np.arange(0, self.num_predicate, 1):
                data[sample_ID][predicate_idx] = {}
                data[sample_ID][predicate_idx]['time'] = []
                data[sample_ID][predicate_idx]['state'] = []

            # generate data (body predicates)
            for body_predicate_idx in self.body_predicate_set:
                t = 0
                while t < time_horizon:
                    time_to_event = np.random.exponential(scale=1.0 / self.body_intensity[body_predicate_idx])
                    t += time_to_event
                    if t >= time_horizon:
                        break
                    data[sample_ID][body_predicate_idx]['time'].append(t)
                    if len(data[sample_ID][body_predicate_idx]['state'])>0:
                        cur_state = 1 - data[sample_ID][body_predicate_idx]['state'][-1]
                    else:
                        cur_state = 1
                    data[sample_ID][body_predicate_idx]['state'].append(cur_state)


            for head_predicate_idx in self.head_predicate_set:
                data[sample_ID][head_predicate_idx] = {}
                data[sample_ID][head_predicate_idx]['time'] = [0,]
                data[sample_ID][head_predicate_idx]['state'] = [0,]

                # obtain the maximal intensity
                intensity_potential = []
                for t in np.arange(0, time_horizon, 0.1):
                    intensity_potential.append(self.intensity(t, head_predicate_idx, data[sample_ID]))
                intensity_max = max(intensity_potential)
                #print(intensity_max)
                # generate events via accept and reject
                t = 0
                while t < time_horizon:
                    time_to_event = np.random.exponential(scale=1.0/intensity_max)
                    t = t + time_to_event
                    if t >= time_horizon:
                        break
                    ratio = min(self.intensity(t, head_predicate_idx, data[sample_ID]) / intensity_max, 1)
                    flag = np.random.binomial(1, ratio)     # if flag = 1, accept, if flag = 0, regenerate
                    if flag == 1: # accept
                        data[sample_ID][head_predicate_idx]['time'].append(t)
                        cur_state = 1 - data[sample_ID][head_predicate_idx]['state'][-1]
                        data[sample_ID][head_predicate_idx]['state'].append(cur_state)
                

        return data


def get_logic_model_0():
    model = Logic_Model_Generator()
    model.body_intensity= {0:0.5, 1:1.0, 2:0.7, 3:0.3}
    model.body_predicate_set = [0,1,2,3]
    model.head_predicate_set = [4]
    model.predicate_notation = ['A','B','C','D','E']
    model.num_predicate = len(model.body_predicate_set)
    
    # define weights and base
    model.model_parameter = dict()
    head_predicate_idx = 4
    model.model_parameter[head_predicate_idx] = {'base':0.0}
    weights = [0.3, 0.4, 0.5, 0.6, 0.7]
    model.num_formula = len(weights)
    for idx, w in enumerate(weights):
        model.model_parameter[head_predicate_idx][idx] = {'weight': w}
   
    # encode rule information
    logic_template = {}

    head_predicate_idx = 4
    logic_template[head_predicate_idx] = {} 

    # A->E  A Before E.
    formula_idx = 0
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0, 4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE]

    # A ^ B --> E,  A Before E, B Equal E.
    formula_idx = 1
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0,1]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0,4], [1,4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE,model.EQUAL]

    # B Equal E
    formula_idx = 2
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1,4] ]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.EQUAL]

    # D --> E,  D BEFORE E.
    formula_idx = 3
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [3]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[3, 4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE]

    # Not C --> E, Not C Before E
    formula_idx = 4
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [2]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [0]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[2,4] ]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE]

    model.logic_template = logic_template
    
    return model
    


if __name__ == "__main__":
    logic_model_generator = get_logic_model_0()
    data = logic_model_generator.generate_data(num_sample=1000, time_horizon=10)
    np.save('data-0.npy', data)

def init():
    self.head_predicate_idx = 1
    self.train_dataset = None
    self.test_datset = None
    self.num_formula = 0
    #claim parameters 
    self.model_parameter = {}
    # init base parameters
    for idx in head_predicate_idx:
        self.model_parameter[idx] = {}
        self.model_parameter[idx]['base'] = torch.autograd.Variable((torch.ones(1) * -0.2).double(), requires_grad=True)

def add_rule(self, new_rule_table, idx):
    self.logic_template[head_predicate_idx][self.num_formula] = {}
    self.logic_template[head_predicate_idx][self.num_formula]['body_predicate_idx'] = new_rule_table[head_predicate_idx]['body_predicate_idx'][idx]
    self.logic_template[head_predicate_idx][self.num_formula]['body_predicate_sign'] = new_rule_table[head_predicate_idx]['body_predicate_sign'][idx]
    self.logic_template[head_predicate_idx][self.num_formula]['head_predicate_sign'] = new_rule_table[head_predicate_idx]['head_predicate_sign'][idx]
    self.logic_template[head_predicate_idx][self.num_formula]['temporal_relation_idx'] = new_rule_table[head_predicate_idx]['temporal_relation_idx'][idx]
    self.logic_template[head_predicate_idx][self.num_formula]['temporal_relation_type'] = new_rule_table[head_predicate_idx]['temporal_relation_type'][idx]

    #init weight 
    self.model_parameter[head_predicate_idx][self.num_formula] = {}
    self.model_parameter[head_predicate_idx][self.num_formula]['weight'] = torch.autograd.Variable((torch.ones(1) * 0.01).double(), requires_grad=True)

    self.num_formula += 1


def initialize_rule_set():
    rand_rule = generate_rand_rule()
    self.add_rule(rand_rule)
    # update model parameter
    self.optimize_log_likelihood()
    



def generate_rule_via_column_generation(extend_old_rules = False):
    new_rule_table = init_new_rule_table()
    
    if extend_old_rules: # extend existing rules.
        rules = ..
    else:#create new rules
        rules = ..

    for idx, rule in enumerate(rules):
        new_rule_table.append(rule)
        new_rule_table[idx]['gain'] = optimize_log_likelihood_gradient(new_rule_table, idx)
    
    idx = np.argmax(new_rule_table[head_predicate_idx]['performance_gain'])
    best_gain = new_rule_table[head_predicate_idx]['performance_gain'][idx]
    if best_gain > 0:
        add_rule(self, new_rule_table, idx)
        # update model parameter
        self.optimize_log_likelihood()
        return True 
    else:
        return False

    

def algorithm(self):
    initialize_rule_set()
    while first_level: #generate rules with len=1
        generate_rule_via_column_generation()
    
    while second_level:#generate rules with len>1, by extending existing rules.
        add_one_predicate_to_existing_rule()
    

    
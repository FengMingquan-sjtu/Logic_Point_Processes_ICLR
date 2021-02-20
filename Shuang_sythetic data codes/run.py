from logic_learning_2 import Logic_Learning_Model
from generate_synthetic_data import Logic_Model_Generator
import numpy as np
if __name__ == "__main__":
    num_sample = 20000 #dataset size
    T_max = 10
    logic_model_generator = Logic_Model_Generator()
    data = logic_model_generator.generate_data(num_sample=num_sample, time_horizon=T_max)
    np.save('data.npy', data)

    print("++++++ Generated data, start train. ++++++", flush=1)

    head_predicate_idx = [4]
    model = Logic_Learning_Model(head_predicate_idx = head_predicate_idx)
    model.predicate_set= [0, 1, 2, 3, 4] # the set of all meaningful predicates
    model.predicate_notation = ['A', 'B', 'C', 'D', 'E']
    
    dataset = np.load('data.npy', allow_pickle='TRUE').item()
    
    small_dataset = {i:dataset[i] for i in range(num_sample)}
    model.batch_size_cp = num_sample  # sample used by cp
    
    model.search_algorithm(head_predicate_idx[0], small_dataset, T_max)

    with open("model.pkl",'wb') as f:
        pickle.dump(model, f)
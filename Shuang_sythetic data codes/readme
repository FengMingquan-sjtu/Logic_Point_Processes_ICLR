# Temporal_Logic_Point_Processes
codes of paper "Temporal Logic Point Processes", ICML 2020

Synthetic data demo:

function: "generate_synthetic_data.py" -- it is used to generated synthetic event data given prespecified rules

function: " logic_template.py" --- it is used to learn model parameters given the prespecified logic templates and the input event data.

Logic templates introduced here look like: 

# For example, consider three rules:

    # A and B and Equal(A,B), and Before(A, D), then D;
    
    # C and Before(C, Not D), then  Not D
    
    # D Then  E, and Equal(D, E)

# For the transity intensity, it has the general form:

intensity = exp(base + weight_1 * feature due to logic_1 + weight_2 * feature due to logic_2 + ....)

Each head predicate will have an intensity to capture its dynamics. The body predicates define the evidence to construct the features. The weights indicates the importance of each logic rules.

# For the base intensity

First, note that the "base" term in our model is learnable (identifiable)

Second, tote that in this simple demo example, we assume the base transition intensity term for 0 --> 1 and 1 --> 0 are the same (just for simplicity of demonstration). 
In practice, the base transition intensity of for 0 --> 1 and 1 --> 0 can be quite different. We can just introduce more base variables to indicate this.

# For the logic rule weights
We have probability simplex type of constraints for the weights. 

For example, suppose for one specific head predicates, there are two temporal logic rules associated with this head predicate. Then the head predicate has transition intensity as

intensity = exp(base + weight_1 * feature due to logic_1 + weight_2 * feature due to logic_2 )

We add constraints such as 

weight_1 + weight_2 = 1

weight_1 >= 0

weight_2 >= 0

for the weights. Then these weights are learnable (identifiable) and can be interpreted as the importance of the rules. 
